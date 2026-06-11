"""
Dashboard API endpoints
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.web.auth import verify_credentials_or_cookie as verify_credentials
from src.web.state import get_dashboard_stats, get_signals_history, get_paper_positions

router = APIRouter()

# Base path for strategy data
DATA_BASE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
SCHEDULER_STATE_FILE = os.path.join(DATA_BASE_PATH, 'adaptive_hybrid', 'scheduler_state.json')
LIGHT_CHECK_STATE_FILE = os.path.join(DATA_BASE_PATH, 'adaptive_hybrid', 'light_check_state.json')
INITIAL_BALANCE = 500.0


def _get_strategy_folder() -> str:
    """Get the correct data folder for the active strategy."""
    from src.config import ACTIVE_STRATEGY
    folder_map = {
        'sniper': 'sniper',
        'adaptive_hybrid': 'adaptive_hybrid',
        'hybrid': 'sniper',  # hybrid uses sniper's infrastructure
    }
    return folder_map.get(ACTIVE_STRATEGY, 'ramf')


def _get_paper_trades_csv() -> str:
    """Get the correct paper trades CSV path based on active strategy."""
    return os.path.join(DATA_BASE_PATH, _get_strategy_folder(), 'paper_trades.csv')


def _get_leverage() -> int:
    """Get the leverage for the active strategy."""
    from src.config import ACTIVE_STRATEGY
    if ACTIVE_STRATEGY == 'sniper':
        from src.config import SNIPER_LEVERAGE
        return SNIPER_LEVERAGE
    elif ACTIVE_STRATEGY == 'adaptive_hybrid':
        from src.config import ADAPTIVE_HYBRID_LEVERAGE
        return ADAPTIVE_HYBRID_LEVERAGE
    else:
        from src.config import RAMF_LEVERAGE
        return RAMF_LEVERAGE


def _trade_pnl_series(df: pd.DataFrame) -> pd.Series:
    """Per-trade realized PnL matching the strategies' real paper balance.

    Conventions verified on prod CSVs (2026-06-10):
    - every strategy logs `pnl` net of exit fee but EXCLUDING the entry fee
      (the entry fee is deducted from the balance at open)
    - adaptive_hybrid scale-outs land in `partial_pnl_realized`; its
      closed_trades.csv also has `total_pnl` = pnl + partial_pnl_realized

    Real PnL = total_pnl (or pnl + partial_pnl_realized) - entry_fee.
    """
    if 'total_pnl' in df.columns:
        pnl = pd.to_numeric(df['total_pnl'], errors='coerce').fillna(0.0)
    else:
        if 'pnl' in df.columns:
            pnl = pd.to_numeric(df['pnl'], errors='coerce').fillna(0.0)
        else:
            pnl = pd.Series(0.0, index=df.index)
        if 'partial_pnl_realized' in df.columns:
            pnl = pnl + pd.to_numeric(df['partial_pnl_realized'], errors='coerce').fillna(0.0)
    if 'entry_fee' in df.columns:
        pnl = pnl - pd.to_numeric(df['entry_fee'], errors='coerce').fillna(0.0)
    return pnl


def _get_stats_from_csv() -> Dict:
    """Calculate stats from paper_trades.csv file with real-time unrealized PnL."""
    leverage = _get_leverage()
    PAPER_TRADES_CSV = _get_paper_trades_csv()

    stats = {
        "balance": INITIAL_BALANCE,
        "daily_pnl": 0.0,
        "total_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "realized_pnl": 0.0,
        "open_positions": 0,
        "used_margin": 0.0,
        "total_exposure": 0.0,
        "effective_leverage": 0.0,
        "available_margin": INITIAL_BALANCE,
    }

    if not os.path.exists(PAPER_TRADES_CSV):
        return stats

    try:
        df = pd.read_csv(PAPER_TRADES_CSV)
        if df.empty:
            return stats

        # Count open positions and calculate used margin
        open_positions = df[df['status'] == 'OPEN']
        stats["open_positions"] = len(open_positions)

        # Calculate unrealized PnL from open positions
        unrealized_pnl = 0.0
        if not open_positions.empty and 'position_size' in open_positions.columns:
            # Margin = position_size / leverage, Exposure = sum of notional
            stats["total_exposure"] = round(
                open_positions['position_size'].sum(), 2
            )
            stats["used_margin"] = round(
                (open_positions['position_size'] / leverage).sum(), 2
            )

            # Get market data provider singleton (uses 30s cache)
            provider = None
            try:
                from src.data_providers.market_data import get_market_data_provider
                provider = get_market_data_provider()
            except Exception:
                pass

            # Calculate unrealized PnL for each open position
            for _, row in open_positions.iterrows():
                symbol = row.get('symbol', '')
                entry_price = float(row.get('entry_price', 0))
                position_size = float(row.get('position_size', 0))
                direction = row.get('direction', 'BUY')

                current_price = entry_price
                if provider:
                    try:
                        price = provider.get_current_price(symbol)
                        if price:
                            current_price = price
                    except Exception:
                        pass

                if entry_price > 0:
                    if direction == "BUY":
                        unrealized_pnl += position_size * (current_price - entry_price) / entry_price
                    else:
                        unrealized_pnl += position_size * (entry_price - current_price) / entry_price

        stats["unrealized_pnl"] = round(unrealized_pnl, 2)

        # Realized PnL: prefer closed_trades.csv (has total_pnl incl. partials);
        # fall back to closed rows of paper_trades.
        realized_pnl = 0.0
        daily_realized_pnl = 0.0
        today = datetime.now().strftime('%Y-%m-%d')

        closed_csv = os.path.join(DATA_BASE_PATH, _get_strategy_folder(), 'closed_trades.csv')
        closed_positions = None
        if os.path.exists(closed_csv):
            try:
                closed_positions = pd.read_csv(closed_csv)
            except Exception:
                closed_positions = None
        if closed_positions is None or closed_positions.empty:
            closed_positions = df[df['status'] != 'OPEN']

        if not closed_positions.empty:
            pnl_series = _trade_pnl_series(closed_positions)
            realized_pnl = float(pnl_series.sum())

            # Daily PnL (trades closed today)
            time_col = 'exit_time' if 'exit_time' in closed_positions.columns else 'close_timestamp' if 'close_timestamp' in closed_positions.columns else None
            if time_col:
                today_mask = closed_positions[time_col].astype(str).str.startswith(today, na=False)
                daily_realized_pnl = float(pnl_series[today_mask].sum())

        stats["realized_pnl"] = round(realized_pnl, 2)

        # Total PnL = realized + unrealized
        stats["total_pnl"] = round(realized_pnl + unrealized_pnl, 2)

        # Daily PnL = realized today + current unrealized
        stats["daily_pnl"] = round(daily_realized_pnl + unrealized_pnl, 2)

        # Events invisible to the balance_after ledger (they happen AFTER the
        # last close): scale-out partials already credited on still-open
        # positions, and entry fees deducted for positions opened since.
        open_partial_pnl = 0.0
        open_entry_fees = 0.0
        if not open_positions.empty:
            if 'partial_pnl_realized' in open_positions.columns:
                open_partial_pnl = float(pd.to_numeric(
                    open_positions['partial_pnl_realized'], errors='coerce').fillna(0.0).sum())
            if 'entry_fee' in open_positions.columns:
                open_entry_fees = float(pd.to_numeric(
                    open_positions['entry_fee'], errors='coerce').fillna(0.0).sum())

        # Balance (equity): reconstruct the strategy's live paper_balance the
        # same way AdaptiveHybridStrategy._load_state does — INITIAL + realized
        # (incl. closed partials, net of all closed-trade entry fees) + partials
        # realized on open positions - entry fees of open positions. Using the
        # last balance_after alone ignored everything after the last close.
        balance_base = INITIAL_BALANCE + realized_pnl + open_partial_pnl - open_entry_fees
        stats["balance"] = round(balance_base + unrealized_pnl, 2)
        stats["available_margin"] = round(max(0, stats["balance"] - stats["used_margin"]), 2)

        # Effective leverage = total notional exposure / equity
        if stats["balance"] > 0:
            stats["effective_leverage"] = round(stats["total_exposure"] / stats["balance"], 1)

    except Exception as e:
        print(f"[Dashboard API] Error reading CSV: {e}")

    return stats


def _aggregate_strategy_stats(strategy_key: str) -> Dict:
    """Compute basic stats for an independent strategy from its CSVs.

    Each independent strategy persists to:
        src/data/<strategy_key>/paper_trades.csv  (open trades)
        src/data/<strategy_key>/closed_trades.csv (closed trades)
    """
    folder = os.path.join(DATA_BASE_PATH, strategy_key)
    paper_csv = os.path.join(folder, 'paper_trades.csv')
    closed_csv = os.path.join(folder, 'closed_trades.csv')

    stats = {
        "key": strategy_key,
        "open_positions": 0,
        "closed_trades": 0,
        "total_pnl": 0.0,
        "daily_pnl": 0.0,
        "win_rate": 0.0,
        "last_trade_at": None,
        "data_dir_exists": os.path.isdir(folder),
    }

    today_str = datetime.now().strftime('%Y-%m-%d')

    # Open positions
    if os.path.exists(paper_csv):
        try:
            df = pd.read_csv(paper_csv)
            if not df.empty and 'status' in df.columns:
                stats["open_positions"] = int((df['status'] == 'OPEN').sum())
        except Exception:
            pass

    # Closed trades
    if os.path.exists(closed_csv):
        try:
            df = pd.read_csv(closed_csv)
            if not df.empty:
                stats["closed_trades"] = int(len(df))
                # Real PnL (incl. partials, net of entry fee) — see _trade_pnl_series
                pnl_series = _trade_pnl_series(df)
                stats["total_pnl"] = round(float(pnl_series.sum()), 2)
                wins = int((pnl_series > 0).sum())
                stats["win_rate"] = round(wins / len(df) * 100, 1)
                time_col = 'exit_time' if 'exit_time' in df.columns else (
                    'close_timestamp' if 'close_timestamp' in df.columns else None
                )
                if time_col:
                    today_mask = df[time_col].astype(str).str.startswith(today_str, na=False)
                    stats["daily_pnl"] = round(float(pnl_series[today_mask].sum()), 2)
                    last_ts = df[time_col].astype(str).max()
                    stats["last_trade_at"] = last_ts if last_ts else None
        except Exception:
            pass

    return stats


# Names + display labels for all known strategies.
# Independent strategies match folder names in src/data/.
_KNOWN_STRATEGIES = [
    ("adaptive_hybrid", "Adaptive Hybrid"),
    ("ote_scalp", "OTE Scalper"),
    ("funding_mr", "Funding Mean Reversion"),
    ("vol_breakout", "Volatility Breakout"),
    ("liq_cascade", "Liquidation Cascade Fade"),
]


@router.get("/strategies")
async def get_strategies(username: str = Depends(verify_credentials)) -> Dict:
    """Aggregate stats per active strategy (Adaptive Hybrid + independents).

    Fixes the prior bug where the dashboard only ever showed
    `{Adaptive Hybrid: 100}` and never reported the OTE Scalper, FundingMR,
    VolBreakout, LiqCascade strategies.
    """
    strategies = []
    for key, label in _KNOWN_STRATEGIES:
        s = _aggregate_strategy_stats(key)
        s["label"] = label
        strategies.append(s)

    total_pnl = round(sum(s["total_pnl"] for s in strategies), 2)
    total_open = sum(s["open_positions"] for s in strategies)
    total_closed = sum(s["closed_trades"] for s in strategies)

    return {
        "strategies": strategies,
        "summary": {
            "total_pnl": total_pnl,
            "total_open_positions": total_open,
            "total_closed_trades": total_closed,
            "active_count": sum(1 for s in strategies
                                if s["closed_trades"] > 0 or s["open_positions"] > 0),
        },
    }


@router.get("/stats")
async def get_stats(username: str = Depends(verify_credentials)) -> Dict:
    """Get dashboard statistics."""
    from src.web.state import is_strategy_running
    from src.config import ACTIVE_STRATEGY

    # Get max daily trades based on active strategy
    if ACTIVE_STRATEGY == 'sniper':
        from src.config import SNIPER_MAX_DAILY_TRADES
        max_daily_trades = SNIPER_MAX_DAILY_TRADES
    elif ACTIVE_STRATEGY == 'adaptive_hybrid':
        from src.config import ADAPTIVE_HYBRID_MAX_DAILY_TRADES
        max_daily_trades = ADAPTIVE_HYBRID_MAX_DAILY_TRADES
    else:
        from src.config import RAMF_MAX_DAILY_TRADES
        max_daily_trades = RAMF_MAX_DAILY_TRADES

    # First try to read from CSV file (shared between processes)
    stats = _get_stats_from_csv()
    stats["running"] = is_strategy_running()
    stats["max_daily_trades"] = max_daily_trades
    stats["strategy_name"] = ACTIVE_STRATEGY.replace('_', ' ').title()

    # Read regime from scheduler state
    scheduler = _read_scheduler_status()
    for tok in scheduler.get('tokens', []):
        if tok.get('regime'):
            stats['regime'] = tok['regime']
            break

    # Count today's trades
    today_str = datetime.now().strftime('%Y-%m-%d')
    strategy_folder = os.path.join(DATA_BASE_PATH, _get_strategy_folder())
    closed_csv = os.path.join(strategy_folder, 'closed_trades.csv')
    if os.path.exists(closed_csv):
        try:
            closed_df = pd.read_csv(closed_csv)
            time_col = 'exit_time' if 'exit_time' in closed_df.columns else 'timestamp'
            if time_col in closed_df.columns:
                stats['daily_trades'] = int(closed_df[closed_df[time_col].str.startswith(today_str, na=False)].shape[0])
        except Exception:
            pass

    # Multi-strategy breakdown (Adaptive Hybrid + independents)
    # Previously the dashboard only showed `{Adaptive Hybrid: 100}` and ignored
    # the independent strategies that write to their own CSV folders.
    try:
        per_strat = {}
        for key, label in _KNOWN_STRATEGIES:
            s = _aggregate_strategy_stats(key)
            if s["closed_trades"] > 0 or s["open_positions"] > 0:
                per_strat[label] = {
                    "key": key,
                    "open": s["open_positions"],
                    "closed": s["closed_trades"],
                    "pnl": s["total_pnl"],
                    "daily_pnl": s["daily_pnl"],
                    "win_rate": s["win_rate"],
                }
        if per_strat:
            stats["strategies"] = per_strat
    except Exception as e:
        print(f"[Dashboard API] Error aggregating strategies: {e}")

    # If CSV has data, return it
    if stats["total_pnl"] != 0 or stats["open_positions"] > 0:
        return stats

    # Fallback to in-memory singleton (works when API and strategy in same process)
    try:
        if ACTIVE_STRATEGY == 'sniper':
            from src.strategies.custom.sniper_ai_strategy import SniperAIStrategy
            strategy = SniperAIStrategy._instance if hasattr(SniperAIStrategy, '_instance') else None
        elif ACTIVE_STRATEGY == 'adaptive_hybrid':
            from src.strategies.custom.adaptive_hybrid_strategy import AdaptiveHybridStrategy
            strategy = AdaptiveHybridStrategy._instance if hasattr(AdaptiveHybridStrategy, '_instance') else None
        else:
            from src.strategies.custom.ramf_strategy import RAMFStrategy
            strategy = RAMFStrategy._instance if hasattr(RAMFStrategy, '_instance') else None

        if strategy:
            paper_status = strategy.get_paper_status()
            stats.update({
                "balance": paper_status.get("paper_balance", stats["balance"]),
                "daily_pnl": paper_status.get("daily_pnl", stats["daily_pnl"]),
                "total_pnl": paper_status.get("total_pnl", stats["total_pnl"]),
                "open_positions": paper_status.get("open_positions", stats["open_positions"]),
            })
    except Exception:
        pass

    return stats


@router.get("/pnl")
async def get_pnl_history(
    days: int = 7,
    username: str = Depends(verify_credentials)
) -> Dict:
    """Get PnL history for chart."""
    pnl_by_date = {}

    # Try to read from paper_trades.csv first (dynamic path based on active strategy)
    strategy_folder = os.path.join(DATA_BASE_PATH, _get_strategy_folder())
    paper_trades_csv = _get_paper_trades_csv()
    if os.path.exists(paper_trades_csv):
        try:
            df = pd.read_csv(paper_trades_csv)
            closed = df[df['status'] != 'OPEN']
            if not closed.empty and 'pnl' in closed.columns:
                # Try exit_time first, fall back to close_timestamp
                time_col = 'exit_time' if 'exit_time' in closed.columns else 'close_timestamp' if 'close_timestamp' in closed.columns else None
                if time_col:
                    pnl_series = _trade_pnl_series(closed)
                    for idx, row in closed.iterrows():
                        exit_time = str(row.get(time_col, ''))
                        if exit_time and len(exit_time) >= 10:
                            date = exit_time[:10]
                            if date not in pnl_by_date:
                                pnl_by_date[date] = 0.0
                            pnl_by_date[date] += float(pnl_series.loc[idx])
        except Exception:
            pass

    # Also check closed_trades.csv (separate log of closed trades)
    if not pnl_by_date:
        closed_trades_csv = os.path.join(strategy_folder, 'closed_trades.csv')
        if os.path.exists(closed_trades_csv):
            try:
                df = pd.read_csv(closed_trades_csv)
                if not df.empty and 'pnl' in df.columns:
                    time_col = 'exit_time' if 'exit_time' in df.columns else 'close_timestamp' if 'close_timestamp' in df.columns else None
                    if time_col:
                        pnl_series = _trade_pnl_series(df)
                        for idx, row in df.iterrows():
                            exit_time = str(row.get(time_col, ''))
                            if exit_time and len(exit_time) >= 10:
                                date = exit_time[:10]
                                if date not in pnl_by_date:
                                    pnl_by_date[date] = 0.0
                                pnl_by_date[date] += float(pnl_series.loc[idx])
            except Exception:
                pass

    # Fallback to signals history
    if not pnl_by_date:
        signals = get_signals_history(limit=100)
        for signal in reversed(signals):
            if "pnl" in signal:
                ts = signal.get("timestamp", "")
                if ts:
                    date = ts[:10]
                    if date not in pnl_by_date:
                        pnl_by_date[date] = 0.0
                    pnl_by_date[date] += signal.get("pnl", 0)

    # Build chart data
    today = datetime.now()
    labels = []
    data = []
    cumulative = 0.0

    for i in range(days - 1, -1, -1):
        date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        labels.append(date[-5:])  # MM-DD format
        cumulative += pnl_by_date.get(date, 0)
        data.append(round(cumulative, 2))

    return {
        "labels": labels,
        "data": data,
    }


@router.get("/trades")
async def get_closed_trades(
    limit: int = 50,
    username: str = Depends(verify_credentials),
) -> Dict:
    """Get closed trade history from closed_trades.csv."""
    strategy_folder = os.path.join(DATA_BASE_PATH, _get_strategy_folder())
    closed_trades_csv = os.path.join(strategy_folder, 'closed_trades.csv')
    trades = []

    if os.path.exists(closed_trades_csv):
        try:
            df = pd.read_csv(closed_trades_csv)
            if not df.empty:
                # Sort by exit_time descending (most recent first)
                time_col = 'exit_time' if 'exit_time' in df.columns else 'timestamp'
                if time_col in df.columns:
                    df = df.sort_values(time_col, ascending=False)
                df = df.head(limit)

                # Real per-trade PnL (incl. partials, net of entry fee)
                pnl_series = _trade_pnl_series(df)
                for idx, row in df.iterrows():
                    trades.append({
                        "symbol": row.get("symbol", ""),
                        "direction": row.get("direction", ""),
                        "entry_price": float(row.get("entry_price", 0) or 0),
                        "close_price": float(row.get("close_price", 0) or 0),
                        "position_size": float(row.get("position_size", 0) or 0),
                        "pnl": round(float(pnl_series.loc[idx]), 2),
                        "pnl_pct": round(float(row.get("pnl_pct", 0) or 0), 2),
                        "close_reason": row.get("close_reason", ""),
                        "entry_time": str(row.get("timestamp", row.get("entry_time", ""))),
                        "exit_time": str(row.get("exit_time", "")),
                        "score": round(float(row.get("score", 0) or 0), 1),
                        "leverage": float(row.get("leverage", 1) or 1),
                        "memory_decision_id": str(row.get("memory_decision_id", "")) if pd.notna(row.get("memory_decision_id")) else None,
                    })
        except Exception as e:
            print(f"[Dashboard API] Error reading closed trades: {e}")

    # Summary stats
    total_pnl = sum(t["pnl"] for t in trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]

    return {
        "trades": trades,
        "summary": {
            "total": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / len(trades), 2) if trades else 0,
        },
    }


def _read_scheduler_status() -> Dict:
    """Read scheduler and light check state files, return formatted status."""
    now = datetime.now().timestamp()

    result = {
        "tokens": [],
        "queue_size": 0,
        "light_check": {
            "last_prices": {},
            "spike_count": 0,
        },
    }

    # Read scheduler state
    if os.path.exists(SCHEDULER_STATE_FILE):
        try:
            with open(SCHEDULER_STATE_FILE, 'r') as f:
                state = json.load(f)

            last_check = state.get('last_check', {})
            last_result = state.get('last_result', {})
            queue_items = state.get('queue', [])

            result["queue_size"] = len(queue_items)

            # Build next_recheck lookup from queue
            next_recheck: Dict[str, float] = {}
            next_priority: Dict[str, int] = {}
            for item in queue_items:
                sym = item.get('symbol', '')
                scheduled = item.get('scheduled_at', 0)
                prio = item.get('priority', 4)
                if sym not in next_recheck or scheduled < next_recheck[sym]:
                    next_recheck[sym] = scheduled
                    next_priority[sym] = prio

            # Merge all known symbols
            all_symbols = set(last_check.keys()) | set(last_result.keys()) | set(next_recheck.keys())
            for symbol in sorted(all_symbols):
                lc_ts = last_check.get(symbol, 0)
                lr = last_result.get(symbol, {})
                nr = next_recheck.get(symbol)

                token_info = {
                    "symbol": symbol,
                    "last_check_ago_s": round(now - lc_ts) if lc_ts else None,
                    "score": lr.get('score'),
                    "threshold": lr.get('threshold'),
                    "regime": lr.get('regime'),
                    "next_recheck_s": round(nr - now) if nr else None,
                    "priority": next_priority.get(symbol),
                }
                result["tokens"].append(token_info)
        except Exception:
            pass

    # Read light check state
    if os.path.exists(LIGHT_CHECK_STATE_FILE):
        try:
            with open(LIGHT_CHECK_STATE_FILE, 'r') as f:
                lc_state = json.load(f)
            result["light_check"]["last_prices"] = lc_state.get('last_prices', {})
            result["light_check"]["spike_count"] = lc_state.get('spike_count', 0)
        except Exception:
            pass

    return result


@router.get("/scheduler")
async def get_scheduler_status(username: str = Depends(verify_credentials)) -> Dict:
    """Get smart scheduler status."""
    return _read_scheduler_status()


@router.get("/sse")
async def sse_updates(username: str = Depends(verify_credentials)):
    """Server-Sent Events for real-time updates."""
    from src.web.api.positions import _get_positions_from_csv

    async def event_generator():
        from src.web.state import is_strategy_running

        while True:
            # Use CSV-based functions for cross-process data sharing
            from src.config import ACTIVE_STRATEGY
            stats = _get_stats_from_csv()
            stats["running"] = is_strategy_running()
            stats["strategy_name"] = ACTIVE_STRATEGY.replace('_', ' ').title()

            positions = _get_positions_from_csv()
            signals = get_signals_history(limit=5)

            # Send scheduler status (also used for regime below)
            scheduler = _read_scheduler_status()

            # Get regime from scheduler state (already read above for scheduler event)
            if 'regime' not in stats:
                for tok in scheduler.get('tokens', []):
                    if tok.get('regime'):
                        stats['regime'] = tok['regime']
                        break

            # Send stats update
            yield f"event: stats\ndata: {_json_dumps(stats)}\n\n"

            # Send positions update
            yield f"event: positions\ndata: {_json_dumps(positions)}\n\n"

            # Send latest signals
            yield f"event: signals\ndata: {_json_dumps(signals)}\n\n"
            yield f"event: scheduler\ndata: {_json_dumps(scheduler)}\n\n"

            # Wait before next update
            await asyncio.sleep(5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


def _json_dumps(obj) -> str:
    """Convert object to JSON string."""
    return json.dumps(obj, default=str)
