"""
Dashboard API endpoints
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.web.auth import verify_credentials
from src.web.state import get_dashboard_stats, get_signals_history, get_paper_positions

router = APIRouter()

# Path to paper trades CSV file
PAPER_TRADES_CSV = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ramf', 'paper_trades.csv')
INITIAL_BALANCE = 500.0


def _get_stats_from_csv() -> Dict:
    """Calculate stats from paper_trades.csv file with real-time unrealized PnL."""
    from src.config import RAMF_LEVERAGE

    stats = {
        "balance": INITIAL_BALANCE,
        "daily_pnl": 0.0,
        "total_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "realized_pnl": 0.0,
        "open_positions": 0,
        "used_margin": 0.0,
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
            # Margin = position_size / leverage
            stats["used_margin"] = round(
                (open_positions['position_size'] / RAMF_LEVERAGE).sum(), 2
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

        # Calculate realized PnL from closed positions
        closed_positions = df[df['status'] != 'OPEN']
        realized_pnl = 0.0
        daily_realized_pnl = 0.0

        if not closed_positions.empty and 'pnl' in closed_positions.columns:
            realized_pnl = closed_positions['pnl'].sum()

            # Daily PnL (trades closed today)
            today = datetime.now().strftime('%Y-%m-%d')
            if 'exit_time' in closed_positions.columns:
                today_closed = closed_positions[closed_positions['exit_time'].str.startswith(today, na=False)]
                if not today_closed.empty:
                    daily_realized_pnl = today_closed['pnl'].sum()

        stats["realized_pnl"] = round(realized_pnl, 2)

        # Total PnL = realized + unrealized
        stats["total_pnl"] = round(realized_pnl + unrealized_pnl, 2)

        # Daily PnL = realized today + current unrealized
        stats["daily_pnl"] = round(daily_realized_pnl + unrealized_pnl, 2)

        # Balance (equity) = initial + realized + unrealized
        stats["balance"] = round(INITIAL_BALANCE + realized_pnl + unrealized_pnl, 2)
        stats["available_margin"] = round(max(0, stats["balance"] - stats["used_margin"]), 2)

    except Exception as e:
        print(f"[Dashboard API] Error reading CSV: {e}")

    return stats


@router.get("/stats")
async def get_stats(username: str = Depends(verify_credentials)) -> Dict:
    """Get dashboard statistics."""
    from src.web.state import is_strategy_running
    from src.config import RAMF_MAX_DAILY_TRADES

    # First try to read from CSV file (shared between processes)
    stats = _get_stats_from_csv()
    stats["running"] = is_strategy_running()
    stats["max_daily_trades"] = RAMF_MAX_DAILY_TRADES

    # If CSV has data, return it
    if stats["total_pnl"] != 0 or stats["open_positions"] > 0:
        return stats

    # Fallback to in-memory singleton (works when API and strategy in same process)
    try:
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

    # Try to read from CSV file first
    if os.path.exists(PAPER_TRADES_CSV):
        try:
            df = pd.read_csv(PAPER_TRADES_CSV)
            closed = df[df['status'] != 'OPEN']
            if not closed.empty and 'pnl' in closed.columns and 'exit_time' in closed.columns:
                for _, row in closed.iterrows():
                    exit_time = str(row.get('exit_time', ''))
                    if exit_time and len(exit_time) >= 10:
                        date = exit_time[:10]
                        if date not in pnl_by_date:
                            pnl_by_date[date] = 0.0
                        pnl_by_date[date] += float(row.get('pnl', 0))
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


@router.get("/sse")
async def sse_updates(username: str = Depends(verify_credentials)):
    """Server-Sent Events for real-time updates."""
    from src.web.api.positions import _get_positions_from_csv

    async def event_generator():
        from src.web.state import is_strategy_running

        while True:
            # Use CSV-based functions for cross-process data sharing
            stats = _get_stats_from_csv()
            stats["running"] = is_strategy_running()
            positions = _get_positions_from_csv()
            signals = get_signals_history(limit=5)

            # Send stats update
            yield f"event: stats\ndata: {_json_dumps(stats)}\n\n"

            # Send positions update
            yield f"event: positions\ndata: {_json_dumps(positions)}\n\n"

            # Send latest signals
            yield f"event: signals\ndata: {_json_dumps(signals)}\n\n"

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
    import json
    return json.dumps(obj, default=str)
