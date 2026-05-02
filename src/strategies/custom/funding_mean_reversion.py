"""
Funding Rate Mean Reversion Strategy

Edge: Extreme funding rates on perpetual futures predict short-term mean reversion.
When funding is extremely positive (longs pay shorts), the market is overleveraged
long -> expect a pullback. When extremely negative, expect a bounce.

Tokens: 14 monitored tokens from SNIPER_ASSETS.
Data: HyperLiquid funding rates + 1h OHLCV candles.
"""

import os
import time as _time
import pandas as pd
import numpy as np
from collections import deque
from datetime import datetime
from termcolor import cprint
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator

from src.data_providers.market_data import MarketDataProvider
from src.data.trade_memory import TradeMemory
from src.config import PAPER_TRADING_BALANCE, SNIPER_ASSETS, PAPER_TAKER_FEE_V2, PAPER_SLIPPAGE_V2
from src.strategies.base_strategy import BaseStrategy

# --- Config ---
ZSCORE_THRESHOLDS = {'major': 1.5, 'mid': 1.3, 'alt': 1.0}  # Loosened (was 2.0/1.8/1.5) to permit more signals in low-volatility funding regimes
TOKEN_CLASS = {
    'BTC': 'major', 'ETH': 'major',
    'SOL': 'mid', 'XRP': 'mid', 'AVAX': 'mid', 'SUI': 'mid',
    'LINK': 'mid', 'ADA': 'mid', 'AAVE': 'mid', 'NEAR': 'mid', 'TAO': 'mid',
    'DOGE': 'alt', 'kPEPE': 'alt', 'ENA': 'alt',
}
LEVERAGE = {'major': 3, 'mid': 3, 'alt': 2}
ATR_SL_MULT = {'major': 3.0, 'mid': 2.5, 'alt': 2.5}
RISK_PER_TRADE_PCT = 1.5
# Concurrent open positions cap (was conflated with the daily-trade cap).
MAX_CONCURRENT_POSITIONS = 3
# Number of new entries allowed per UTC day.
MAX_DAILY_TRADES = 6
MAX_DAILY_LOSS_USD = 15.0
MAX_HOLD_HOURS = 12
# Minimum time a position must be held before the funding-normalization exit can fire.
MIN_HOLD_HOURS_FOR_FUNDING_EXIT = 1.0
TRAILING_ACTIVATE_ATR = 1.5
TRAILING_DISTANCE_ATR = 1.0
FUNDING_EXIT_ZSCORE = 0.5
RSI_LONG_MAX = 35
RSI_SHORT_MIN = 65
VOLUME_FILTER_RATIO = 0.15
FUNDING_HISTORY_LEN = 168


class FundingMeanReversionStrategy(BaseStrategy):
    """Funding rate mean reversion paper trading strategy."""

    def __init__(self):
        self.assets = SNIPER_ASSETS
        self.tokens = self.assets
        self.paper_balance = PAPER_TRADING_BALANCE
        self.paper_positions = {}
        self.closed_positions = []
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_date = datetime.utcnow().date()
        self._position_counter = 0
        self._funding_history = {}
        self._trailing = {}

        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data', 'funding_mr')
        os.makedirs(self.data_dir, exist_ok=True)

        self._market_data = MarketDataProvider(start_liquidation_stream=False)
        self._trade_memory = TradeMemory.get_instance()
        self._load_state()

        cprint(f"[FundingMR] Initialized | Balance: ${self.paper_balance:,.2f} | "
               f"Open: {len(self.paper_positions)}", "cyan")

    # --- Data ---

    def _fetch_candles(self, symbol, interval='1h', candles=100):
        """Fetch OHLCV candles from HyperLiquid."""
        try:
            from hyperliquid.info import Info
            info = Info(skip_ws=True, timeout=15)
            end_ms = int(_time.time() * 1000)
            iv_ms = {'1h': 3_600_000, '15m': 900_000, '4h': 14_400_000}.get(interval, 3_600_000)
            _time.sleep(0.15)
            data = info.candles_snapshot(symbol, interval, end_ms - candles * iv_ms, end_ms)
            if not data:
                return None
            df = pd.DataFrame(data).rename(columns={
                't': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
            for c in ['open', 'high', 'low', 'close', 'volume']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            return df
        except Exception as e:
            cprint(f"[FundingMR] Candle fetch error {symbol}: {e}", "yellow")
            return None

    def _get_funding_zscore(self, symbol):
        """Z-score of current funding rate against rolling 7-day history."""
        rd = self._market_data.get_funding_rate(symbol)
        if not rd or rd.get('funding_rate') is None:
            return 0.0
        rate = rd['funding_rate']
        if symbol not in self._funding_history:
            self._funding_history[symbol] = deque(maxlen=FUNDING_HISTORY_LEN)
        self._funding_history[symbol].append(rate)
        hist = list(self._funding_history[symbol])
        if len(hist) < 10:
            z = self._market_data.get_funding_zscore(symbol)
            return z if z is not None else 0.0
        mean, std = np.mean(hist), np.std(hist)
        return (rate - mean) / std if std > 1e-10 else 0.0

    # --- Helpers ---

    def _cls(self, s):
        return TOKEN_CLASS.get(s, 'mid')

    def _slippage(self, s):
        m = {'major': 'btc', 'mid': 'mid', 'alt': 'alt'}
        return PAPER_SLIPPAGE_V2.get(m.get(self._cls(s), 'mid'), 0.0012)

    def _check_daily_reset(self):
        today = datetime.utcnow().date()
        if today != self.daily_date:
            self.daily_pnl, self.daily_trades, self.daily_date = 0.0, 0, today

    # --- Position Management ---

    def _manage_positions(self):
        """Check all open positions for exit conditions."""
        to_close = []
        for pid, pos in list(self.paper_positions.items()):
            price = self._market_data.get_current_price(pos['symbol'])
            if price is None:
                continue
            d, entry, atr = pos['direction'], pos['entry_price'], pos.get('atr', 0)
            reason = None

            # Stop loss
            if (d == 'BUY' and price <= pos['stop_loss']) or (d == 'SELL' and price >= pos['stop_loss']):
                reason = 'stop_loss'
            # Take profit
            if not reason:
                if (d == 'BUY' and price >= pos['take_profit']) or (d == 'SELL' and price <= pos['take_profit']):
                    reason = 'take_profit'
            # Compute hold duration once (used by both funding-norm and time-stop checks).
            try:
                hrs_held = (datetime.utcnow() - datetime.fromisoformat(pos['entry_time'])).total_seconds() / 3600
            except Exception:
                hrs_held = 0.0
            # Funding normalization (guard against premature exit via min-hold).
            if (not reason
                    and hrs_held >= MIN_HOLD_HOURS_FOR_FUNDING_EXIT
                    and abs(self._get_funding_zscore(pos['symbol'])) <= FUNDING_EXIT_ZSCORE):
                reason = 'funding_normalized'
            # Time stop
            if not reason and hrs_held >= MAX_HOLD_HOURS:
                reason = 'time_stop'
            # Trailing stop
            if not reason and atr > 0:
                ts = self._trailing.get(pid, {'best_price': entry})
                if d == 'BUY':
                    ts['best_price'] = max(ts['best_price'], price)
                    prof = (ts['best_price'] - entry) / atr
                    if prof >= TRAILING_ACTIVATE_ATR and price <= ts['best_price'] - TRAILING_DISTANCE_ATR * atr:
                        reason = 'trailing_stop'
                else:
                    ts['best_price'] = min(ts['best_price'], price)
                    prof = (entry - ts['best_price']) / atr
                    if prof >= TRAILING_ACTIVATE_ATR and price >= ts['best_price'] + TRAILING_DISTANCE_ATR * atr:
                        reason = 'trailing_stop'
                self._trailing[pid] = ts

            if reason:
                to_close.append((pid, price, reason))

        for pid, px, reason in to_close:
            self._close_position(pid, px, reason)

    def _close_position(self, position_id, exit_price, reason):
        """Close a position and update balance."""
        pos = self.paper_positions.get(position_id)
        if not pos:
            return
        d, entry, size = pos['direction'], pos['entry_price'], pos['position_size']
        slip = self._slippage(pos['symbol'])
        eff = exit_price * (1 - slip) if d == 'BUY' else exit_price * (1 + slip)
        pnl_pct = (eff - entry) / entry if d == 'BUY' else (entry - eff) / entry
        pnl_usd = pnl_pct * size
        exit_fee = size * PAPER_TAKER_FEE_V2
        net = pnl_usd - exit_fee
        self.paper_balance += net
        self.daily_pnl += net
        hrs = (datetime.utcnow() - datetime.fromisoformat(pos['entry_time'])).total_seconds() / 3600

        pos.update({
            'status': 'CLOSED', 'exit_price': exit_price,
            'exit_time': datetime.utcnow().isoformat(),
            'pnl': round(net, 4), 'pnl_pct': round(pnl_pct * 100, 2),
            'close_reason': reason, 'hold_hours': round(hrs, 2),
            'exit_fee': round(exit_fee, 6),
        })
        self.closed_positions.append(pos)
        del self.paper_positions[position_id]
        self._trailing.pop(position_id, None)
        self._log_closed_trade(pos)
        self._update_position_csv(position_id, pos)

        try:
            if 'memory_decision_id' in pos:
                self._trade_memory.update_outcome(
                    int(pos['memory_decision_id']), pnl=net,
                    hold_duration_hours=hrs, close_reason=reason)
        except Exception:
            pass

        c = "green" if net > 0 else "red"
        cprint(f"[FundingMR] CLOSED {d} {pos['symbol']} | PnL: ${net:+.2f} ({pnl_pct*100:+.1f}%) | "
               f"{reason} | {hrs:.1f}h | Bal: ${self.paper_balance:,.2f}", c, attrs=['bold'])

    # --- Entry Scanning ---

    def _scan_for_entries(self):
        """Scan all symbols for new entry signals."""
        self._check_daily_reset()
        if self.daily_trades >= MAX_DAILY_TRADES:
            return
        if self.daily_pnl <= -MAX_DAILY_LOSS_USD:
            cprint(f"[FundingMR] Daily loss limit hit (${self.daily_pnl:+.2f})", "red")
            return
        if len(self.paper_positions) >= MAX_CONCURRENT_POSITIONS:
            return
        open_syms = {p['symbol'] for p in self.paper_positions.values()}
        for sym in self.assets:
            if sym in open_syms or self.daily_trades >= MAX_DAILY_TRADES:
                continue
            if len(self.paper_positions) >= MAX_CONCURRENT_POSITIONS:
                break
            sig = self._evaluate_entry(sym)
            if sig:
                self._open_position(sig)

    def _evaluate_entry(self, symbol):
        """Evaluate a single symbol for entry. Returns signal dict or None."""
        z = self._get_funding_zscore(symbol)
        thr = ZSCORE_THRESHOLDS.get(self._cls(symbol), 1.8)
        if abs(z) < thr:
            return None
        direction = 'SELL' if z > 0 else 'BUY'

        df = self._fetch_candles(symbol, '1h', 100)
        if df is None or len(df) < 20:
            return None
        close = df['close']
        rsi = RSIIndicator(close=close, window=14).rsi().iloc[-1]

        if direction == 'BUY' and rsi > RSI_LONG_MAX:
            return None
        if direction == 'SELL' and rsi < RSI_SHORT_MIN:
            return None

        vol, vol_avg = df['volume'].iloc[-1], df['volume'].iloc[-20:].mean()
        if vol_avg > 0 and vol < VOLUME_FILTER_RATIO * vol_avg:
            return None

        atr = AverageTrueRange(high=df['high'], low=df['low'], close=close, window=14).average_true_range().iloc[-1]
        price = close.iloc[-1]
        if atr <= 0 or price <= 0:
            return None

        return {'symbol': symbol, 'direction': direction, 'price': price, 'atr': atr,
                'rsi': rsi, 'funding_zscore': z,
                'volume_ratio': round(vol / vol_avg, 2) if vol_avg > 0 else 0}

    # --- Trade Execution ---

    def _open_position(self, signal):
        """Open a paper position."""
        sym, d, price, atr = signal['symbol'], signal['direction'], signal['price'], signal['atr']
        cls = self._cls(sym)
        lev = LEVERAGE.get(cls, 3)
        sl_mult = ATR_SL_MULT.get(cls, 2.5)
        slip = self._slippage(sym)

        entry = price * (1 + slip) if d == 'BUY' else price * (1 - slip)
        sl_dist = sl_mult * atr
        tp_dist = sl_dist * 2.0
        sl = entry - sl_dist if d == 'BUY' else entry + sl_dist
        tp = entry + tp_dist if d == 'BUY' else entry - tp_dist

        risk = self.paper_balance * (RISK_PER_TRADE_PCT / 100)
        sl_pct = sl_dist / entry
        if sl_pct <= 0:
            return
        size = min((risk / sl_pct) * lev, self.paper_balance * 0.25)
        fee = size * PAPER_TAKER_FEE_V2
        if self.paper_balance < fee + 1:
            return

        self._position_counter += 1
        pid = f"fmr_{sym}_{datetime.utcnow().strftime('%Y%m%d')}_{self._position_counter}"
        trade = {
            'position_id': pid, 'timestamp': datetime.utcnow().isoformat(),
            'symbol': sym, 'direction': d, 'entry_price': entry,
            'position_size': round(size, 2), 'leverage': lev,
            'stop_loss': sl, 'take_profit': tp, 'sl_pct': round(sl_pct * 100, 2),
            'atr': atr, 'entry_fee': round(fee, 6), 'status': 'OPEN',
            'entry_time': datetime.utcnow().isoformat(),
            'funding_zscore': signal['funding_zscore'], 'rsi': signal['rsi'],
        }
        self.paper_positions[pid] = trade
        self.paper_balance -= fee
        self.daily_trades += 1
        self._log_open_trade(trade)

        try:
            did = self._trade_memory.log_decision(
                symbol=sym, direction=d,
                confidence=min(abs(signal['funding_zscore']) / 3.0 * 100, 100),
                source='funding_mr',
                reasoning=f"Z={signal['funding_zscore']:.2f} RSI={signal['rsi']:.1f} Vol={signal['volume_ratio']}",
                key_indicators={'funding_zscore': signal['funding_zscore'], 'rsi': signal['rsi'], 'atr': atr})
            trade['memory_decision_id'] = did
        except Exception:
            pass

        cprint(f"\n[FundingMR] OPENED {d} {sym}", "magenta", attrs=['bold'])
        cprint(f"  Entry: ${entry:,.4f} | Size: ${size:,.2f} | Lev: {lev}x", "white")
        cprint(f"  SL: ${sl:,.4f} | TP: ${tp:,.4f} | Z: {signal['funding_zscore']:+.2f} | RSI: {signal['rsi']:.1f}", "white")

    # --- Main Cycle ---

    def run_cycle(self, symbols=None):
        """Run one full cycle: manage positions, then scan for entries."""
        t0 = _time.time()
        if symbols:
            self.assets = symbols
        self._check_daily_reset()
        cprint(f"\n[FundingMR] === Cycle {datetime.utcnow().strftime('%H:%M:%S')} UTC | "
               f"Bal: ${self.paper_balance:,.2f} | Open: {len(self.paper_positions)} | "
               f"Trades: {self.daily_trades}/{MAX_DAILY_TRADES} | PnL: ${self.daily_pnl:+.2f} ===", "cyan")
        if self.paper_positions:
            self._manage_positions()
        self._scan_for_entries()
        cprint(f"[FundingMR] Cycle done in {_time.time()-t0:.1f}s", "cyan")

    # --- CSV Persistence ---

    def _log_open_trade(self, trade):
        f = os.path.join(self.data_dir, 'paper_trades.csv')
        df = pd.DataFrame([trade])
        df.to_csv(f, mode='a', header=not os.path.exists(f), index=False)

    def _log_closed_trade(self, trade):
        f = os.path.join(self.data_dir, 'closed_trades.csv')
        df = pd.DataFrame([trade])
        df.to_csv(f, mode='a', header=not os.path.exists(f), index=False)

    def _update_position_csv(self, position_id, trade):
        try:
            p = os.path.join(self.data_dir, 'paper_trades.csv')
            if not os.path.exists(p):
                return
            df = pd.read_csv(p)
            m = df['position_id'] == position_id
            if m.any():
                for k in ['status', 'exit_price', 'exit_time', 'pnl', 'close_reason']:
                    df.loc[m, k] = trade.get(k, '')
                df.to_csv(p, index=False)
        except Exception as e:
            cprint(f"[FundingMR] CSV update error: {e}", "yellow")

    def _load_state(self):
        """Load positions and balance from CSV files."""
        pf = os.path.join(self.data_dir, 'paper_trades.csv')
        cf = os.path.join(self.data_dir, 'closed_trades.csv')

        if os.path.exists(pf):
            try:
                df = pd.read_csv(pf)
                for _, r in df[df['status'] == 'OPEN'].iterrows() if not df.empty else []:
                    pid = r.get('position_id', '')
                    if not pid:
                        continue
                    self.paper_positions[pid] = {
                        'position_id': pid, 'timestamp': r.get('timestamp', ''),
                        'symbol': r.get('symbol', ''), 'direction': r.get('direction', 'BUY'),
                        'entry_price': float(r.get('entry_price', 0) or 0),
                        'position_size': float(r.get('position_size', 0) or 0),
                        'leverage': float(r.get('leverage', 3) or 3),
                        'stop_loss': float(r.get('stop_loss', 0) or 0),
                        'take_profit': float(r.get('take_profit', 0) or 0),
                        'sl_pct': float(r.get('sl_pct', 1.5) or 1.5),
                        'atr': float(r.get('atr', 0) or 0),
                        'entry_fee': float(r.get('entry_fee', 0) or 0),
                        'status': 'OPEN',
                        'entry_time': r.get('timestamp', datetime.utcnow().isoformat()),
                        'funding_zscore': float(r.get('funding_zscore', 0) or 0),
                        'rsi': float(r.get('rsi', 50) or 50),
                    }
                if self.paper_positions:
                    mx = 0
                    for pid in self.paper_positions:
                        parts = pid.split('_')
                        try:
                            mx = max(mx, int(parts[-1])) if len(parts) >= 4 else mx
                        except ValueError:
                            pass
                    self._position_counter = mx
            except Exception as e:
                cprint(f"[FundingMR] Load error: {e}", "yellow")

        realized, cfees = 0.0, 0.0
        if os.path.exists(cf):
            try:
                cdf = pd.read_csv(cf)
                if not cdf.empty and 'pnl' in cdf.columns:
                    realized = cdf['pnl'].sum()
                    cfees = cdf['entry_fee'].fillna(0).sum() if 'entry_fee' in cdf.columns else 0
                    self.closed_positions = cdf.to_dict('records')
            except Exception as e:
                cprint(f"[FundingMR] Load error: {e}", "yellow")

        ofees = sum(t.get('entry_fee', 0) for t in self.paper_positions.values())
        self.paper_balance = PAPER_TRADING_BALANCE + realized - cfees - ofees
        today_str = datetime.utcnow().date().isoformat()
        all_trades = list(self.paper_positions.values()) + self.closed_positions
        self.daily_trades = sum(1 for p in all_trades if str(p.get('timestamp', ''))[:10] == today_str)
        self.daily_pnl = sum(p.get('pnl', 0) for p in self.closed_positions
                             if str(p.get('exit_time', ''))[:10] == today_str)

    def get_status(self):
        """Return current strategy status (for dashboard integration)."""
        return {
            'strategy': 'funding_mean_reversion',
            'paper_balance': round(self.paper_balance, 2),
            'initial_balance': PAPER_TRADING_BALANCE,
            'total_pnl': round(self.paper_balance - PAPER_TRADING_BALANCE, 2),
            'daily_pnl': round(self.daily_pnl, 2),
            'daily_trades': self.daily_trades,
            'open_positions': len(self.paper_positions),
            'total_closed': len(self.closed_positions),
            'positions': [p.copy() for p in self.paper_positions.values()],
        }


# --- Standalone ---
if __name__ == '__main__':
    import signal, sys
    strategy = FundingMeanReversionStrategy()

    def _exit(sig, frame):
        cprint("\n[FundingMR] Shutting down...", "yellow")
        sys.exit(0)
    signal.signal(signal.SIGINT, _exit)

    cprint("[FundingMR] Starting Funding Mean Reversion Strategy", "cyan", attrs=['bold'])
    cprint(f"  Assets: {', '.join(SNIPER_ASSETS)}", "white")
    cprint(f"  Balance: ${strategy.paper_balance:,.2f} | MaxTrades: {MAX_DAILY_TRADES} | Risk: {RISK_PER_TRADE_PCT}%", "white")

    while True:
        try:
            strategy.run_cycle()
        except Exception as e:
            cprint(f"[FundingMR] Cycle error: {e}", "red")
            import traceback; traceback.print_exc()
        cprint(f"[FundingMR] Next cycle in 300s...", "white")
        _time.sleep(300)
