"""
Volatility Compression Breakout Strategy

Detects Bollinger Band squeezes followed by volume expansion breakouts.
Rides the trend with an ATR trailing stop. No fixed TP -- lets winners run.

Entry: BB squeeze (width < 20th pctl) + breakout + volume > 2x avg + ADX in [22, 32] + 4H EMA alignment
Exit:  2.5x ATR trailing stop | 24h time stop | ADX < 20 exhaustion
"""

import os
import time
import threading
import pandas as pd
try:
    import pandas_ta as ta
except ImportError:
    import pandas_ta_classic as ta
from datetime import datetime, timedelta
from termcolor import cprint

from ..base_strategy import BaseStrategy

try:
    from src.config import (
        PAPER_TRADING, PAPER_TRADING_BALANCE, PAPER_SLIPPAGE_V2,
        PAPER_TAKER_FEE_V2, RISK_MAX_DRAWDOWN_PCT, CASH_PERCENTAGE,
    )
except ImportError:
    PAPER_TRADING = True
    PAPER_TRADING_BALANCE = 500
    PAPER_SLIPPAGE_V2 = {'btc': 0.0003, 'eth': 0.0005, 'mid': 0.0012, 'alt': 0.003}
    PAPER_TAKER_FEE_V2 = 0.00045
    RISK_MAX_DRAWDOWN_PCT = 15
    CASH_PERCENTAGE = 20

# Strategy config
# Universe / daily cap / ADX max come from config.py (single source of truth);
# the rest are local tuned constants (do NOT touch: edge is fragile, n=29).
try:
    from src.config import (
        VOL_BREAKOUT_TOKENS as _CFG_VB_TOKENS,
        VOL_BREAKOUT_MAX_DAILY_TRADES as _CFG_VB_MAX_TRADES,
        VOL_BREAKOUT_ADX_MAX as _CFG_VB_ADX_MAX,
    )
except ImportError:
    _CFG_VB_TOKENS = ['BTC', 'ETH', 'SOL', 'XRP', 'AVAX', 'SUI', 'TAO', 'NEAR', 'AAVE', 'LINK',
                      'DOGE', 'ADA', 'LTC', 'ARB', 'OP', 'INJ']
    _CFG_VB_MAX_TRADES = 4
    _CFG_VB_ADX_MAX = 32

VOL_BREAKOUT_SYMBOLS = list(_CFG_VB_TOKENS)
VOL_BREAKOUT_LEVERAGE = {'btc': 3, 'eth': 3, 'mid': 3, 'alt': 2}
VOL_BREAKOUT_TOKEN_CLASSES = {
    'btc': ['BTC'], 'eth': ['ETH'],
    'mid': ['SOL', 'XRP', 'AVAX', 'LINK', 'AAVE', 'NEAR', 'SUI', 'TAO',
            'ADA', 'LTC', 'ARB', 'OP', 'INJ'],
    'alt': ['DOGE'],  # higher slippage class for the meme
}
VOL_BREAKOUT_RISK_PCT = 0.015
VOL_BREAKOUT_TRAILING_ATR_MULT = 2.5
VOL_BREAKOUT_MAX_HOLD_HOURS = 24
VOL_BREAKOUT_MAX_DAILY_TRADES = _CFG_VB_MAX_TRADES
VOL_BREAKOUT_MAX_DAILY_LOSS = 15.0
VOL_BREAKOUT_SQUEEZE_PERCENTILE = 0.20
VOL_BREAKOUT_VOLUME_MULT = 2.0
VOL_BREAKOUT_VOLUME_MIN = 0.15
VOL_BREAKOUT_ADX_ENTRY = 22
VOL_BREAKOUT_ADX_MAX = _CFG_VB_ADX_MAX  # Reject late entries (ADX already extended)
VOL_BREAKOUT_ADX_EXIT = 20


def _token_class(symbol: str) -> str:
    for cls, tokens in VOL_BREAKOUT_TOKEN_CLASSES.items():
        if symbol in tokens:
            return cls
    return 'mid'


def _price_dec(price: float) -> int:
    if price >= 100: return 2
    if price >= 1: return 3
    if price >= 0.01: return 5
    return 7


class VolatilityBreakoutStrategy(BaseStrategy):
    """Volatility Compression Breakout -- paper trading strategy."""

    def __init__(self):
        super().__init__("Volatility Breakout")
        self.tokens = VOL_BREAKOUT_SYMBOLS
        self.paper_balance = PAPER_TRADING_BALANCE
        self.paper_positions = {}
        self.closed_positions = []
        self._position_counter = 0
        self._lock = threading.RLock()
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self._last_trade_date = None
        self._trailing = {}
        self.peak_balance = PAPER_TRADING_BALANCE
        self._trade_memory = None
        try:
            from src.data.trade_memory import TradeMemory
            self._trade_memory = TradeMemory.get_instance()
        except Exception:
            pass
        self._candle_cache = {}
        self._cache_ttl = 300
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data', 'vol_breakout')
        os.makedirs(self.data_dir, exist_ok=True)
        self._load_state()
        cprint(f"[VolBreakout] Init | ${self.paper_balance:,.2f} | {len(self.paper_positions)} open | "
               f"{len(VOL_BREAKOUT_SYMBOLS)} symbols", "cyan")

    # -- Candle fetching (HyperLiquid) --

    def _fetch_candles(self, symbol: str, interval: str = '1h', candles: int = 200):
        ck = (symbol, interval)
        cached = self._candle_cache.get(ck)
        if cached and (datetime.now() - cached[1]).total_seconds() < self._cache_ttl:
            return cached[0].copy()
        try:
            from hyperliquid.info import Info
            info = Info(skip_ws=True, timeout=15)
            end_ms = int(time.time() * 1000)
            iv = {'1m': 60_000, '5m': 300_000, '15m': 900_000,
                  '1h': 3_600_000, '4h': 14_400_000}.get(interval, 3_600_000)
            time.sleep(0.15)
            data = info.candles_snapshot(symbol, interval, end_ms - candles * iv, end_ms)
            if not data:
                return None
            df = pd.DataFrame(data).rename(columns={
                't': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
            for c in ['open', 'high', 'low', 'close', 'volume']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            self._candle_cache[ck] = (df.copy(), datetime.now())
            return df
        except Exception as e:
            cprint(f"[VolBreakout] Candle error {symbol}/{interval}: {e}", "yellow")
            return None

    # -- Setup detection --

    def _compute_setup(self, symbol: str):
        df = self._fetch_candles(symbol, '1h', 200)
        if df is None or len(df) < 100:
            return None
        close, high, low, volume = df['close'], df['high'], df['low'], df['volume']

        bb = ta.bbands(close, length=20, std=2.0)
        if bb is None or bb.empty:
            return None
        bbu, bbl, bbm = bb['BBU_20_2.0'], bb['BBL_20_2.0'], bb['BBM_20_2.0']
        bb_width = (bbu - bbl) / bbm.replace(0, 1)

        # At least one of the 2 last pre-breakout bars must have been in squeeze
        sq_thresh = bb_width.iloc[-100:].quantile(VOL_BREAKOUT_SQUEEZE_PERCENTILE)
        if len(bb_width) < 3 or bb_width.iloc[-3:-1].min() > sq_thresh:
            return None

        # Regime filter: skip breakouts in chop (low Kaufman Efficiency Ratio),
        # where the trailing-stop edge inverts. Same metric as Adaptive Hybrid's gate.
        try:
            from src import config as _cfg
            er_min = getattr(_cfg, 'VOL_BREAKOUT_ER_MIN', 0.0)
        except ImportError:
            er_min = 0.0
        if er_min > 0:
            from src.strategies.modules.regime_gate import efficiency_ratio
            er = efficiency_ratio(close.values, period=20)
            if er < er_min:
                return None

        vol_avg = volume.rolling(20).mean()
        vol_ratio = volume.iloc[-1] / vol_avg.iloc[-1] if vol_avg.iloc[-1] > 0 else 0
        # Dead market filter
        if vol_ratio < VOL_BREAKOUT_VOLUME_MIN:
            return None

        last_close = close.iloc[-1]
        if last_close > bbu.iloc[-1]:
            direction = 'BUY'
        elif last_close < bbl.iloc[-1]:
            direction = 'SELL'
        else:
            return None

        # Volume spike confirmation: breakout needs high volume
        if vol_ratio < VOL_BREAKOUT_VOLUME_MULT:
            return None

        adx = ta.adx(high, low, close, length=14)
        if adx is None or adx.empty:
            return None
        adx_val = adx['ADX_14'].iloc[-1]
        # Trend-strength gate (ADX > 25). The "ADX rising" check was rejecting
        # ~50% of valid squeezes intra-bar and has been dropped.
        if adx_val < VOL_BREAKOUT_ADX_ENTRY:
            return None
        # Late-entry gate: ADX already extended = breakout mostly done
        # (7 entries with ADX > 30 lost -$10.42 over 32 days)
        if adx_val > VOL_BREAKOUT_ADX_MAX:
            return None

        df4 = self._fetch_candles(symbol, '4h', 30)
        if df4 is None or len(df4) < 20:
            return None
        ema4 = ta.ema(df4['close'], length=20)
        if ema4 is None or ema4.empty:
            return None
        e4v = ema4.iloc[-1]
        if (direction == 'BUY' and last_close < e4v) or (direction == 'SELL' and last_close > e4v):
            return None

        atr_s = ta.atr(high, low, close, length=14)
        if atr_s is None or atr_s.empty:
            return None

        return {'symbol': symbol, 'direction': direction, 'price': last_close,
                'atr': atr_s.iloc[-1], 'adx': adx_val, 'bb_width': bb_width.iloc[-1],
                'volume_ratio': vol_ratio, 'ema_4h': e4v}

    # -- Position management --

    def _manage_positions(self):
        if not self.paper_positions:
            return
        to_close = []
        with self._lock:
            for pid, trade in self.paper_positions.items():
                sym, direction, entry_px, atr = trade['symbol'], trade['direction'], trade['entry_price'], trade['atr']
                try:
                    from src.nice_funcs_hyperliquid import ask_bid
                    a, b, _ = ask_bid(sym)
                    cpx = (a + b) / 2
                except Exception:
                    df = self._fetch_candles(sym, '1h', 5)
                    if df is None or len(df) == 0:
                        continue
                    cpx = float(df['close'].iloc[-1])
                trade['current_price'] = cpx
                trail_dist = atr * VOL_BREAKOUT_TRAILING_ATR_MULT

                if pid not in self._trailing:
                    self._trailing[pid] = {'highest': entry_px, 'lowest': entry_px}
                ts = self._trailing[pid]
                ts['highest'] = max(ts['highest'], cpx)
                ts['lowest'] = min(ts['lowest'], cpx)

                reason = None
                cl_px = cpx
                if direction == 'BUY' and cpx <= ts['highest'] - trail_dist:
                    reason, cl_px = 'TRAILING_STOP', ts['highest'] - trail_dist
                elif direction == 'SELL' and cpx >= ts['lowest'] + trail_dist:
                    reason, cl_px = 'TRAILING_STOP', ts['lowest'] + trail_dist

                if not reason:
                    et = trade.get('entry_time')
                    if isinstance(et, str):
                        try: et = datetime.fromisoformat(et)
                        except (ValueError, TypeError): et = None
                    if et and (datetime.now() - et).total_seconds() / 3600 >= VOL_BREAKOUT_MAX_HOLD_HOURS:
                        reason = 'TIME_STOP'

                if not reason:
                    et = trade.get('entry_time')
                    if isinstance(et, str):
                        try: et = datetime.fromisoformat(et)
                        except (ValueError, TypeError): et = None
                    held_secs = (datetime.now() - et).total_seconds() if et else float('inf')
                    # Min hold 1h before allowing ADX-based exit (avoid false exits on
                    # mechanical ADX decay right after breakout pic).
                    if held_secs >= 3600:
                        df = self._fetch_candles(sym, '1h', 30)
                        if df is not None and len(df) >= 17:
                            adx = ta.adx(df['high'], df['low'], df['close'], length=14)
                            # Multi-bar confirmation: 3 last 1H bars all sub-18.
                            if (adx is not None and not adx.empty
                                    and len(adx['ADX_14'].dropna()) >= 3
                                    and adx['ADX_14'].iloc[-3:].max() < 18):
                                reason = 'ADX_EXHAUSTION'

                if reason:
                    to_close.append((pid, cl_px, reason))

        for pid, px, reason in to_close:
            self._close_position(pid, px, reason)
            self._trailing.pop(pid, None)

    def _close_position(self, pid, close_px, reason):
        with self._lock:
            if pid not in self.paper_positions:
                return
            t = self.paper_positions[pid].copy()
            ep, d, sz, sym = t['entry_price'], t['direction'], t['position_size'], t['symbol']
            tc = _token_class(sym)
            slip = PAPER_SLIPPAGE_V2.get(tc, 0.001)
            efee = sz * PAPER_TAKER_FEE_V2
            ecp = close_px * (1 - slip) if d == 'BUY' else close_px * (1 + slip)
            pct = (ecp - ep) / ep if d == 'BUY' else (ep - ecp) / ep
            pnl = sz * pct - efee
            if abs(pnl) > sz * 1.5:
                pnl = max(-sz, min(sz, pnl))

            t.update({'close_price': close_px, 'effective_close_price': round(ecp, 6),
                       'exit_time': datetime.now().isoformat(), 'close_reason': reason,
                       'exit_fee': round(efee, 4), 'total_fees': round(t.get('entry_fee', 0) + efee, 4),
                       'pnl': round(pnl, 2), 'pnl_pct': round(pct * 100, 2), 'status': 'CLOSED'})
            self.daily_pnl += pnl
            self.paper_balance += pnl
            del self.paper_positions[pid]
            self.closed_positions.append(t)

        col = 'green' if pnl > 0 else 'red'
        pd_ = _price_dec(ep)
        cprint(f"\n[VolBreakout] CLOSED {sym} ({reason})", col, attrs=['bold'])
        cprint(f"  {ep:,.{pd_}f} -> {close_px:,.{pd_}f} | PnL ${pnl:+,.2f} ({pct*100:+.2f}%) | Bal ${self.paper_balance:,.2f}", col)
        self._log_closed_trade(t)
        self._update_paper_csv(pid)

        if self._trade_memory and 'memory_decision_id' in t:
            try:
                et = t.get('entry_time')
                if isinstance(et, str):
                    try: et = datetime.fromisoformat(et)
                    except: et = None
                hh = (datetime.now() - et).total_seconds() / 3600 if et else None
                self._trade_memory.update_outcome(t['memory_decision_id'], pnl=pnl,
                                                   hold_duration_hours=hh, close_reason=reason)
            except Exception:
                pass

    # -- Trade execution --

    def _open_trade(self, s):
        sym, d, px, atr = s['symbol'], s['direction'], s['price'], s['atr']
        tc = _token_class(sym)
        lev = VOL_BREAKOUT_LEVERAGE.get(tc, 3)
        trail_dist = atr * VOL_BREAKOUT_TRAILING_ATR_MULT
        sl_frac = max(trail_dist / px, 0.001)
        risk = self.paper_balance * VOL_BREAKOUT_RISK_PCT
        pos_sz = min(risk / sl_frac * lev, self.paper_balance * 0.25)

        with self._lock:
            used = sum(p['position_size'] / p.get('leverage', 3) for p in self.paper_positions.values())
            avail = max(0, self.paper_balance - used - self.paper_balance * CASH_PERCENTAGE / 100)
        pos_sz = min(pos_sz, avail * 0.9 * lev)
        if pos_sz < 10:
            return

        slip = PAPER_SLIPPAGE_V2.get(tc, 0.001)
        fill = px * (1 + slip) if d == 'BUY' else px * (1 - slip)
        fee = pos_sz * PAPER_TAKER_FEE_V2

        with self._lock:
            if self.paper_balance - fee < 0:
                return
            if any(p['symbol'] == sym for p in self.paper_positions.values()):
                return
            self._position_counter += 1
            pid = f"VB_{sym}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._position_counter}"
            trade = {
                'position_id': pid, 'timestamp': datetime.now().isoformat(),
                'entry_time': datetime.now(), 'symbol': sym, 'direction': d,
                'entry_price': fill, 'position_size': round(pos_sz, 2), 'leverage': lev,
                'atr': atr, 'entry_fee': round(fee, 4), 'adx': s.get('adx', 0),
                'bb_width': s.get('bb_width', 0), 'volume_ratio': s.get('volume_ratio', 0),
                'status': 'OPEN',
            }
            self.paper_positions[pid] = trade
            self.paper_balance -= fee
            self.daily_trades += 1

        pd_ = _price_dec(fill)
        cprint(f"\n[VolBreakout] OPENED {d} {sym} ({pid})", "magenta", attrs=['bold'])
        cprint(f"  ${fill:,.{pd_}f} | Size ${pos_sz:,.2f} | {lev}x | ADX {s['adx']:.1f} | Vol {s['volume_ratio']:.1f}x", "white")
        self._log_open_trade(trade)

        if self._trade_memory:
            try:
                did = self._trade_memory.log_decision(
                    symbol=sym, direction=d, confidence=min(100, s.get('adx', 50)),
                    source='volatility_breakout',
                    reasoning=f"Squeeze breakout: BB_w={s['bb_width']:.4f}, vol={s['volume_ratio']:.1f}x, ADX={s['adx']:.1f}")
                with self._lock:
                    if pid in self.paper_positions:
                        self.paper_positions[pid]['memory_decision_id'] = did
            except Exception:
                pass

    # -- Main cycle --

    def _check_daily_reset(self):
        """Reset daily counters at midnight UTC.

        Also called from main.independent_strategy_loop BEFORE the global
        daily-loss breaker is evaluated: when the breaker trips, run_cycle()
        is skipped, so without this hook daily_pnl would latch forever.
        """
        today = datetime.utcnow().date()
        if self._last_trade_date != today:
            self.daily_trades, self.daily_pnl, self._last_trade_date = 0, 0.0, today

    def run_cycle(self, symbols=None):
        symbols = symbols or VOL_BREAKOUT_SYMBOLS
        self._check_daily_reset()

        cprint(f"\n{'='*60}\n  [VolBreakout] {datetime.now():%Y-%m-%d %H:%M:%S} | "
               f"${self.paper_balance:,.2f} | {len(self.paper_positions)} open | "
               f"{self.daily_trades}/{VOL_BREAKOUT_MAX_DAILY_TRADES} trades | PnL ${self.daily_pnl:+,.2f}\n{'='*60}", "cyan")

        self._manage_positions()

        if self.daily_trades >= VOL_BREAKOUT_MAX_DAILY_TRADES:
            cprint("[VolBreakout] Daily trade limit reached", "yellow"); return
        if self.daily_pnl <= -VOL_BREAKOUT_MAX_DAILY_LOSS:
            cprint(f"[VolBreakout] Daily loss limit (${self.daily_pnl:+,.2f})", "red"); return
        self.peak_balance = max(self.peak_balance, self.paper_balance)
        dd = (self.peak_balance - self.paper_balance) / self.peak_balance * 100
        if dd >= RISK_MAX_DRAWDOWN_PCT:
            cprint(f"[VolBreakout] Drawdown breaker: {dd:.1f}%", "red"); return

        setups = []
        for sym in symbols:
            if any(p['symbol'] == sym for p in self.paper_positions.values()):
                continue
            try:
                s = self._compute_setup(sym)
                if s:
                    setups.append(s)
                    cprint(f"  [SETUP] {sym} {s['direction']} | ADX={s['adx']:.1f} | Vol={s['volume_ratio']:.1f}x", "green")
            except Exception as e:
                cprint(f"  [ERR] {sym}: {e}", "red")

        if not setups:
            cprint("[VolBreakout] No setups", "white"); return

        setups.sort(key=lambda x: x['adx'], reverse=True)
        for s in setups:
            if self.daily_trades >= VOL_BREAKOUT_MAX_DAILY_TRADES or self.daily_pnl <= -VOL_BREAKOUT_MAX_DAILY_LOSS:
                break
            self._open_trade(s)

    def generate_signals(self) -> dict:
        self.run_cycle()
        return {'token': 'MULTI', 'signal': 0, 'direction': 'NEUTRAL', 'metadata': self.get_status()}

    # -- CSV persistence --

    def _log_open_trade(self, t):
        try:
            f = os.path.join(self.data_dir, 'paper_trades.csv')
            safe = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in t.items()}
            df = pd.DataFrame([safe])
            df.to_csv(f, mode='a', header=not os.path.exists(f), index=False)
        except Exception as e:
            cprint(f"[VolBreakout] CSV err: {e}", "yellow")

    def _log_closed_trade(self, t):
        try:
            f = os.path.join(self.data_dir, 'closed_trades.csv')
            safe = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in t.items()}
            df = pd.DataFrame([safe])
            df.to_csv(f, mode='a', header=not os.path.exists(f), index=False)
        except Exception as e:
            cprint(f"[VolBreakout] CSV err: {e}", "yellow")

    def _update_paper_csv(self, closed_pid):
        try:
            f = os.path.join(self.data_dir, 'paper_trades.csv')
            if not os.path.exists(f): return
            df = pd.read_csv(f)
            if 'position_id' in df.columns:
                df.loc[df['position_id'] == closed_pid, 'status'] = 'CLOSED'
                df.to_csv(f, index=False)
        except Exception:
            pass

    def _load_state(self):
        try:
            pf = os.path.join(self.data_dir, 'paper_trades.csv')
            cf = os.path.join(self.data_dir, 'closed_trades.csv')
            if os.path.exists(pf):
                df = pd.read_csv(pf)
                if not df.empty:
                    for _, r in df[df['status'] == 'OPEN'].iterrows():
                        pid = r.get('position_id', '')
                        if pid:
                            self.paper_positions[pid] = {
                                'position_id': pid, 'timestamp': r.get('timestamp', ''),
                                'entry_time': r.get('timestamp', datetime.now().isoformat()),
                                'symbol': r.get('symbol', ''), 'direction': r.get('direction', 'BUY'),
                                'entry_price': float(r.get('entry_price', 0)),
                                'position_size': float(r.get('position_size', 0)),
                                'leverage': float(r.get('leverage', 3)),
                                'atr': float(r.get('atr', 0) or 0),
                                'entry_fee': float(r.get('entry_fee', 0) or 0), 'status': 'OPEN'}
                    if self.paper_positions:
                        mx = 0
                        for p in self.paper_positions:
                            parts = p.split('_')
                            if len(parts) >= 4:
                                try: mx = max(mx, int(parts[-1]))
                                except ValueError: pass
                        self._position_counter = mx

            rpnl, cfees = 0.0, 0.0
            if os.path.exists(cf):
                cdf = pd.read_csv(cf)
                if not cdf.empty and 'pnl' in cdf.columns:
                    rpnl = cdf['pnl'].sum()
                    if 'entry_fee' in cdf.columns:
                        cfees = cdf['entry_fee'].fillna(0).sum()
                    self.closed_positions = cdf.to_dict('records')
            ofees = sum(t.get('entry_fee', 0) for t in self.paper_positions.values())
            self.paper_balance = PAPER_TRADING_BALANCE + rpnl - cfees - ofees
            cprint(f"[VolBreakout] Loaded: {len(self.paper_positions)} open, "
                   f"{len(self.closed_positions)} closed, ${self.paper_balance:,.2f}", "cyan")
        except Exception as e:
            cprint(f"[VolBreakout] Load err: {e}", "yellow")

    def get_status(self):
        with self._lock:
            return {
                'balance': round(self.paper_balance, 2),
                'initial_balance': PAPER_TRADING_BALANCE,
                'total_pnl': round(self.paper_balance - PAPER_TRADING_BALANCE, 2),
                'daily_pnl': round(self.daily_pnl, 2),
                'daily_trades': self.daily_trades,
                'open_positions': len(self.paper_positions),
                'total_closed': len(self.closed_positions),
                'positions': [p.copy() for p in self.paper_positions.values()]}


if __name__ == '__main__':
    import sys
    cprint("\n" + "=" * 60, "cyan")
    cprint("  Volatility Compression Breakout Strategy", "cyan", attrs=['bold'])
    cprint("=" * 60 + "\n", "cyan")
    strat = VolatilityBreakoutStrategy()
    once = '--once' in sys.argv
    try:
        while True:
            strat.run_cycle()
            if once: break
            cprint(f"\n[VolBreakout] Sleeping 5m...\n", "white")
            time.sleep(300)
    except KeyboardInterrupt:
        s = strat.get_status()
        cprint(f"\n[VolBreakout] Final: ${s['balance']:,.2f} | PnL ${s['total_pnl']:+,.2f} | {s['total_closed']} closed", "yellow")
