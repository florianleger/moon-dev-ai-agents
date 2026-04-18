"""
Fibonacci OTE (Optimal Trade Entry) Scalping Strategy

Edge: In a trending market, pullbacks into the 0.618-0.786 Fibonacci retracement
zone of the latest impulse provide high-probability scalp entries in the direction
of the higher-timeframe trend.

Pipeline:
 - H1 trend filter (EMA20 / EMA50 / EMA200 alignment + ADX >= 20)
 - M5 swing detection -> Fib OTE zone
 - Volume + session + funding-window filters
 - Fixed 1:2 R/R, time stop 60min, BOS -> breakeven SL
"""

import os
import time as _time
import threading
from datetime import datetime, timedelta
from collections import deque
from termcolor import cprint

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    try:
        import pandas_ta_classic as ta
    except ImportError:
        ta = None

from src.strategies.base_strategy import BaseStrategy
from src.strategies.modules.swing_detector import detect_swings
from src.strategies.modules.fib_ote import compute_fib_levels, is_price_in_ote
from src.strategies.modules.bos_detector import detect_bos

try:
    from src.data_providers.market_data import MarketDataProvider
except Exception:
    MarketDataProvider = None

try:
    from src.data.trade_memory import TradeMemory
except Exception:
    TradeMemory = None

import src.config as config


def _price_dec(price: float) -> int:
    """Return suitable decimal count for log formatting."""
    if price is None or price <= 0:
        return 4
    if price >= 100:
        return 2
    if price >= 1:
        return 3
    if price >= 0.01:
        return 5
    return 7


# Token classification (mirrors the pattern used by sibling strategies)
_TOKEN_CLASS = {
    'BTC': 'btc', 'ETH': 'eth',
    'SOL': 'mid', 'XRP': 'mid', 'AVAX': 'mid', 'LINK': 'mid',
    'ADA': 'mid', 'AAVE': 'mid', 'NEAR': 'mid', 'SUI': 'mid', 'TAO': 'mid',
    'DOGE': 'alt', 'kPEPE': 'alt', 'ENA': 'alt',
}


class OteScalperStrategy(BaseStrategy):
    """Fibonacci OTE (Optimal Trade Entry) scalping strategy.

    Entry: 0.618-0.786 Fib retracement zone in direction of H1 trend.
    Exit:  Fixed 1:2 R/R or time stop (60min).
    Management: Move SL to breakeven on BOS (break of structure).
    """

    # --- Initialisation ---
    def __init__(self):
        super().__init__("OTE Scalper")

        # -- Config (loaded via getattr w/ safe defaults) --
        self.tokens = list(getattr(config, 'OTE_SCALP_TOKENS', ['BTC', 'ETH', 'SOL']))
        self.assets = self.tokens  # compatibility alias

        self.trend_tf = getattr(config, 'OTE_SCALP_TREND_TIMEFRAME', '1h')
        self.entry_tf = getattr(config, 'OTE_SCALP_ENTRY_TIMEFRAME', '5m')
        self.ema_fast = getattr(config, 'OTE_SCALP_TREND_EMA_FAST', 20)
        self.ema_slow = getattr(config, 'OTE_SCALP_TREND_EMA_SLOW', 50)
        self.ema_filter = getattr(config, 'OTE_SCALP_TREND_EMA_FILTER', 200)
        self.adx_min = getattr(config, 'OTE_SCALP_ADX_MIN', 20)
        self.swing_lookback = getattr(config, 'OTE_SCALP_SWING_LOOKBACK', 24)
        self.swing_min_range_pct = getattr(config, 'OTE_SCALP_SWING_MIN_RANGE_PCT', 0.004)
        self.rr_ratio = getattr(config, 'OTE_SCALP_RR_RATIO', 2.0)
        self.sl_buffer_atr = getattr(config, 'OTE_SCALP_SL_BUFFER_ATR', 0.2)
        self.risk_pct = getattr(config, 'OTE_SCALP_RISK_PCT', 0.005)
        self.max_hold_minutes = getattr(config, 'OTE_SCALP_MAX_HOLD_MINUTES', 60)
        self.max_daily_trades = getattr(config, 'OTE_SCALP_MAX_DAILY_TRADES', 10)
        self.max_daily_loss_usd = getattr(config, 'OTE_SCALP_MAX_DAILY_LOSS_USD', 20.0)
        self.max_positions = getattr(config, 'OTE_SCALP_MAX_POSITIONS', 2)
        self.cooldown_minutes = getattr(config, 'OTE_SCALP_COOLDOWN_MINUTES', 15)
        self.leverage_map = getattr(config, 'OTE_SCALP_LEVERAGE',
                                    {'btc': 3, 'eth': 3, 'mid': 3, 'alt': 2})
        self.volume_min = getattr(config, 'OTE_SCALP_VOLUME_MIN', 0.15)
        self.avoid_hours = set(getattr(config, 'OTE_SCALP_AVOID_HOURS_UTC', [1, 2, 3]))
        self.funding_window_min = getattr(config, 'OTE_SCALP_FUNDING_AVOID_WINDOW_MIN', 5)
        self.bos_buffer_pct = getattr(config, 'OTE_SCALP_BOS_BUFFER_PCT', 0.001)
        self.max_position_pct = getattr(config, 'OTE_SCALP_MAX_POSITION_PCT', 25) / 100.0

        self.paper_balance_initial = getattr(config, 'PAPER_TRADING_BALANCE', 500.0)
        self.slippage_map = getattr(config, 'PAPER_SLIPPAGE_V2',
                                    {'btc': 0.0003, 'eth': 0.0005, 'mid': 0.0012, 'alt': 0.003})
        self.taker_fee = getattr(config, 'PAPER_TAKER_FEE_V2', 0.00045)

        # -- Runtime state --
        self.paper_balance = self.paper_balance_initial
        self.positions = {}           # active positions keyed by position_id
        self.closed_positions = []
        self._position_counter = 0
        self._lock = threading.RLock()

        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.utcnow().date()

        self._last_close_time = {}    # symbol -> datetime of last exit (cooldowns)
        self._trend_cache = {}        # symbol -> (trend, datetime) cached 15min

        # Data providers / trade memory
        self._market_data = None
        if MarketDataProvider is not None:
            try:
                self._market_data = MarketDataProvider(start_liquidation_stream=False)
            except Exception as e:
                cprint(f"[OteScalp] MarketDataProvider init failed: {e}", "yellow")

        self._trade_memory = None
        if TradeMemory is not None:
            try:
                self._trade_memory = TradeMemory.get_instance()
            except Exception:
                pass

        # Persistence directory
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data', 'ote_scalp')
        os.makedirs(self.data_dir, exist_ok=True)

        self._load_state()

        cprint(f"[OteScalp] Init | ${self.paper_balance:,.2f} | "
               f"{len(self.positions)} open | {len(self.tokens)} tokens "
               f"| RR {self.rr_ratio} | Risk {self.risk_pct*100:.2f}%", "cyan")

    # --- Main cycle ---
    def run_cycle(self, symbols=None):
        """Single pass: reset counters, manage open trades, then scan for entries."""
        if symbols:
            self.tokens = list(symbols)
            self.assets = self.tokens

        self._reset_daily_if_new_day()

        t0 = _time.time()
        cprint(f"\n[OteScalp] === {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC | "
               f"Bal ${self.paper_balance:,.2f} | Open {len(self.positions)}/{self.max_positions} | "
               f"Trades {self.daily_trades}/{self.max_daily_trades} | "
               f"PnL ${self.daily_pnl:+.2f} ===", "cyan")

        # 1) Manage existing positions first (SL/TP/time stop/BOS -> BE)
        if self.positions:
            self._manage_positions()

        # 2) Kill-switches
        if self.daily_pnl <= -self.max_daily_loss_usd:
            cprint(f"[OteScalp] Daily loss limit hit (${self.daily_pnl:+.2f}) — no new entries",
                   "red")
            return
        if self.daily_trades >= self.max_daily_trades:
            cprint("[OteScalp] Daily trade limit reached — no new entries", "yellow")
            return
        if len(self.positions) >= self.max_positions:
            return

        # 3) Scan for new entries
        self._scan_for_entries()

        cprint(f"[OteScalp] Cycle done in {_time.time() - t0:.1f}s", "cyan")

    def _reset_daily_if_new_day(self):
        today = datetime.utcnow().date()
        if today != self.last_reset_date:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_reset_date = today

    # --- Entry scanning ---
    def _scan_for_entries(self):
        # Skip global no-trade windows first
        if self._is_avoid_hour():
            cprint(f"[OteScalp] In avoid-hour window ({datetime.utcnow().hour} UTC) — skip",
                   "white")
            return
        if self._is_funding_window():
            cprint("[OteScalp] Near funding settlement — skip", "white")
            return

        open_syms = {p['symbol'] for p in self.positions.values()}
        for sym in self.tokens:
            if sym in open_syms:
                continue
            if len(self.positions) >= self.max_positions:
                break
            if self.daily_trades >= self.max_daily_trades:
                break

            # Per-symbol cooldown
            lc = self._last_close_time.get(sym)
            if lc and (datetime.utcnow() - lc).total_seconds() < self.cooldown_minutes * 60:
                continue

            try:
                sig = self._check_entry(sym)
            except Exception as e:
                cprint(f"[OteScalp] check_entry {sym} err: {e}", "yellow")
                continue

            if sig:
                self._execute_trade(sig)

    def _check_entry(self, symbol):
        """Return a signal dict or None if no qualifying setup on this symbol."""
        trend = self._detect_trend_h1(symbol)
        if trend == 'NEUTRAL':
            return None

        m5 = self._fetch_candles(symbol, self.entry_tf, 60)
        if m5 is None or len(m5) < max(self.swing_lookback + 5, 25):
            return None

        swings = detect_swings(m5, lookback=self.swing_lookback,
                               min_range_pct=self.swing_min_range_pct)
        if not swings.get('valid'):
            return None

        swing_dir = swings['direction']
        if trend == 'BULLISH' and swing_dir != 'UP':
            return None
        if trend == 'BEARISH' and swing_dir != 'DOWN':
            return None

        fib = compute_fib_levels(swings['swing_low'], swings['swing_high'],
                                 'UP' if trend == 'BULLISH' else 'DOWN')
        if fib.get('impulse_range', 0) <= 0:
            return None

        current_price = float(m5['close'].iloc[-1])
        if not is_price_in_ote(current_price, fib):
            return None

        # Volume filter: last bar vs 20-period avg
        vol = float(m5['volume'].iloc[-1])
        vol_avg = float(m5['volume'].iloc[-20:].mean()) if len(m5) >= 20 else 0.0
        vol_ratio = (vol / vol_avg) if vol_avg > 0 else 0.0
        if vol_ratio < self.volume_min:
            return None

        atr = self._compute_atr(m5, length=14)
        if atr <= 0:
            return None

        direction = 'BUY' if trend == 'BULLISH' else 'SELL'

        # SL/TP compute
        if direction == 'BUY':
            sl = swings['swing_low'] - (atr * self.sl_buffer_atr)
            risk_dist = current_price - sl
        else:
            sl = swings['swing_high'] + (atr * self.sl_buffer_atr)
            risk_dist = sl - current_price

        if risk_dist <= 0:
            return None

        tp = current_price + risk_dist * self.rr_ratio if direction == 'BUY' \
            else current_price - risk_dist * self.rr_ratio

        leverage = self._get_leverage(symbol)

        return {
            'symbol': symbol,
            'direction': direction,
            'entry_price': current_price,
            'swing_low': swings['swing_low'],
            'swing_high': swings['swing_high'],
            'ote_mid': fib['ote_mid'],
            'atr': atr,
            'stop_loss': sl,
            'take_profit': tp,
            'risk_dist': risk_dist,
            'leverage': leverage,
            'trend': trend,
            'vol_ratio': vol_ratio,
            'range_pct': swings.get('range_pct', 0.0),
        }

    # --- Trend detection (H1) ---
    def _detect_trend_h1(self, symbol):
        """Return 'BULLISH', 'BEARISH' or 'NEUTRAL'. Cached 15min per symbol."""
        cached = self._trend_cache.get(symbol)
        if cached and (datetime.utcnow() - cached[1]).total_seconds() < 900:
            return cached[0]

        df = self._fetch_candles(symbol, self.trend_tf, 250)
        if df is None or len(df) < max(self.ema_filter, 50):
            self._trend_cache[symbol] = ('NEUTRAL', datetime.utcnow())
            return 'NEUTRAL'

        close = df['close']
        high = df['high']
        low = df['low']

        ema_fast = close.ewm(span=self.ema_fast, adjust=False).mean().iloc[-1]
        ema_slow = close.ewm(span=self.ema_slow, adjust=False).mean().iloc[-1]
        ema_filter = close.ewm(span=self.ema_filter, adjust=False).mean().iloc[-1]
        last_close = float(close.iloc[-1])

        adx_val = 0.0
        if ta is not None:
            try:
                adx_df = ta.adx(high, low, close, length=14)
                if adx_df is not None and not adx_df.empty:
                    # column name is 'ADX_14'
                    col = [c for c in adx_df.columns if c.upper().startswith('ADX')]
                    if col:
                        val = adx_df[col[0]].iloc[-1]
                        adx_val = float(val) if pd.notna(val) else 0.0
            except Exception:
                adx_val = 0.0
        if adx_val <= 0:
            # Manual ADX fallback via Wilder's smoothing
            adx_val = self._manual_adx(high, low, close, length=14)

        if (last_close > ema_filter and ema_fast > ema_slow > ema_filter
                and adx_val >= self.adx_min):
            trend = 'BULLISH'
        elif (last_close < ema_filter and ema_fast < ema_slow < ema_filter
              and adx_val >= self.adx_min):
            trend = 'BEARISH'
        else:
            trend = 'NEUTRAL'

        self._trend_cache[symbol] = (trend, datetime.utcnow())
        return trend

    @staticmethod
    def _manual_adx(high, low, close, length=14):
        """Simple ADX fallback if pandas_ta is unavailable/errored."""
        try:
            h = high.astype(float)
            l = low.astype(float)
            c = close.astype(float)
            prev_close = c.shift(1)
            tr = pd.concat([
                (h - l),
                (h - prev_close).abs(),
                (l - prev_close).abs(),
            ], axis=1).max(axis=1)
            up = h.diff()
            dn = -l.diff()
            plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
            minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)

            atr = tr.rolling(length).mean()
            plus_di = 100 * (pd.Series(plus_dm, index=h.index).rolling(length).mean() / atr)
            minus_di = 100 * (pd.Series(minus_dm, index=h.index).rolling(length).mean() / atr)
            di_diff = pd.Series(plus_di - minus_di)
            di_sum = pd.Series(plus_di + minus_di)
            dx = (100 * di_diff.abs()) / di_sum.replace(0, np.nan)
            adx = dx.rolling(length).mean()
            val = adx.iloc[-1]
            return float(val) if pd.notna(val) else 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Session filters
    # ------------------------------------------------------------------
    def _is_avoid_hour(self):
        return datetime.utcnow().hour in self.avoid_hours

    def _is_funding_window(self):
        """Skip a window around funding settlement (00/08/16 UTC)."""
        now = datetime.utcnow()
        for hr in (0, 8, 16):
            boundary = now.replace(hour=hr, minute=0, second=0, microsecond=0)
            diff_min = abs((now - boundary).total_seconds()) / 60.0
            if diff_min <= self.funding_window_min:
                return True
        return False

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------
    def _manage_positions(self):
        to_close = []
        with self._lock:
            for pid, pos in list(self.positions.items()):
                sym = pos['symbol']
                price = self._get_price(sym)
                if price is None:
                    continue

                d = pos['direction']
                # SL / TP hit?
                if d == 'BUY':
                    if price <= pos['stop_loss']:
                        to_close.append((pid, pos['stop_loss'], 'STOP_LOSS'))
                        continue
                    if price >= pos['take_profit']:
                        to_close.append((pid, pos['take_profit'], 'TAKE_PROFIT'))
                        continue
                else:  # SELL
                    if price >= pos['stop_loss']:
                        to_close.append((pid, pos['stop_loss'], 'STOP_LOSS'))
                        continue
                    if price <= pos['take_profit']:
                        to_close.append((pid, pos['take_profit'], 'TAKE_PROFIT'))
                        continue

                # Time stop?
                try:
                    et = datetime.fromisoformat(pos['entry_time'])
                    elapsed_min = (datetime.utcnow() - et).total_seconds() / 60.0
                except Exception:
                    elapsed_min = 0.0
                if elapsed_min >= self.max_hold_minutes:
                    to_close.append((pid, price, 'TIME_STOP'))
                    continue

                # BOS -> move SL to breakeven (once)
                if not pos.get('bos_triggered'):
                    self._check_bos_and_move_sl(pos)

        for pid, px, reason in to_close:
            self._close_position(pid, px, reason)

    def _check_bos_and_move_sl(self, position):
        """If price breaks structure past the impulse reference, move SL to breakeven."""
        sym = position['symbol']
        df = self._fetch_candles(sym, self.entry_tf, 20)
        if df is None or len(df) < 5:
            return

        d = position['direction']
        ref_high = position.get('swing_high', 0.0)
        ref_low = position.get('swing_low', 0.0)
        bos = detect_bos(df, reference_high=ref_high, reference_low=ref_low,
                        direction=d, lookback=10)
        if not bos.get('bos_detected'):
            return

        entry = position['entry_price']
        buf = entry * self.bos_buffer_pct
        new_sl = entry + buf if d == 'BUY' else entry - buf

        # Only ratchet SL in a favourable direction
        with self._lock:
            cur_sl = position.get('stop_loss')
            if d == 'BUY' and new_sl > cur_sl:
                position['stop_loss'] = new_sl
                position['bos_triggered'] = True
                cprint(f"[OteScalp] BOS {sym} BUY — SL -> BE ${new_sl:,.{_price_dec(new_sl)}f}",
                       "cyan")
            elif d == 'SELL' and new_sl < cur_sl:
                position['stop_loss'] = new_sl
                position['bos_triggered'] = True
                cprint(f"[OteScalp] BOS {sym} SELL — SL -> BE ${new_sl:,.{_price_dec(new_sl)}f}",
                       "cyan")

    def _close_position(self, position_id, exit_price, reason):
        with self._lock:
            pos = self.positions.get(position_id)
            if not pos:
                return
            sym = pos['symbol']
            d = pos['direction']
            entry = pos['entry_price']
            size = pos['position_size']
            tc = self._get_token_class(sym)
            slip = self.slippage_map.get(tc, 0.0012)

            eff_exit = exit_price * (1 - slip) if d == 'BUY' else exit_price * (1 + slip)
            pnl_pct = (eff_exit - entry) / entry if d == 'BUY' else (entry - eff_exit) / entry
            pnl_gross = pnl_pct * size
            exit_fee = size * self.taker_fee
            pnl_net = pnl_gross - exit_fee

            # Sanity clamp (cannot lose more than the notional)
            if abs(pnl_net) > size * 1.5:
                pnl_net = max(-size, min(size, pnl_net))

            self.paper_balance += pnl_net
            self.daily_pnl += pnl_net
            self._last_close_time[sym] = datetime.utcnow()

            try:
                et = datetime.fromisoformat(pos['entry_time'])
                hold_min = (datetime.utcnow() - et).total_seconds() / 60.0
            except Exception:
                hold_min = 0.0

            pos.update({
                'status': 'CLOSED',
                'exit_price': exit_price,
                'effective_exit_price': round(eff_exit, 6),
                'exit_time': datetime.utcnow().isoformat(),
                'pnl': round(pnl_net, 4),
                'pnl_pct': round(pnl_pct * 100, 2),
                'close_reason': reason,
                'hold_minutes': round(hold_min, 1),
                'exit_fee': round(exit_fee, 6),
            })
            closed = pos.copy()
            self.closed_positions.append(closed)
            del self.positions[position_id]

        col = 'green' if pnl_net > 0 else 'red'
        pd_ = _price_dec(entry)
        cprint(f"[OteScalp] CLOSED {d} {sym} ({reason}) | "
               f"{entry:,.{pd_}f} -> {exit_price:,.{pd_}f} | "
               f"PnL ${pnl_net:+,.2f} ({pnl_pct*100:+.2f}%) | "
               f"{hold_min:.0f}m | Bal ${self.paper_balance:,.2f}",
               col, attrs=['bold'])

        self._log_closed_trade(closed)
        self._update_open_csv(position_id, closed)

        if self._trade_memory and 'memory_decision_id' in closed:
            try:
                self._trade_memory.update_outcome(
                    int(closed['memory_decision_id']),
                    pnl=pnl_net,
                    hold_duration_hours=hold_min / 60.0,
                    close_reason=reason,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def _execute_trade(self, signal):
        sym = signal['symbol']
        d = signal['direction']
        px = signal['entry_price']
        atr = signal['atr']
        sl = signal['stop_loss']
        tp = signal['take_profit']
        lev = signal['leverage']

        tc = self._get_token_class(sym)
        slip = self.slippage_map.get(tc, 0.0012)
        fill_price = px * (1 + slip) if d == 'BUY' else px * (1 - slip)

        size = self._calculate_position_size(signal)
        if size < 10:
            cprint(f"[OteScalp] {sym} size ${size:.2f} < $10 min — skip", "yellow")
            return

        entry_fee = size * self.taker_fee
        with self._lock:
            if self.paper_balance - entry_fee < 1:
                cprint("[OteScalp] Insufficient paper balance for fee — skip", "yellow")
                return
            self._position_counter += 1
            pid = f"ote_{sym}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{self._position_counter}"

            trade = {
                'position_id': pid,
                'timestamp': datetime.utcnow().isoformat(),
                'entry_time': datetime.utcnow().isoformat(),
                'symbol': sym,
                'direction': d,
                'entry_price': fill_price,
                'position_size': round(size, 2),
                'leverage': lev,
                'stop_loss': sl,
                'take_profit': tp,
                'atr': atr,
                'swing_high': signal['swing_high'],
                'swing_low': signal['swing_low'],
                'ote_mid': signal['ote_mid'],
                'trend': signal['trend'],
                'vol_ratio': round(signal['vol_ratio'], 2),
                'range_pct': round(signal['range_pct'] * 100, 2),
                'entry_fee': round(entry_fee, 6),
                'status': 'OPEN',
                'bos_triggered': False,
            }
            self.positions[pid] = trade
            self.paper_balance -= entry_fee
            self.daily_trades += 1

        pd_ = _price_dec(fill_price)
        cprint(f"\n[OteScalp] OPENED {d} {sym} ({pid})", "magenta", attrs=['bold'])
        cprint(f"  Entry ${fill_price:,.{pd_}f} | Size ${size:,.2f} | {lev}x | "
               f"Trend {signal['trend']}", "white")
        cprint(f"  SL ${sl:,.{pd_}f} | TP ${tp:,.{pd_}f} | ATR {atr:.4f} | "
               f"Vol {signal['vol_ratio']:.1f}x | Range {signal['range_pct']*100:.2f}%",
               "white")

        self._log_open_trade(trade)

        if self._trade_memory:
            try:
                did = self._trade_memory.log_decision(
                    symbol=sym, direction=d,
                    confidence=min(100, 50 + signal['vol_ratio'] * 20),
                    source='ote_scalp',
                    reasoning=(f"OTE {d} | {signal['trend']} H1 trend | "
                               f"ATR {atr:.4f} | Vol {signal['vol_ratio']:.1f}x | "
                               f"Range {signal['range_pct']*100:.2f}%"),
                    key_indicators={
                        'atr': atr, 'vol_ratio': signal['vol_ratio'],
                        'range_pct': signal['range_pct'],
                    })
                with self._lock:
                    if pid in self.positions:
                        self.positions[pid]['memory_decision_id'] = did
            except Exception:
                pass

    def _calculate_position_size(self, signal):
        """Risk-based sizing capped by max_position_pct of balance."""
        risk_usd = self.paper_balance * self.risk_pct
        entry = signal['entry_price']
        risk_dist = signal['risk_dist']
        lev = signal['leverage']
        if risk_dist <= 0 or entry <= 0:
            return 0.0
        sl_fraction = risk_dist / entry
        raw_size = (risk_usd / sl_fraction) * lev
        cap = self.paper_balance * self.max_position_pct * lev
        return max(0.0, min(raw_size, cap))

    def _get_token_class(self, symbol):
        return _TOKEN_CLASS.get(symbol, 'mid')

    def _get_leverage(self, symbol):
        return self.leverage_map.get(self._get_token_class(symbol), 3)

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------
    def _fetch_candles(self, symbol, timeframe, limit):
        """Fetch OHLCV from HyperLiquid (same pattern as sibling strategies)."""
        try:
            from hyperliquid.info import Info
            info = Info(skip_ws=True, timeout=15)
            iv_ms = {
                '1m': 60_000, '5m': 300_000, '15m': 900_000,
                '1h': 3_600_000, '4h': 14_400_000, '1d': 86_400_000,
            }.get(timeframe, 300_000)
            end_ms = int(_time.time() * 1000)
            start_ms = end_ms - (limit * iv_ms)
            _time.sleep(0.15)
            raw = info.candles_snapshot(symbol, timeframe, start_ms, end_ms)
            if not raw:
                return None
            df = pd.DataFrame(raw).rename(columns={
                't': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low',
                'c': 'close', 'v': 'volume',
            })
            for c in ('open', 'high', 'low', 'close', 'volume'):
                df[c] = pd.to_numeric(df[c], errors='coerce')
            return df.dropna(subset=['close']).reset_index(drop=True)
        except Exception as e:
            cprint(f"[OteScalp] Fetch candles {symbol}/{timeframe} err: {e}", "yellow")
            return None

    def _compute_atr(self, ohlcv_df, length=14):
        """ATR via Wilder's smoothing (pandas-native, no dependency on ta)."""
        try:
            h = ohlcv_df['high'].astype(float)
            l = ohlcv_df['low'].astype(float)
            c = ohlcv_df['close'].astype(float)
            prev_c = c.shift(1)
            tr = pd.concat([
                (h - l),
                (h - prev_c).abs(),
                (l - prev_c).abs(),
            ], axis=1).max(axis=1)
            atr = tr.ewm(alpha=1.0 / length, adjust=False).mean()
            val = atr.iloc[-1]
            return float(val) if pd.notna(val) else 0.0
        except Exception:
            return 0.0

    def _get_price(self, symbol):
        """Latest price (market_data provider first, candle fallback)."""
        if self._market_data is not None:
            try:
                p = self._market_data.get_current_price(symbol)
                if p is not None and p > 0:
                    return float(p)
            except Exception:
                pass
        df = self._fetch_candles(symbol, self.entry_tf, 2)
        if df is not None and len(df) > 0:
            return float(df['close'].iloc[-1])
        return None

    # ------------------------------------------------------------------
    # BaseStrategy hook
    # ------------------------------------------------------------------
    def generate_signals(self) -> dict:
        self.run_cycle()
        return {
            'token': 'MULTI',
            'signal': 0,
            'direction': 'NEUTRAL',
            'metadata': self.get_status(),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _log_open_trade(self, trade):
        try:
            f = os.path.join(self.data_dir, 'paper_trades.csv')
            df = pd.DataFrame([trade])
            df.to_csv(f, mode='a', header=not os.path.exists(f), index=False)
        except Exception as e:
            cprint(f"[OteScalp] CSV open log err: {e}", "yellow")

    def _log_closed_trade(self, trade):
        try:
            f = os.path.join(self.data_dir, 'closed_trades.csv')
            df = pd.DataFrame([trade])
            df.to_csv(f, mode='a', header=not os.path.exists(f), index=False)
        except Exception as e:
            cprint(f"[OteScalp] CSV closed log err: {e}", "yellow")

    def _update_open_csv(self, position_id, trade):
        try:
            p = os.path.join(self.data_dir, 'paper_trades.csv')
            if not os.path.exists(p):
                return
            df = pd.read_csv(p)
            if 'position_id' not in df.columns:
                return
            mask = df['position_id'] == position_id
            if mask.any():
                for k in ('status', 'exit_price', 'exit_time', 'pnl',
                          'pnl_pct', 'close_reason', 'hold_minutes'):
                    if k in trade:
                        df.loc[mask, k] = trade.get(k, '')
                df.to_csv(p, index=False)
        except Exception as e:
            cprint(f"[OteScalp] CSV update err: {e}", "yellow")

    def _save_state(self):
        """No-op — we persist trade-by-trade via _log_* methods."""
        pass

    def _load_state(self):
        pf = os.path.join(self.data_dir, 'paper_trades.csv')
        cf = os.path.join(self.data_dir, 'closed_trades.csv')

        if os.path.exists(pf):
            try:
                df = pd.read_csv(pf)
                if not df.empty and 'status' in df.columns:
                    open_rows = df[df['status'] == 'OPEN']
                    for _, r in open_rows.iterrows():
                        pid = r.get('position_id', '')
                        if not pid:
                            continue
                        self.positions[pid] = {
                            'position_id': pid,
                            'timestamp': r.get('timestamp', ''),
                            'entry_time': r.get('entry_time',
                                                 r.get('timestamp',
                                                       datetime.utcnow().isoformat())),
                            'symbol': r.get('symbol', ''),
                            'direction': r.get('direction', 'BUY'),
                            'entry_price': float(r.get('entry_price', 0) or 0),
                            'position_size': float(r.get('position_size', 0) or 0),
                            'leverage': float(r.get('leverage', 3) or 3),
                            'stop_loss': float(r.get('stop_loss', 0) or 0),
                            'take_profit': float(r.get('take_profit', 0) or 0),
                            'atr': float(r.get('atr', 0) or 0),
                            'swing_high': float(r.get('swing_high', 0) or 0),
                            'swing_low': float(r.get('swing_low', 0) or 0),
                            'ote_mid': float(r.get('ote_mid', 0) or 0),
                            'trend': r.get('trend', 'NEUTRAL'),
                            'vol_ratio': float(r.get('vol_ratio', 0) or 0),
                            'range_pct': float(r.get('range_pct', 0) or 0),
                            'entry_fee': float(r.get('entry_fee', 0) or 0),
                            'status': 'OPEN',
                            'bos_triggered': bool(r.get('bos_triggered', False)),
                        }
                    if self.positions:
                        mx = 0
                        for pid in self.positions:
                            parts = str(pid).split('_')
                            try:
                                mx = max(mx, int(parts[-1]))
                            except (ValueError, IndexError):
                                pass
                        self._position_counter = mx
            except Exception as e:
                cprint(f"[OteScalp] Load open trades err: {e}", "yellow")

        realized, cfees = 0.0, 0.0
        if os.path.exists(cf):
            try:
                cdf = pd.read_csv(cf)
                if not cdf.empty:
                    if 'pnl' in cdf.columns:
                        realized = cdf['pnl'].fillna(0).sum()
                    if 'entry_fee' in cdf.columns:
                        cfees = cdf['entry_fee'].fillna(0).sum()
                    self.closed_positions = cdf.to_dict('records')
            except Exception as e:
                cprint(f"[OteScalp] Load closed trades err: {e}", "yellow")

        open_fees = sum(t.get('entry_fee', 0) for t in self.positions.values())
        self.paper_balance = self.paper_balance_initial + realized - cfees - open_fees

        today_str = datetime.utcnow().date().isoformat()
        self.daily_trades = sum(
            1 for p in list(self.positions.values()) + self.closed_positions
            if str(p.get('timestamp', ''))[:10] == today_str)
        self.daily_pnl = sum(
            p.get('pnl', 0) for p in self.closed_positions
            if str(p.get('exit_time', ''))[:10] == today_str)

    # ------------------------------------------------------------------
    # Status (dashboard)
    # ------------------------------------------------------------------
    def get_status(self):
        with self._lock:
            return {
                'strategy': 'ote_scalp',
                'balance': round(self.paper_balance, 2),
                'initial_balance': self.paper_balance_initial,
                'total_pnl': round(self.paper_balance - self.paper_balance_initial, 2),
                'daily_pnl': round(self.daily_pnl, 2),
                'daily_trades': self.daily_trades,
                'open_positions': len(self.positions),
                'total_closed': len(self.closed_positions),
                'positions': [p.copy() for p in self.positions.values()],
            }


# ----------------------------------------------------------------------
# Standalone entry point
# ----------------------------------------------------------------------
if __name__ == '__main__':
    import signal
    import sys

    cprint("\n" + "=" * 60, "cyan")
    cprint("  Fibonacci OTE Scalping Strategy", "cyan", attrs=['bold'])
    cprint("=" * 60 + "\n", "cyan")

    strat = OteScalperStrategy()

    def _exit(sig, frame):
        s = strat.get_status()
        cprint(f"\n[OteScalp] Final: ${s['balance']:,.2f} | "
               f"PnL ${s['total_pnl']:+,.2f} | "
               f"{s['total_closed']} closed | "
               f"{s['open_positions']} still open", "yellow")
        sys.exit(0)

    signal.signal(signal.SIGINT, _exit)

    once = '--once' in sys.argv

    while True:
        try:
            strat.run_cycle()
        except Exception as e:
            cprint(f"[OteScalp] Cycle error: {e}", "red")
            import traceback
            traceback.print_exc()
        if once:
            break
        cprint("[OteScalp] Next cycle in 60s...\n", "white")
        _time.sleep(60)
