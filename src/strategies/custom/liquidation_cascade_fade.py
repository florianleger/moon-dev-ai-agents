"""
Liquidation Cascade Fade Strategy

Opportunistic strategy that fades large liquidation cascades once exhaustion is detected.
Large cascades create temporary price dislocations when forced sellers/buyers push price
beyond fair value. By trading against the cascade after exhaustion, we capture the snapback.

Tokens: BTC, ETH, SOL, XRP, AVAX, SUI (deep order books required)
Edge: High conviction, infrequent trades, tight risk management
"""

import os
import threading
import time as _time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from termcolor import cprint
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator

from ..base_strategy import BaseStrategy
from src.data.trade_memory import TradeMemory
from src.strategies.modules.liquidation_cascade import SYMBOL_MAP

# Config imports with defaults
try:
    from src.config import (
        PAPER_TRADING, PAPER_TRADING_BALANCE,
        LIQ_CASCADE_TOKENS, LIQ_CASCADE_SIGMA_THRESHOLD,
        LIQ_CASCADE_RISK_PCT, LIQ_CASCADE_MAX_HOLD_HOURS,
        LIQ_CASCADE_SL_ATR_MULT, LIQ_CASCADE_TP_ATR_MULT,
        LIQ_CASCADE_MAX_DAILY_TRADES, LIQ_CASCADE_MAX_DAILY_LOSS_USD,
        LIQ_CASCADE_LEVERAGE, LIQ_CASCADE_COOLDOWN_MINUTES,
        PAPER_SLIPPAGE_V2, PAPER_TAKER_FEE_V2,
    )
except ImportError:
    PAPER_TRADING = True
    PAPER_TRADING_BALANCE = 500
    LIQ_CASCADE_TOKENS = ['BTC', 'ETH', 'SOL', 'XRP', 'AVAX', 'SUI']
    LIQ_CASCADE_SIGMA_THRESHOLD = 3.0
    LIQ_CASCADE_RISK_PCT = 0.01
    LIQ_CASCADE_MAX_HOLD_HOURS = 4
    LIQ_CASCADE_SL_ATR_MULT = 2.0
    LIQ_CASCADE_TP_ATR_MULT = 3.0
    LIQ_CASCADE_MAX_DAILY_TRADES = 2
    LIQ_CASCADE_MAX_DAILY_LOSS_USD = 10.0
    LIQ_CASCADE_LEVERAGE = {'btc': 3, 'eth': 3, 'mid': 2, 'alt': 2}
    LIQ_CASCADE_COOLDOWN_MINUTES = 30
    PAPER_SLIPPAGE_V2 = {'btc': 0.0003, 'eth': 0.0005, 'mid': 0.0012, 'alt': 0.003}
    PAPER_TAKER_FEE_V2 = 0.00045

# Absolute volume thresholds (USD) for cascade detection per token class
# Loosened by 50% (was BTC 5M / ETH 2M / mid 500k) to detect smaller cascades in current low-vol regime.
CASCADE_VOLUME_THRESHOLDS = {
    'BTC': 2_500_000,
    'ETH': 1_000_000,
    'SOL': 250_000, 'XRP': 250_000, 'AVAX': 250_000, 'SUI': 250_000,
}


class LiquidationCascadeFadeStrategy(BaseStrategy):
    """
    Fades liquidation cascades once exhaustion is detected.

    Flow per cycle:
    1. Manage existing positions (TP at VWAP, SL, time stops)
    2. Monitor liquidation feeds for cascade events
    3. Wait for exhaustion signals
    4. Execute fade trades (paper)
    """

    def __init__(self):
        super().__init__("Liquidation Cascade Fade")

        self.assets = list(LIQ_CASCADE_TOKENS)
        self.tokens = self.assets

        # Paper trading state
        self.paper_balance = PAPER_TRADING_BALANCE
        self.paper_positions = {}
        self.closed_positions = []
        self._position_counter = 0
        self._position_lock = threading.RLock()

        # Daily tracking
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_date = None

        # Cooldown tracking: {symbol: datetime of last cascade trade}
        self._cooldowns = {}

        # Cascade state: {symbol: cascade_info} for pending exhaustion checks
        self._pending_cascades = {}

        # Liquidation stream
        self._liq_stream = None
        self._market_data = None
        self._init_providers()

        # Trade memory
        self._trade_memory = TradeMemory.get_instance()

        # Data directory
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data', 'liq_cascade'
        )
        os.makedirs(self.data_dir, exist_ok=True)

        # Candle cache: {(symbol, interval): (DataFrame, datetime)}
        self._candle_cache = {}
        self._candle_cache_ttl = 120

        # Load existing state
        self._load_state()

        cprint(f"[LiqCascadeFade] Strategy initialized", "cyan")
        cprint(f"  - Tokens: {self.assets}", "white")
        cprint(f"  - Sigma threshold: {LIQ_CASCADE_SIGMA_THRESHOLD}", "white")
        cprint(f"  - Balance: ${self.paper_balance:,.2f}", "white")

    # =========================================================================
    # PROVIDERS
    # =========================================================================

    def _init_providers(self):
        """Initialize data providers."""
        try:
            from src.data_providers.binance_futures import get_liquidation_stream
            self._liq_stream = get_liquidation_stream()
            if not self._liq_stream.is_connected:
                self._liq_stream.start_stream()
                _time.sleep(2)
        except Exception as e:
            cprint(f"[LiqCascadeFade] Liquidation stream unavailable: {e}", "yellow")

        try:
            from src.data_providers.market_data import MarketDataProvider
            self._market_data = MarketDataProvider(start_liquidation_stream=False)
        except Exception as e:
            cprint(f"[LiqCascadeFade] Market data provider unavailable: {e}", "yellow")

    # =========================================================================
    # DATA FETCHING
    # =========================================================================

    def _fetch_candles(self, symbol: str, interval: str = '5m', candles: int = 200) -> pd.DataFrame:
        """Fetch candle data from HyperLiquid with caching."""
        cache_key = (symbol, interval)
        cached = self._candle_cache.get(cache_key)
        if cached:
            df_cached, cached_at = cached
            if (datetime.now() - cached_at).total_seconds() < self._candle_cache_ttl:
                return df_cached.copy()

        try:
            from hyperliquid.info import Info
            info = Info(skip_ws=True, timeout=15)
            end_time = int(_time.time() * 1000)
            interval_map = {'1m': 60_000, '5m': 300_000, '15m': 900_000, '1h': 3_600_000}
            interval_ms = interval_map.get(interval, 300_000)
            start_time = end_time - (candles * interval_ms)
            _time.sleep(0.15)

            data = info.candles_snapshot(symbol, interval, start_time, end_time)
            if not data:
                return None

            df = pd.DataFrame(data)
            df = df.rename(columns={'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            self._candle_cache[cache_key] = (df.copy(), datetime.now())
            return df
        except Exception as e:
            cprint(f"[LiqCascadeFade] Error fetching candles for {symbol}: {e}", "yellow")
            return None

    def _get_current_price(self, symbol: str) -> float:
        """Get current price from HyperLiquid."""
        if self._market_data:
            price = self._market_data.get_current_price(symbol)
            if price:
                return price
        # Fallback: last candle close
        df = self._fetch_candles(symbol, '1m', 5)
        if df is not None and len(df) > 0:
            return float(df['close'].iloc[-1])
        return 0.0

    # =========================================================================
    # CASCADE DETECTION
    # =========================================================================

    def _detect_cascade(self, symbol: str, lookback_minutes: int = 15) -> dict:
        """Detect liquidation cascade from Binance data.

        Returns cascade info dict or None if no cascade detected.
        """
        if not self._liq_stream:
            return None

        binance_symbol = SYMBOL_MAP.get(symbol.upper(), f'{symbol.upper()}USDT')

        try:
            df = self._liq_stream.get_recent_liquidations(minutes=lookback_minutes)
        except Exception:
            return None

        if df is None or df.empty:
            return None

        # Filter for this symbol
        symbol_df = df[df['symbol'] == binance_symbol]
        if symbol_df.empty:
            return None

        recent_liq_volume = float(symbol_df['usd_value'].sum())

        # Method 1: Z-score against 24h rolling baseline
        zscore = 0.0
        try:
            baseline_df = self._liq_stream.get_recent_liquidations(minutes=1440)
            if baseline_df is not None and not baseline_df.empty:
                baseline_symbol = baseline_df[baseline_df['symbol'] == binance_symbol]
                if not baseline_symbol.empty and 'timestamp' in baseline_symbol.columns:
                    # Compute 15-min window volumes over 24h
                    baseline_symbol = baseline_symbol.copy()
                    baseline_symbol['window'] = baseline_symbol['timestamp'].dt.floor(f'{lookback_minutes}min')
                    window_volumes = baseline_symbol.groupby('window')['usd_value'].sum().values
                    if len(window_volumes) >= 3:
                        mean_vol = np.mean(window_volumes)
                        std_vol = np.std(window_volumes)
                        if std_vol > 0:
                            zscore = (recent_liq_volume - mean_vol) / std_vol
        except Exception:
            pass

        # Method 2: Absolute volume threshold
        abs_threshold = CASCADE_VOLUME_THRESHOLDS.get(symbol, 500_000)
        absolute_trigger = recent_liq_volume >= abs_threshold

        threshold = LIQ_CASCADE_SIGMA_THRESHOLD
        if zscore < threshold and not absolute_trigger:
            return None

        # Determine cascade direction
        # SELL side = long positions liquidated, BUY side = short positions liquidated
        long_liqs = float(symbol_df[symbol_df['side'] == 'SELL']['usd_value'].sum())
        short_liqs = float(symbol_df[symbol_df['side'] == 'BUY']['usd_value'].sum())

        if long_liqs + short_liqs == 0:
            return None

        cascade_side = 'LONG_LIQUIDATED' if long_liqs > short_liqs else 'SHORT_LIQUIDATED'
        fade_direction = 'BUY' if cascade_side == 'LONG_LIQUIDATED' else 'SELL'

        return {
            'detected': True,
            'zscore': round(zscore, 2),
            'total_volume': recent_liq_volume,
            'cascade_side': cascade_side,
            'fade_direction': fade_direction,
            'long_liqs': long_liqs,
            'short_liqs': short_liqs,
            'absolute_trigger': absolute_trigger,
            'detected_at': datetime.now(),
        }

    def _check_exhaustion(self, symbol: str, cascade: dict) -> bool:
        """Check if cascade has exhausted (safe to enter fade).

        Exhaustion signals (any one is sufficient):
        1. Price stops making new extremes for 2 consecutive 5m candles
        2. RSI(14) on 5m reaches extreme (<20 or >80)
        3. Volume drops below 50% of cascade peak volume
        """
        df = self._fetch_candles(symbol, '5m', 30)
        if df is None or len(df) < 5:
            return False

        fade_dir = cascade['fade_direction']

        # Signal 1: Price stopped making new extremes (2 consecutive candles)
        if fade_dir == 'BUY':
            # Longs liquidated -> price dumped -> check if stopped making new lows
            recent_lows = df['low'].iloc[-3:].values
            if len(recent_lows) == 3 and recent_lows[-1] >= recent_lows[-2] and recent_lows[-2] >= recent_lows[-3]:
                return True
        else:
            # Shorts liquidated -> price pumped -> check if stopped making new highs
            recent_highs = df['high'].iloc[-3:].values
            if len(recent_highs) == 3 and recent_highs[-1] <= recent_highs[-2] and recent_highs[-2] <= recent_highs[-3]:
                return True

        # Signal 2: RSI extreme
        rsi = RSIIndicator(close=df['close'], window=14).rsi()
        last_rsi = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50
        if fade_dir == 'BUY' and last_rsi < 20:
            return True
        if fade_dir == 'SELL' and last_rsi > 80:
            return True

        # Signal 3: Volume declining from peak
        vol_peak = df['volume'].iloc[-10:].max()
        vol_recent = df['volume'].iloc[-2:].mean()
        if vol_peak > 0 and vol_recent < vol_peak * 0.5:
            return True

        return False

    def _compute_pre_cascade_vwap(self, symbol: str) -> float:
        """Calculate VWAP from candles before the cascade started (anchor price)."""
        df = self._fetch_candles(symbol, '15m', 100)
        if df is None or len(df) < 20:
            return 0.0

        # Use candles from -20 to -5 (before recent cascade window)
        pre_df = df.iloc[-20:-5].copy()
        if pre_df.empty:
            return 0.0

        try:
            typical_price = (pre_df['high'] + pre_df['low'] + pre_df['close']) / 3
            vwap = (typical_price * pre_df['volume']).sum() / pre_df['volume'].sum()
            return float(vwap)
        except Exception:
            return float(pre_df['close'].mean())

    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================

    def _token_class(self, symbol: str) -> str:
        """Determine token class for leverage/slippage."""
        sym = symbol.upper()
        if sym == 'BTC':
            return 'btc'
        elif sym == 'ETH':
            return 'eth'
        return 'mid'

    def _price_decimals(self, price: float) -> int:
        if price > 1000:
            return 2
        elif price > 10:
            return 3
        elif price > 0.1:
            return 4
        return 6

    def _manage_positions(self) -> list:
        """Check all open positions for TP/SL/time stops. Returns list of closed trades."""
        closed = []
        positions_to_close = []

        with self._position_lock:
            for position_id, pos in list(self.paper_positions.items()):
                symbol = pos['symbol']
                current_price = self._get_current_price(symbol)
                if current_price <= 0:
                    continue

                entry_price = pos['entry_price']
                direction = pos['direction']
                close_reason = None
                close_price = current_price

                # Stop Loss
                if direction == 'BUY' and current_price <= pos['stop_loss']:
                    close_reason = 'STOP_LOSS'
                    close_price = pos['stop_loss']
                elif direction == 'SELL' and current_price >= pos['stop_loss']:
                    close_reason = 'STOP_LOSS'
                    close_price = pos['stop_loss']

                # Take Profit: price returns to pre-cascade VWAP
                vwap_target = pos.get('vwap_target')
                if not close_reason and vwap_target:
                    if direction == 'BUY' and current_price >= vwap_target:
                        close_reason = 'TP_VWAP'
                        close_price = vwap_target
                    elif direction == 'SELL' and current_price <= vwap_target:
                        close_reason = 'TP_VWAP'
                        close_price = vwap_target

                # ATR-based take profit (fallback)
                if not close_reason:
                    if direction == 'BUY' and current_price >= pos['take_profit']:
                        close_reason = 'TAKE_PROFIT_ATR'
                        close_price = pos['take_profit']
                    elif direction == 'SELL' and current_price <= pos['take_profit']:
                        close_reason = 'TAKE_PROFIT_ATR'
                        close_price = pos['take_profit']

                # Time stop: 4 hours max
                if not close_reason:
                    entry_time = pos.get('entry_time')
                    if isinstance(entry_time, str):
                        entry_time = datetime.fromisoformat(entry_time)
                    if entry_time and (datetime.now() - entry_time).total_seconds() > LIQ_CASCADE_MAX_HOLD_HOURS * 3600:
                        close_reason = 'TIME_STOP'

                if close_reason:
                    positions_to_close.append((position_id, close_price, close_reason))

        for position_id, close_price, reason in positions_to_close:
            closed_trade = self._close_position(position_id, close_price, reason)
            if closed_trade:
                closed.append(closed_trade)

        return closed

    def _close_position(self, position_id: str, close_price: float, reason: str) -> dict:
        """Close a paper position with slippage and fees."""
        with self._position_lock:
            if position_id not in self.paper_positions:
                return None

            trade = self.paper_positions[position_id].copy()
            entry_price = trade['entry_price']
            direction = trade['direction']
            position_size = trade['position_size']
            symbol = trade['symbol']
            entry_fee = trade.get('entry_fee', 0)

            tc = self._token_class(symbol)
            slippage = PAPER_SLIPPAGE_V2.get(tc, 0.001)
            exit_fee = position_size * PAPER_TAKER_FEE_V2

            if direction == 'BUY':
                effective_close = close_price * (1 - slippage)
                price_change_pct = (effective_close - entry_price) / entry_price
            else:
                effective_close = close_price * (1 + slippage)
                price_change_pct = (entry_price - effective_close) / entry_price

            pnl = position_size * price_change_pct - exit_fee

            # Sanity clamp
            if abs(pnl) > position_size * 1.5:
                pnl = max(-position_size, min(position_size, pnl))

            trade['close_price'] = close_price
            trade['effective_close_price'] = round(effective_close, 6)
            trade['exit_time'] = datetime.now().isoformat()
            trade['close_reason'] = reason
            trade['exit_fee'] = round(exit_fee, 4)
            trade['total_fees'] = round(entry_fee + exit_fee, 4)
            trade['pnl'] = round(pnl, 2)
            trade['pnl_pct'] = round(price_change_pct * 100, 2)
            trade['status'] = 'CLOSED'

            self.daily_pnl += pnl
            self.paper_balance += pnl
            del self.paper_positions[position_id]
            self.closed_positions.append(trade)

        color = 'green' if pnl > 0 else 'red'
        _pd = self._price_decimals(entry_price)
        cprint(f"\n[LIQ CASCADE FADE] Closed {symbol} ({reason})", color, attrs=['bold'])
        cprint(f"  Entry: ${entry_price:,.{_pd}f} -> Exit: ${close_price:,.{_pd}f}", "white")
        cprint(f"  PnL: ${pnl:+,.2f} ({price_change_pct*100:+.2f}%)", color)
        cprint(f"  Balance: ${self.paper_balance:,.2f}", "white")

        # Log closed trade
        self._log_closed_trade(trade)

        # Update trade memory
        if 'memory_decision_id' in trade:
            try:
                entry_time = trade.get('entry_time')
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time)
                hold_hours = (datetime.now() - entry_time).total_seconds() / 3600 if entry_time else None
                self._trade_memory.update_outcome(
                    decision_id=trade['memory_decision_id'],
                    pnl=pnl,
                    hold_duration_hours=hold_hours,
                    close_reason=reason,
                )
            except Exception:
                pass

        return trade

    # =========================================================================
    # TRADE EXECUTION
    # =========================================================================

    def _execute_fade_trade(self, symbol: str, cascade: dict) -> dict:
        """Execute a fade trade against the cascade."""
        direction = cascade['fade_direction']
        current_price = self._get_current_price(symbol)
        if current_price <= 0:
            return None

        # VWAP target (equilibrium price before cascade)
        vwap_target = self._compute_pre_cascade_vwap(symbol)

        # Validate VWAP: entry should be within 1% of deviation from VWAP
        # (if VWAP is valid and price hasn't already snapped back)
        if vwap_target > 0:
            vwap_dist_pct = abs(current_price - vwap_target) / vwap_target
            if vwap_dist_pct < 0.003:
                cprint(f"  [{symbol}] Price already at VWAP (dist={vwap_dist_pct:.3%}), skipping", "yellow")
                return None

        # Get ATR for SL/TP sizing
        df = self._fetch_candles(symbol, '5m', 100)
        if df is None or len(df) < 20:
            return None

        atr_ind = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
        atr = float(atr_ind.average_true_range().iloc[-1])
        if atr <= 0:
            return None

        # Position sizing
        tc = self._token_class(symbol)
        leverage = LIQ_CASCADE_LEVERAGE.get(tc, 2)
        risk_amount = self.paper_balance * LIQ_CASCADE_RISK_PCT
        sl_distance = atr * LIQ_CASCADE_SL_ATR_MULT
        sl_pct = sl_distance / current_price

        if sl_pct <= 0:
            return None

        position_size = min(
            risk_amount / sl_pct * leverage,
            self.paper_balance * 0.25  # Max 25% of balance per trade
        )

        if position_size < 10:
            cprint(f"  [{symbol}] Position too small (${position_size:.2f}), skipping", "yellow")
            return None

        # SL/TP prices
        tp_distance = atr * LIQ_CASCADE_TP_ATR_MULT
        if direction == 'BUY':
            stop_loss = current_price - sl_distance
            take_profit = current_price + tp_distance
        else:
            stop_loss = current_price + sl_distance
            take_profit = current_price - tp_distance

        # Entry slippage + fee
        slippage = PAPER_SLIPPAGE_V2.get(tc, 0.001)
        if direction == 'BUY':
            entry_price = current_price * (1 + slippage)
        else:
            entry_price = current_price * (1 - slippage)
        entry_fee = position_size * PAPER_TAKER_FEE_V2

        if self.paper_balance - entry_fee < 0:
            return None

        with self._position_lock:
            self._position_counter += 1
            position_id = f"LC_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._position_counter}"

            trade = {
                'position_id': position_id,
                'strategy': 'liq_cascade_fade',
                'timestamp': datetime.now().isoformat(),
                'entry_time': datetime.now(),
                'symbol': symbol,
                'direction': direction,
                'entry_price': round(entry_price, self._price_decimals(entry_price)),
                'position_size': round(position_size, 2),
                'leverage': leverage,
                'stop_loss': round(stop_loss, self._price_decimals(current_price)),
                'take_profit': round(take_profit, self._price_decimals(current_price)),
                'vwap_target': round(vwap_target, self._price_decimals(current_price)) if vwap_target > 0 else None,
                'sl_pct': round(sl_pct * 100, 2),
                'tp_pct': round(tp_distance / current_price * 100, 2),
                'atr': round(atr, self._price_decimals(current_price)),
                'entry_fee': round(entry_fee, 4),
                'cascade_zscore': cascade.get('zscore', 0),
                'cascade_volume': cascade.get('total_volume', 0),
                'cascade_side': cascade.get('cascade_side', ''),
                'status': 'OPEN',
            }

            self.paper_positions[position_id] = trade
            self.paper_balance -= entry_fee
            self.daily_trades += 1
            self._cooldowns[symbol] = datetime.now()

        # Log to CSV
        self._log_open_trade(trade)

        _pd = self._price_decimals(entry_price)
        cprint(f"\n[LIQ CASCADE FADE] Opened {direction} {symbol} (ID: {position_id})", "magenta", attrs=['bold'])
        cprint(f"  Entry: ${entry_price:,.{_pd}f} | Size: ${position_size:,.2f} | Leverage: {leverage}x", "white")
        cprint(f"  SL: ${stop_loss:,.{_pd}f} ({sl_pct*100:.2f}%) | TP: ${take_profit:,.{_pd}f}", "white")
        if vwap_target:
            cprint(f"  VWAP target: ${vwap_target:,.{_pd}f}", "cyan")
        cprint(f"  Cascade: z={cascade['zscore']:.1f}, vol=${cascade['total_volume']:,.0f}, side={cascade['cascade_side']}", "white")

        # Log to trade memory
        try:
            decision_id = self._trade_memory.log_decision(
                symbol=symbol,
                direction=direction,
                confidence=min(100, cascade.get('zscore', 0) * 20),
                source='liq_cascade_fade',
                reasoning=f"Cascade fade: {cascade['cascade_side']}, z={cascade['zscore']:.1f}, vol=${cascade['total_volume']:,.0f}",
                market_regime=None,
                key_indicators={'atr': atr, 'zscore': cascade.get('zscore', 0)},
            )
            trade['memory_decision_id'] = decision_id
            with self._position_lock:
                if position_id in self.paper_positions:
                    self.paper_positions[position_id]['memory_decision_id'] = decision_id
        except Exception:
            pass

        return trade

    # =========================================================================
    # MAIN CYCLE
    # =========================================================================

    def _reset_daily_counters(self):
        """Reset daily counters at midnight."""
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_trade_date = today

    def run_cycle(self, symbols: list = None) -> dict:
        """Main strategy cycle.

        1. Reset daily counters
        2. Manage existing positions
        3. Scan for cascade events
        4. Check exhaustion and execute fades

        Returns summary dict.
        """
        symbols = symbols or self.assets
        self._reset_daily_counters()

        result = {
            'closed': [],
            'opened': [],
            'cascades_detected': 0,
            'exhaustion_signals': 0,
        }

        # Step 1: Manage existing positions
        closed = self._manage_positions()
        result['closed'] = closed

        # Daily limits check
        if self.daily_trades >= LIQ_CASCADE_MAX_DAILY_TRADES:
            cprint(f"[LiqCascadeFade] Daily trade limit reached ({self.daily_trades})", "yellow")
            return result

        if self.daily_pnl <= -LIQ_CASCADE_MAX_DAILY_LOSS_USD:
            cprint(f"[LiqCascadeFade] Daily loss limit reached (${self.daily_pnl:+,.2f})", "red")
            return result

        # Step 2: Scan each symbol for cascade events
        for symbol in symbols:
            # Skip if already in position for this symbol
            with self._position_lock:
                has_position = any(
                    p['symbol'] == symbol for p in self.paper_positions.values()
                )
            if has_position:
                continue

            # Cooldown check
            last_trade = self._cooldowns.get(symbol)
            if last_trade and (datetime.now() - last_trade).total_seconds() < LIQ_CASCADE_COOLDOWN_MINUTES * 60:
                continue

            # Detect cascade
            cascade = self._detect_cascade(symbol)
            if not cascade:
                continue

            result['cascades_detected'] += 1
            cprint(f"\n[LIQ CASCADE] Detected for {symbol}: z={cascade['zscore']:.1f}, "
                   f"vol=${cascade['total_volume']:,.0f}, side={cascade['cascade_side']}", "yellow", attrs=['bold'])

            # Check exhaustion before entering
            if self._check_exhaustion(symbol, cascade):
                result['exhaustion_signals'] += 1
                cprint(f"  [{symbol}] Exhaustion confirmed, executing fade {cascade['fade_direction']}", "cyan")

                trade = self._execute_fade_trade(symbol, cascade)
                if trade:
                    result['opened'].append(trade)

                    # Recheck daily limits
                    if self.daily_trades >= LIQ_CASCADE_MAX_DAILY_TRADES:
                        break
            else:
                cprint(f"  [{symbol}] Cascade active but no exhaustion yet, waiting...", "white")
                self._pending_cascades[symbol] = cascade

        # Step 3: Recheck pending cascades for exhaustion
        expired = []
        for symbol, cascade in list(self._pending_cascades.items()):
            # Expire after 30 minutes
            detected_at = cascade.get('detected_at', datetime.now())
            if (datetime.now() - detected_at).total_seconds() > 1800:
                expired.append(symbol)
                continue

            if self._check_exhaustion(symbol, cascade):
                result['exhaustion_signals'] += 1
                trade = self._execute_fade_trade(symbol, cascade)
                if trade:
                    result['opened'].append(trade)
                expired.append(symbol)

        for s in expired:
            self._pending_cascades.pop(s, None)

        # Summary
        if result['cascades_detected'] > 0 or result['closed'] or result['opened']:
            cprint(f"\n[LiqCascadeFade] Cycle summary: "
                   f"cascades={result['cascades_detected']}, "
                   f"exhausted={result['exhaustion_signals']}, "
                   f"opened={len(result['opened'])}, "
                   f"closed={len(result['closed'])}, "
                   f"balance=${self.paper_balance:,.2f}", "cyan")

        return result

    def generate_signals(self) -> dict:
        """BaseStrategy interface — delegates to run_cycle."""
        result = self.run_cycle()
        if result['opened']:
            trade = result['opened'][0]
            return {
                'token': trade['symbol'],
                'signal': 0.8,
                'direction': trade['direction'],
                'metadata': {'strategy': 'liq_cascade_fade', 'cascade_zscore': trade.get('cascade_zscore', 0)},
            }
        return {'token': '', 'signal': 0, 'direction': 'NEUTRAL', 'metadata': {}}

    # =========================================================================
    # CSV PERSISTENCE
    # =========================================================================

    def _log_open_trade(self, trade: dict):
        try:
            log_file = os.path.join(self.data_dir, 'paper_trades.csv')
            df = pd.DataFrame([trade])
            if os.path.exists(log_file):
                df.to_csv(log_file, mode='a', header=False, index=False)
            else:
                df.to_csv(log_file, index=False)
        except Exception as e:
            cprint(f"[LiqCascadeFade] Error logging open trade: {e}", "yellow")

    def _log_closed_trade(self, trade: dict):
        try:
            log_file = os.path.join(self.data_dir, 'closed_trades.csv')
            df = pd.DataFrame([trade])
            if os.path.exists(log_file):
                df.to_csv(log_file, mode='a', header=False, index=False)
            else:
                df.to_csv(log_file, index=False)
        except Exception as e:
            cprint(f"[LiqCascadeFade] Error logging closed trade: {e}", "yellow")

    def _load_state(self):
        """Load open positions and balance from CSV."""
        paper_file = os.path.join(self.data_dir, 'paper_trades.csv')
        closed_file = os.path.join(self.data_dir, 'closed_trades.csv')

        # Reconstruct balance from closed trades
        realized_pnl = 0.0
        if os.path.exists(closed_file):
            try:
                df = pd.read_csv(closed_file)
                if 'pnl' in df.columns:
                    realized_pnl = df['pnl'].sum()
                    self.closed_positions = df.to_dict('records')
            except Exception as e:
                cprint(f"[LiqCascadeFade] Error loading closed trades: {e}", "yellow")

        # Load open positions
        if os.path.exists(paper_file):
            try:
                df = pd.read_csv(paper_file)
                if 'status' in df.columns:
                    open_df = df[df['status'] == 'OPEN']
                    for _, row in open_df.iterrows():
                        pos = row.to_dict()
                        pid = pos.get('position_id', '')
                        if pid:
                            self.paper_positions[pid] = pos
                    entry_fees = sum(p.get('entry_fee', 0) for p in self.paper_positions.values())
                    self.paper_balance = PAPER_TRADING_BALANCE + realized_pnl - entry_fees
            except Exception as e:
                cprint(f"[LiqCascadeFade] Error loading open positions: {e}", "yellow")
        else:
            self.paper_balance = PAPER_TRADING_BALANCE + realized_pnl

        if self.paper_positions:
            cprint(f"[LiqCascadeFade] Loaded {len(self.paper_positions)} open positions, balance=${self.paper_balance:,.2f}", "cyan")

    def get_paper_status(self) -> dict:
        """Get current paper trading status."""
        with self._position_lock:
            return {
                'strategy': 'liq_cascade_fade',
                'paper_balance': round(self.paper_balance, 2),
                'initial_balance': PAPER_TRADING_BALANCE,
                'total_pnl': round(self.paper_balance - PAPER_TRADING_BALANCE, 2),
                'daily_pnl': round(self.daily_pnl, 2),
                'daily_trades': self.daily_trades,
                'open_positions': len(self.paper_positions),
                'total_closed': len(self.closed_positions),
                'positions': [pos.copy() for pos in self.paper_positions.values()],
                'pending_cascades': list(self._pending_cascades.keys()),
            }


# =========================================================================
# STANDALONE EXECUTION
# =========================================================================

if __name__ == '__main__':
    import signal
    import sys

    cprint("\n" + "=" * 60, "cyan")
    cprint("  Liquidation Cascade Fade Strategy", "cyan", attrs=['bold'])
    cprint("  Fading forced liquidation cascades", "cyan")
    cprint("=" * 60, "cyan")

    strategy = LiquidationCascadeFadeStrategy()

    running = True

    def handle_exit(sig, frame):
        global running
        cprint("\n[LiqCascadeFade] Shutting down...", "yellow")
        running = False

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    cycle_interval = 60  # Check every 60 seconds (cascades are time-sensitive)

    while running:
        try:
            result = strategy.run_cycle()
            status = strategy.get_paper_status()
            cprint(f"[LiqCascadeFade] Balance: ${status['paper_balance']:,.2f} | "
                   f"Open: {status['open_positions']} | "
                   f"Closed: {status['total_closed']} | "
                   f"Daily PnL: ${status['daily_pnl']:+,.2f}", "white")
        except Exception as e:
            cprint(f"[LiqCascadeFade] Cycle error: {e}", "red")

        for _ in range(cycle_interval):
            if not running:
                break
            _time.sleep(1)

    cprint("[LiqCascadeFade] Strategy stopped.", "yellow")
