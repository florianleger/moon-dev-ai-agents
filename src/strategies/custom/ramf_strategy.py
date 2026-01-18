"""
Regime Adaptive Momentum Fade (RAMF) Strategy

A contrarian intraday strategy for HyperLiquid perpetuals that:
- Trades AGAINST exhausted momentum (fading moves)
- Adapts to volatility regime (mean-reversion in high vol, trend-follow in low vol)
- Uses funding rates and liquidation data as unique signals
- Requires high confidence (70%+) multi-factor alignment

Designed for small accounts ($500) with conservative risk management.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from termcolor import cprint
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import VolumeWeightedAveragePrice
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator

from ..base_strategy import BaseStrategy

# Import config - with defaults if not configured yet
try:
    from src.config import (
        RAMF_ASSETS,
        RAMF_LEVERAGE,
        RAMF_STOP_LOSS_PCT,
        RAMF_TAKE_PROFIT_PCT,
        RAMF_MIN_CONFIDENCE,
        RAMF_MAX_DAILY_TRADES,
        RAMF_MAX_DAILY_LOSS_USD,
        RAMF_MAX_DAILY_GAIN_USD,
        PAPER_TRADING,
        PAPER_TRADING_BALANCE
    )
except ImportError:
    # Default values
    RAMF_ASSETS = ['BTC', 'ETH', 'SOL']
    RAMF_LEVERAGE = 3
    RAMF_STOP_LOSS_PCT = 1.0
    RAMF_TAKE_PROFIT_PCT = 2.0
    RAMF_MIN_CONFIDENCE = 70
    RAMF_MAX_DAILY_TRADES = 6
    RAMF_MAX_DAILY_LOSS_USD = 25
    RAMF_MAX_DAILY_GAIN_USD = 25
    PAPER_TRADING = True
    PAPER_TRADING_BALANCE = 500


class RAMFStrategy(BaseStrategy):
    """
    Regime Adaptive Momentum Fade Strategy

    Core concept: Trade against exhausted momentum in high-volatility regimes,
    using funding rates and liquidation data as unique confirmation signals.

    This strategy is designed to be anti-correlated with typical bot behavior:
    - Most bots chase momentum; we fade it when exhausted
    - Most bots use fixed indicators; we adapt to volatility regime
    - Most bots ignore funding/liquidations; we use them as core signals
    """

    def __init__(self):
        super().__init__("RAMF Strategy")

        # Assets to trade
        self.assets = RAMF_ASSETS

        # Volatility regime thresholds
        self.volatility_high_percentile = 75
        self.volatility_low_percentile = 25

        # Momentum exhaustion parameters
        self.atr_extension_threshold = 2.0  # ATRs from VWAP for exhaustion
        self.consecutive_bar_threshold = 5  # Consecutive bars in same direction

        # Volume spike threshold
        self.volume_spike_threshold = 2.0  # 2x average volume

        # Funding rate thresholds
        self.funding_zscore_threshold = 1.5  # Z-score for extreme funding

        # Liquidation imbalance threshold
        self.liquidation_ratio_threshold = 1.5  # Long/short liq ratio

        # Confidence threshold
        self.min_confidence = RAMF_MIN_CONFIDENCE

        # BTC trend cache
        self._btc_trend_cache = None
        self._btc_trend_timestamp = None

        # Daily tracking
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_date = None

        # Paper trading state
        self.paper_balance = PAPER_TRADING_BALANCE
        self.paper_positions = {}

        # Data directory for logging
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data', 'ramf'
        )
        os.makedirs(self.data_dir, exist_ok=True)

        cprint(f"[RAMF] Strategy initialized", "cyan")
        cprint(f"  - Assets: {self.assets}", "white")
        cprint(f"  - Min Confidence: {self.min_confidence}%", "white")
        cprint(f"  - Paper Trading: {PAPER_TRADING}", "white")

    def _reset_daily_counters(self):
        """Reset daily counters if new day."""
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_trade_date = today
            cprint(f"[RAMF] New trading day - counters reset", "cyan")

    def _fetch_candles(self, symbol: str, interval: str = '15m', candles: int = 300) -> pd.DataFrame:
        """
        Fetch candle data from HyperLiquid.

        Args:
            symbol: Crypto symbol (e.g., 'BTC')
            interval: Candle interval
            candles: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        try:
            from hyperliquid.info import Info
            import time

            info = Info(skip_ws=True)

            # Calculate time range
            end_time = int(time.time() * 1000)
            interval_ms = {
                '1m': 60 * 1000,
                '5m': 5 * 60 * 1000,
                '15m': 15 * 60 * 1000,
                '1h': 60 * 60 * 1000,
                '4h': 4 * 60 * 60 * 1000,
                '1d': 24 * 60 * 60 * 1000
            }.get(interval, 15 * 60 * 1000)

            start_time = end_time - (candles * interval_ms)

            candle_data = info.candles_snapshot(symbol, interval, start_time, end_time)
            if not candle_data:
                return None

            df = pd.DataFrame(candle_data)
            df = df.rename(columns={
                't': 'timestamp',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            })

            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            return df

        except Exception as e:
            cprint(f"[RAMF] Error fetching candles for {symbol}: {e}", "yellow")
            return None

    def calculate_volatility_regime(self, df: pd.DataFrame) -> dict:
        """
        Determine current volatility regime using ATR percentile.

        Returns:
            dict: {'regime': 'high'|'low'|'normal', 'atr_percentile': float, 'current_atr': float}
        """
        try:
            atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
            atr_values = atr.average_true_range()
            current_atr = atr_values.iloc[-1]

            # Calculate percentile over last 100 bars
            lookback = min(100, len(atr_values))
            atr_series = atr_values.tail(lookback)
            percentile = (atr_series < current_atr).mean() * 100

            if percentile > self.volatility_high_percentile:
                regime = 'high'
            elif percentile < self.volatility_low_percentile:
                regime = 'low'
            else:
                regime = 'normal'

            return {
                'regime': regime,
                'atr_percentile': round(percentile, 1),
                'current_atr': round(current_atr, 6)
            }

        except Exception as e:
            cprint(f"[RAMF] Error calculating volatility regime: {e}", "yellow")
            return {'regime': 'normal', 'atr_percentile': 50.0, 'current_atr': 0.0}

    def calculate_momentum_exhaustion(self, df: pd.DataFrame) -> dict:
        """
        Detect momentum exhaustion using VWAP deviation and consecutive bars.

        Returns:
            dict with exhaustion metrics and signals
        """
        try:
            # VWAP calculation
            vwap = VolumeWeightedAveragePrice(
                df['high'], df['low'], df['close'], df['volume']
            )
            vwap_values = vwap.volume_weighted_average_price()
            current_vwap = vwap_values.iloc[-1]
            current_price = df['close'].iloc[-1]

            # ATR for normalization
            atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
            current_atr = atr.average_true_range().iloc[-1]

            # Distance from VWAP in ATR units
            if current_atr > 0:
                vwap_distance_atr = (current_price - current_vwap) / current_atr
            else:
                vwap_distance_atr = 0.0

            # Count consecutive directional bars
            closes = df['close'].tail(10).values
            consecutive_up = 0
            consecutive_down = 0

            for i in range(1, len(closes)):
                if closes[i] > closes[i-1]:
                    if consecutive_down > 0:
                        consecutive_down = 0
                    consecutive_up += 1
                elif closes[i] < closes[i-1]:
                    if consecutive_up > 0:
                        consecutive_up = 0
                    consecutive_down += 1

            # Determine exhaustion state
            overbought_exhaustion = (
                vwap_distance_atr > self.atr_extension_threshold and
                consecutive_up >= self.consecutive_bar_threshold
            )
            oversold_exhaustion = (
                vwap_distance_atr < -self.atr_extension_threshold and
                consecutive_down >= self.consecutive_bar_threshold
            )

            return {
                'vwap_distance_atr': round(vwap_distance_atr, 2),
                'consecutive_up': consecutive_up,
                'consecutive_down': consecutive_down,
                'overbought_exhaustion': overbought_exhaustion,
                'oversold_exhaustion': oversold_exhaustion,
                'current_vwap': round(current_vwap, 2),
                'current_price': round(current_price, 2)
            }

        except Exception as e:
            cprint(f"[RAMF] Error calculating momentum exhaustion: {e}", "yellow")
            return {
                'vwap_distance_atr': 0.0,
                'consecutive_up': 0,
                'consecutive_down': 0,
                'overbought_exhaustion': False,
                'oversold_exhaustion': False,
                'current_vwap': 0.0,
                'current_price': 0.0
            }

    def calculate_volume_profile(self, df: pd.DataFrame) -> dict:
        """
        Detect volume spikes that often mark local extremes.

        Returns:
            dict with volume metrics
        """
        try:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].tail(20).mean()

            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
            else:
                volume_ratio = 1.0

            is_volume_spike = volume_ratio > self.volume_spike_threshold

            return {
                'volume_ratio': round(volume_ratio, 2),
                'is_volume_spike': is_volume_spike,
                'current_volume': round(current_volume, 2),
                'avg_volume': round(avg_volume, 2)
            }

        except Exception as e:
            cprint(f"[RAMF] Error calculating volume profile: {e}", "yellow")
            return {
                'volume_ratio': 1.0,
                'is_volume_spike': False,
                'current_volume': 0.0,
                'avg_volume': 0.0
            }

    def get_funding_zscore(self, symbol: str) -> float:
        """
        Calculate Z-score approximation for current funding rate.

        Uses HyperLiquid funding rate data.

        Returns:
            float: Z-score approximation (-3 to +3 typical range)
        """
        try:
            from src.nice_funcs_hyperliquid import get_funding_rates

            funding_data = get_funding_rates(symbol)
            if funding_data and 'funding_rate' in funding_data:
                # Current hourly funding rate
                current_rate = funding_data['funding_rate']

                # Convert to annual percentage for Z-score approximation
                # Funding is typically quoted as hourly rate
                annual_rate = current_rate * 24 * 365 * 100

                # Typical perpetual funding ranges:
                # Mean: ~10% annual (slight long bias)
                # Std: ~15% annual
                mean_funding = 10.0
                std_funding = 15.0

                zscore = (annual_rate - mean_funding) / std_funding
                return round(zscore, 2)

            return 0.0

        except Exception as e:
            # Silently return neutral if funding data unavailable
            return 0.0

    def get_liquidation_imbalance(self) -> float:
        """
        Calculate ratio of long to short liquidations.

        Uses Moon Dev API for liquidation data if available.

        Returns:
            float: Long/Short liquidation ratio (1.0 = balanced)
        """
        try:
            from src.agents.api import MoonDevAPI

            api = MoonDevAPI()
            df = api.get_liquidation_data(limit=10000)

            if df is not None and not df.empty and 'side' in df.columns:
                # SELL side = long liquidations, BUY side = short liquidations
                longs_liq = df[df['side'] == 'SELL']['usd_value'].sum() if 'usd_value' in df.columns else 0
                shorts_liq = df[df['side'] == 'BUY']['usd_value'].sum() if 'usd_value' in df.columns else 0

                if shorts_liq > 0:
                    ratio = longs_liq / shorts_liq
                else:
                    ratio = 1.0 if longs_liq == 0 else 2.0

                return round(ratio, 2)

            return 1.0

        except Exception:
            # Return neutral if liquidation data unavailable
            return 1.0

    def check_btc_trend(self) -> bool:
        """
        Check if BTC is in an uptrend (price > EMA200).
        Used as a global market filter for long positions.

        Returns:
            bool: True if BTC is bullish
        """
        # Cache for 15 minutes
        if self._btc_trend_cache is not None and self._btc_trend_timestamp:
            age = (datetime.now() - self._btc_trend_timestamp).total_seconds()
            if age < 900:
                return self._btc_trend_cache

        try:
            df = self._fetch_candles('BTC', interval='1h', candles=250)
            if df is None or len(df) < 200:
                return True  # Default to allowing trades

            close = df['close']
            ema200 = EMAIndicator(close, window=200).ema_indicator()

            current_price = close.iloc[-1]
            current_ema200 = ema200.iloc[-1]

            is_bullish = current_price > current_ema200

            # Cache result
            self._btc_trend_cache = is_bullish
            self._btc_trend_timestamp = datetime.now()

            return is_bullish

        except Exception as e:
            cprint(f"[RAMF] Error checking BTC trend: {e}", "yellow")
            return True

    def calculate_rsi(self, df: pd.DataFrame) -> float:
        """Calculate current RSI value."""
        try:
            rsi = RSIIndicator(df['close'], window=14).rsi()
            return round(rsi.iloc[-1], 1)
        except Exception:
            return 50.0

    def generate_signals(self, symbol: str = None, df: pd.DataFrame = None) -> dict:
        """
        Generate trading signal for the given symbol.

        Args:
            symbol: Crypto symbol (e.g., 'BTC')
            df: DataFrame with OHLCV data (optional - will fetch if not provided)

        Returns:
            dict: Signal with token, direction, strength, and metadata
        """
        # Reset daily counters if new day
        self._reset_daily_counters()

        # Check daily limits
        if self.daily_trades >= RAMF_MAX_DAILY_TRADES:
            cprint(f"[RAMF] Daily trade limit reached ({RAMF_MAX_DAILY_TRADES})", "yellow")
            return None

        if self.daily_pnl <= -RAMF_MAX_DAILY_LOSS_USD:
            cprint(f"[RAMF] Daily loss limit reached (${RAMF_MAX_DAILY_LOSS_USD})", "red")
            return None

        if self.daily_pnl >= RAMF_MAX_DAILY_GAIN_USD:
            cprint(f"[RAMF] Daily gain limit reached (${RAMF_MAX_DAILY_GAIN_USD})", "green")
            return None

        # If no symbol provided, iterate through assets
        if symbol is None:
            for asset in self.assets:
                try:
                    asset_df = self._fetch_candles(asset, interval='15m', candles=300)
                    if asset_df is None or len(asset_df) < 200:
                        continue

                    signal = self._generate_signal_for_asset(asset, asset_df)
                    if signal and signal['direction'] != 'NEUTRAL':
                        return signal

                except Exception as e:
                    cprint(f"[RAMF] Error processing {asset}: {e}", "yellow")
                    continue

            return None

        # Generate signal for specific symbol
        if df is None:
            df = self._fetch_candles(symbol, interval='15m', candles=300)

        if df is None or len(df) < 200:
            return None

        return self._generate_signal_for_asset(symbol, df)

    def _generate_signal_for_asset(self, symbol: str, df: pd.DataFrame) -> dict:
        """
        Generate signal for a specific asset.

        This is where the RAMF magic happens:
        - In HIGH volatility: Look for momentum exhaustion (contrarian)
        - In LOW volatility: Look for trend continuation
        """
        try:
            # 1. Calculate volatility regime
            regime = self.calculate_volatility_regime(df)

            # 2. Calculate momentum exhaustion
            exhaustion = self.calculate_momentum_exhaustion(df)

            # 3. Check volume profile
            volume = self.calculate_volume_profile(df)

            # 4. Get funding rate Z-score
            funding_zscore = self.get_funding_zscore(symbol)

            # 5. Get liquidation imbalance (optional - may not have API key)
            liq_ratio = self.get_liquidation_imbalance()

            # 6. Check BTC macro trend
            btc_bullish = self.check_btc_trend()

            # 7. Get RSI for additional context
            rsi = self.calculate_rsi(df)

            # Initialize scores
            long_score = 0
            short_score = 0

            # Scoring logic depends on volatility regime
            if regime['regime'] == 'high':
                # HIGH VOLATILITY MODE: Mean-reversion / Fade exhaustion
                cprint(f"[RAMF] {symbol}: High volatility mode (ATR {regime['atr_percentile']:.0f}th pctl)", "cyan")

                # LONG conditions (fade oversold exhaustion)
                if exhaustion['oversold_exhaustion']:
                    long_score += 30
                    cprint(f"  + Oversold exhaustion detected (+30)", "green")

                if volume['is_volume_spike']:
                    long_score += 15
                    cprint(f"  + Volume spike ({volume['volume_ratio']:.1f}x) (+15)", "green")

                if funding_zscore < -self.funding_zscore_threshold:
                    long_score += 20
                    cprint(f"  + Negative funding Z={funding_zscore:.2f} (+20)", "green")

                if liq_ratio > self.liquidation_ratio_threshold:
                    long_score += 15
                    cprint(f"  + Long liquidation cascade ({liq_ratio:.2f}) (+15)", "green")

                if btc_bullish:
                    long_score += 15
                    cprint(f"  + BTC macro bullish (+15)", "green")

                if rsi < 30:
                    long_score += 5
                    cprint(f"  + RSI oversold ({rsi:.0f}) (+5)", "green")

                # SHORT conditions (fade overbought exhaustion)
                if exhaustion['overbought_exhaustion']:
                    short_score += 30
                    cprint(f"  + Overbought exhaustion detected (+30)", "red")

                if volume['is_volume_spike']:
                    short_score += 15
                    cprint(f"  + Volume spike ({volume['volume_ratio']:.1f}x) (+15)", "red")

                if funding_zscore > self.funding_zscore_threshold:
                    short_score += 20
                    cprint(f"  + Positive funding Z={funding_zscore:.2f} (+20)", "red")

                if liq_ratio < (1 / self.liquidation_ratio_threshold):
                    short_score += 15
                    cprint(f"  + Short liquidation cascade ({liq_ratio:.2f}) (+15)", "red")

                if rsi > 70:
                    short_score += 5
                    cprint(f"  + RSI overbought ({rsi:.0f}) (+5)", "red")

                # Shorts don't need BTC filter
                short_score += 5

            elif regime['regime'] == 'low':
                # LOW VOLATILITY MODE: Trend following (smaller positions)
                cprint(f"[RAMF] {symbol}: Low volatility mode (ATR {regime['atr_percentile']:.0f}th pctl)", "cyan")

                # Simple trend following based on price vs VWAP
                if exhaustion['vwap_distance_atr'] > 0.5:
                    long_score += 40
                    if btc_bullish:
                        long_score += 30
                elif exhaustion['vwap_distance_atr'] < -0.5:
                    short_score += 40

                # Scale down for low vol (less confident)
                long_score = int(long_score * 0.7)
                short_score = int(short_score * 0.7)

            else:
                # NORMAL VOLATILITY: No clear edge, stay out
                cprint(f"[RAMF] {symbol}: Normal volatility - no clear edge", "white")
                return {
                    'token': symbol,
                    'signal': 0.0,
                    'direction': 'NEUTRAL',
                    'metadata': {
                        'strategy_type': 'ramf',
                        'regime': regime,
                        'reason': 'Normal volatility - no clear edge'
                    }
                }

            # Determine direction and signal strength
            if long_score >= self.min_confidence:
                direction = 'BUY'
                signal_strength = min(1.0, long_score / 100)
            elif short_score >= self.min_confidence:
                direction = 'SELL'
                signal_strength = min(1.0, short_score / 100)
            else:
                direction = 'NEUTRAL'
                signal_strength = 0.0

            # Build signal
            current_price = float(df['close'].iloc[-1])

            signal = {
                'token': symbol,
                'signal': round(signal_strength, 3),
                'direction': direction,
                'metadata': {
                    'strategy_type': 'ramf',
                    'regime': regime,
                    'exhaustion': exhaustion,
                    'volume': volume,
                    'funding_zscore': funding_zscore,
                    'liquidation_ratio': liq_ratio,
                    'btc_macro_bullish': btc_bullish,
                    'rsi': rsi,
                    'long_score': long_score,
                    'short_score': short_score,
                    'current_price': current_price,
                    'stop_loss_pct': RAMF_STOP_LOSS_PCT,
                    'take_profit_pct': RAMF_TAKE_PROFIT_PCT,
                    'leverage': RAMF_LEVERAGE
                }
            }

            # Log actionable signals
            if direction != 'NEUTRAL':
                color = 'green' if direction == 'BUY' else 'red'
                score = long_score if direction == 'BUY' else short_score

                cprint("=" * 60, color)
                cprint(f"[RAMF] SIGNAL: {direction} {symbol} @ ${current_price:,.2f}", color, attrs=['bold'])
                cprint(f"  Confidence: {score}%", color)
                cprint(f"  Regime: {regime['regime']} ({regime['atr_percentile']:.0f}th pctl)", 'white')
                cprint(f"  VWAP Distance: {exhaustion['vwap_distance_atr']:.2f} ATR", 'white')
                cprint(f"  Funding Z-Score: {funding_zscore:.2f}", 'white')
                cprint(f"  Liquidation Ratio: {liq_ratio:.2f}", 'white')
                cprint(f"  RSI: {rsi:.0f}", 'white')
                cprint(f"  Stop-Loss: {RAMF_STOP_LOSS_PCT}% | Take-Profit: {RAMF_TAKE_PROFIT_PCT}%", 'white')
                cprint("=" * 60, color)

                # Log to file
                self._log_signal(signal)

            return signal

        except Exception as e:
            cprint(f"[RAMF] Error generating signal for {symbol}: {e}", "red")
            import traceback
            traceback.print_exc()
            return None

    def _log_signal(self, signal: dict):
        """Log signal to CSV file."""
        try:
            log_file = os.path.join(self.data_dir, 'signals.csv')

            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'symbol': signal['token'],
                'direction': signal['direction'],
                'confidence': signal['signal'],
                'price': signal['metadata'].get('current_price', 0),
                'regime': signal['metadata'].get('regime', {}).get('regime', 'unknown'),
                'funding_zscore': signal['metadata'].get('funding_zscore', 0),
                'rsi': signal['metadata'].get('rsi', 50),
                'paper_trading': PAPER_TRADING
            }

            df = pd.DataFrame([log_entry])

            if os.path.exists(log_file):
                df.to_csv(log_file, mode='a', header=False, index=False)
            else:
                df.to_csv(log_file, index=False)

        except Exception as e:
            cprint(f"[RAMF] Error logging signal: {e}", "yellow")

    def execute_paper_trade(self, signal: dict) -> dict:
        """
        Execute a paper trade (simulation).

        Args:
            signal: Signal dict from generate_signals()

        Returns:
            dict: Trade execution result
        """
        if not PAPER_TRADING:
            cprint("[RAMF] Paper trading disabled - skipping", "yellow")
            return None

        try:
            symbol = signal['token']
            direction = signal['direction']
            price = signal['metadata'].get('current_price', 0)
            confidence = signal['signal']

            # Calculate position size (2% risk per trade)
            risk_amount = self.paper_balance * 0.02
            position_size = risk_amount / (RAMF_STOP_LOSS_PCT / 100) * RAMF_LEVERAGE

            # Record trade
            trade = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'direction': direction,
                'entry_price': price,
                'position_size': round(position_size, 2),
                'leverage': RAMF_LEVERAGE,
                'stop_loss': round(price * (1 - RAMF_STOP_LOSS_PCT/100) if direction == 'BUY' else price * (1 + RAMF_STOP_LOSS_PCT/100), 2),
                'take_profit': round(price * (1 + RAMF_TAKE_PROFIT_PCT/100) if direction == 'BUY' else price * (1 - RAMF_TAKE_PROFIT_PCT/100), 2),
                'confidence': confidence,
                'status': 'OPEN'
            }

            # Store position
            self.paper_positions[symbol] = trade
            self.daily_trades += 1

            # Log to file
            log_file = os.path.join(self.data_dir, 'paper_trades.csv')
            df = pd.DataFrame([trade])

            if os.path.exists(log_file):
                df.to_csv(log_file, mode='a', header=False, index=False)
            else:
                df.to_csv(log_file, index=False)

            cprint(f"[RAMF PAPER] Opened {direction} {symbol}", "magenta")
            cprint(f"  Entry: ${price:,.2f} | Size: ${position_size:,.2f}", "white")
            cprint(f"  SL: ${trade['stop_loss']:,.2f} | TP: ${trade['take_profit']:,.2f}", "white")

            return trade

        except Exception as e:
            cprint(f"[RAMF] Error executing paper trade: {e}", "red")
            return None


# For standalone testing
if __name__ == "__main__":
    cprint("=" * 60, "cyan")
    cprint("  Testing RAMF Strategy", "cyan", attrs=['bold'])
    cprint("=" * 60, "cyan")

    strategy = RAMFStrategy()

    for symbol in ['BTC', 'ETH', 'SOL']:
        try:
            cprint(f"\nAnalyzing {symbol}...", "white", attrs=['bold'])

            df = strategy._fetch_candles(symbol, interval='15m', candles=300)

            if df is not None and len(df) >= 200:
                signal = strategy.generate_signals(symbol, df)

                if signal:
                    if signal['direction'] != 'NEUTRAL':
                        cprint(f"Signal generated!", "green")
                    else:
                        cprint(f"No actionable signal", "white")
            else:
                cprint(f"Not enough data for {symbol}", "yellow")

        except Exception as e:
            cprint(f"Error testing {symbol}: {e}", "red")

    cprint("\nTest completed!", "cyan")
