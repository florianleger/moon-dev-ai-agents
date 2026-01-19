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
        PAPER_TRADING_BALANCE,
        # v2.0 Advanced Settings
        RAMF_USE_ADAPTIVE_SL_TP,
        RAMF_ATR_SL_MULTIPLIER,
        RAMF_ATR_TP_MULTIPLIER,
        RAMF_MIN_SL_PCT,
        RAMF_MAX_SL_PCT,
        RAMF_USE_TIME_WINDOWS,
        RAMF_OPTIMAL_HOURS,
        RAMF_AVOID_HOURS,
        RAMF_OPTIMAL_HOUR_BONUS,
        RAMF_AVOID_HOUR_PENALTY,
        RAMF_USE_MTF,
        RAMF_MTF_TIMEFRAMES,
        RAMF_MTF_AGREEMENT_BONUS,
        RAMF_MTF_MIN_AGREEMENT,
        RAMF_USE_FUNDING_DIVERGENCE,
        RAMF_FUNDING_DIV_LOOKBACK,
        RAMF_FUNDING_DIV_THRESHOLD,
        RAMF_FUNDING_DIV_BONUS,
        RAMF_USE_LIQ_CLUSTERS,
        RAMF_LIQ_CLUSTER_THRESHOLD,
        RAMF_LIQ_CLUSTER_BONUS,
        # Volatility regime thresholds
        RAMF_VOLATILITY_HIGH_PERCENTILE,
        RAMF_VOLATILITY_LOW_PERCENTILE
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
    # v2.0 defaults
    RAMF_USE_ADAPTIVE_SL_TP = True
    RAMF_ATR_SL_MULTIPLIER = 1.5
    RAMF_ATR_TP_MULTIPLIER = 3.0
    RAMF_MIN_SL_PCT = 0.5
    RAMF_MAX_SL_PCT = 2.0
    RAMF_USE_TIME_WINDOWS = True
    RAMF_OPTIMAL_HOURS = [7, 8, 9, 13, 14, 15, 19, 20, 21]
    RAMF_AVOID_HOURS = [0, 1, 2, 3, 4, 5]
    RAMF_OPTIMAL_HOUR_BONUS = 15
    RAMF_AVOID_HOUR_PENALTY = 20
    RAMF_USE_MTF = True
    RAMF_MTF_TIMEFRAMES = ['5m', '15m', '1h', '4h']
    RAMF_MTF_AGREEMENT_BONUS = 10
    RAMF_MTF_MIN_AGREEMENT = 2
    RAMF_USE_FUNDING_DIVERGENCE = True
    RAMF_FUNDING_DIV_LOOKBACK = 24
    RAMF_FUNDING_DIV_THRESHOLD = 0.3
    RAMF_FUNDING_DIV_BONUS = 20
    RAMF_USE_LIQ_CLUSTERS = True
    RAMF_LIQ_CLUSTER_THRESHOLD = 2.0
    RAMF_LIQ_CLUSTER_BONUS = 15
    # Volatility regime defaults (widened for more signals)
    RAMF_VOLATILITY_HIGH_PERCENTILE = 60
    RAMF_VOLATILITY_LOW_PERCENTILE = 40

# Lower confidence threshold for low volatility regime (0.7x multiplier makes 70 impossible)
RAMF_MIN_CONFIDENCE_LOW_VOL = 35


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

    # Singleton instance for web dashboard access
    _instance = None

    def __init__(self):
        # Set singleton instance for web API access
        RAMFStrategy._instance = self
        super().__init__("RAMF Strategy")

        # Assets to trade
        self.assets = RAMF_ASSETS

        # Volatility regime thresholds (configurable)
        self.volatility_high_percentile = RAMF_VOLATILITY_HIGH_PERCENTILE
        self.volatility_low_percentile = RAMF_VOLATILITY_LOW_PERCENTILE

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
        self.paper_positions = {}  # {position_id: trade_dict}
        self.closed_positions = []  # List of closed positions for logging
        self._position_counter = 0  # Unique position ID counter

        # Initialize market data provider (replaces Moon Dev API)
        try:
            from src.data_providers.market_data import MarketDataProvider
            self._market_data = MarketDataProvider(start_liquidation_stream=True)
            cprint("[RAMF] Market data provider initialized", "green")
        except Exception as e:
            self._market_data = None
            cprint(f"[RAMF] Warning: Could not initialize market data provider: {e}", "yellow")

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
                'atr_percentile': round(float(percentile), 1),
                'current_atr': round(float(current_atr), 6)
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
                'vwap_distance_atr': round(float(vwap_distance_atr), 2),
                'consecutive_up': int(consecutive_up),
                'consecutive_down': int(consecutive_down),
                'overbought_exhaustion': bool(overbought_exhaustion),
                'oversold_exhaustion': bool(oversold_exhaustion),
                'current_vwap': round(float(current_vwap), 2),
                'current_price': round(float(current_price), 2)
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
                'volume_ratio': round(float(volume_ratio), 2),
                'is_volume_spike': bool(is_volume_spike),
                'current_volume': round(float(current_volume), 2),
                'avg_volume': round(float(avg_volume), 2)
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

        Uses Binance Futures WebSocket for real-time liquidation data.

        Ratio > 1.0 = More longs liquidated (bearish pressure)
        Ratio < 1.0 = More shorts liquidated (bullish pressure)
        Ratio = 1.0 = Balanced (or no data)

        Returns:
            float: Long/Short liquidation ratio (1.0 = balanced)
        """
        try:
            if self._market_data is not None:
                ratio = self._market_data.get_liquidation_ratio(minutes=15)
                return round(float(ratio), 2)

            # Fallback: try direct import
            from src.data_providers.market_data import get_market_data_provider
            provider = get_market_data_provider()
            ratio = provider.get_liquidation_ratio(minutes=15)
            return round(float(ratio), 2)

        except Exception as e:
            cprint(f"[RAMF] Warning: Could not get liquidation ratio: {e}", "yellow")
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

    # =========================================================================
    # v2.0 IMPROVEMENTS - New Analysis Methods
    # =========================================================================

    def get_funding_divergence(self, symbol: str, df: pd.DataFrame) -> dict:
        """
        Detect divergence between funding rate and price action.

        Bullish divergence: Price falling but funding very negative (shorts paying longs)
        Bearish divergence: Price rising but funding very positive (longs paying shorts)

        Returns:
            dict: {
                'score': float (-1 to +1, positive = bullish divergence),
                'signal': str ('bullish_div', 'bearish_div', 'neutral'),
                'funding_rate': float,
                'price_change_pct': float
            }
        """
        if not RAMF_USE_FUNDING_DIVERGENCE:
            return {'score': 0.0, 'signal': 'neutral', 'funding_rate': 0.0, 'price_change_pct': 0.0}

        try:
            # Get current funding rate
            funding_zscore = self.get_funding_zscore(symbol)

            # Calculate price change over lookback period
            lookback_candles = min(len(df), RAMF_FUNDING_DIV_LOOKBACK * 4)  # Assuming 15m candles
            if lookback_candles < 10:
                return {'score': 0.0, 'signal': 'neutral', 'funding_rate': 0.0, 'price_change_pct': 0.0}

            start_price = df['close'].iloc[-lookback_candles]
            end_price = df['close'].iloc[-1]
            price_change_pct = (end_price - start_price) / start_price * 100

            # Detect divergence
            # Bullish: Price down but funding extremely negative (shorts are overcrowded)
            # Bearish: Price up but funding extremely positive (longs are overcrowded)

            score = 0.0
            signal = 'neutral'

            if price_change_pct < -2 and funding_zscore < -1.5:
                # Price falling, funding very negative = bullish divergence
                score = min(1.0, abs(funding_zscore) * 0.3)
                signal = 'bullish_div'
            elif price_change_pct > 2 and funding_zscore > 1.5:
                # Price rising, funding very positive = bearish divergence
                score = -min(1.0, abs(funding_zscore) * 0.3)
                signal = 'bearish_div'
            elif abs(funding_zscore) > 2.0:
                # Extreme funding without price confirmation - still valuable
                if funding_zscore < -2.0:
                    score = 0.5
                    signal = 'bullish_div'
                elif funding_zscore > 2.0:
                    score = -0.5
                    signal = 'bearish_div'

            return {
                'score': round(score, 2),
                'signal': signal,
                'funding_rate': funding_zscore,
                'price_change_pct': round(price_change_pct, 2)
            }

        except Exception as e:
            cprint(f"[RAMF] Error calculating funding divergence: {e}", "yellow")
            return {'score': 0.0, 'signal': 'neutral', 'funding_rate': 0.0, 'price_change_pct': 0.0}

    def calculate_adaptive_sl_tp(self, df: pd.DataFrame, direction: str) -> dict:
        """
        Calculate dynamic stop-loss and take-profit based on ATR.

        In high volatility: wider SL/TP to avoid noise
        In low volatility: tighter SL/TP for better R:R

        Args:
            df: OHLCV DataFrame
            direction: 'BUY' or 'SELL'

        Returns:
            dict: {'sl_pct': float, 'tp_pct': float, 'atr_value': float}
        """
        if not RAMF_USE_ADAPTIVE_SL_TP:
            return {
                'sl_pct': RAMF_STOP_LOSS_PCT,
                'tp_pct': RAMF_TAKE_PROFIT_PCT,
                'atr_value': 0.0
            }

        try:
            # Calculate ATR
            atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
            current_atr = atr.average_true_range().iloc[-1]
            current_price = df['close'].iloc[-1]

            # ATR as percentage of price
            atr_pct = (current_atr / current_price) * 100

            # Calculate SL based on ATR multiplier
            sl_pct = atr_pct * RAMF_ATR_SL_MULTIPLIER

            # Apply floor and ceiling
            sl_pct = max(RAMF_MIN_SL_PCT, min(RAMF_MAX_SL_PCT, sl_pct))

            # TP based on ATR multiplier (maintains R:R ratio)
            tp_pct = atr_pct * RAMF_ATR_TP_MULTIPLIER

            return {
                'sl_pct': round(sl_pct, 2),
                'tp_pct': round(tp_pct, 2),
                'atr_value': round(current_atr, 4),
                'atr_pct': round(atr_pct, 2)
            }

        except Exception as e:
            cprint(f"[RAMF] Error calculating adaptive SL/TP: {e}", "yellow")
            return {
                'sl_pct': RAMF_STOP_LOSS_PCT,
                'tp_pct': RAMF_TAKE_PROFIT_PCT,
                'atr_value': 0.0
            }

    def get_time_window_modifier(self) -> dict:
        """
        Calculate confidence modifier based on current trading hour (UTC).

        Optimal hours: London open (7-9), NY open (13-15), Asia close (19-21)
        Avoid hours: Late US session and early Asia (0-5 UTC)

        Returns:
            dict: {'modifier': int, 'hour': int, 'session': str}
        """
        if not RAMF_USE_TIME_WINDOWS:
            return {'modifier': 0, 'hour': 0, 'session': 'any'}

        try:
            current_hour = datetime.utcnow().hour

            if current_hour in RAMF_OPTIMAL_HOURS:
                # Determine session
                if current_hour in [7, 8, 9]:
                    session = 'london_open'
                elif current_hour in [13, 14, 15]:
                    session = 'ny_open'
                else:
                    session = 'asia_close'

                return {
                    'modifier': RAMF_OPTIMAL_HOUR_BONUS,
                    'hour': current_hour,
                    'session': session
                }

            elif current_hour in RAMF_AVOID_HOURS:
                return {
                    'modifier': -RAMF_AVOID_HOUR_PENALTY,
                    'hour': current_hour,
                    'session': 'low_liquidity'
                }

            else:
                return {
                    'modifier': 0,
                    'hour': current_hour,
                    'session': 'normal'
                }

        except Exception as e:
            cprint(f"[RAMF] Error calculating time window: {e}", "yellow")
            return {'modifier': 0, 'hour': 0, 'session': 'unknown'}

    def get_mtf_confluence(self, symbol: str, direction: str) -> dict:
        """
        Check for multi-timeframe confluence.

        Looks for alignment across multiple timeframes to confirm trade direction.

        Args:
            symbol: Crypto symbol
            direction: Expected direction ('BUY' or 'SELL')

        Returns:
            dict: {
                'agreements': int (number of timeframes agreeing),
                'total': int (total timeframes checked),
                'bonus': int (confidence bonus),
                'details': dict (per-timeframe results)
            }
        """
        if not RAMF_USE_MTF:
            return {'agreements': 0, 'total': 0, 'bonus': 0, 'details': {}}

        try:
            agreements = 0
            details = {}

            for tf in RAMF_MTF_TIMEFRAMES:
                try:
                    # Fetch data for this timeframe
                    tf_df = self._fetch_candles(symbol, interval=tf, candles=100)
                    if tf_df is None or len(tf_df) < 50:
                        details[tf] = 'no_data'
                        continue

                    # Simple trend check: price vs EMA20
                    close = tf_df['close']
                    ema20 = EMAIndicator(close, window=20).ema_indicator()
                    current_price = close.iloc[-1]
                    current_ema = ema20.iloc[-1]

                    # Check RSI for confirmation
                    rsi = RSIIndicator(close, window=14).rsi().iloc[-1]

                    tf_bullish = current_price > current_ema and rsi > 40
                    tf_bearish = current_price < current_ema and rsi < 60

                    if direction == 'BUY' and tf_bullish:
                        agreements += 1
                        details[tf] = 'bullish'
                    elif direction == 'SELL' and tf_bearish:
                        agreements += 1
                        details[tf] = 'bearish'
                    else:
                        details[tf] = 'neutral'

                except Exception as e:
                    details[tf] = f'error: {str(e)[:20]}'

            # Calculate bonus
            bonus = 0
            if agreements >= RAMF_MTF_MIN_AGREEMENT:
                bonus = (agreements - RAMF_MTF_MIN_AGREEMENT + 1) * RAMF_MTF_AGREEMENT_BONUS

            return {
                'agreements': agreements,
                'total': len(RAMF_MTF_TIMEFRAMES),
                'bonus': bonus,
                'details': details
            }

        except Exception as e:
            cprint(f"[RAMF] Error calculating MTF confluence: {e}", "yellow")
            return {'agreements': 0, 'total': 0, 'bonus': 0, 'details': {}}

    def predict_liquidation_clusters(self, df: pd.DataFrame, direction: str) -> dict:
        """
        Predict areas where liquidations might cluster based on:
        - Recent swing highs/lows (common SL placement)
        - Round numbers
        - Previous liquidation imbalance direction

        Args:
            df: OHLCV DataFrame
            direction: Trade direction ('BUY' or 'SELL')

        Returns:
            dict: {
                'near_cluster': bool,
                'cluster_price': float,
                'distance_pct': float,
                'bonus': int
            }
        """
        if not RAMF_USE_LIQ_CLUSTERS:
            return {'near_cluster': False, 'cluster_price': 0.0, 'distance_pct': 0.0, 'bonus': 0}

        try:
            current_price = df['close'].iloc[-1]

            # Find recent swing points (potential liquidation levels)
            highs = df['high'].tail(50)
            lows = df['low'].tail(50)

            # Recent high/low that could be a liquidation cluster
            recent_high = highs.max()
            recent_low = lows.min()

            # Get liquidation ratio to understand current pressure
            liq_ratio = self.get_liquidation_imbalance()

            near_cluster = False
            cluster_price = 0.0
            bonus = 0

            if direction == 'BUY':
                # For longs: shorts might be liquidated above recent high
                # If we're near recent high and liq_ratio shows short liquidations
                distance_to_high = (recent_high - current_price) / current_price * 100

                if 0 < distance_to_high < 2 and liq_ratio < 1.0:
                    # Price approaching recent high, shorts being liquidated
                    # This could trigger a cascade - bullish!
                    near_cluster = True
                    cluster_price = recent_high
                    bonus = RAMF_LIQ_CLUSTER_BONUS

                distance_pct = distance_to_high

            else:  # SELL
                # For shorts: longs might be liquidated below recent low
                distance_to_low = (current_price - recent_low) / current_price * 100

                if 0 < distance_to_low < 2 and liq_ratio > 1.0:
                    # Price approaching recent low, longs being liquidated
                    # This could trigger a cascade - bearish!
                    near_cluster = True
                    cluster_price = recent_low
                    bonus = RAMF_LIQ_CLUSTER_BONUS

                distance_pct = distance_to_low

            return {
                'near_cluster': near_cluster,
                'cluster_price': round(cluster_price, 2),
                'distance_pct': round(distance_pct, 2),
                'bonus': bonus,
                'liq_ratio': liq_ratio
            }

        except Exception as e:
            cprint(f"[RAMF] Error predicting liquidation clusters: {e}", "yellow")
            return {'near_cluster': False, 'cluster_price': 0.0, 'distance_pct': 0.0, 'bonus': 0}

    # =========================================================================
    # END v2.0 IMPROVEMENTS
    # =========================================================================

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
        - v2.0: Enhanced with funding divergence, MTF, time windows, adaptive SL/TP
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

            # =========================================================================
            # v2.0: Get additional analysis data
            # =========================================================================
            # 8. Funding rate divergence
            funding_div = self.get_funding_divergence(symbol, df)

            # 9. Time window modifier
            time_window = self.get_time_window_modifier()

            # Initialize scores
            long_score = 0
            short_score = 0

            # =========================================================================
            # BASE SCORING (depends on volatility regime)
            # =========================================================================
            if regime['regime'] == 'high':
                # HIGH VOLATILITY MODE: Mean-reversion / Fade exhaustion
                cprint(f"[RAMF] {symbol}: High volatility mode (ATR {regime['atr_percentile']:.0f}th pctl)", "cyan")

                # LONG conditions (fade oversold exhaustion)
                if exhaustion['oversold_exhaustion']:
                    long_score += 25
                    cprint(f"  + Oversold exhaustion detected (+25)", "green")

                if volume['is_volume_spike']:
                    long_score += 10
                    cprint(f"  + Volume spike ({volume['volume_ratio']:.1f}x) (+10)", "green")

                if funding_zscore < -self.funding_zscore_threshold:
                    long_score += 15
                    cprint(f"  + Negative funding Z={funding_zscore:.2f} (+15)", "green")

                if liq_ratio > self.liquidation_ratio_threshold:
                    long_score += 10
                    cprint(f"  + Long liquidation cascade ({liq_ratio:.2f}) (+10)", "green")

                if btc_bullish:
                    long_score += 10
                    cprint(f"  + BTC macro bullish (+10)", "green")

                if rsi < 30:
                    long_score += 5
                    cprint(f"  + RSI oversold ({rsi:.0f}) (+5)", "green")

                # SHORT conditions (fade overbought exhaustion)
                if exhaustion['overbought_exhaustion']:
                    short_score += 25
                    cprint(f"  + Overbought exhaustion detected (+25)", "red")

                if volume['is_volume_spike']:
                    short_score += 10
                    cprint(f"  + Volume spike ({volume['volume_ratio']:.1f}x) (+10)", "red")

                if funding_zscore > self.funding_zscore_threshold:
                    short_score += 15
                    cprint(f"  + Positive funding Z={funding_zscore:.2f} (+15)", "red")

                if liq_ratio < (1 / self.liquidation_ratio_threshold):
                    short_score += 10
                    cprint(f"  + Short liquidation cascade ({liq_ratio:.2f}) (+10)", "red")

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
                    long_score += 35
                    cprint(f"  + Price above VWAP (+35)", "green")
                    if btc_bullish:
                        long_score += 25
                        cprint(f"  + BTC macro bullish (+25)", "green")
                elif exhaustion['vwap_distance_atr'] < -0.5:
                    short_score += 35
                    cprint(f"  + Price below VWAP (+35)", "red")

                # Scale down for low vol (less confident) - use lower threshold
                long_score = int(long_score * 0.7)
                short_score = int(short_score * 0.7)
                # NOTE: Using RAMF_MIN_CONFIDENCE_LOW_VOL (35) for this regime

            else:
                # NORMAL VOLATILITY: No clear edge, stay out
                cprint(f"[RAMF] {symbol}: Normal volatility - no clear edge", "white")
                return {
                    'token': symbol,
                    'signal': 0.0,
                    'direction': 'NEUTRAL',
                    'metadata': {
                        'strategy_type': 'ramf_v2',
                        'regime': regime,
                        'reason': 'Normal volatility - no clear edge'
                    }
                }

            # =========================================================================
            # v2.0: APPLY ENHANCEMENT MODIFIERS
            # =========================================================================

            # Funding divergence bonus
            if funding_div['signal'] == 'bullish_div' and funding_div['score'] >= RAMF_FUNDING_DIV_THRESHOLD:
                bonus = int(RAMF_FUNDING_DIV_BONUS * abs(funding_div['score']))
                long_score += bonus
                cprint(f"  + Bullish funding divergence (+{bonus})", "green")
            elif funding_div['signal'] == 'bearish_div' and abs(funding_div['score']) >= RAMF_FUNDING_DIV_THRESHOLD:
                bonus = int(RAMF_FUNDING_DIV_BONUS * abs(funding_div['score']))
                short_score += bonus
                cprint(f"  + Bearish funding divergence (+{bonus})", "red")

            # Time window modifier (applies to both)
            if time_window['modifier'] != 0:
                if time_window['modifier'] > 0:
                    long_score += time_window['modifier']
                    short_score += time_window['modifier']
                    cprint(f"  + Optimal trading hour ({time_window['session']}) (+{time_window['modifier']})", "cyan")
                else:
                    long_score += time_window['modifier']
                    short_score += time_window['modifier']
                    cprint(f"  - Low liquidity hour ({time_window['session']}) ({time_window['modifier']})", "yellow")

            # Determine preliminary direction for MTF and liquidation cluster checks
            prelim_direction = 'BUY' if long_score > short_score else 'SELL' if short_score > long_score else 'NEUTRAL'

            # MTF confluence (only check if we have a preliminary direction)
            mtf_data = {'agreements': 0, 'total': 0, 'bonus': 0, 'details': {}}
            if prelim_direction != 'NEUTRAL':
                mtf_data = self.get_mtf_confluence(symbol, prelim_direction)
                if mtf_data['bonus'] > 0:
                    if prelim_direction == 'BUY':
                        long_score += mtf_data['bonus']
                    else:
                        short_score += mtf_data['bonus']
                    cprint(f"  + MTF confluence ({mtf_data['agreements']}/{mtf_data['total']} TFs) (+{mtf_data['bonus']})", "magenta")

            # Liquidation cluster prediction
            liq_cluster = {'near_cluster': False, 'bonus': 0}
            if prelim_direction != 'NEUTRAL':
                liq_cluster = self.predict_liquidation_clusters(df, prelim_direction)
                if liq_cluster['bonus'] > 0:
                    if prelim_direction == 'BUY':
                        long_score += liq_cluster['bonus']
                    else:
                        short_score += liq_cluster['bonus']
                    cprint(f"  + Near liquidation cluster @ ${liq_cluster['cluster_price']:,.0f} (+{liq_cluster['bonus']})", "magenta")

            # =========================================================================
            # FINAL DIRECTION DETERMINATION
            # =========================================================================
            effective_min_confidence = RAMF_MIN_CONFIDENCE_LOW_VOL if regime['regime'] == 'low' else self.min_confidence

            if long_score >= effective_min_confidence:
                direction = 'BUY'
                signal_strength = min(1.0, long_score / 100)
            elif short_score >= effective_min_confidence:
                direction = 'SELL'
                signal_strength = min(1.0, short_score / 100)
            else:
                direction = 'NEUTRAL'
                signal_strength = 0.0

            # =========================================================================
            # v2.0: Calculate adaptive SL/TP
            # =========================================================================
            adaptive_levels = self.calculate_adaptive_sl_tp(df, direction)

            # Build signal
            current_price = float(df['close'].iloc[-1])

            signal = {
                'token': symbol,
                'signal': round(float(signal_strength), 3),
                'direction': direction,
                'metadata': {
                    'strategy_type': 'ramf_v2',
                    'regime': regime,
                    'exhaustion': exhaustion,
                    'volume': volume,
                    'funding_zscore': float(funding_zscore),
                    'liquidation_ratio': float(liq_ratio),
                    'btc_macro_bullish': bool(btc_bullish),
                    'rsi': float(rsi),
                    'long_score': int(long_score),
                    'short_score': int(short_score),
                    'current_price': float(current_price),
                    # v2.0: Adaptive SL/TP
                    'stop_loss_pct': float(adaptive_levels['sl_pct']),
                    'take_profit_pct': float(adaptive_levels['tp_pct']),
                    'atr_pct': float(adaptive_levels.get('atr_pct', 0)),
                    'leverage': int(RAMF_LEVERAGE),
                    # v2.0: New metadata
                    'funding_divergence': funding_div,
                    'time_window': time_window,
                    'mtf_confluence': mtf_data,
                    'liq_cluster': liq_cluster
                }
            }

            # Log actionable signals
            if direction != 'NEUTRAL':
                color = 'green' if direction == 'BUY' else 'red'
                score = long_score if direction == 'BUY' else short_score

                cprint("=" * 60, color)
                cprint(f"[RAMF v2.0] SIGNAL: {direction} {symbol} @ ${current_price:,.2f}", color, attrs=['bold'])
                cprint(f"  Confidence: {score}%", color)
                cprint(f"  Regime: {regime['regime']} ({regime['atr_percentile']:.0f}th pctl)", 'white')
                cprint(f"  VWAP Distance: {exhaustion['vwap_distance_atr']:.2f} ATR", 'white')
                cprint(f"  Funding Z-Score: {funding_zscore:.2f} | Divergence: {funding_div['signal']}", 'white')
                cprint(f"  Liquidation Ratio: {liq_ratio:.2f}", 'white')
                cprint(f"  RSI: {rsi:.0f}", 'white')
                cprint(f"  MTF Agreement: {mtf_data['agreements']}/{mtf_data['total']}", 'white')
                cprint(f"  Time Window: {time_window['session']} (UTC {time_window['hour']}h)", 'white')
                cprint(f"  Adaptive SL: {adaptive_levels['sl_pct']:.2f}% | TP: {adaptive_levels['tp_pct']:.2f}%", 'cyan')
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
        """Log signal to CSV file with v2.0 metadata."""
        try:
            log_file = os.path.join(self.data_dir, 'signals.csv')

            # v2.0: Enhanced logging with all new features
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'symbol': signal['token'],
                'direction': signal['direction'],
                'confidence': signal['signal'],
                'price': signal['metadata'].get('current_price', 0),
                'regime': signal['metadata'].get('regime', {}).get('regime', 'unknown'),
                'funding_zscore': signal['metadata'].get('funding_zscore', 0),
                'rsi': signal['metadata'].get('rsi', 50),
                'long_score': signal['metadata'].get('long_score', 0),
                'short_score': signal['metadata'].get('short_score', 0),
                # v2.0: Adaptive SL/TP
                'sl_pct': signal['metadata'].get('stop_loss_pct', RAMF_STOP_LOSS_PCT),
                'tp_pct': signal['metadata'].get('take_profit_pct', RAMF_TAKE_PROFIT_PCT),
                'atr_pct': signal['metadata'].get('atr_pct', 0),
                # v2.0: Funding divergence
                'funding_div_signal': signal['metadata'].get('funding_divergence', {}).get('signal', 'neutral'),
                'funding_div_score': signal['metadata'].get('funding_divergence', {}).get('score', 0),
                # v2.0: Time window
                'time_session': signal['metadata'].get('time_window', {}).get('session', 'unknown'),
                'time_modifier': signal['metadata'].get('time_window', {}).get('modifier', 0),
                # v2.0: MTF confluence
                'mtf_agreements': signal['metadata'].get('mtf_confluence', {}).get('agreements', 0),
                'mtf_total': signal['metadata'].get('mtf_confluence', {}).get('total', 0),
                # v2.0: Liquidation cluster
                'near_liq_cluster': signal['metadata'].get('liq_cluster', {}).get('near_cluster', False),
                # Paper trading status
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

        v2.0: Uses adaptive SL/TP from signal metadata when available.

        Args:
            signal: Signal dict from generate_signals()

        Returns:
            dict: Trade execution result
        """
        if not PAPER_TRADING:
            cprint("[RAMF] Paper trading disabled - skipping", "yellow")
            return None

        try:
            # Extract signal fields with safe defaults
            symbol = signal.get('token', '')
            direction = signal.get('direction', 'NEUTRAL')

            if not symbol or direction == 'NEUTRAL':
                cprint(f"[RAMF] Invalid signal: token={symbol}, direction={direction}", "yellow")
                return None

            # Get metadata safely (may be empty dict from strategy_agent wrapper)
            metadata = signal.get('metadata', {})

            # Get price from metadata, or fetch current price if not available
            price = metadata.get('current_price', 0)
            if price == 0 or price is None:
                cprint(f"[RAMF] No price in metadata, fetching current price for {symbol}...", "yellow")
                try:
                    df = self._fetch_candles(symbol, interval='15m', candles=5)
                    if df is not None and len(df) > 0:
                        price = float(df['close'].iloc[-1])
                        cprint(f"[RAMF] Fetched price: ${price:,.2f}", "white")
                except Exception as e:
                    cprint(f"[RAMF] Could not fetch price: {e}", "red")
                    return None

            if price <= 0:
                cprint(f"[RAMF] Cannot execute trade with price={price}", "red")
                return None

            # Get confidence (support both 'signal' key and 'confidence' key)
            confidence = signal.get('signal', signal.get('confidence', 0))
            if isinstance(confidence, (int, float)) and confidence > 1:
                confidence = confidence / 100  # Convert percentage to decimal

            # v2.0: Get adaptive SL/TP from signal metadata (falls back to config defaults)
            sl_pct = metadata.get('stop_loss_pct', RAMF_STOP_LOSS_PCT)
            tp_pct = metadata.get('take_profit_pct', RAMF_TAKE_PROFIT_PCT)

            # Ensure SL/TP are valid
            if sl_pct <= 0:
                sl_pct = RAMF_STOP_LOSS_PCT
            if tp_pct <= 0:
                tp_pct = RAMF_TAKE_PROFIT_PCT

            cprint(f"[RAMF] Executing paper trade: {direction} {symbol} @ ${price:,.2f}", "cyan")

            # Calculate margin already used by open positions
            used_margin = sum(
                pos.get('position_size', 0) / pos.get('leverage', RAMF_LEVERAGE)
                for pos in self.paper_positions.values()
            )
            available_margin = max(0, self.paper_balance - used_margin)

            # Calculate position size (2% risk per trade)
            # Position size is adjusted for the adaptive SL
            risk_amount = self.paper_balance * 0.02
            position_size = risk_amount / (sl_pct / 100) * RAMF_LEVERAGE

            # Calculate margin required for this position
            margin_required = position_size / RAMF_LEVERAGE

            # Cap position size to available margin (leave 10% buffer)
            max_position_for_margin = available_margin * 0.9 * RAMF_LEVERAGE
            if position_size > max_position_for_margin:
                old_size = position_size
                position_size = max_position_for_margin
                margin_required = position_size / RAMF_LEVERAGE
                cprint(f"[RAMF] Position capped: ${old_size:,.2f} -> ${position_size:,.2f} (margin limit)", "yellow")

            # Minimum viable position size check
            if position_size < 10:
                cprint(f"[RAMF] Insufficient margin for trade. Available: ${available_margin:.2f}, Required: ${margin_required:.2f}", "red")
                return None

            # Generate unique position ID
            self._position_counter += 1
            position_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._position_counter}"

            # Calculate SL/TP prices using adaptive percentages
            if direction == 'BUY':
                stop_loss_price = price * (1 - sl_pct / 100)
                take_profit_price = price * (1 + tp_pct / 100)
            else:  # SELL
                stop_loss_price = price * (1 + sl_pct / 100)
                take_profit_price = price * (1 - tp_pct / 100)

            # Record trade with v2.0 metadata
            trade = {
                'position_id': position_id,
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'direction': direction,
                'entry_price': price,
                'position_size': round(position_size, 2),
                'leverage': RAMF_LEVERAGE,
                'stop_loss': round(stop_loss_price, 2),
                'take_profit': round(take_profit_price, 2),
                'sl_pct': sl_pct,
                'tp_pct': tp_pct,
                'confidence': confidence,
                'status': 'OPEN',
                # v2.0: Store additional metadata for analysis (with safe access)
                'regime': metadata.get('regime', {}).get('regime', 'unknown') if isinstance(metadata.get('regime'), dict) else 'unknown',
                'mtf_agreements': metadata.get('mtf_confluence', {}).get('agreements', 0) if isinstance(metadata.get('mtf_confluence'), dict) else 0,
                'time_session': metadata.get('time_window', {}).get('session', 'unknown') if isinstance(metadata.get('time_window'), dict) else 'unknown',
                'funding_div': metadata.get('funding_divergence', {}).get('signal', 'neutral') if isinstance(metadata.get('funding_divergence'), dict) else 'neutral'
            }

            # Store position with unique ID (no overwrite!)
            self.paper_positions[position_id] = trade
            self.daily_trades += 1

            # Log to file
            log_file = os.path.join(self.data_dir, 'paper_trades.csv')
            df = pd.DataFrame([trade])

            if os.path.exists(log_file):
                df.to_csv(log_file, mode='a', header=False, index=False)
            else:
                df.to_csv(log_file, index=False)

            cprint(f"[RAMF v2.0 PAPER] Opened {direction} {symbol} (ID: {position_id})", "magenta")
            cprint(f"  Entry: ${price:,.2f} | Size: ${position_size:,.2f}", "white")
            cprint(f"  Adaptive SL: ${trade['stop_loss']:,.2f} ({sl_pct:.2f}%)", "white")
            cprint(f"  Adaptive TP: ${trade['take_profit']:,.2f} ({tp_pct:.2f}%)", "white")
            cprint(f"  Regime: {trade['regime']} | MTF: {trade['mtf_agreements']} | Session: {trade['time_session']}", "white")

            return trade

        except Exception as e:
            cprint(f"[RAMF] Error executing paper trade: {e}", "red")
            import traceback
            traceback.print_exc()
            return None

    def monitor_paper_positions(self) -> list:
        """
        Monitor all open paper positions and close those that hit SL/TP.

        This should be called periodically (e.g., every cycle in main.py).

        Returns:
            list: List of closed positions
        """
        if not PAPER_TRADING or not self.paper_positions:
            return []

        closed = []

        # Get current prices for all symbols with open positions
        symbols_to_check = set(pos['symbol'] for pos in self.paper_positions.values())
        current_prices = {}

        for symbol in symbols_to_check:
            try:
                df = self._fetch_candles(symbol, interval='15m', candles=5)
                if df is not None and len(df) > 0:
                    current_prices[symbol] = float(df['close'].iloc[-1])
            except Exception as e:
                cprint(f"[RAMF] Could not fetch price for {symbol}: {e}", "yellow")

        # Check each position
        positions_to_close = []

        for position_id, trade in self.paper_positions.items():
            symbol = trade['symbol']
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            direction = trade['direction']
            entry_price = trade['entry_price']
            stop_loss = trade['stop_loss']
            take_profit = trade['take_profit']

            close_reason = None

            if direction == 'BUY':
                # Long position
                if current_price <= stop_loss:
                    close_reason = 'STOP_LOSS'
                elif current_price >= take_profit:
                    close_reason = 'TAKE_PROFIT'
            else:
                # Short position
                if current_price >= stop_loss:
                    close_reason = 'STOP_LOSS'
                elif current_price <= take_profit:
                    close_reason = 'TAKE_PROFIT'

            if close_reason:
                positions_to_close.append((position_id, current_price, close_reason))

        # Close positions and calculate PnL
        for position_id, close_price, reason in positions_to_close:
            closed_trade = self._close_paper_position(position_id, close_price, reason)
            if closed_trade:
                closed.append(closed_trade)

        return closed

    def _close_paper_position(self, position_id: str, close_price: float, reason: str) -> dict:
        """
        Close a paper position and update PnL.

        Args:
            position_id: Unique position ID
            close_price: Price at which to close
            reason: Closure reason (STOP_LOSS, TAKE_PROFIT, MANUAL)

        Returns:
            dict: Closed position with PnL
        """
        if position_id not in self.paper_positions:
            return None

        try:
            trade = self.paper_positions[position_id].copy()
            entry_price = trade['entry_price']
            direction = trade['direction']
            position_size = trade['position_size']

            # Calculate PnL
            if direction == 'BUY':
                price_change_pct = (close_price - entry_price) / entry_price
            else:
                price_change_pct = (entry_price - close_price) / entry_price

            # PnL = position_size * price_change_pct (leverage already factored in position_size)
            pnl = position_size * price_change_pct

            # Update trade record
            trade['close_price'] = close_price
            trade['close_timestamp'] = datetime.now().isoformat()
            trade['close_reason'] = reason
            trade['pnl'] = round(pnl, 2)
            trade['pnl_pct'] = round(price_change_pct * 100, 2)
            trade['status'] = 'CLOSED'

            # Update daily PnL
            self.daily_pnl += pnl

            # Update paper balance
            self.paper_balance += pnl

            # Remove from open positions
            del self.paper_positions[position_id]

            # Add to closed positions
            self.closed_positions.append(trade)

            # Log closure
            color = 'green' if pnl > 0 else 'red'
            cprint(f"[RAMF PAPER] Closed {trade['symbol']} ({reason})", color, attrs=['bold'])
            cprint(f"  Entry: ${entry_price:,.2f} → Exit: ${close_price:,.2f}", "white")
            cprint(f"  PnL: ${pnl:+,.2f} ({price_change_pct*100:+.2f}%)", color)
            cprint(f"  Daily PnL: ${self.daily_pnl:+,.2f} | Balance: ${self.paper_balance:,.2f}", "white")

            # Log to closed trades file
            self._log_closed_trade(trade)

            return trade

        except Exception as e:
            cprint(f"[RAMF] Error closing position {position_id}: {e}", "red")
            import traceback
            traceback.print_exc()
            return None

    def _log_closed_trade(self, trade: dict):
        """Log closed trade to CSV file."""
        try:
            log_file = os.path.join(self.data_dir, 'closed_trades.csv')
            df = pd.DataFrame([trade])

            if os.path.exists(log_file):
                df.to_csv(log_file, mode='a', header=False, index=False)
            else:
                df.to_csv(log_file, index=False)

        except Exception as e:
            cprint(f"[RAMF] Error logging closed trade: {e}", "yellow")

    def get_paper_status(self) -> dict:
        """
        Get current paper trading status.

        Returns:
            dict: Status including balance, open positions, daily PnL
        """
        return {
            'paper_balance': round(self.paper_balance, 2),
            'initial_balance': PAPER_TRADING_BALANCE,
            'total_pnl': round(self.paper_balance - PAPER_TRADING_BALANCE, 2),
            'daily_pnl': round(self.daily_pnl, 2),
            'daily_trades': self.daily_trades,
            'open_positions': len(self.paper_positions),
            'total_closed': len(self.closed_positions),
            'positions': list(self.paper_positions.values())
        }

    def close_all_paper_positions(self) -> list:
        """
        Force close all open paper positions at current market price.

        Returns:
            list: List of closed positions
        """
        if not self.paper_positions:
            return []

        closed = []
        position_ids = list(self.paper_positions.keys())

        for position_id in position_ids:
            trade = self.paper_positions[position_id]
            symbol = trade['symbol']

            try:
                df = self._fetch_candles(symbol, interval='15m', candles=5)
                if df is not None and len(df) > 0:
                    current_price = float(df['close'].iloc[-1])
                    closed_trade = self._close_paper_position(position_id, current_price, 'MANUAL')
                    if closed_trade:
                        closed.append(closed_trade)
            except Exception as e:
                cprint(f"[RAMF] Error closing {position_id}: {e}", "red")

        return closed


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
