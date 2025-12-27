"""
Multifactor Strategy - Combines technical analysis with social sentiment
"""
import pandas as pd
import numpy as np
from termcolor import cprint
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volume import VolumeWeightedAveragePrice

from ..base_strategy import BaseStrategy
from ..utils.sentiment_reader import SentimentReader

# Import config - will be added
try:
    from src.config import (
        MULTIFACTOR_ASSETS,
        MULTIFACTOR_SMALLCAPS,
        MULTIFACTOR_TIMEFRAME,
        MULTIFACTOR_RISK_PER_TRADE
    )
except ImportError:
    # Default values if config not updated yet
    MULTIFACTOR_ASSETS = ['BTC', 'ETH', 'SOL']
    MULTIFACTOR_SMALLCAPS = []
    MULTIFACTOR_TIMEFRAME = '15m'
    MULTIFACTOR_RISK_PER_TRADE = 0.04


class MultifactorStrategy(BaseStrategy):
    """
    Multi-factor trading strategy combining:
    - Trend analysis (EMA alignment)
    - Momentum (MACD)
    - RSI (overbought/oversold)
    - Volume confirmation
    - Social sentiment (Twitter)
    """

    def __init__(self):
        super().__init__("Multifactor Strategy")

        # Factor weights (must sum to 1.0)
        self.weights = {
            'trend': 0.25,
            'momentum': 0.20,
            'rsi': 0.20,
            'volume': 0.15,
            'sentiment': 0.20
        }

        # Thresholds
        self.buy_threshold = 0.6
        self.sell_threshold = -0.6

        # EMA periods
        self.ema_fast = 20
        self.ema_mid = 50
        self.ema_slow = 200

        # Assets to trade
        self.assets = MULTIFACTOR_ASSETS + MULTIFACTOR_SMALLCAPS

        # Sentiment reader
        self.sentiment_reader = SentimentReader()

        # Global trend filter
        self.use_btc_filter = True
        self._btc_trend_cache = None
        self._btc_trend_timestamp = None

    def check_btc_trend(self) -> bool:
        """
        Check if BTC is in an uptrend (price > EMA200).
        Used as a global market filter.

        Returns:
            bool: True if BTC is bullish, False otherwise
        """
        import time
        from datetime import datetime

        # Cache for 15 minutes
        if self._btc_trend_cache is not None and self._btc_trend_timestamp:
            age = (datetime.now() - self._btc_trend_timestamp).total_seconds()
            if age < 900:  # 15 minutes
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

            trend_status = "BULLISH" if is_bullish else "BEARISH"
            cprint(f"[BTC Filter] Price: ${current_price:,.0f} vs EMA200: ${current_ema200:,.0f} -> {trend_status}", "cyan")

            return is_bullish

        except Exception as e:
            cprint(f"[BTC Filter] Error: {e}", "yellow")
            return True  # Default to allowing trades

    def calculate_trend_score(self, df: pd.DataFrame) -> float:
        """
        Calculate trend score based on EMA alignment.

        Returns:
            float: Score from -1.0 (strong downtrend) to +1.0 (strong uptrend)
        """
        try:
            close = df['close']

            ema20 = EMAIndicator(close, window=self.ema_fast).ema_indicator()
            ema50 = EMAIndicator(close, window=self.ema_mid).ema_indicator()
            ema200 = EMAIndicator(close, window=self.ema_slow).ema_indicator()

            current_price = close.iloc[-1]
            current_ema20 = ema20.iloc[-1]
            current_ema50 = ema50.iloc[-1]
            current_ema200 = ema200.iloc[-1]

            # Strong uptrend: price > EMA20 > EMA50 > EMA200
            if current_price > current_ema20 > current_ema50 > current_ema200:
                return 1.0

            # Moderate uptrend: price > EMA20 > EMA50
            if current_price > current_ema20 > current_ema50:
                return 0.5

            # Strong downtrend: price < EMA20 < EMA50 < EMA200
            if current_price < current_ema20 < current_ema50 < current_ema200:
                return -1.0

            # Moderate downtrend: price < EMA20 < EMA50
            if current_price < current_ema20 < current_ema50:
                return -0.5

            # Range/indecisive
            return 0.0

        except Exception as e:
            cprint(f"Error calculating trend: {e}", "yellow")
            return 0.0

    def calculate_momentum_score(self, df: pd.DataFrame) -> float:
        """
        Calculate momentum score based on MACD.

        Returns:
            float: Score from -1.0 to +1.0
        """
        try:
            close = df['close']

            macd_indicator = MACD(close)
            macd = macd_indicator.macd()
            signal = macd_indicator.macd_signal()
            histogram = macd_indicator.macd_diff()

            current_macd = macd.iloc[-1]
            current_signal = signal.iloc[-1]
            current_hist = histogram.iloc[-1]
            prev_hist = histogram.iloc[-2] if len(histogram) > 1 else current_hist

            # MACD above signal with increasing histogram
            if current_macd > current_signal and current_hist > prev_hist:
                return 1.0

            # MACD above signal
            if current_macd > current_signal:
                return 0.5

            # MACD below signal with decreasing histogram
            if current_macd < current_signal and current_hist < prev_hist:
                return -1.0

            # MACD below signal
            if current_macd < current_signal:
                return -0.5

            return 0.0

        except Exception as e:
            cprint(f"Error calculating momentum: {e}", "yellow")
            return 0.0

    def calculate_rsi_score(self, df: pd.DataFrame) -> float:
        """
        Calculate RSI score with dynamic zones.

        Returns:
            float: Score from -1.0 (overbought) to +1.0 (oversold opportunity)
        """
        try:
            close = df['close']

            rsi = RSIIndicator(close, window=14).rsi()
            current_rsi = rsi.iloc[-1]

            # Oversold - buying opportunity
            if current_rsi < 30:
                return 1.0
            elif current_rsi < 45:
                return 0.5

            # Overbought - selling opportunity
            if current_rsi > 70:
                return -1.0
            elif current_rsi > 55:
                return -0.5

            # Neutral zone
            return 0.0

        except Exception as e:
            cprint(f"Error calculating RSI: {e}", "yellow")
            return 0.0

    def calculate_volume_score(self, df: pd.DataFrame) -> float:
        """
        Calculate volume score based on volume vs 20-period average.

        Returns:
            float: Score from -1.0 to +1.0
        """
        try:
            volume = df['volume']
            close = df['close']

            avg_volume = volume.rolling(window=20).mean()
            current_volume = volume.iloc[-1]
            current_avg = avg_volume.iloc[-1]

            # Price direction
            price_change = close.iloc[-1] - close.iloc[-2]
            price_up = price_change > 0

            # Volume ratio
            volume_ratio = current_volume / current_avg if current_avg > 0 else 1.0

            # High volume with price direction
            if volume_ratio > 2.0:
                return 1.0 if price_up else -1.0
            elif volume_ratio > 1.5:
                return 0.5 if price_up else -0.5

            # Normal volume
            return 0.0

        except Exception as e:
            cprint(f"Error calculating volume: {e}", "yellow")
            return 0.0

    def get_sentiment_score(self, symbol: str) -> float:
        """
        Get sentiment score using contrarian logic for Fear & Greed.

        Contrarian approach:
        - Extreme Fear (< -0.4) = Buying opportunity (+0.5 to +1.0)
        - Extreme Greed (> 0.4) = Selling signal (-0.5 to -1.0)
        - Neutral = No strong signal

        Args:
            symbol: Crypto symbol (e.g., 'BTC')

        Returns:
            float: Score from -1.0 to +1.0
        """
        try:
            # Check if we have recent data
            if not self.sentiment_reader.has_recent_data(max_age_hours=2.0):
                return 0.0

            raw_sentiment = self.sentiment_reader.get_current_sentiment(symbol)

            # Apply contrarian logic to Fear & Greed
            # Extreme Fear (-1 to -0.4) -> Buying opportunity (inverted to +0.5 to +1)
            # Extreme Greed (+0.4 to +1) -> Selling signal (inverted to -0.5 to -1)
            # Neutral (-0.4 to +0.4) -> Keep as is but dampen

            if raw_sentiment < -0.4:
                # Extreme Fear = Contrarian BUY signal
                # Convert -1.0 to +1.0, -0.4 to +0.5
                contrarian_score = 0.5 + ((-0.4 - raw_sentiment) / 0.6) * 0.5
                cprint(f"[Sentiment] Extreme Fear ({raw_sentiment:.2f}) -> Contrarian BUY: {contrarian_score:.2f}", "green")
                return contrarian_score

            elif raw_sentiment > 0.4:
                # Extreme Greed = Contrarian SELL signal
                # Convert +0.4 to -0.5, +1.0 to -1.0
                contrarian_score = -0.5 - ((raw_sentiment - 0.4) / 0.6) * 0.5
                cprint(f"[Sentiment] Extreme Greed ({raw_sentiment:.2f}) -> Contrarian SELL: {contrarian_score:.2f}", "red")
                return contrarian_score

            else:
                # Neutral zone - dampen the signal
                return raw_sentiment * 0.5

        except Exception as e:
            cprint(f"Error getting sentiment: {e}", "yellow")
            return 0.0

    def calculate_combined_score(self, df: pd.DataFrame, symbol: str) -> dict:
        """
        Calculate combined score from all factors.

        Returns:
            dict: Factor scores and combined score
        """
        scores = {
            'trend': self.calculate_trend_score(df),
            'momentum': self.calculate_momentum_score(df),
            'rsi': self.calculate_rsi_score(df),
            'volume': self.calculate_volume_score(df),
            'sentiment': self.get_sentiment_score(symbol)
        }

        combined = sum(
            scores[factor] * self.weights[factor]
            for factor in scores
        )

        return {
            'factors': scores,
            'combined': combined
        }

    def generate_signals(self, symbol: str = None, df: pd.DataFrame = None) -> dict:
        """
        Generate trading signals for the given symbol and data.

        Args:
            symbol: Crypto symbol (e.g., 'BTC')
            df: DataFrame with OHLCV data

        Returns:
            dict: Signal with token, direction, strength, and metadata
        """
        if symbol is None or df is None:
            # If called without args, iterate through all assets
            # This maintains compatibility with the base interface
            for asset in self.assets:
                try:
                    df = self._fetch_candles(asset)
                    if df is None or df.empty:
                        continue

                    signal = self._generate_signal_for_asset(asset, df)
                    if signal and signal['direction'] != 'NEUTRAL':
                        return signal

                except Exception as e:
                    cprint(f"Error processing {asset}: {e}", "yellow")
                    continue

            return None

        return self._generate_signal_for_asset(symbol, df)

    def _fetch_candles(self, symbol: str, interval: str = '1h', candles: int = 300) -> pd.DataFrame:
        """
        Fetch candle data from HyperLiquid.

        Args:
            symbol: Crypto symbol
            interval: Candle interval (e.g., '1h', '15m')
            candles: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        try:
            from hyperliquid.info import Info
            import time

            info = Info(skip_ws=True)

            # Calculate time range
            end_time = int(time.time() * 1000)  # Current time in ms
            # Approximate interval in ms
            interval_ms = {
                '1m': 60 * 1000,
                '5m': 5 * 60 * 1000,
                '15m': 15 * 60 * 1000,
                '1h': 60 * 60 * 1000,
                '4h': 4 * 60 * 60 * 1000,
                '1d': 24 * 60 * 60 * 1000
            }.get(interval, 60 * 60 * 1000)

            start_time = end_time - (candles * interval_ms)

            candle_data = info.candles_snapshot(symbol, interval, start_time, end_time)
            if not candle_data:
                return None

            # HyperLiquid returns dict format: {t, T, s, i, o, c, h, l, v, n}
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
            cprint(f"Error fetching candles for {symbol}: {e}", "yellow")
            return None

    def _generate_signal_for_asset(self, symbol: str, df: pd.DataFrame) -> dict:
        """
        Generate signal for a specific asset.
        """
        try:
            # Need at least 200 candles for EMA200
            if len(df) < 200:
                return None

            # Calculate combined score
            result = self.calculate_combined_score(df, symbol)
            combined_score = result['combined']
            factors = result['factors']

            # Check BTC trend filter for BUY signals
            btc_bullish = True
            if self.use_btc_filter and combined_score >= self.buy_threshold:
                btc_bullish = self.check_btc_trend()
                if not btc_bullish:
                    cprint(f"[BTC Filter] Blocking BUY signal for {symbol} - BTC in downtrend", "yellow")

            # Determine direction
            if combined_score >= self.buy_threshold and btc_bullish:
                direction = 'BUY'
                signal_strength = min(1.0, (combined_score - self.buy_threshold) / (1.0 - self.buy_threshold) + 0.5)
            elif combined_score <= self.sell_threshold:
                direction = 'SELL'
                signal_strength = min(1.0, (self.sell_threshold - combined_score) / (1.0 + self.sell_threshold) + 0.5)
            else:
                direction = 'NEUTRAL'
                signal_strength = 0.0

            signal = {
                'token': symbol,
                'signal': signal_strength,
                'direction': direction,
                'metadata': {
                    'strategy_type': 'multifactor',
                    'combined_score': round(combined_score, 3),
                    'factors': {k: round(v, 3) for k, v in factors.items()},
                    'weights': self.weights,
                    'thresholds': {
                        'buy': self.buy_threshold,
                        'sell': self.sell_threshold
                    },
                    'current_price': float(df['close'].iloc[-1]),
                    'btc_filter': 'bullish' if btc_bullish else 'bearish'
                }
            }

            # Log signal if not neutral
            if direction != 'NEUTRAL':
                color = 'green' if direction == 'BUY' else 'red'
                cprint(f"[Multifactor] {symbol}: {direction} (score: {combined_score:.3f})", color)
                for factor, score in factors.items():
                    cprint(f"  - {factor}: {score:.2f}", 'white')

            return signal

        except Exception as e:
            cprint(f"Error generating signal for {symbol}: {e}", "red")
            return None


# For standalone testing
if __name__ == "__main__":
    cprint("Testing Multifactor Strategy...", "cyan")

    strategy = MultifactorStrategy()

    for symbol in ['BTC', 'ETH', 'SOL']:
        try:
            cprint(f"\nAnalyzing {symbol}...", "white")
            df = strategy._fetch_candles(symbol)

            if df is not None and len(df) >= 200:
                signal = strategy.generate_signals(symbol, df)
                if signal:
                    color = 'green' if signal['direction'] == 'BUY' else ('red' if signal['direction'] == 'SELL' else 'white')
                    cprint(f"  Direction: {signal['direction']}", color)
                    cprint(f"  Strength: {signal['signal']:.2f}", "white")
                    cprint(f"  Combined Score: {signal['metadata']['combined_score']}", "white")
                    cprint(f"  Factors:", "white")
                    for factor, score in signal['metadata']['factors'].items():
                        cprint(f"    - {factor}: {score}", "white")
            else:
                cprint(f"  Not enough data for {symbol}", "yellow")

        except Exception as e:
            cprint(f"Error testing {symbol}: {e}", "red")

    cprint("\nTest completed!", "cyan")
