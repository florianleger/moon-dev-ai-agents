"""
Adaptive Hybrid Strategy

Multi-module scoring strategy that aggregates 8 independent signal generators.
Instead of requiring ALL conditions (AND logic), it requires enough convergence
(weighted score > threshold) for a trade signal.

Modules:
1. Mean Reversion (Bollinger Bands + RSI)
2. Momentum Breakout (range breakout + volume)
3. EMA Crossover (9/21 crossover)
4. Funding Rate Contrarian (extreme funding Z-score)
5. RSI Divergence (price vs RSI divergence)
6. Sniper Lite (relaxed extreme move + funding)
7. Trend Rider Lite (relaxed trend + pullback)
8. RAMF Lite (volatility regime + exhaustion)

Target: 1-3 trades/day with 55%+ win rate.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from termcolor import cprint
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import EMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volume import VolumeWeightedAveragePrice

from ..base_strategy import BaseStrategy

# Import config with defaults
try:
    from src.config import (
        PAPER_TRADING,
        PAPER_TRADING_BALANCE,
        ADAPTIVE_HYBRID_BASE_THRESHOLD,
        ADAPTIVE_HYBRID_URGENCY_START_HOURS,
        ADAPTIVE_HYBRID_URGENCY_FLOOR,
        ADAPTIVE_HYBRID_MAX_DAILY_TRADES,
        ADAPTIVE_HYBRID_MAX_DAILY_LOSS_USD,
        ADAPTIVE_HYBRID_LEVERAGE,
        ADAPTIVE_HYBRID_ATR_SL_MULT,
        ADAPTIVE_HYBRID_ATR_TP_MULT,
        ADAPTIVE_HYBRID_SKIP_LLM,
        ADAPTIVE_HYBRID_WEIGHTS,
        SNIPER_ASSETS,
    )
except ImportError:
    PAPER_TRADING = True
    PAPER_TRADING_BALANCE = 500
    ADAPTIVE_HYBRID_BASE_THRESHOLD = 45
    ADAPTIVE_HYBRID_URGENCY_START_HOURS = 4
    ADAPTIVE_HYBRID_URGENCY_FLOOR = 30
    ADAPTIVE_HYBRID_MAX_DAILY_TRADES = 5
    ADAPTIVE_HYBRID_MAX_DAILY_LOSS_USD = 30
    ADAPTIVE_HYBRID_LEVERAGE = 3
    ADAPTIVE_HYBRID_ATR_SL_MULT = 1.5
    ADAPTIVE_HYBRID_ATR_TP_MULT = 2.5
    ADAPTIVE_HYBRID_SKIP_LLM = True
    ADAPTIVE_HYBRID_WEIGHTS = {
        'mean_reversion': 0.15, 'momentum_breakout': 0.12,
        'ema_crossover': 0.10, 'funding_contrarian': 0.10,
        'rsi_divergence': 0.10, 'sniper_lite': 0.18,
        'trend_rider_lite': 0.15, 'ramf_lite': 0.10,
    }
    SNIPER_ASSETS = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'LINK',
                     'DOT', 'MATIC', 'ARB', 'OP', 'RENDER', 'AAVE', 'CRV', 'FIL']


class AdaptiveHybridStrategy(BaseStrategy):
    """
    Adaptive Hybrid Strategy - Multi-module convergence scoring.

    Runs 8 independent signal modules, aggregates scores by weighted average,
    and triggers trades when enough modules converge on a direction.
    """

    _instance = None

    def __init__(self):
        AdaptiveHybridStrategy._instance = self
        super().__init__("Adaptive Hybrid")

        self.assets = SNIPER_ASSETS
        self.weights = ADAPTIVE_HYBRID_WEIGHTS

        # Daily tracking
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_date = None
        self.last_trade_time = None

        # Paper trading state
        self.paper_balance = PAPER_TRADING_BALANCE
        self.paper_positions = {}
        self.closed_positions = []
        self._position_counter = 0

        # Market data provider (for funding rates)
        self._market_data = None
        try:
            from src.data_providers.market_data import MarketDataProvider
            self._market_data = MarketDataProvider(start_liquidation_stream=False)
        except Exception as e:
            cprint(f"[AdaptiveHybrid] Warning: Market data provider unavailable: {e}", "yellow")

        # Data directory
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data', 'adaptive_hybrid'
        )
        os.makedirs(self.data_dir, exist_ok=True)

        # Load existing state
        self._load_state_from_csv()

        cprint(f"[AdaptiveHybrid] Strategy initialized", "cyan")
        cprint(f"  - Assets: {len(self.assets)} symbols", "white")
        cprint(f"  - Base threshold: {ADAPTIVE_HYBRID_BASE_THRESHOLD}/100", "white")
        cprint(f"  - Paper Trading: {PAPER_TRADING}", "white")
        cprint(f"  - Balance: ${self.paper_balance:,.2f}", "white")

    # =========================================================================
    # DATA FETCHING
    # =========================================================================

    def _fetch_candles(self, symbol: str, interval: str = '1h', candles: int = 200) -> pd.DataFrame:
        """Fetch candle data from HyperLiquid."""
        try:
            from hyperliquid.info import Info
            import time

            info = Info(skip_ws=True)
            end_time = int(time.time() * 1000)

            interval_map = {
                '1m': 60_000, '5m': 300_000, '15m': 900_000,
                '30m': 1_800_000, '1h': 3_600_000, '4h': 14_400_000
            }
            interval_ms = interval_map.get(interval, 3_600_000)
            start_time = end_time - (candles * interval_ms)

            data = info.candles_snapshot(symbol, interval, start_time, end_time)
            if not data:
                return None

            df = pd.DataFrame(data)
            df = df.rename(columns={
                't': 'timestamp', 'o': 'open', 'h': 'high',
                'l': 'low', 'c': 'close', 'v': 'volume'
            })
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            return df

        except Exception as e:
            cprint(f"[AdaptiveHybrid] Error fetching candles for {symbol}: {e}", "yellow")
            return None

    def _compute_indicators(self, df: pd.DataFrame) -> dict:
        """Compute all technical indicators needed by modules."""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # RSI
        rsi_ind = RSIIndicator(close=close, window=14)
        df['rsi'] = rsi_ind.rsi()

        # ADX
        adx_ind = ADXIndicator(high=high, low=low, close=close, window=14)
        df['adx'] = adx_ind.adx()

        # EMAs
        df['ema_9'] = EMAIndicator(close=close, window=9).ema_indicator()
        df['ema_21'] = EMAIndicator(close=close, window=21).ema_indicator()
        df['ema_50'] = EMAIndicator(close=close, window=50).ema_indicator()
        df['ema_200'] = EMAIndicator(close=close, window=200).ema_indicator() if len(df) >= 200 else close

        # Bollinger Bands
        bb = BollingerBands(close=close, window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()

        # ATR
        atr_ind = AverageTrueRange(high=high, low=low, close=close, window=14)
        df['atr'] = atr_ind.average_true_range()

        # MACD
        macd_ind = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd_ind.macd()
        df['macd_signal'] = macd_ind.macd_signal()
        df['macd_diff'] = macd_ind.macd_diff()

        # Volume metrics
        vol_avg = volume.rolling(20).mean()
        df['volume_ratio'] = volume / vol_avg.replace(0, 1)

        # VWAP approximation (intraday)
        typical_price = (high + low + close) / 3
        df['vwap'] = (typical_price * volume).cumsum() / volume.cumsum().replace(0, 1)

        # Return last row values as dict for easy access
        last = df.iloc[-1]
        return {
            'close': float(last['close']),
            'high': float(last['high']),
            'low': float(last['low']),
            'volume': float(last['volume']),
            'rsi': float(last['rsi']) if pd.notna(last['rsi']) else 50,
            'adx': float(last['adx']) if pd.notna(last['adx']) else 20,
            'ema_9': float(last['ema_9']) if pd.notna(last['ema_9']) else float(last['close']),
            'ema_21': float(last['ema_21']) if pd.notna(last['ema_21']) else float(last['close']),
            'ema_50': float(last['ema_50']) if pd.notna(last['ema_50']) else float(last['close']),
            'ema_200': float(last['ema_200']) if pd.notna(last['ema_200']) else float(last['close']),
            'bb_upper': float(last['bb_upper']) if pd.notna(last['bb_upper']) else float(last['close']),
            'bb_lower': float(last['bb_lower']) if pd.notna(last['bb_lower']) else float(last['close']),
            'bb_mid': float(last['bb_mid']) if pd.notna(last['bb_mid']) else float(last['close']),
            'atr': float(last['atr']) if pd.notna(last['atr']) else 0,
            'macd': float(last['macd']) if pd.notna(last['macd']) else 0,
            'macd_signal': float(last['macd_signal']) if pd.notna(last['macd_signal']) else 0,
            'macd_diff': float(last['macd_diff']) if pd.notna(last['macd_diff']) else 0,
            'volume_ratio': float(last['volume_ratio']) if pd.notna(last['volume_ratio']) else 1.0,
            'vwap': float(last['vwap']) if pd.notna(last['vwap']) else float(last['close']),
        }

    # =========================================================================
    # 8 SIGNAL MODULES
    # =========================================================================

    def _score_mean_reversion(self, df: pd.DataFrame, ind: dict) -> dict:
        """Module 1: Mean reversion using Bollinger Bands + RSI in ranging markets."""
        long_score = 0
        short_score = 0

        bb_range = ind['bb_upper'] - ind['bb_lower']
        if bb_range <= 0:
            return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'No BB range'}

        bb_pct = (ind['close'] - ind['bb_lower']) / bb_range

        # Long: Price near lower band
        if bb_pct < 0.10:
            long_score += 45
        elif bb_pct < 0.25:
            long_score += 25

        # RSI confirmation
        if ind['rsi'] < 30:
            long_score += 35
        elif ind['rsi'] < 40:
            long_score += 20

        # Ranging market bonus (ADX < 25)
        if ind['adx'] < 20:
            long_score += 20
        elif ind['adx'] < 30:
            long_score += 10

        # Short: Price near upper band
        if bb_pct > 0.90:
            short_score += 45
        elif bb_pct > 0.75:
            short_score += 25

        if ind['rsi'] > 70:
            short_score += 35
        elif ind['rsi'] > 60:
            short_score += 20

        if ind['adx'] < 20:
            short_score += 20
        elif ind['adx'] < 30:
            short_score += 10

        best = max(long_score, short_score)
        direction = 'BUY' if long_score > short_score else 'SELL' if short_score > long_score else 'NEUTRAL'
        return {'score': min(100, best), 'direction': direction,
                'reason': f'BB%={bb_pct:.2f} RSI={ind["rsi"]:.0f} ADX={ind["adx"]:.0f}'}

    def _score_momentum_breakout(self, df: pd.DataFrame, ind: dict) -> dict:
        """Module 2: Breakout above/below recent range with volume confirmation."""
        long_score = 0
        short_score = 0

        high_20 = df['high'].tail(20).max()
        low_20 = df['low'].tail(20).min()
        close = ind['close']
        vol_ratio = ind['volume_ratio']

        # Upside breakout
        if close >= high_20:
            long_score += 40
            if vol_ratio > 1.5:
                long_score += 30
            elif vol_ratio > 1.2:
                long_score += 15
            if ind['adx'] > 25:
                long_score += 20
        elif close >= high_20 * 0.997:
            long_score += 20
            if vol_ratio > 1.3:
                long_score += 15

        # Downside breakout
        if close <= low_20:
            short_score += 40
            if vol_ratio > 1.5:
                short_score += 30
            elif vol_ratio > 1.2:
                short_score += 15
            if ind['adx'] > 25:
                short_score += 20
        elif close <= low_20 * 1.003:
            short_score += 20
            if vol_ratio > 1.3:
                short_score += 15

        best = max(long_score, short_score)
        direction = 'BUY' if long_score > short_score else 'SELL' if short_score > long_score else 'NEUTRAL'
        return {'score': min(100, best), 'direction': direction,
                'reason': f'H20={high_20:.2f} L20={low_20:.2f} Vol={vol_ratio:.1f}x'}

    def _score_ema_crossover(self, df: pd.DataFrame, ind: dict) -> dict:
        """Module 3: EMA 9/21 crossover signals."""
        long_score = 0
        short_score = 0

        if len(df) < 3 or 'ema_9' not in df.columns:
            return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'Insufficient data'}

        ema_9_now = ind['ema_9']
        ema_21_now = ind['ema_21']
        ema_9_prev = float(df['ema_9'].iloc[-2]) if pd.notna(df['ema_9'].iloc[-2]) else ema_9_now
        ema_21_prev = float(df['ema_21'].iloc[-2]) if pd.notna(df['ema_21'].iloc[-2]) else ema_21_now

        # Bullish crossover
        if ema_9_now > ema_21_now and ema_9_prev <= ema_21_prev:
            long_score += 60
        elif ema_9_now > ema_21_now:
            gap_pct = (ema_9_now - ema_21_now) / ema_21_now
            if gap_pct < 0.002:
                long_score += 30  # Recently crossed

        if ind['close'] > ema_9_now and long_score > 0:
            long_score += 25

        # MACD confirmation
        if ind['macd_diff'] > 0 and long_score > 0:
            long_score += 15

        # Bearish crossover
        if ema_9_now < ema_21_now and ema_9_prev >= ema_21_prev:
            short_score += 60
        elif ema_9_now < ema_21_now:
            gap_pct = (ema_21_now - ema_9_now) / ema_21_now
            if gap_pct < 0.002:
                short_score += 30

        if ind['close'] < ema_9_now and short_score > 0:
            short_score += 25

        if ind['macd_diff'] < 0 and short_score > 0:
            short_score += 15

        best = max(long_score, short_score)
        direction = 'BUY' if long_score > short_score else 'SELL' if short_score > long_score else 'NEUTRAL'
        return {'score': min(100, best), 'direction': direction,
                'reason': f'EMA9={ema_9_now:.2f} EMA21={ema_21_now:.2f} MACD={ind["macd_diff"]:.4f}'}

    def _score_funding_contrarian(self, symbol: str, ind: dict) -> dict:
        """Module 4: Contrarian signal based on extreme funding rates."""
        if not self._market_data:
            return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'No market data provider'}

        try:
            zscore = self._market_data.get_funding_zscore(symbol)
        except Exception:
            return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'Funding data unavailable'}

        long_score = 0
        short_score = 0

        # Extreme negative funding = shorts paying longs = go long (contrarian)
        if zscore <= -2.0:
            long_score += 80
        elif zscore <= -1.5:
            long_score += 55
        elif zscore <= -1.0:
            long_score += 30

        # Extreme positive funding = longs paying shorts = go short (contrarian)
        if zscore >= 2.0:
            short_score += 80
        elif zscore >= 1.5:
            short_score += 55
        elif zscore >= 1.0:
            short_score += 30

        best = max(long_score, short_score)
        direction = 'BUY' if long_score > short_score else 'SELL' if short_score > long_score else 'NEUTRAL'
        return {'score': min(100, best), 'direction': direction,
                'reason': f'Funding Z={zscore:.2f}'}

    def _score_rsi_divergence(self, df: pd.DataFrame, ind: dict) -> dict:
        """Module 5: RSI divergence detection (price vs RSI disagreement)."""
        long_score = 0
        short_score = 0

        if len(df) < 20 or 'rsi' not in df.columns:
            return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'Insufficient data'}

        # Look at last 10 bars for divergence
        lookback = min(10, len(df) - 1)
        prices = df['close'].tail(lookback + 1).values
        rsis = df['rsi'].tail(lookback + 1).values

        # Filter out NaN
        valid = ~np.isnan(rsis)
        if valid.sum() < 5:
            return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'Not enough RSI data'}

        prices = prices[valid]
        rsis = rsis[valid]

        # Bullish divergence: price making lower lows but RSI making higher lows
        price_min_idx = np.argmin(prices[-5:])
        price_prev_min_idx = np.argmin(prices[:5])

        if prices[-5:][price_min_idx] < prices[:5][price_prev_min_idx]:
            # Price is lower
            if rsis[-5:].min() > rsis[:5].min():
                # RSI is higher = bullish divergence
                long_score += 60
                if ind['rsi'] < 40:
                    long_score += 25

        # Bearish divergence: price making higher highs but RSI making lower highs
        price_max_idx = np.argmax(prices[-5:])
        price_prev_max_idx = np.argmax(prices[:5])

        if prices[-5:][price_max_idx] > prices[:5][price_prev_max_idx]:
            if rsis[-5:].max() < rsis[:5].max():
                short_score += 60
                if ind['rsi'] > 60:
                    short_score += 25

        best = max(long_score, short_score)
        direction = 'BUY' if long_score > short_score else 'SELL' if short_score > long_score else 'NEUTRAL'
        return {'score': min(100, best), 'direction': direction,
                'reason': f'RSI={ind["rsi"]:.0f} div_long={long_score} div_short={short_score}'}

    def _score_sniper_lite(self, df: pd.DataFrame, ind: dict) -> dict:
        """Module 6: Relaxed version of Sniper AI (extreme move + funding check)."""
        long_score = 0
        short_score = 0

        # Check Z-score (relaxed: 1.5 sigma instead of 1.8)
        window = 20
        close_series = df['close'].tail(window + 5)
        rolling_mean = close_series.rolling(window=window).mean()
        rolling_std = close_series.rolling(window=window).std()

        current_price = ind['close']
        mean = float(rolling_mean.iloc[-1]) if pd.notna(rolling_mean.iloc[-1]) else current_price
        std = float(rolling_std.iloc[-1]) if pd.notna(rolling_std.iloc[-1]) else 1

        z_score = (current_price - mean) / std if std > 0 else 0

        # Extreme move down = potential long (fade the move)
        if z_score <= -1.5:
            long_score += 45
            if z_score <= -2.0:
                long_score += 15
        # Extreme move up = potential short
        if z_score >= 1.5:
            short_score += 45
            if z_score >= 2.0:
                short_score += 15

        # Volume confirmation (relaxed: 2x instead of 3x)
        if ind['volume_ratio'] > 2.0:
            if long_score > 0:
                long_score += 20
            if short_score > 0:
                short_score += 20
        elif ind['volume_ratio'] > 1.5:
            if long_score > 0:
                long_score += 10
            if short_score > 0:
                short_score += 10

        # RSI confirmation
        if ind['rsi'] < 35 and long_score > 0:
            long_score += 20
        if ind['rsi'] > 65 and short_score > 0:
            short_score += 20

        best = max(long_score, short_score)
        direction = 'BUY' if long_score > short_score else 'SELL' if short_score > long_score else 'NEUTRAL'
        return {'score': min(100, best), 'direction': direction,
                'reason': f'Z={z_score:.2f} Vol={ind["volume_ratio"]:.1f}x RSI={ind["rsi"]:.0f}'}

    def _score_trend_rider_lite(self, df: pd.DataFrame, ind: dict) -> dict:
        """Module 7: Relaxed trend following (ADX>25, EMA20>EMA50, pullback)."""
        long_score = 0
        short_score = 0

        # Trend alignment (relaxed: ADX>25 instead of 35, only 2 EMAs needed)
        bullish_trend = ind['ema_9'] > ind['ema_50'] and ind['adx'] > 25
        bearish_trend = ind['ema_9'] < ind['ema_50'] and ind['adx'] > 25

        if bullish_trend:
            long_score += 35
            # Pullback to EMA (relaxed RSI zone: 35-50 instead of 30-45)
            if 35 <= ind['rsi'] <= 50:
                long_score += 30
            elif 30 <= ind['rsi'] <= 55:
                long_score += 15
            # Price near EMA 21 (within 1%)
            ema_dist = abs(ind['close'] - ind['ema_21']) / ind['ema_21']
            if ema_dist < 0.01:
                long_score += 20
            elif ema_dist < 0.02:
                long_score += 10
            # MACD positive
            if ind['macd_diff'] > 0:
                long_score += 15

        if bearish_trend:
            short_score += 35
            if 50 <= ind['rsi'] <= 65:
                short_score += 30
            elif 45 <= ind['rsi'] <= 70:
                short_score += 15
            ema_dist = abs(ind['close'] - ind['ema_21']) / ind['ema_21']
            if ema_dist < 0.01:
                short_score += 20
            elif ema_dist < 0.02:
                short_score += 10
            if ind['macd_diff'] < 0:
                short_score += 15

        best = max(long_score, short_score)
        direction = 'BUY' if long_score > short_score else 'SELL' if short_score > long_score else 'NEUTRAL'
        return {'score': min(100, best), 'direction': direction,
                'reason': f'ADX={ind["adx"]:.0f} EMA9{">" if ind["ema_9"] > ind["ema_50"] else "<"}EMA50 RSI={ind["rsi"]:.0f}'}

    def _score_ramf_lite(self, df: pd.DataFrame, ind: dict) -> dict:
        """Module 8: Volatility regime + momentum exhaustion (no dead zone)."""
        long_score = 0
        short_score = 0

        if 'atr' not in df.columns or len(df) < 50:
            return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'Insufficient data'}

        # Volatility regime (NO dead zone - always classify)
        atr_values = df['atr'].dropna().tail(50)
        if len(atr_values) < 20:
            return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'Not enough ATR data'}

        current_atr = ind['atr']
        atr_percentile = (atr_values < current_atr).sum() / len(atr_values) * 100

        is_high_vol = atr_percentile >= 50
        regime = 'HIGH' if is_high_vol else 'LOW'

        if is_high_vol:
            # High vol: mean reversion / exhaustion
            # Check VWAP distance (relaxed: 1.0 ATR instead of 1.5)
            vwap_dist = abs(ind['close'] - ind['vwap']) / current_atr if current_atr > 0 else 0

            if vwap_dist >= 1.0:
                if ind['close'] < ind['vwap']:
                    long_score += 40  # Extended below VWAP
                else:
                    short_score += 40  # Extended above VWAP

            # Consecutive bars check (relaxed: 2 instead of 3)
            recent = df.tail(3)
            up_bars = (recent['close'] > recent['open']).sum()
            down_bars = (recent['close'] < recent['open']).sum()

            if down_bars >= 2 and long_score > 0:
                long_score += 30
            if up_bars >= 2 and short_score > 0:
                short_score += 30

            # RSI confirmation
            if ind['rsi'] < 35:
                long_score += 30
            elif ind['rsi'] < 45:
                long_score += 15
            if ind['rsi'] > 65:
                short_score += 30
            elif ind['rsi'] > 55:
                short_score += 15
        else:
            # Low vol: trend following
            if ind['ema_9'] > ind['ema_21']:
                long_score += 35
                if ind['macd_diff'] > 0:
                    long_score += 25
                if ind['rsi'] > 50:
                    long_score += 20
            elif ind['ema_9'] < ind['ema_21']:
                short_score += 35
                if ind['macd_diff'] < 0:
                    short_score += 25
                if ind['rsi'] < 50:
                    short_score += 20

        best = max(long_score, short_score)
        direction = 'BUY' if long_score > short_score else 'SELL' if short_score > long_score else 'NEUTRAL'
        return {'score': min(100, best), 'direction': direction,
                'reason': f'Regime={regime} ATR%={atr_percentile:.0f} VWAP_dist={abs(ind["close"] - ind["vwap"]):.2f}'}

    # =========================================================================
    # AGGREGATION
    # =========================================================================

    def _aggregate_scores(self, module_results: dict) -> dict:
        """Aggregate all module scores into a final trading decision."""
        weights = self.weights

        # Separate by direction
        long_modules = {}
        short_modules = {}

        for name, result in module_results.items():
            if result['score'] <= 0 or result['direction'] == 'NEUTRAL':
                continue
            if result['direction'] == 'BUY':
                long_modules[name] = result
            elif result['direction'] == 'SELL':
                short_modules[name] = result

        # Calculate weighted sums for each direction
        long_weighted = sum(r['score'] * weights.get(n, 0) for n, r in long_modules.items())
        short_weighted = sum(r['score'] * weights.get(n, 0) for n, r in short_modules.items())
        long_weight_total = sum(weights.get(n, 0) for n in long_modules)
        short_weight_total = sum(weights.get(n, 0) for n in short_modules)

        if long_weighted == 0 and short_weighted == 0:
            return {'direction': 'NEUTRAL', 'score': 0, 'agreement': 0,
                    'details': 'No modules produced directional signals'}

        # Determine winning direction
        if long_weighted >= short_weighted:
            direction = 'BUY'
            winning_modules = long_modules
            losing_weight = short_weight_total
            winning_weight = long_weight_total
        else:
            direction = 'SELL'
            winning_modules = short_modules
            losing_weight = long_weight_total
            winning_weight = short_weight_total

        # Weighted average of ACTIVE modules in winning direction
        active_weighted_sum = sum(r['score'] * weights.get(n, 0) for n, r in winning_modules.items())
        active_weight_total = sum(weights.get(n, 0) for n in winning_modules)

        if active_weight_total <= 0:
            return {'direction': 'NEUTRAL', 'score': 0, 'agreement': 0,
                    'details': 'No weight in winning direction'}

        raw_score = active_weighted_sum / active_weight_total
        coverage = active_weight_total / sum(weights.values())
        final_score = raw_score * (0.5 + 0.5 * coverage)

        # Conflict penalty: if opposing direction has significant weight
        if losing_weight > 0.25 * winning_weight:
            conflict_factor = 0.70
            final_score *= conflict_factor
        else:
            conflict_factor = 1.0

        # Build details string
        module_details = []
        for name, result in winning_modules.items():
            module_details.append(f"{name}={result['score']}")

        return {
            'direction': direction,
            'score': round(final_score, 1),
            'raw_score': round(raw_score, 1),
            'coverage': round(coverage * 100, 1),
            'conflict_factor': conflict_factor,
            'active_modules': len(winning_modules),
            'total_modules_fired': len(long_modules) + len(short_modules),
            'agreement': round(coverage, 2),
            'details': f"{direction} score={final_score:.1f} ({len(winning_modules)} modules: {', '.join(module_details)})",
            'module_scores': {n: r['score'] for n, r in module_results.items() if r['score'] > 0},
        }

    # =========================================================================
    # URGENCY SYSTEM
    # =========================================================================

    def _get_urgency_multiplier(self) -> float:
        """Progressive threshold relaxation if no trades recently."""
        if self.last_trade_time is None:
            # Never traded = maximum urgency
            hours_since = 24
        else:
            hours_since = (datetime.now() - self.last_trade_time).total_seconds() / 3600

        start = ADAPTIVE_HYBRID_URGENCY_START_HOURS
        if hours_since < start:
            return 1.0

        # Reduce by 5 points per hour after start, down to floor
        steps = hours_since - start
        reduction = min(steps * 0.05, 0.30)  # Max 30% reduction
        return max(0.70, 1.0 - reduction)

    def _get_effective_threshold(self) -> float:
        """Get current threshold adjusted by urgency."""
        base = ADAPTIVE_HYBRID_BASE_THRESHOLD
        urgency = self._get_urgency_multiplier()
        effective = base * urgency
        return max(effective, ADAPTIVE_HYBRID_URGENCY_FLOOR)

    # =========================================================================
    # MAIN SIGNAL GENERATION
    # =========================================================================

    def generate_signals(self, symbol: str = None, df: pd.DataFrame = None) -> dict:
        """Generate trading signal for the given symbol."""
        self._reset_daily_counters()

        # Daily limits
        if self.daily_trades >= ADAPTIVE_HYBRID_MAX_DAILY_TRADES:
            return None
        if self.daily_pnl <= -ADAPTIVE_HYBRID_MAX_DAILY_LOSS_USD:
            return None

        # If no symbol, return None (main.py iterates symbols)
        if symbol is None:
            return None

        if symbol not in self.assets:
            return None

        # Fetch data
        if df is None:
            df = self._fetch_candles(symbol, interval='1h', candles=200)

        if df is None or len(df) < 50:
            return None

        # Compute all indicators
        ind = self._compute_indicators(df)

        # Run all 8 modules
        module_results = {
            'mean_reversion': self._score_mean_reversion(df, ind),
            'momentum_breakout': self._score_momentum_breakout(df, ind),
            'ema_crossover': self._score_ema_crossover(df, ind),
            'funding_contrarian': self._score_funding_contrarian(symbol, ind),
            'rsi_divergence': self._score_rsi_divergence(df, ind),
            'sniper_lite': self._score_sniper_lite(df, ind),
            'trend_rider_lite': self._score_trend_rider_lite(df, ind),
            'ramf_lite': self._score_ramf_lite(df, ind),
        }

        # Aggregate scores
        aggregated = self._aggregate_scores(module_results)

        # Get threshold
        threshold = self._get_effective_threshold()
        urgency = self._get_urgency_multiplier()

        # Log analysis
        fired_modules = {n: r for n, r in module_results.items() if r['score'] > 0 and r['direction'] != 'NEUTRAL'}
        cprint(f"  [{symbol}] Score: {aggregated['score']:.1f}/{threshold:.0f} | "
               f"Modules: {len(fired_modules)}/8 | Direction: {aggregated['direction']} | "
               f"Urgency: {urgency:.0%}", "white")

        for name, result in fired_modules.items():
            cprint(f"    {name}: {result['direction']} {result['score']} - {result['reason']}", "white")

        # Decision
        if aggregated['direction'] != 'NEUTRAL' and aggregated['score'] >= threshold:
            # Map score to signal strength (0-1)
            score = aggregated['score']
            if score >= 70:
                strength = 0.85
            elif score >= 55:
                strength = 0.70
            else:
                strength = 0.55

            # Calculate ATR-based SL/TP
            atr = ind['atr']
            if atr > 0:
                sl_pct = (atr * ADAPTIVE_HYBRID_ATR_SL_MULT / ind['close']) * 100
                tp_pct = (atr * ADAPTIVE_HYBRID_ATR_TP_MULT / ind['close']) * 100
            else:
                sl_pct = 1.5
                tp_pct = 2.5

            cprint(f"  [{symbol}] SIGNAL: {aggregated['direction']} (score={score:.1f}, "
                   f"threshold={threshold:.0f}, strength={strength:.0%})", "green", attrs=['bold'])

            return {
                'token': symbol,
                'signal': strength,
                'direction': aggregated['direction'],
                'metadata': {
                    'strategy': 'Adaptive Hybrid',
                    'score': aggregated['score'],
                    'threshold': threshold,
                    'urgency_multiplier': urgency,
                    'active_modules': aggregated['active_modules'],
                    'total_fired': aggregated['total_modules_fired'],
                    'coverage': aggregated.get('coverage', 0),
                    'module_scores': aggregated.get('module_scores', {}),
                    'reason': aggregated['details'],
                    'current_price': ind['close'],
                    'stop_loss_pct': round(sl_pct, 2),
                    'take_profit_pct': round(tp_pct, 2),
                    'atr': atr,
                    'rsi': ind['rsi'],
                    'adx': ind['adx'],
                }
            }

        # NEUTRAL signal with diagnostic info
        return {
            'token': symbol,
            'signal': 0,
            'direction': 'NEUTRAL',
            'metadata': {
                'strategy': 'Adaptive Hybrid',
                'score': aggregated['score'],
                'threshold': threshold,
                'urgency_multiplier': urgency,
                'reason': f"Score {aggregated['score']:.1f} < threshold {threshold:.0f}",
                'diagnostic': aggregated.get('details', ''),
                'module_scores': aggregated.get('module_scores', {}),
                'current_price': ind['close'],
            }
        }

    # =========================================================================
    # PAPER TRADING
    # =========================================================================

    def _reset_daily_counters(self):
        """Reset daily counters if new day."""
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_trade_date = today

    def execute_paper_trade(self, signal: dict) -> dict:
        """Execute a paper trade (simulation)."""
        if not PAPER_TRADING:
            return None

        symbol = signal.get('token', '')
        direction = signal.get('direction', 'NEUTRAL')

        if not symbol or direction == 'NEUTRAL':
            return None

        metadata = signal.get('metadata', {})
        price = metadata.get('current_price', 0)

        if price == 0:
            df = self._fetch_candles(symbol, interval='1h', candles=5)
            if df is not None and len(df) > 0:
                price = float(df['close'].iloc[-1])

        if price <= 0:
            cprint(f"[AdaptiveHybrid] Cannot execute trade with price={price}", "red")
            return None

        sl_pct = metadata.get('stop_loss_pct', 1.5)
        tp_pct = metadata.get('take_profit_pct', 2.5)

        # Calculate margin
        used_margin = sum(
            pos.get('position_size', 0) / pos.get('leverage', ADAPTIVE_HYBRID_LEVERAGE)
            for pos in self.paper_positions.values()
        )
        available_margin = max(0, self.paper_balance - used_margin)

        # Position sizing: 2% risk per trade
        risk_amount = self.paper_balance * 0.02
        position_size = risk_amount / (sl_pct / 100) * ADAPTIVE_HYBRID_LEVERAGE
        max_position = available_margin * 0.9 * ADAPTIVE_HYBRID_LEVERAGE
        position_size = min(position_size, max_position)

        if position_size < 10:
            cprint(f"[AdaptiveHybrid] Insufficient margin", "red")
            return None

        # Generate position ID
        self._position_counter += 1
        position_id = f"AH_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._position_counter}"

        # Calculate SL/TP prices
        if direction == 'BUY':
            stop_loss_price = price * (1 - sl_pct / 100)
            take_profit_price = price * (1 + tp_pct / 100)
        else:
            stop_loss_price = price * (1 + sl_pct / 100)
            take_profit_price = price * (1 - tp_pct / 100)

        trade = {
            'position_id': position_id,
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'direction': direction,
            'entry_price': price,
            'position_size': round(position_size, 2),
            'leverage': ADAPTIVE_HYBRID_LEVERAGE,
            'stop_loss': round(stop_loss_price, 2),
            'take_profit': round(take_profit_price, 2),
            'sl_pct': sl_pct,
            'tp_pct': tp_pct,
            'confidence': round(float(signal.get('signal', 0)) * 100, 1),
            'status': 'OPEN',
            'score': metadata.get('score', 0),
            'modules': str(metadata.get('module_scores', {})),
        }

        self.paper_positions[position_id] = trade
        self.daily_trades += 1
        self.last_trade_time = datetime.now()

        # Log to file
        log_file = os.path.join(self.data_dir, 'paper_trades.csv')
        df = pd.DataFrame([trade])
        if os.path.exists(log_file):
            df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            df.to_csv(log_file, index=False)

        cprint(f"\n[ADAPTIVE HYBRID] Opened {direction} {symbol} (ID: {position_id})", "magenta", attrs=['bold'])
        cprint(f"  Entry: ${price:,.2f} | Size: ${position_size:,.2f} | Score: {metadata.get('score', 0):.1f}", "white")
        cprint(f"  SL: ${trade['stop_loss']:,.2f} ({sl_pct:.2f}%)", "white")
        cprint(f"  TP: ${trade['take_profit']:,.2f} ({tp_pct:.2f}%)", "white")

        return trade

    def monitor_paper_positions(self) -> list:
        """Monitor all open paper positions and close those that hit SL/TP."""
        if not PAPER_TRADING or not self.paper_positions:
            return []

        closed = []
        symbols_to_check = set(pos['symbol'] for pos in self.paper_positions.values())
        current_prices = {}

        for symbol in symbols_to_check:
            try:
                df = self._fetch_candles(symbol, interval='15m', candles=5)
                if df is not None and len(df) > 0:
                    current_prices[symbol] = float(df['close'].iloc[-1])
            except Exception:
                pass

        positions_to_close = []

        for position_id, trade in self.paper_positions.items():
            symbol = trade['symbol']
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            direction = trade['direction']
            stop_loss = trade['stop_loss']
            take_profit = trade['take_profit']

            close_reason = None
            if direction == 'BUY':
                if current_price <= stop_loss:
                    close_reason = 'STOP_LOSS'
                elif current_price >= take_profit:
                    close_reason = 'TAKE_PROFIT'
            else:
                if current_price >= stop_loss:
                    close_reason = 'STOP_LOSS'
                elif current_price <= take_profit:
                    close_reason = 'TAKE_PROFIT'

            if close_reason:
                positions_to_close.append((position_id, current_price, close_reason))

        for position_id, close_price, reason in positions_to_close:
            closed_trade = self._close_paper_position(position_id, close_price, reason)
            if closed_trade:
                closed.append(closed_trade)

        return closed

    def _close_paper_position(self, position_id: str, close_price: float, reason: str) -> dict:
        """Close a paper position and update PnL."""
        if position_id not in self.paper_positions:
            return None

        trade = self.paper_positions[position_id].copy()
        entry_price = trade['entry_price']
        direction = trade['direction']
        position_size = trade['position_size']

        if direction == 'BUY':
            price_change_pct = (close_price - entry_price) / entry_price
        else:
            price_change_pct = (entry_price - close_price) / entry_price

        pnl = position_size * price_change_pct

        trade['close_price'] = close_price
        trade['close_timestamp'] = datetime.now().isoformat()
        trade['close_reason'] = reason
        trade['pnl'] = round(pnl, 2)
        trade['pnl_pct'] = round(price_change_pct * 100, 2)
        trade['status'] = 'CLOSED'

        self.daily_pnl += pnl
        self.paper_balance += pnl

        del self.paper_positions[position_id]
        self.closed_positions.append(trade)

        color = 'green' if pnl > 0 else 'red'
        cprint(f"\n[ADAPTIVE HYBRID] Closed {trade['symbol']} ({reason})", color, attrs=['bold'])
        cprint(f"  Entry: ${entry_price:,.2f} -> Exit: ${close_price:,.2f}", "white")
        cprint(f"  PnL: ${pnl:+,.2f} ({price_change_pct*100:+.2f}%)", color)
        cprint(f"  Balance: ${self.paper_balance:,.2f}", "white")

        # Log closed trade
        self._log_closed_trade(trade)
        self._update_position_status_in_csv(position_id, trade)

        return trade

    def _update_position_status_in_csv(self, position_id: str, trade: dict):
        """Update a position's status in paper_trades.csv when closed."""
        try:
            paper_trades_file = os.path.join(self.data_dir, 'paper_trades.csv')
            if not os.path.exists(paper_trades_file):
                return
            df = pd.read_csv(paper_trades_file)
            if df.empty:
                return
            mask = df['position_id'] == position_id
            if mask.any():
                df.loc[mask, 'status'] = trade.get('status', 'CLOSED')
                df.to_csv(paper_trades_file, index=False)
        except Exception as e:
            cprint(f"[AdaptiveHybrid] Warning: Could not update CSV: {e}", "yellow")

    def _log_closed_trade(self, trade: dict):
        """Log closed trade to separate CSV file."""
        try:
            log_file = os.path.join(self.data_dir, 'closed_trades.csv')
            df = pd.DataFrame([trade])
            if os.path.exists(log_file):
                df.to_csv(log_file, mode='a', header=False, index=False)
            else:
                df.to_csv(log_file, index=False)
        except Exception as e:
            cprint(f"[AdaptiveHybrid] Error logging closed trade: {e}", "yellow")

    def get_paper_status(self) -> dict:
        """Get current paper trading status (for dashboard)."""
        return {
            'paper_balance': round(self.paper_balance, 2),
            'initial_balance': PAPER_TRADING_BALANCE,
            'total_pnl': round(self.paper_balance - PAPER_TRADING_BALANCE, 2),
            'daily_pnl': round(self.daily_pnl, 2),
            'daily_trades': self.daily_trades,
            'open_positions': len(self.paper_positions),
            'total_closed': len(self.closed_positions),
            'positions': list(self.paper_positions.values()),
        }

    def close_all_paper_positions(self) -> list:
        """Force close all open paper positions at current market price."""
        if not self.paper_positions:
            return []

        closed = []
        for position_id in list(self.paper_positions.keys()):
            trade = self.paper_positions[position_id]
            try:
                df = self._fetch_candles(trade['symbol'], interval='15m', candles=5)
                if df is not None and len(df) > 0:
                    current_price = float(df['close'].iloc[-1])
                    closed_trade = self._close_paper_position(position_id, current_price, 'MANUAL')
                    if closed_trade:
                        closed.append(closed_trade)
            except Exception as e:
                cprint(f"[AdaptiveHybrid] Error closing {position_id}: {e}", "red")

        return closed

    def _load_state_from_csv(self):
        """Load existing positions and balance from CSV files."""
        try:
            paper_trades_file = os.path.join(self.data_dir, 'paper_trades.csv')
            closed_trades_file = os.path.join(self.data_dir, 'closed_trades.csv')

            if os.path.exists(paper_trades_file):
                df = pd.read_csv(paper_trades_file)
                if not df.empty:
                    open_df = df[df['status'] == 'OPEN']
                    for _, row in open_df.iterrows():
                        position_id = row.get('position_id', '')
                        if position_id:
                            self.paper_positions[position_id] = {
                                'position_id': position_id,
                                'timestamp': row.get('timestamp', ''),
                                'symbol': row.get('symbol', ''),
                                'direction': row.get('direction', 'BUY'),
                                'entry_price': float(row.get('entry_price', 0)),
                                'position_size': float(row.get('position_size', 0)),
                                'leverage': float(row.get('leverage', ADAPTIVE_HYBRID_LEVERAGE)),
                                'stop_loss': float(row.get('stop_loss', 0)),
                                'take_profit': float(row.get('take_profit', 0)),
                                'sl_pct': float(row.get('sl_pct', 1.5)),
                                'tp_pct': float(row.get('tp_pct', 2.5)),
                                'confidence': float(row.get('confidence', 0)),
                                'status': 'OPEN',
                            }

                    if self.paper_positions:
                        max_counter = 0
                        for pos_id in self.paper_positions.keys():
                            parts = pos_id.split('_')
                            if len(parts) >= 4:
                                try:
                                    max_counter = max(max_counter, int(parts[-1]))
                                except ValueError:
                                    pass
                        self._position_counter = max_counter

            realized_pnl = 0.0
            if os.path.exists(closed_trades_file):
                closed_df = pd.read_csv(closed_trades_file)
                if not closed_df.empty and 'pnl' in closed_df.columns:
                    realized_pnl = closed_df['pnl'].sum()
                    self.closed_positions = closed_df.to_dict('records')

            self.paper_balance = PAPER_TRADING_BALANCE + realized_pnl

        except Exception as e:
            cprint(f"[AdaptiveHybrid] Warning: Could not load state: {e}", "yellow")


# For standalone testing
if __name__ == "__main__":
    cprint("=" * 60, "cyan")
    cprint("  Testing Adaptive Hybrid Strategy", "cyan", attrs=['bold'])
    cprint("=" * 60, "cyan")

    strategy = AdaptiveHybridStrategy()

    for symbol in ['BTC', 'ETH', 'SOL', 'LINK', 'ARB', 'DOGE']:
        try:
            cprint(f"\nAnalyzing {symbol}...", "white", attrs=['bold'])
            signal = strategy.generate_signals(symbol=symbol)

            if signal:
                if signal['direction'] != 'NEUTRAL':
                    cprint(f"  SIGNAL: {signal['direction']} (strength={signal['signal']:.0%})", "green")
                else:
                    score = signal['metadata'].get('score', 0)
                    threshold = signal['metadata'].get('threshold', 45)
                    cprint(f"  NEUTRAL (score={score:.1f}/{threshold:.0f})", "yellow")

        except Exception as e:
            cprint(f"Error testing {symbol}: {e}", "red")
            import traceback
            traceback.print_exc()

    cprint("\nTest completed!", "cyan")
