#!/usr/bin/env python3
"""
Backtest runner for the Adaptive Hybrid Strategy.

WARNING: This backtester only runs 6 pure-technical modules (mean_reversion,
momentum_breakout, ema_trend, rsi_divergence, sniper_lite, ramf_lite).
Live trading uses 24+ modules including real-time data modules (funding, OI,
sentiment, liquidation, CVD, etc.) that are NOT validated here.
Backtest results represent ~49% of the live signal mix.

Modules requiring live data (funding_contrarian, oi_delta, sentiment,
squeeze_detector, order_imbalance) are excluded since they depend on
real-time API data unavailable in historical replay.

Usage:
    python src/backtesting/backtest_adaptive_hybrid.py --symbol BTC --timeframe 15m --days 180
    python src/backtesting/backtest_adaptive_hybrid.py --symbol ETH --timeframe 1h --days 90 --walk-forward
"""

import argparse
import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import EMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volume import VolumeWeightedAveragePrice

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.backtesting.backtest_engine import BacktestEngine
from src.config import (
    ADAPTIVE_HYBRID_BASE_THRESHOLD,
    ADAPTIVE_HYBRID_ATR_PROFILES,
    PAPER_TAKER_FEE_V2,
    PAPER_SLIPPAGE_V2,
)

# ---------------------------------------------------------------------------
# Config defaults (read from src/config.py for single-source-of-truth)
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    'mean_reversion': 0.12, 'momentum_breakout': 0.10,
    'ema_trend': 0.12, 'rsi_divergence': 0.08,
    'sniper_lite': 0.14, 'ramf_lite': 0.08,
}

WEIGHT_PROFILES = {
    'ranging': {
        'mean_reversion': 0.18, 'momentum_breakout': 0.08, 'ema_trend': 0.12,
        'rsi_divergence': 0.12, 'sniper_lite': 0.16, 'ramf_lite': 0.10,
    },
    'trending': {
        'mean_reversion': 0.08, 'momentum_breakout': 0.16, 'ema_trend': 0.18,
        'rsi_divergence': 0.08, 'sniper_lite': 0.14, 'ramf_lite': 0.10,
    },
}

# Normalize so each profile sums to 1.0
for _profile in [DEFAULT_WEIGHTS] + list(WEIGHT_PROFILES.values()):
    _total = sum(_profile.values())
    if _total > 0:
        for _k in _profile:
            _profile[_k] /= _total

ATR_PROFILES = ADAPTIVE_HYBRID_ATR_PROFILES  # Read from config (single source of truth)

BASE_THRESHOLD = ADAPTIVE_HYBRID_BASE_THRESHOLD  # Read from config (was hardcoded to 55)
MIN_CONVERGENT_MODULES = 2
MIN_RR_RATIO = 1.5
LEVERAGE = 3
MAX_POSITION_PCT = 25
CASH_PCT = 20
# BacktestEngine expects fee/slippage as percentages (it divides by 100 internally)
PAPER_TAKER_FEE = PAPER_TAKER_FEE_V2 * 100       # Convert decimal 0.00045 -> 0.045%
# Slippage is now token-class-specific; default used when symbol not found
PAPER_SLIPPAGE_DEFAULT = PAPER_SLIPPAGE_V2.get('mid', 0.0012) * 100  # Convert decimal -> %


def get_token_slippage_pct(symbol: str) -> float:
    """Get slippage % for a symbol based on its token class in ATR_PROFILES.

    BacktestEngine expects a percentage value (e.g. 0.12 means 0.12%).
    PAPER_SLIPPAGE_V2 stores decimals (e.g. 0.0012 means 0.12%), so we multiply by 100.
    """
    for profile_name, profile in ATR_PROFILES.items():
        if symbol in profile.get('tokens', []):
            return PAPER_SLIPPAGE_V2.get(profile_name, 0.0012) * 100
    return PAPER_SLIPPAGE_DEFAULT

REGIME_ADX_TRENDING = 30
REGIME_ADX_RANGING = 20

OPTIMAL_HOURS = [7, 8, 9, 13, 14, 15, 19, 20, 21]
AVOID_HOURS = [0, 1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def fetch_ohlcv_hyperliquid(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Fetch OHLCV data from HyperLiquid API."""
    from hyperliquid.info import Info

    info = Info(skip_ws=True)
    end_time = int(time.time() * 1000)

    interval_map = {
        '1m': 60_000, '5m': 300_000, '15m': 900_000,
        '30m': 1_800_000, '1h': 3_600_000, '4h': 14_400_000,
        '1d': 86_400_000,
    }
    interval_ms = interval_map.get(interval, 3_600_000)
    start_time = end_time - (days * 86_400_000)

    # HyperLiquid has a max of ~5000 candles per request, chunk if needed
    max_candles_per_request = 5000
    chunk_ms = max_candles_per_request * interval_ms
    all_data = []

    t = start_time
    while t < end_time:
        chunk_end = min(t + chunk_ms, end_time)
        data = info.candles_snapshot(symbol, interval, t, chunk_end)
        if data:
            all_data.extend(data)
        t = chunk_end
        if data and len(data) < max_candles_per_request:
            break  # No more data available

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df = df.rename(columns={
        't': 'timestamp', 'T': 'close_timestamp',
        'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
    })
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    else:
        df['datetime'] = pd.date_range(end=datetime.utcnow(), periods=len(df), freq='h')

    df = df.drop_duplicates(subset='datetime').sort_values('datetime').reset_index(drop=True)
    return df


def load_csv(filepath: str) -> pd.DataFrame:
    """Load OHLCV from a local CSV file."""
    df = pd.read_csv(filepath)
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'datetime'})
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['open', 'high', 'low', 'close']).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Technical indicators (mirrors AdaptiveHybridStrategy._compute_indicators)
# ---------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicator columns to the DataFrame."""
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume'].fillna(0)

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
    if len(df) >= 200:
        df['ema_200'] = EMAIndicator(close=close, window=200).ema_indicator()
    else:
        df['ema_200'] = close

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

    # VWAP rolling (14-period)
    vwap_ind = VolumeWeightedAveragePrice(high=high, low=low, close=close, volume=volume.clip(lower=1e-8), window=14)
    df['vwap'] = vwap_ind.volume_weighted_average_price()
    df['vwap'] = df['vwap'].fillna((high + low + close) / 3)

    # BB %B
    bb_range = df['bb_upper'] - df['bb_lower']
    df['bb_pct'] = (df['close'] - df['bb_lower']) / bb_range.replace(0, 1)

    return df


def get_indicator_dict(df: pd.DataFrame, idx: int) -> dict:
    """Extract indicator values at a given row index."""
    row = df.iloc[idx]

    # Dynamic RSI thresholds from lookback window
    rsi_lookback = df['rsi'].iloc[max(0, idx - 100):idx + 1].dropna()
    if len(rsi_lookback) >= 30:
        rsi_oversold = max(25, min(40, float(np.percentile(rsi_lookback, 15))))
        rsi_overbought = max(60, min(75, float(np.percentile(rsi_lookback, 85))))
    else:
        rsi_oversold = 30
        rsi_overbought = 70

    def safe(val, default):
        return float(val) if pd.notna(val) else default

    return {
        'close': safe(row['close'], 0),
        'open': safe(row['open'], 0),
        'high': safe(row['high'], 0),
        'low': safe(row['low'], 0),
        'volume': safe(row['volume'], 0),
        'rsi': safe(row['rsi'], 50),
        'rsi_oversold': rsi_oversold,
        'rsi_overbought': rsi_overbought,
        'adx': safe(row['adx'], 20),
        'ema_9': safe(row['ema_9'], safe(row['close'], 0)),
        'ema_21': safe(row['ema_21'], safe(row['close'], 0)),
        'ema_50': safe(row['ema_50'], safe(row['close'], 0)),
        'ema_200': safe(row['ema_200'], safe(row['close'], 0)),
        'bb_upper': safe(row['bb_upper'], safe(row['close'], 0)),
        'bb_lower': safe(row['bb_lower'], safe(row['close'], 0)),
        'bb_mid': safe(row['bb_mid'], safe(row['close'], 0)),
        'bb_pct': safe(row['bb_pct'], 0.5),
        'atr': safe(row['atr'], 0),
        'macd': safe(row['macd'], 0),
        'macd_signal': safe(row['macd_signal'], 0),
        'macd_diff': safe(row['macd_diff'], 0),
        'volume_ratio': safe(row['volume_ratio'], 1.0),
        'vwap': safe(row['vwap'], safe(row['close'], 0)),
    }


# ---------------------------------------------------------------------------
# Scoring modules (exact port from adaptive_hybrid_strategy.py)
# ---------------------------------------------------------------------------

def score_mean_reversion(df: pd.DataFrame, ind: dict) -> dict:
    """Module 1: Mean reversion using Bollinger Bands + RSI."""
    long_score = 0
    short_score = 0

    bb_range = ind['bb_upper'] - ind['bb_lower']
    if bb_range <= 0:
        return {'score': 0, 'direction': 'NEUTRAL'}

    bb_pct = (ind['close'] - ind['bb_lower']) / bb_range

    if bb_pct < 0.10:
        long_score += 45
    elif bb_pct < 0.25:
        long_score += 25

    if ind['rsi'] < ind['rsi_oversold']:
        long_score += 35
    elif ind['rsi'] < (ind['rsi_oversold'] + 50) / 2:
        long_score += 20

    if ind['adx'] < 20:
        long_score += 20
    elif ind['adx'] < 30:
        long_score += 10

    if bb_pct > 0.90:
        short_score += 45
    elif bb_pct > 0.75:
        short_score += 25

    if ind['rsi'] > ind['rsi_overbought']:
        short_score += 35
    elif ind['rsi'] > (ind['rsi_overbought'] + 50) / 2:
        short_score += 20

    if ind['adx'] < 20:
        short_score += 20
    elif ind['adx'] < 30:
        short_score += 10

    best = max(long_score, short_score)
    direction = 'BUY' if long_score > short_score else 'SELL' if short_score > long_score else 'NEUTRAL'
    return {'score': min(100, best), 'direction': direction}


def score_momentum_breakout(df: pd.DataFrame, ind: dict, idx: int) -> dict:
    """Module 2: Breakout above/below 20-bar range with volume."""
    long_score = 0
    short_score = 0

    start = max(0, idx - 19)
    high_20 = float(df['high'].iloc[start:idx + 1].max())
    low_20 = float(df['low'].iloc[start:idx + 1].min())
    close = ind['close']
    vol_ratio = ind['volume_ratio']

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
    return {'score': min(100, best), 'direction': direction}


def score_ema_trend(ind: dict) -> dict:
    """Module 3: EMA alignment + ADX + MACD."""
    long_score = 0
    short_score = 0

    bullish = ind['ema_9'] > ind['ema_21'] > ind['ema_50']
    bearish = ind['ema_9'] < ind['ema_21'] < ind['ema_50']

    if bullish:
        long_score += 30
        if ind['adx'] > 25:
            long_score += 20
        if ind['macd_diff'] > 0:
            long_score += 15
        ema_dist = abs(ind['close'] - ind['ema_21']) / ind['ema_21'] if ind['ema_21'] > 0 else 1
        if ema_dist < 0.01:
            long_score += 20

    if bearish:
        short_score += 30
        if ind['adx'] > 25:
            short_score += 20
        if ind['macd_diff'] < 0:
            short_score += 15
        ema_dist = abs(ind['close'] - ind['ema_21']) / ind['ema_21'] if ind['ema_21'] > 0 else 1
        if ema_dist < 0.01:
            short_score += 20

    best = max(long_score, short_score)
    direction = 'BUY' if long_score > short_score else ('SELL' if short_score > long_score else 'NEUTRAL')
    return {'score': min(100, best), 'direction': direction}


def score_rsi_divergence(df: pd.DataFrame, ind: dict, idx: int) -> dict:
    """Module 5: RSI divergence using swing pivots."""
    lookback = 30
    if idx < lookback:
        return {'score': 0, 'direction': 'NEUTRAL'}

    prices = df['close'].values[idx - lookback:idx + 1]
    rsis = df['rsi'].values[idx - lookback:idx + 1]

    valid = ~np.isnan(rsis)
    if valid.sum() < lookback // 2:
        return {'score': 0, 'direction': 'NEUTRAL'}

    def find_swing_lows(arr, order=3):
        lows = []
        for i in range(order, len(arr) - order):
            if all(arr[i] <= arr[i - j] for j in range(1, order + 1)) and all(arr[i] <= arr[i + j] for j in range(1, order + 1)):
                lows.append(i)
        return lows

    def find_swing_highs(arr, order=3):
        highs = []
        for i in range(order, len(arr) - order):
            if all(arr[i] >= arr[i - j] for j in range(1, order + 1)) and all(arr[i] >= arr[i + j] for j in range(1, order + 1)):
                highs.append(i)
        return highs

    long_score = 0
    short_score = 0

    price_lows = find_swing_lows(prices)
    if len(price_lows) >= 2:
        i, j = price_lows[-2], price_lows[-1]
        if prices[j] < prices[i] and rsis[j] > rsis[i]:
            long_score += 60
            if ind['rsi'] < ind['rsi_oversold'] + 10:
                long_score += 25

    price_highs = find_swing_highs(prices)
    if len(price_highs) >= 2:
        i, j = price_highs[-2], price_highs[-1]
        if prices[j] > prices[i] and rsis[j] < rsis[i]:
            short_score += 60
            if ind['rsi'] > ind['rsi_overbought'] - 10:
                short_score += 25

    best = max(long_score, short_score)
    direction = 'BUY' if long_score > short_score else ('SELL' if short_score > long_score else 'NEUTRAL')
    return {'score': min(100, best), 'direction': direction}


def score_sniper_lite(df: pd.DataFrame, ind: dict, idx: int) -> dict:
    """Module 6: Extreme Z-score move + volume + RSI."""
    long_score = 0
    short_score = 0

    window = 50
    threshold = 2.0
    if idx < window:
        return {'score': 0, 'direction': 'NEUTRAL'}

    close_series = df['close'].iloc[max(0, idx - window - 4):idx + 1]
    rolling_mean = close_series.rolling(window=window).mean()
    rolling_std = close_series.rolling(window=window).std()

    current_price = ind['close']
    mean = float(rolling_mean.iloc[-1]) if pd.notna(rolling_mean.iloc[-1]) else current_price
    std = float(rolling_std.iloc[-1]) if pd.notna(rolling_std.iloc[-1]) else 1
    z_score = (current_price - mean) / std if std > 0 else 0

    if z_score <= -threshold:
        long_score += 45
        if z_score <= -(threshold + 0.5):
            long_score += 15
    if z_score >= threshold:
        short_score += 45
        if z_score >= threshold + 0.5:
            short_score += 15

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

    if ind['rsi'] < 35 and long_score > 0:
        long_score += 20
    if ind['rsi'] > 65 and short_score > 0:
        short_score += 20

    best = max(long_score, short_score)
    direction = 'BUY' if long_score > short_score else 'SELL' if short_score > long_score else 'NEUTRAL'
    return {'score': min(100, best), 'direction': direction}


def score_ramf_lite(df: pd.DataFrame, ind: dict, idx: int) -> dict:
    """Module 7: Volatility regime + momentum exhaustion."""
    long_score = 0
    short_score = 0

    if idx < 50:
        return {'score': 0, 'direction': 'NEUTRAL'}

    atr_values = df['atr'].iloc[max(0, idx - 49):idx + 1].dropna()
    if len(atr_values) < 20:
        return {'score': 0, 'direction': 'NEUTRAL'}

    current_atr = ind['atr']
    atr_percentile = (atr_values < current_atr).sum() / len(atr_values) * 100
    is_high_vol = atr_percentile >= 50

    if is_high_vol:
        vwap_dist = abs(ind['close'] - ind['vwap']) / current_atr if current_atr > 0 else 0
        if vwap_dist >= 1.0:
            if ind['close'] < ind['vwap']:
                long_score += 40
            else:
                short_score += 40

        start = max(0, idx - 2)
        recent = df.iloc[start:idx + 1]
        up_bars = (recent['close'] > recent['open']).sum()
        down_bars = (recent['close'] < recent['open']).sum()

        if down_bars >= 2 and long_score > 0:
            long_score += 30
        if up_bars >= 2 and short_score > 0:
            short_score += 30

        if ind['rsi'] < 35:
            long_score += 30
        elif ind['rsi'] < 45:
            long_score += 15
        if ind['rsi'] > 65:
            short_score += 30
        elif ind['rsi'] > 55:
            short_score += 15
    else:
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
    return {'score': min(100, best), 'direction': direction}


# ---------------------------------------------------------------------------
# Aggregation (mirrors _aggregate_scores)
# ---------------------------------------------------------------------------

def get_weights_for_adx(adx_val: float) -> dict:
    """Select weight profile based on ADX value."""
    if adx_val > REGIME_ADX_TRENDING:
        return WEIGHT_PROFILES['trending']
    elif adx_val < REGIME_ADX_RANGING:
        return WEIGHT_PROFILES['ranging']
    return DEFAULT_WEIGHTS


def aggregate_scores(module_results: dict, adx_val: float, hour: int) -> dict:
    """Aggregate module scores into a final signal."""
    weights = get_weights_for_adx(adx_val)

    long_modules = {}
    short_modules = {}

    for name, result in module_results.items():
        if result['score'] <= 0 or result['direction'] == 'NEUTRAL':
            continue
        if result['direction'] == 'BUY':
            long_modules[name] = result
        elif result['direction'] == 'SELL':
            short_modules[name] = result

    long_weighted = sum(r['score'] * weights.get(n, 0) for n, r in long_modules.items())
    short_weighted = sum(r['score'] * weights.get(n, 0) for n, r in short_modules.items())
    long_weight_total = sum(weights.get(n, 0) for n in long_modules)
    short_weight_total = sum(weights.get(n, 0) for n in short_modules)

    if long_weighted == 0 and short_weighted == 0:
        return {'direction': 'NEUTRAL', 'score': 0, 'active_modules': 0}

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

    if len(winning_modules) < MIN_CONVERGENT_MODULES:
        return {'direction': 'NEUTRAL', 'score': 0, 'active_modules': len(winning_modules)}

    active_weighted_sum = sum(r['score'] * weights.get(n, 0) for n, r in winning_modules.items())
    active_weight_total = sum(weights.get(n, 0) for n in winning_modules)

    if active_weight_total <= 0:
        return {'direction': 'NEUTRAL', 'score': 0, 'active_modules': 0}

    raw_score = active_weighted_sum / active_weight_total

    # Coverage penalty
    data_available = sum(1 for r in module_results.values() if r.get('score', 0) > 0)
    n_active = len(winning_modules)
    convergence_ratio = n_active / max(data_available, 1)
    coverage_penalty = convergence_ratio ** 0.5
    final_score = raw_score * coverage_penalty

    # Conflict penalty
    conflict_ratio = losing_weight / max(winning_weight, 0.01)
    if conflict_ratio > 0.15:
        conflict_factor = max(0.50, 1.0 - conflict_ratio * 0.60)
    else:
        conflict_factor = 1.0
    final_score *= conflict_factor

    # Session filter
    if hour in AVOID_HOURS:
        final_score *= 0.80
    elif hour in OPTIMAL_HOURS:
        final_score *= 1.10

    return {
        'direction': direction,
        'score': round(final_score, 1),
        'active_modules': n_active,
    }


# ---------------------------------------------------------------------------
# Position sizing helpers
# ---------------------------------------------------------------------------

def get_atr_profile(symbol: str):
    """Return (sl_mult, tp_mult) for the symbol's tier."""
    for profile in ATR_PROFILES.values():
        if symbol in profile['tokens']:
            return profile['sl_mult'], profile['tp_mult']
    return 1.5, 3.0


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

def run_backtest(df: pd.DataFrame, symbol: str, initial_balance: float = 500.0,
                 verbose: bool = True) -> 'BacktestResult':
    """Run a single backtest pass on the provided DataFrame."""
    df = compute_indicators(df.copy())

    engine = BacktestEngine(
        initial_balance=initial_balance,
        fee_pct=PAPER_TAKER_FEE,
        slippage_pct=get_token_slippage_pct(symbol)
    )

    warmup = 55  # Need at least 50 bars for indicators
    sl_mult, tp_mult = get_atr_profile(symbol)

    daily_trades = 0
    last_trade_date = None
    max_daily_trades = 5

    for idx in range(warmup, len(df)):
        row = df.iloc[idx]
        ts = row['datetime'] if 'datetime' in df.columns else pd.Timestamp.now()
        candle = {'high': float(row['high']), 'low': float(row['low']), 'close': float(row['close'])}

        # Reset daily counter
        current_date = ts.date() if hasattr(ts, 'date') else None
        if current_date and current_date != last_trade_date:
            daily_trades = 0
            last_trade_date = current_date

        # Check exits on current candle
        engine.check_exits(candle, ts)

        # Skip signal generation if daily limit reached or position already open
        if daily_trades >= max_daily_trades:
            continue
        if len(engine.positions) > 0:
            continue

        ind = get_indicator_dict(df, idx)

        # Run 6 technical modules
        module_results = {
            'mean_reversion': score_mean_reversion(df, ind),
            'momentum_breakout': score_momentum_breakout(df, ind, idx),
            'ema_trend': score_ema_trend(ind),
            'rsi_divergence': score_rsi_divergence(df, ind, idx),
            'sniper_lite': score_sniper_lite(df, ind, idx),
            'ramf_lite': score_ramf_lite(df, ind, idx),
        }

        hour = ts.hour if hasattr(ts, 'hour') else 12
        aggregated = aggregate_scores(module_results, ind['adx'], hour)

        if aggregated['direction'] == 'NEUTRAL' or aggregated['score'] < BASE_THRESHOLD:
            continue

        direction = aggregated['direction']
        price = ind['close']
        atr = ind['atr']

        if atr <= 0:
            continue

        # SL/TP prices
        sl_pct = (atr * sl_mult / price) * 100
        tp_pct = (atr * tp_mult / price) * 100
        if tp_pct < sl_pct * MIN_RR_RATIO:
            tp_pct = sl_pct * MIN_RR_RATIO

        if direction == 'BUY':
            sl_price = price * (1 - sl_pct / 100)
            tp_price = price * (1 + tp_pct / 100)
        else:
            sl_price = price * (1 + sl_pct / 100)
            tp_price = price * (1 - tp_pct / 100)

        # Position sizing: 2% risk, leveraged
        risk_amount = engine.balance * 0.02
        sl_fraction = max(sl_pct / 100, 0.001)
        position_size = risk_amount / sl_fraction * LEVERAGE
        # Cap at max position %
        max_pos = engine.balance * (MAX_POSITION_PCT / 100)
        position_size = min(position_size, max_pos)

        if position_size < 10:
            continue

        engine.open_position(symbol, direction, price, position_size, sl_price, tp_price, ts)
        daily_trades += 1

    # Close any remaining positions at last close
    if len(engine.positions) > 0 and len(df) > 0:
        last_close = float(df['close'].iloc[-1])
        last_ts = df['datetime'].iloc[-1] if 'datetime' in df.columns else pd.Timestamp.now()
        engine.close_all_positions(last_close, last_ts, reason='END')

    benchmark_start = float(df['close'].iloc[warmup]) if warmup < len(df) else None
    benchmark_end = float(df['close'].iloc[-1]) if len(df) > 0 else None

    result = engine.get_results(benchmark_start, benchmark_end)

    if verbose:
        print(result.summary())
        # Trade breakdown
        if result.trades:
            wins = sum(1 for t in result.trades if t.pnl > 0)
            sl_exits = sum(1 for t in result.trades if t.exit_reason == 'SL')
            tp_exits = sum(1 for t in result.trades if t.exit_reason == 'TP')
            end_exits = sum(1 for t in result.trades if t.exit_reason == 'END')
            print(f"Exit breakdown: SL={sl_exits} TP={tp_exits} END={end_exits}")
            print(f"Avg MAE: {np.mean([t.max_adverse for t in result.trades]) * 100:.2f}%")
            print(f"Avg MFE: {np.mean([t.max_favorable for t in result.trades]) * 100:.2f}%")

    return result


# ---------------------------------------------------------------------------
# Walk-forward analysis
# ---------------------------------------------------------------------------

def run_walk_forward(df: pd.DataFrame, symbol: str, train_days: int = 90,
                     test_days: int = 30, initial_balance: float = 500.0):
    """Walk-forward optimization: train on N days, test on M days, slide forward."""
    if 'datetime' not in df.columns:
        print("Walk-forward requires datetime column")
        return

    df = df.sort_values('datetime').reset_index(drop=True)
    start_date = df['datetime'].iloc[0]
    end_date = df['datetime'].iloc[-1]

    window_start = start_date
    fold = 0
    all_oos_results = []

    print(f"\n{'='*60}")
    print(f"WALK-FORWARD ANALYSIS: {symbol}")
    print(f"Train: {train_days}d | Test: {test_days}d")
    print(f"Data: {start_date.date()} to {end_date.date()}")
    print(f"{'='*60}\n")

    while window_start + timedelta(days=train_days + test_days) <= end_date:
        train_end = window_start + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)

        train_df = df[(df['datetime'] >= window_start) & (df['datetime'] < train_end)].copy()
        test_df = df[(df['datetime'] >= train_end) & (df['datetime'] < test_end)].copy()

        fold += 1
        print(f"--- Fold {fold} ---")
        print(f"  Train: {window_start.date()} to {train_end.date()} ({len(train_df)} bars)")
        print(f"  Test:  {train_end.date()} to {test_end.date()} ({len(test_df)} bars)")

        if len(train_df) < 100 or len(test_df) < 20:
            print("  SKIP: insufficient data")
            window_start += timedelta(days=test_days)
            continue

        # Train pass (for statistics only)
        train_result = run_backtest(train_df, symbol, initial_balance, verbose=False)
        print(f"  Train: {train_result.total_trades} trades, "
              f"return={train_result.total_return_pct:.2f}%, "
              f"WR={train_result.win_rate:.1f}%, "
              f"Sharpe={train_result.sharpe_ratio:.2f}")

        # Out-of-sample test
        test_result = run_backtest(test_df, symbol, initial_balance, verbose=False)
        print(f"  Test:  {test_result.total_trades} trades, "
              f"return={test_result.total_return_pct:.2f}%, "
              f"WR={test_result.win_rate:.1f}%, "
              f"Sharpe={test_result.sharpe_ratio:.2f}, "
              f"DD={test_result.max_drawdown_pct:.1f}%")

        all_oos_results.append(test_result)
        window_start += timedelta(days=test_days)

    # Aggregate OOS results
    if all_oos_results:
        total_trades = sum(r.total_trades for r in all_oos_results)
        oos_returns = [r.total_return_pct for r in all_oos_results]
        oos_sharpes = [r.sharpe_ratio for r in all_oos_results if r.total_trades > 0]
        oos_win_rates = [r.win_rate for r in all_oos_results if r.total_trades > 0]
        oos_drawdowns = [r.max_drawdown_pct for r in all_oos_results]

        print(f"\n{'='*60}")
        print(f"WALK-FORWARD SUMMARY ({len(all_oos_results)} folds)")
        print(f"{'='*60}")
        print(f"Total OOS trades:     {total_trades}")
        print(f"Avg OOS return:       {np.mean(oos_returns):.2f}% (+/- {np.std(oos_returns):.2f}%)")
        print(f"Cumulative return:    {sum(oos_returns):.2f}%")
        print(f"Avg OOS Sharpe:       {np.mean(oos_sharpes):.2f}" if oos_sharpes else "Avg OOS Sharpe: N/A")
        print(f"Avg OOS Win Rate:     {np.mean(oos_win_rates):.1f}%" if oos_win_rates else "Avg OOS Win Rate: N/A")
        print(f"Max OOS Drawdown:     {max(oos_drawdowns):.1f}%")
        print(f"Profitable folds:     {sum(1 for r in oos_returns if r > 0)}/{len(oos_returns)}")
    else:
        print("No walk-forward folds completed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Backtest Adaptive Hybrid Strategy')
    parser.add_argument('--symbol', type=str, default='BTC', help='Trading symbol (default: BTC)')
    parser.add_argument('--timeframe', type=str, default='1h', help='Candle timeframe (default: 1h)')
    parser.add_argument('--days', type=int, default=180, help='Days of history (default: 180)')
    parser.add_argument('--balance', type=float, default=500.0, help='Initial balance (default: 500)')
    parser.add_argument('--csv', type=str, default=None, help='Path to local CSV file (optional)')
    parser.add_argument('--walk-forward', action='store_true', help='Run walk-forward analysis')
    parser.add_argument('--train-days', type=int, default=90, help='Walk-forward train period (default: 90)')
    parser.add_argument('--test-days', type=int, default=30, help='Walk-forward test period (default: 30)')

    args = parser.parse_args()

    print(f"\nAdaptive Hybrid Backtest")
    print(f"Symbol: {args.symbol} | TF: {args.timeframe} | Days: {args.days}")
    print(f"Balance: ${args.balance:,.2f}")
    print(f"-" * 50)

    # Load data
    if args.csv:
        print(f"Loading from CSV: {args.csv}")
        df = load_csv(args.csv)
    else:
        print(f"Fetching {args.days} days of {args.timeframe} data from HyperLiquid...")
        df = fetch_ohlcv_hyperliquid(args.symbol, args.timeframe, args.days)

    if df.empty or len(df) < 100:
        print(f"ERROR: Insufficient data ({len(df)} rows). Need at least 100.")
        sys.exit(1)

    print(f"Loaded {len(df)} candles: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")

    if args.walk_forward:
        run_walk_forward(df, args.symbol, args.train_days, args.test_days, args.balance)
    else:
        run_backtest(df, args.symbol, args.balance)


if __name__ == '__main__':
    main()
