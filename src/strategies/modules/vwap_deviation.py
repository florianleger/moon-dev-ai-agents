"""VWAP Deviation Bands scoring module."""

import pandas as pd
import numpy as np


def score_vwap_deviation(df: pd.DataFrame, indicators: dict, config: dict = None) -> dict:
    """Score based on VWAP deviation bands and crossover signals.

    Args:
        df: OHLCV DataFrame with indicators computed.
        indicators: Dict of last-row indicator values (must include 'vwap', 'close').
        config: Optional config overrides.

    Returns:
        dict with 'score' (0-100), 'direction', and 'reason'.
    """
    try:
        return _score_vwap_deviation_inner(df, indicators, config)
    except Exception as e:
        return {'score': 0, 'direction': 'NEUTRAL', 'reason': f'VWAP error: {e}'}


def _score_vwap_deviation_inner(df: pd.DataFrame, indicators: dict, config: dict = None) -> dict:
    vwap = indicators.get('vwap')
    close = indicators.get('close')

    if vwap is None or close is None or vwap == 0:
        return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'No VWAP data'}

    # Compute volume-weighted standard deviation from df
    if 'vwap' not in df.columns or len(df) < 20:
        return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'Insufficient VWAP history'}

    # Volume-weighted deviation: sqrt(sum(vol * (close - vwap)^2) / sum(vol))
    vol = df['volume'].values[-100:]
    closes = df['close'].values[-100:]
    vwaps = df['vwap'].values[-100:]

    total_vol = vol.sum()
    if total_vol == 0:
        return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'No volume for VWAP bands'}

    vw_std = np.sqrt(np.sum(vol * (closes - vwaps) ** 2) / total_vol)
    if vw_std == 0:
        return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'Zero VWAP std'}

    z_vwap = (close - vwap) / vw_std

    long_score = 0
    short_score = 0

    # Signal 1: Z-score extremes (mean reversion from VWAP bands)
    if z_vwap < -2.0:
        long_score += 55
    elif z_vwap < -1.5:
        long_score += 40
    elif z_vwap < -1.0:
        long_score += 20

    if z_vwap > 2.0:
        short_score += 55
    elif z_vwap > 1.5:
        short_score += 40
    elif z_vwap > 1.0:
        short_score += 20

    # Signal 2: VWAP crossover with volume confirmation
    if len(df) >= 2:
        prev_close = df['close'].values[-2]
        prev_vwap = df['vwap'].values[-2]
        volume_ratio = indicators.get('volume_ratio', 1.0)

        # Bullish crossover: prev below VWAP, now above, with volume
        if prev_close < prev_vwap and close > vwap and volume_ratio > 1.3:
            long_score += 25
        # Bearish crossover: prev above VWAP, now below, with volume
        if prev_close > prev_vwap and close < vwap and volume_ratio > 1.3:
            short_score += 25

    best = max(long_score, short_score)
    direction = 'BUY' if long_score > short_score else ('SELL' if short_score > long_score else 'NEUTRAL')
    return {'score': min(100, best), 'direction': direction,
            'reason': f'z_VWAP={z_vwap:+.2f} VWAP={vwap:,.1f} std={vw_std:,.1f}'}
