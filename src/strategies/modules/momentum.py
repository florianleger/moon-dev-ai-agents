"""Momentum Breakout scoring module (range breakout + volume)."""

import pandas as pd


def score_momentum_breakout(df: pd.DataFrame, indicators: dict, config: dict = None) -> dict:
    """Score breakout above/below recent range with volume confirmation.

    Args:
        df: OHLCV DataFrame.
        indicators: Dict of last-row indicator values.
        config: Optional config overrides.

    Returns:
        dict with 'score' (0-100), 'direction', and 'reason'.
    """
    long_score = 0
    short_score = 0

    high_20 = df['high'].tail(20).max()
    low_20 = df['low'].tail(20).min()
    close = indicators['close']
    vol_ratio = indicators['volume_ratio']

    # Upside breakout
    if close >= high_20:
        long_score += 40
        if vol_ratio > 1.5:
            long_score += 30
        elif vol_ratio > 1.2:
            long_score += 15
        if indicators['adx'] > 25:
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
        if indicators['adx'] > 25:
            short_score += 20
    elif close <= low_20 * 1.003:
        short_score += 20
        if vol_ratio > 1.3:
            short_score += 15

    best = max(long_score, short_score)
    direction = 'BUY' if long_score > short_score else 'SELL' if short_score > long_score else 'NEUTRAL'
    return {'score': min(100, best), 'direction': direction,
            'reason': f'H20={high_20:.2f} L20={low_20:.2f} Vol={vol_ratio:.1f}x'}
