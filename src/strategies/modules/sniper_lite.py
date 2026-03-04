"""Sniper Lite scoring module (extreme move + volume + RSI)."""

import pandas as pd


def score_sniper_lite(df: pd.DataFrame, indicators: dict, config: dict = None) -> dict:
    """Relaxed version of Sniper AI (extreme move + volume check).

    Args:
        df: OHLCV DataFrame.
        indicators: Dict of last-row indicator values.
        config: Optional config overrides.

    Returns:
        dict with 'score' (0-100), 'direction', and 'reason'.
    """
    long_score = 0
    short_score = 0

    # Check Z-score (2.0 sigma, 50-bar window)
    window = 50
    threshold = 2.0
    close_series = df['close'].tail(window + 5)
    rolling_mean = close_series.rolling(window=window).mean()
    rolling_std = close_series.rolling(window=window).std()

    current_price = indicators['close']
    mean = float(rolling_mean.iloc[-1]) if pd.notna(rolling_mean.iloc[-1]) else current_price
    std = float(rolling_std.iloc[-1]) if pd.notna(rolling_std.iloc[-1]) else 1

    z_score = (current_price - mean) / std if std > 0 else 0

    # Extreme move down = potential long (fade the move)
    if z_score <= -threshold:
        long_score += 45
        if z_score <= -(threshold + 0.5):
            long_score += 15
    # Extreme move up = potential short
    if z_score >= threshold:
        short_score += 45
        if z_score >= threshold + 0.5:
            short_score += 15

    # Volume confirmation (relaxed: 2x instead of 3x)
    if indicators['volume_ratio'] > 2.0:
        if long_score > 0:
            long_score += 20
        if short_score > 0:
            short_score += 20
    elif indicators['volume_ratio'] > 1.5:
        if long_score > 0:
            long_score += 10
        if short_score > 0:
            short_score += 10

    # RSI confirmation
    if indicators['rsi'] < 35 and long_score > 0:
        long_score += 20
    if indicators['rsi'] > 65 and short_score > 0:
        short_score += 20

    best = max(long_score, short_score)
    direction = 'BUY' if long_score > short_score else 'SELL' if short_score > long_score else 'NEUTRAL'
    return {'score': min(100, best), 'direction': direction,
            'reason': f'Z={z_score:.2f} Vol={indicators["volume_ratio"]:.1f}x RSI={indicators["rsi"]:.0f}'}
