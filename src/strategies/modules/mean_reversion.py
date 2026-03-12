"""Mean Reversion scoring module (Bollinger Bands + RSI)."""

import pandas as pd


def score_mean_reversion(df: pd.DataFrame, indicators: dict, config: dict = None) -> dict:
    """Score mean reversion signal from BB + RSI.

    Args:
        df: OHLCV DataFrame with indicators computed.
        indicators: Dict of last-row indicator values.
        config: Optional config overrides (unused for now).

    Returns:
        dict with 'score' (0-100), 'direction', and 'reason'.
    """
    long_score = 0
    short_score = 0

    bb_range = indicators['bb_upper'] - indicators['bb_lower']
    if bb_range <= 0:
        return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'No BB range'}

    bb_pct = (indicators['close'] - indicators['bb_lower']) / bb_range

    # Long: Price near lower band
    if bb_pct < 0.10:
        long_score += 45
    elif bb_pct < 0.25:
        long_score += 25

    # RSI confirmation
    if indicators['rsi'] < indicators['rsi_oversold']:
        long_score += 35
    elif indicators['rsi'] < (indicators['rsi_oversold'] + 50) / 2:
        long_score += 20

    # Short: Price near upper band
    if bb_pct > 0.90:
        short_score += 45
    elif bb_pct > 0.75:
        short_score += 25

    if indicators['rsi'] > indicators['rsi_overbought']:
        short_score += 35
    elif indicators['rsi'] > (indicators['rsi_overbought'] + 50) / 2:
        short_score += 20

    # Ranging market bonus (ADX < 25) — apply only to the winning direction
    adx_bonus = 0
    if indicators['adx'] < 20:
        adx_bonus = 20
    elif indicators['adx'] < 30:
        adx_bonus = 10

    if adx_bonus > 0:
        if long_score >= short_score:
            long_score += adx_bonus
        else:
            short_score += adx_bonus

    best = max(long_score, short_score)
    direction = 'BUY' if long_score > short_score else 'SELL' if short_score > long_score else 'NEUTRAL'
    return {'score': min(100, best), 'direction': direction,
            'reason': f'BB%={bb_pct:.2f} RSI={indicators["rsi"]:.0f} ADX={indicators["adx"]:.0f}'}
