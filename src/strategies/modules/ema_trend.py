"""EMA Trend scoring module (EMA alignment + ADX + MACD)."""


def score_ema_trend(indicators: dict, config: dict = None) -> dict:
    """Combined EMA trend module.

    Args:
        indicators: Dict of last-row indicator values.
        config: Optional config overrides.

    Returns:
        dict with 'score' (0-100), 'direction', and 'reason'.
    """
    long_score = 0
    short_score = 0

    # EMA alignment (9 > 21 > 50 for bullish)
    bullish_alignment = indicators['ema_9'] > indicators['ema_21'] > indicators['ema_50']
    bearish_alignment = indicators['ema_9'] < indicators['ema_21'] < indicators['ema_50']

    if bullish_alignment:
        long_score += 30
        if indicators['adx'] > 25:
            long_score += 20  # Strong trend confirmation
        if indicators['macd_diff'] > 0:
            long_score += 15
        # Check if close is near EMA 21 (pullback entry)
        ema_dist = abs(indicators['close'] - indicators['ema_21']) / indicators['ema_21']
        if ema_dist < 0.01:
            long_score += 20  # Pullback to EMA

    if bearish_alignment:
        short_score += 30
        if indicators['adx'] > 25:
            short_score += 20
        if indicators['macd_diff'] < 0:
            short_score += 15
        ema_dist = abs(indicators['close'] - indicators['ema_21']) / indicators['ema_21']
        if ema_dist < 0.01:
            short_score += 20

    best = max(long_score, short_score)
    direction = 'BUY' if long_score > short_score else ('SELL' if short_score > long_score else 'NEUTRAL')
    return {'score': min(100, best), 'direction': direction,
            'reason': f'EMA trend alignment ADX={indicators["adx"]:.1f} MACD={indicators["macd_diff"]:.4f}'}
