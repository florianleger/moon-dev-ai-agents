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

    # Partial alignment (2 of 3 EMAs ordered)
    partial_bull = indicators['ema_9'] > indicators['ema_21'] and not bullish_alignment
    partial_bear = indicators['ema_9'] < indicators['ema_21'] and not bearish_alignment

    # EMA spread strength: how spread apart the EMAs are (normalized to ATR)
    atr = indicators.get('atr', 0)
    if atr > 0:
        spread_9_21 = abs(indicators['ema_9'] - indicators['ema_21']) / atr
        spread_21_50 = abs(indicators['ema_21'] - indicators['ema_50']) / atr
    else:
        ema_21 = indicators['ema_21'] if indicators['ema_21'] != 0 else 1
        spread_9_21 = abs(indicators['ema_9'] - indicators['ema_21']) / ema_21 * 100
        spread_21_50 = abs(indicators['ema_21'] - indicators['ema_50']) / ema_21 * 100

    # Spread factor: tight EMAs (< 0.2 ATR) = weak trend, wide (> 1.0) = strong
    spread_factor = max(0.0, min(1.0, (spread_9_21 - 0.2) / 0.8))

    def _score_trend(score_ref, alignment_full, alignment_partial, macd_cond, rsi_cond):
        """Score a trend direction, returning points to add."""
        pts = 0
        if alignment_full:
            pts += int(15 + 15 * spread_factor)  # 15-30 based on spread strength
        elif alignment_partial:
            pts += int(8 + 7 * spread_factor)  # 8-15 for partial
        else:
            return 0  # No alignment, no score

        if indicators['adx'] > 30:
            pts += 15  # Strong trend
        elif indicators['adx'] > 20:
            pts += int(5 + 10 * ((indicators['adx'] - 20) / 10))  # 5-15 scaled

        if macd_cond:
            pts += 10

        # Pullback to EMA bonus
        close = indicators['close']
        ema_21_val = indicators['ema_21']
        if atr > 0:
            ema_dist = abs(close - ema_21_val) / atr
        else:
            ema_dist = abs(close - ema_21_val) / ema_21_val if ema_21_val != 0 else 0
        if ema_dist < 0.5:
            pts += 15  # Pullback to EMA

        return pts

    long_score += _score_trend(long_score, bullish_alignment, partial_bull,
                               indicators['macd_diff'] > 0, indicators['rsi'] > 50)
    short_score += _score_trend(short_score, bearish_alignment, partial_bear,
                                indicators['macd_diff'] < 0, indicators['rsi'] < 50)

    best = max(long_score, short_score)
    direction = 'BUY' if long_score > short_score else ('SELL' if short_score > long_score else 'NEUTRAL')
    return {'score': min(100, best), 'direction': direction,
            'reason': f'EMA trend alignment ADX={indicators["adx"]:.1f} MACD={indicators["macd_diff"]:.4f}'}
