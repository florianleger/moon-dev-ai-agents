"""Squeeze Detector scoring module (funding + OI + volatility compression)."""


def score_squeeze_detector(funding_zscore: float, indicators: dict, config: dict = None) -> dict:
    """Squeeze detection combining funding + volatility compression.

    Args:
        funding_zscore: Per-token historical funding Z-score.
        indicators: Dict of last-row indicator values.
        config: Optional config overrides.

    Returns:
        dict with 'score' (0-100), 'direction', and 'reason'.
    """
    long_score = 0
    short_score = 0

    bb_pct = indicators.get('bb_pct', 0.5)

    # Bollinger squeeze (BB width compression)
    # Measure actual band width relative to price using std dev of close over 20 periods
    df = indicators.get('_df')
    if df is not None and len(df) >= 20:
        close_20 = df['close'].tail(20)
        bb_width = close_20.std() / close_20.mean()
        bb_width_series = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        bb_width_percentile = bb_width_series.rank(pct=True).iloc[-1]
        bb_squeeze = bb_width_percentile < 0.25  # Bottom 25% of recent width = squeeze
    else:
        bb_squeeze = False

    # Short squeeze: very negative funding + compression
    if funding_zscore < -1.5:
        long_score += 40
        if bb_squeeze:
            long_score += 20
        if indicators['rsi'] < indicators['rsi_oversold'] + 10:
            long_score += 15

    # Long squeeze: very positive funding + compression
    if funding_zscore > 1.5:
        short_score += 40
        if bb_squeeze:
            short_score += 20
        if indicators['rsi'] > indicators['rsi_overbought'] - 10:
            short_score += 15

    best = max(long_score, short_score)
    direction = 'BUY' if long_score > short_score else ('SELL' if short_score > long_score else 'NEUTRAL')
    return {'score': min(100, best), 'direction': direction,
            'reason': f'Squeeze: funding_z={funding_zscore:.2f} bb_pct={bb_pct:.2f}'}
