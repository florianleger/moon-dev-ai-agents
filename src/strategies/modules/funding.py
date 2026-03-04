"""Funding Rate Contrarian scoring module."""


def score_funding_contrarian(funding_zscore: float, indicators: dict, config: dict = None) -> dict:
    """Contrarian signal based on extreme funding rates.

    Args:
        funding_zscore: Per-token historical funding Z-score.
        indicators: Dict of last-row indicator values.
        config: Optional config overrides.

    Returns:
        dict with 'score' (0-100), 'direction', and 'reason'.
    """
    long_score = 0
    short_score = 0

    # Extreme negative funding = shorts paying longs = go long (contrarian)
    if funding_zscore <= -2.0:
        long_score += 80
    elif funding_zscore <= -1.5:
        long_score += 55
    elif funding_zscore <= -1.0:
        long_score += 30

    # Extreme positive funding = longs paying shorts = go short (contrarian)
    if funding_zscore >= 2.0:
        short_score += 80
    elif funding_zscore >= 1.5:
        short_score += 55
    elif funding_zscore >= 1.0:
        short_score += 30

    best = max(long_score, short_score)
    direction = 'BUY' if long_score > short_score else 'SELL' if short_score > long_score else 'NEUTRAL'
    return {'score': min(100, best), 'direction': direction,
            'reason': f'Funding Z={funding_zscore:.2f}'}
