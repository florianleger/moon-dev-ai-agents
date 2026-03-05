"""Options Market Sentiment scoring module (Deribit Put/Call + Max Pain)."""

from src.data_providers.deribit_options import DeribitOptionsProvider


def score_options_sentiment(symbol: str, indicators: dict, config: dict = None) -> dict:
    """Score based on options P/C ratio and max pain distance.

    Only works for BTC and ETH (Deribit coverage).

    Args:
        symbol: Token symbol.
        indicators: Dict of last-row indicator values (must include 'close').
        config: Optional config overrides.

    Returns:
        dict with 'score' (0-100), 'direction', and 'reason'.
    """
    if symbol not in ('BTC', 'ETH'):
        return {'score': 0, 'direction': 'NEUTRAL', 'reason': f'{symbol} not covered by Deribit'}

    try:
        provider = DeribitOptionsProvider.get_instance()

        pc_data = provider.get_put_call_ratio(symbol)
        pc_ratio = pc_data['put_call_ratio'] if pc_data else None
        mp_data = provider.get_max_pain(symbol)
        max_pain = mp_data['max_pain'] if mp_data else None
        close = indicators.get('close', 0)

        long_score = 0
        short_score = 0
        parts = []

        # Put/Call ratio signal (contrarian: high P/C = fear = bullish)
        if pc_ratio is not None:
            parts.append(f'P/C={pc_ratio:.2f}')
            if pc_ratio > 1.2:
                long_score += 30  # Heavy put buying = contrarian bullish
            elif pc_ratio > 0.9:
                long_score += 10  # Mild fear
            elif pc_ratio < 0.5:
                short_score += 30  # Very low puts = complacency = bearish
            elif pc_ratio < 0.7:
                short_score += 10  # Mild complacency

        # Max pain distance signal
        if max_pain is not None and close > 0:
            distance_pct = (close - max_pain) / max_pain * 100
            parts.append(f'MaxPain={max_pain:,.0f} dist={distance_pct:+.1f}%')
            if distance_pct > 5.0:
                short_score += 20  # Price well above max pain, gravity pull down
            elif distance_pct < -5.0:
                long_score += 20  # Price well below max pain, gravity pull up

        if not parts:
            return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'No options data available'}

        best = max(long_score, short_score)
        direction = 'BUY' if long_score > short_score else ('SELL' if short_score > long_score else 'NEUTRAL')
        return {'score': min(100, best), 'direction': direction, 'reason': ' '.join(parts)}
    except Exception as e:
        return {'score': 0, 'direction': 'NEUTRAL', 'reason': f'Options error: {e}'}
