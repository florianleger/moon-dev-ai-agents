"""Crowd Positioning scoring module (Binance L/S ratio + Taker volume)."""


def score_crowd_positioning(symbol: str, indicators: dict, config: dict = None) -> dict:
    """Contrarian signal from Binance crowd positioning data.

    Combines:
    - Global L/S account ratio (contrarian: fade the crowd)
    - Top trader L/S ratio (follow smart money)
    - Taker buy/sell volume (aggressive order flow)

    Args:
        symbol: Token symbol (e.g. 'BTC').
        indicators: Dict of last-row indicator values.
        config: Optional config overrides.

    Returns:
        dict with 'score' (0-100), 'direction', and 'reason'.
    """
    try:
        from src.data_providers.binance_sentiment import BinanceSentimentProvider
        provider = BinanceSentimentProvider.get_instance()

        signal = provider.get_composite_signal(symbol)
        if not signal or signal['confidence'] == 0:
            return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'No positioning data'}

        return {
            'score': signal['confidence'],
            'direction': signal['direction'],
            'reason': f'Crowd: {signal["reason"]}',
        }
    except Exception as e:
        return None
