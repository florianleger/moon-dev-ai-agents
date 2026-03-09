"""Social Hype scoring module (CoinGecko trending + global market macro)."""


def score_social_hype(symbol: str, indicators: dict, config: dict = None) -> dict:
    """Social/trending signal from CoinGecko data.

    Combines:
    - Trending coin detection (retail attention / FOMO)
    - Global market cap change (macro momentum)
    - 1h price momentum

    Args:
        symbol: Token symbol (e.g. 'BTC').
        indicators: Dict of last-row indicator values.
        config: Optional config overrides.

    Returns:
        dict with 'score' (0-100), 'direction', and 'reason'.
    """
    try:
        from src.data_providers.coingecko_social import CoinGeckoSocialProvider
        provider = CoinGeckoSocialProvider.get_instance()

        signal = provider.get_social_signal(symbol)
        if not signal or signal['confidence'] == 0:
            return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'No social signal'}

        return {
            'score': signal['confidence'],
            'direction': signal['direction'],
            'reason': f'Social: {signal["reason"]}',
        }
    except Exception as e:
        return None
