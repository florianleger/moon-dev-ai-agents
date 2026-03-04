"""Cross-Exchange Funding Divergence scoring module."""


def score_funding_divergence(symbol: str, indicators: dict, config: dict = None) -> dict:
    """Signal from cross-exchange funding rate divergence (HL vs Binance).

    Detects:
    - Both exchanges extremely positive/negative (crowd leverage extreme)
    - One exchange positive, other negative (true divergence)
    - Significant spread between exchanges (arbitrage pressure)

    Args:
        symbol: Token symbol (e.g. 'BTC').
        indicators: Dict of last-row indicator values.
        config: Optional config overrides.

    Returns:
        dict with 'score' (0-100), 'direction', and 'reason'.
    """
    try:
        from src.data_providers.cross_exchange_funding import CrossExchangeFundingProvider
        provider = CrossExchangeFundingProvider.get_instance()

        signal = provider.get_divergence_signal(symbol)
        if not signal or signal['confidence'] == 0:
            return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'No funding divergence'}

        return {
            'score': signal['confidence'],
            'direction': signal['direction'],
            'reason': f'FundingDiv: {signal["reason"]}',
        }
    except Exception as e:
        return {'score': 0, 'direction': 'NEUTRAL', 'reason': f'FundingDiv error: {e}'}
