"""Stablecoin supply flow scoring module."""

from src.data_providers.stablecoin_flow import StablecoinFlowProvider


def score_stablecoin_flow(symbol: str, indicators: dict, config: dict = None) -> dict:
    """Score based on USDT+USDC supply changes (macro liquidity proxy).

    Args:
        symbol: Token symbol.
        indicators: Dict of last-row indicator values.
        config: Optional config overrides.

    Returns:
        dict with 'score' (0-100), 'direction', and 'reason'.
    """
    try:
        provider = StablecoinFlowProvider.get_instance()
        signal = provider.get_signal()

        direction = signal.get('direction', 'NEUTRAL')
        confidence = signal.get('confidence', 0)
        reason = signal.get('reason', 'No data')

        # Scale confidence (0-70) to score (0-60) - macro signals are confirmation, not primary
        score = int(confidence * 0.85) if direction != 'NEUTRAL' else 0

        return {'score': min(100, score), 'direction': direction, 'reason': reason}
    except Exception as e:
        return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'API unavailable', 'data_quality': 0.0}
