"""Sentiment scoring module (Fear & Greed + Twitter contrarian)."""

import os


def score_sentiment(symbol: str, indicators: dict, config: dict = None) -> dict:
    """Sentiment composite (Fear & Greed + token-specific Twitter).

    Args:
        symbol: Token symbol.
        indicators: Dict of last-row indicator values.
        config: Optional config overrides.

    Returns:
        dict with 'score' (0-100), 'direction', and 'reason'.
    """
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from strategies.utils.sentiment_reader import SentimentReader
        reader = SentimentReader()

        # get_current_sentiment returns float -1 to +1 (fear to greed)
        token_sentiment = reader.get_current_sentiment(symbol)
        # get_fear_greed_score returns float -1 to +1
        fg_score = reader.get_fear_greed_score()

        long_score = 0
        short_score = 0

        # Fear & Greed contrarian: extreme fear = buy, extreme greed = sell
        if fg_score < -0.5:
            long_score += 50  # Extreme fear
        elif fg_score < -0.2:
            long_score += 25  # Fear
        elif fg_score > 0.5:
            short_score += 50  # Extreme greed
        elif fg_score > 0.2:
            short_score += 25  # Greed

        # Token-specific sentiment (contrarian)
        if token_sentiment < -0.3:
            long_score += 20
        elif token_sentiment > 0.3:
            short_score += 20

        best = max(long_score, short_score)
        direction = 'BUY' if long_score > short_score else ('SELL' if short_score > long_score else 'NEUTRAL')
        return {'score': min(100, best), 'direction': direction,
                'reason': f'F&G={fg_score:+.2f} token={token_sentiment:+.2f}'}
    except Exception as e:
        return {'score': 0, 'direction': 'NEUTRAL', 'reason': f'Sentiment error: {e}'}
