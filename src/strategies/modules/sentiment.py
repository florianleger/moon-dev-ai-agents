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

        # get_fear_greed_score returns float -1 to +1
        fg_score = reader.get_fear_greed_score()

        # Get token-specific sentiment (Twitter only, NOT F&G fallback)
        # reader.get_current_sentiment falls back to F&G when no Twitter data,
        # which would double-count. Use Twitter directly instead.
        token_sentiment = reader._get_twitter_sentiment(symbol.upper())
        has_token_data = token_sentiment is not None
        if token_sentiment is None:
            token_sentiment = 0.0

        long_score = 0
        short_score = 0

        # Fear & Greed contrarian: extreme fear = buy, extreme greed = sell
        # Scale proportionally instead of binary thresholds
        fg_abs = abs(fg_score)
        if fg_abs < 0.15:
            pass  # Neutral zone: no signal
        elif fg_score < 0:
            # Fear -> contrarian BUY, scale 10-50 based on intensity
            long_score += int(10 + 40 * min(1.0, (fg_abs - 0.15) / 0.65))
        else:
            # Greed -> contrarian SELL, scale 10-50 based on intensity
            short_score += int(10 + 40 * min(1.0, (fg_abs - 0.15) / 0.65))

        # Token-specific sentiment (contrarian) — only if we have real data
        if has_token_data:
            tok_abs = abs(token_sentiment)
            if tok_abs > 0.2:
                contrib = int(10 + 15 * min(1.0, (tok_abs - 0.2) / 0.6))
                if token_sentiment < 0:
                    long_score += contrib
                else:
                    short_score += contrib

        best = max(long_score, short_score)
        direction = 'BUY' if long_score > short_score else ('SELL' if short_score > long_score else 'NEUTRAL')
        return {'score': min(100, best), 'direction': direction,
                'reason': f'F&G={fg_score:+.2f} token={token_sentiment:+.2f} twitter={"yes" if has_token_data else "no"}'}
    except Exception as e:
        return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'API unavailable', 'data_quality': 0.0}
