"""Open Interest Delta scoring module."""

from datetime import datetime
from collections import deque


def score_oi_delta(indicators: dict, market_data, oi_history: dict,
                   symbol: str, cache_lock, config: dict = None) -> dict:
    """Open Interest delta -- detects positioning pressure.

    Args:
        indicators: Dict of last-row indicator values.
        market_data: MarketDataProvider instance.
        oi_history: Shared dict {symbol: deque of (timestamp, oi_value)}.
        symbol: Token symbol.
        cache_lock: threading.Lock for oi_history access.
        config: Optional config overrides.

    Returns:
        dict with 'score' (0-100), 'direction', and 'reason'.
    """
    if not market_data:
        return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'No market data'}
    try:
        oi_data = market_data.get_open_interest(symbol)
        if oi_data is None:
            return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'No OI data'}

        current_oi = oi_data.get('open_interest', 0)
        if current_oi <= 0:
            return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'OI is zero'}

        # Store in timestamped deque
        now = datetime.now()
        with cache_lock:
            if symbol not in oi_history:
                oi_history[symbol] = deque(maxlen=50)
            oi_history[symbol].append((now, current_oi))
            history = list(oi_history[symbol])

        # Need at least 3 data points to compute meaningful delta
        if len(history) < 3:
            return {'score': 0, 'direction': 'NEUTRAL',
                    'reason': f'Insufficient OI history ({len(history)}/3 points)'}

        # Calculate delta from oldest available point
        oldest_oi = history[0][1]
        oi_change_pct = ((current_oi - oldest_oi) / oldest_oi * 100) if oldest_oi > 0 else 0

        price_change = indicators.get('close', 0) - indicators.get('open', indicators.get('close', 0))

        long_score = 0
        short_score = 0

        if oi_change_pct > 3:  # OI increasing significantly
            if price_change > 0:
                long_score += 50  # New longs entering
            else:
                short_score += 40  # New shorts entering
        elif oi_change_pct < -3:  # OI decreasing
            if price_change > 0:
                long_score += 30  # Short squeeze (shorts closing)
            else:
                short_score += 30  # Long squeeze

        best = max(long_score, short_score)
        direction = 'BUY' if long_score > short_score else ('SELL' if short_score > long_score else 'NEUTRAL')
        return {'score': min(100, best), 'direction': direction,
                'reason': f'OI delta={oi_change_pct:+.1f}% (OI={current_oi:,.0f}, {len(history)} pts) price_chg={price_change:+.2f}'}
    except Exception as e:
        return {'score': 0, 'direction': 'NEUTRAL', 'reason': f'OI error: {e}'}
