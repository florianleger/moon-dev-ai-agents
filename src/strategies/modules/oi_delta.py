"""Open Interest Delta scoring module."""

import json
import os
from datetime import datetime
from collections import deque

_PERSIST_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'adaptive_hybrid', 'oi_history.json')


def _load_oi_history() -> dict:
    """Load persisted OI history from disk."""
    try:
        with open(_PERSIST_PATH, 'r') as f:
            raw = json.load(f)
        result = {}
        for sym, entries in raw.items():
            dq = deque(maxlen=50)
            for ts_str, oi_val in entries:
                dq.append((datetime.fromisoformat(ts_str), oi_val))
            result[sym] = dq
        return result
    except (FileNotFoundError, json.JSONDecodeError, Exception):
        return {}


def _save_oi_history(history: dict):
    """Persist OI history to disk."""
    try:
        os.makedirs(os.path.dirname(_PERSIST_PATH), exist_ok=True)
        raw = {}
        for sym, dq in history.items():
            raw[sym] = [(ts.isoformat(), oi_val) for ts, oi_val in dq]
        with open(_PERSIST_PATH, 'w') as f:
            json.dump(raw, f)
    except Exception:
        pass


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

        # Merge persisted history into oi_history on first call per symbol
        with cache_lock:
            if symbol not in oi_history:
                persisted = _load_oi_history()
                if symbol in persisted:
                    oi_history[symbol] = persisted[symbol]
                else:
                    oi_history[symbol] = deque(maxlen=50)

        # Store in timestamped deque
        now = datetime.now()
        with cache_lock:
            oi_history[symbol].append((now, current_oi))
            history = list(oi_history[symbol])
            # Persist after update
            _save_oi_history(oi_history)

        # Need at least 2 data points to compute meaningful delta (lowered from 3)
        if len(history) < 2:
            return {'score': 0, 'direction': 'NEUTRAL',
                    'reason': f'Insufficient OI history ({len(history)}/2 points)'}

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
