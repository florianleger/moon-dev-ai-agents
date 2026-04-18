"""Fibonacci OTE module.

Pure-math helpers that compute Fibonacci retracement levels and the
Optimal Trade Entry (OTE) zone from a swing high/low pair.

Convention used here (matches ICT / standard retracement convention):
    - For an ``UP`` impulse (bullish), the 0.0 level sits at ``swing_high``
      and the 1.0 level at ``swing_low`` (price retraces DOWN into the
      impulse looking for longs).
    - For a ``DOWN`` impulse (bearish), the 0.0 level sits at ``swing_low``
      and the 1.0 level at ``swing_high`` (price retraces UP looking for
      shorts).

The OTE zone is the 61.8% to 78.6% retracement band, with 70.5% as the
suggested entry midpoint.

Examples:
    >>> fib = compute_fib_levels(100.0, 110.0, 'UP')
    >>> round(fib['level_618'], 3)
    103.82
    >>> is_price_in_ote(103.0, fib)
    True
"""

from typing import Union

Number = Union[int, float]


def _empty_fib(direction: str = 'RANGE') -> dict:
    """Return a neutral fib dict for invalid inputs."""
    return {
        'level_0': 0.0,
        'level_236': 0.0,
        'level_382': 0.0,
        'level_50': 0.0,
        'level_618': 0.0,
        'level_705': 0.0,
        'level_786': 0.0,
        'level_100': 0.0,
        'ote_lower': 0.0,
        'ote_upper': 0.0,
        'ote_mid': 0.0,
        'impulse_range': 0.0,
        'direction': direction,
    }


def compute_fib_levels(
    swing_low: Number,
    swing_high: Number,
    direction: str,
) -> dict:
    """Compute Fibonacci retracement levels and the OTE zone.

    Args:
        swing_low: The low of the impulse leg.
        swing_high: The high of the impulse leg.
        direction: ``'UP'`` (bullish impulse, looking for long pullback)
            or ``'DOWN'`` (bearish impulse, looking for short pullback).

    Returns:
        Dict with all retracement levels, the OTE band (``ote_lower``,
        ``ote_upper``, ``ote_mid``), the raw impulse range, and the
        direction that was passed in. If inputs are invalid (NaN, zero
        range, unknown direction) an "empty" dict with all levels at 0.0
        is returned.
    """
    # Validate numeric inputs
    try:
        sl = float(swing_low)
        sh = float(swing_high)
    except (TypeError, ValueError):
        return _empty_fib(direction)

    # Guard against NaN (NaN != NaN)
    if sl != sl or sh != sh:
        return _empty_fib(direction)

    if sh <= sl:
        return _empty_fib(direction)

    direction = (direction or '').upper()
    if direction not in ('UP', 'DOWN'):
        return _empty_fib('RANGE')

    impulse_range = sh - sl

    if direction == 'UP':
        # Bullish: retracement from swing_high back down toward swing_low
        # 0% = swing_high (start of pullback), 100% = swing_low (deepest)
        level_0 = sh
        level_100 = sl
        level_236 = sh - impulse_range * 0.236
        level_382 = sh - impulse_range * 0.382
        level_50 = sh - impulse_range * 0.500
        level_618 = sh - impulse_range * 0.618
        level_705 = sh - impulse_range * 0.705
        level_786 = sh - impulse_range * 0.786

        # OTE zone: between 61.8% (shallower = higher price) and 78.6% (deeper = lower)
        ote_upper = level_618  # closer to swing_high
        ote_lower = level_786  # closer to swing_low
        ote_mid = level_705
    else:  # 'DOWN'
        # Bearish: retracement from swing_low back up toward swing_high
        # 0% = swing_low (start of pullback), 100% = swing_high (deepest)
        level_0 = sl
        level_100 = sh
        level_236 = sl + impulse_range * 0.236
        level_382 = sl + impulse_range * 0.382
        level_50 = sl + impulse_range * 0.500
        level_618 = sl + impulse_range * 0.618
        level_705 = sl + impulse_range * 0.705
        level_786 = sl + impulse_range * 0.786

        # OTE zone: between 61.8% (shallower = lower price) and 78.6% (deeper = higher)
        ote_lower = level_618  # closer to swing_low
        ote_upper = level_786  # closer to swing_high
        ote_mid = level_705

    return {
        'level_0': float(level_0),
        'level_236': float(level_236),
        'level_382': float(level_382),
        'level_50': float(level_50),
        'level_618': float(level_618),
        'level_705': float(level_705),
        'level_786': float(level_786),
        'level_100': float(level_100),
        'ote_lower': float(ote_lower),
        'ote_upper': float(ote_upper),
        'ote_mid': float(ote_mid),
        'impulse_range': float(impulse_range),
        'direction': direction,
    }


def is_price_in_ote(price: Number, fib_levels: dict) -> bool:
    """Return True if ``price`` sits inside the OTE zone of ``fib_levels``.

    Args:
        price: Price to test.
        fib_levels: Dict returned by :func:`compute_fib_levels`.

    Returns:
        True if ``ote_lower <= price <= ote_upper`` with a valid fib dict
        (non-zero impulse_range). False otherwise.
    """
    if fib_levels is None or not hasattr(fib_levels, 'get'):
        return False
    if fib_levels.get('impulse_range', 0) <= 0:
        return False

    try:
        p = float(price)
    except (TypeError, ValueError):
        return False
    if p != p:  # NaN check
        return False

    lo = fib_levels.get('ote_lower', 0.0)
    hi = fib_levels.get('ote_upper', 0.0)
    if lo > hi:
        lo, hi = hi, lo
    return lo <= p <= hi


if __name__ == '__main__':
    # Test 1: bullish UP retracement
    print('--- UP impulse: swing_low=100, swing_high=110 ---')
    up_fib = compute_fib_levels(100.0, 110.0, 'UP')
    for k, v in up_fib.items():
        print(f'  {k}: {v}')

    # Sanity: entry mid at 70.5% pulls back to 110 - 10 * 0.705 = 102.95
    print(f"  (sanity) ote_mid should be 102.95 -> {up_fib['ote_mid']:.2f}")
    print(f"  is 103.0 in OTE? {is_price_in_ote(103.0, up_fib)}  (expect True)")
    print(f"  is 108.0 in OTE? {is_price_in_ote(108.0, up_fib)}  (expect False)")
    print(f"  is 101.0 in OTE? {is_price_in_ote(101.0, up_fib)}  (expect False)")

    # Test 2: bearish DOWN retracement
    print('\n--- DOWN impulse: swing_low=100, swing_high=110 ---')
    down_fib = compute_fib_levels(100.0, 110.0, 'DOWN')
    for k, v in down_fib.items():
        print(f'  {k}: {v}')

    # Sanity: mid at 70.5% retraces up to 100 + 10 * 0.705 = 107.05
    print(f"  (sanity) ote_mid should be 107.05 -> {down_fib['ote_mid']:.2f}")
    print(f"  is 107.0 in OTE? {is_price_in_ote(107.0, down_fib)}  (expect True)")
    print(f"  is 102.0 in OTE? {is_price_in_ote(102.0, down_fib)}  (expect False)")

    # Test 3: degenerate inputs
    print('\n--- Invalid: swing_high <= swing_low ---')
    print(compute_fib_levels(110.0, 100.0, 'UP'))

    print('\n--- Invalid: unknown direction ---')
    print(compute_fib_levels(100.0, 110.0, 'SIDEWAYS'))

    print('\n--- is_price_in_ote with invalid fib ---')
    print(is_price_in_ote(105.0, _empty_fib()))  # expect False
