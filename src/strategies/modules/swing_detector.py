"""Swing Detector module.

Detects recent swing high and swing low on any timeframe using either a simple
lookback extremum method or a fractal/pivot detection method.

Used by the Fibonacci OTE scalping strategy to identify the impulse leg from
which Fibonacci retracement levels are drawn.

Examples:
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'open':  [100, 101, 102, 103, 104, 105, 104, 103, 102, 101],
    ...     'high':  [101, 102, 103, 104, 106, 107, 105, 104, 103, 102],
    ...     'low':   [ 99, 100, 101, 102, 103, 104, 103, 102, 101, 100],
    ...     'close': [101, 102, 103, 104, 105, 105, 104, 103, 102, 101],
    ...     'volume':[ 10,  10,  10,  10,  10,  10,  10,  10,  10,  10],
    ... })
    >>> res = detect_swings(df, lookback=10, min_range_pct=0.001)
    >>> res['valid']
    True
"""

from typing import Optional

import pandas as pd


def _empty_result() -> dict:
    """Return a neutral empty result for edge cases."""
    return {
        'swing_high': 0.0,
        'swing_low': 0.0,
        'swing_high_idx': -1,
        'swing_low_idx': -1,
        'range_pct': 0.0,
        'valid': False,
        'direction': 'RANGE',
    }


def detect_swings(
    ohlcv_df: pd.DataFrame,
    lookback: int = 30,
    min_range_pct: float = 0.003,
) -> dict:
    """Detect the most recent swing high/low using a simple lookback extremum.

    Looks back ``lookback`` candles and identifies the highest high and
    the lowest low. The ordering of the two extrema (which came most recently)
    determines the implied direction of the last impulse.

    Args:
        ohlcv_df: pandas DataFrame with columns ``open, high, low, close, volume``.
        lookback: Number of most-recent candles to inspect. If greater than
            ``len(ohlcv_df)`` the full DataFrame is used.
        min_range_pct: Minimum (swing_high - swing_low) / swing_low ratio
            required for the swing to be considered ``valid`` (default 0.3%).

    Returns:
        Dict with keys:
            - ``swing_high`` (float): highest high in the window.
            - ``swing_low`` (float): lowest low in the window.
            - ``swing_high_idx`` (int): iloc position of the high within the df.
            - ``swing_low_idx`` (int): iloc position of the low within the df.
            - ``range_pct`` (float): (high - low) / low.
            - ``valid`` (bool): True if range_pct >= min_range_pct.
            - ``direction`` (str): ``'UP'`` if high is more recent than low,
              ``'DOWN'`` otherwise, ``'RANGE'`` if inputs are degenerate.
    """
    if ohlcv_df is None or len(ohlcv_df) == 0:
        return _empty_result()

    required_cols = {'high', 'low'}
    if not required_cols.issubset(set(ohlcv_df.columns)):
        return _empty_result()

    n = len(ohlcv_df)
    lb = min(lookback, n)
    if lb <= 1:
        return _empty_result()

    window = ohlcv_df.iloc[-lb:]
    highs = window['high'].dropna()
    lows = window['low'].dropna()
    if len(highs) == 0 or len(lows) == 0:
        return _empty_result()

    # Relative iloc indices inside window
    rel_high_idx = int(highs.values.argmax())
    rel_low_idx = int(lows.values.argmin())

    swing_high = float(highs.iloc[rel_high_idx])
    swing_low = float(lows.iloc[rel_low_idx])

    # Convert back to absolute iloc positions
    offset = n - lb
    swing_high_idx = offset + rel_high_idx
    swing_low_idx = offset + rel_low_idx

    if swing_low <= 0:
        return _empty_result()

    range_pct = (swing_high - swing_low) / swing_low
    valid = range_pct >= min_range_pct

    if swing_high_idx > swing_low_idx:
        direction = 'UP'
    elif swing_low_idx > swing_high_idx:
        direction = 'DOWN'
    else:
        direction = 'RANGE'

    return {
        'swing_high': swing_high,
        'swing_low': swing_low,
        'swing_high_idx': int(swing_high_idx),
        'swing_low_idx': int(swing_low_idx),
        'range_pct': float(range_pct),
        'valid': bool(valid),
        'direction': direction,
    }


def detect_swings_fractal(
    ohlcv_df: pd.DataFrame,
    left: int = 3,
    right: int = 3,
) -> dict:
    """Detect the most recent swing high/low using fractal (pivot) detection.

    A pivot high is a candle whose ``high`` is strictly greater than the highs
    of the ``left`` candles before it and the ``right`` candles after it.
    A pivot low is defined analogously on ``low``.

    Args:
        ohlcv_df: pandas DataFrame with columns ``high, low``.
        left: Number of candles required on the left of the pivot.
        right: Number of candles required on the right of the pivot
            (the pivot therefore cannot lie in the last ``right`` candles).

    Returns:
        Dict in the same format as :func:`detect_swings`. ``valid`` is True
        if both a pivot high and a pivot low were found and their relative
        range is strictly positive. ``direction`` is derived from which pivot
        is more recent.
    """
    if ohlcv_df is None or len(ohlcv_df) == 0:
        return _empty_result()

    required_cols = {'high', 'low'}
    if not required_cols.issubset(set(ohlcv_df.columns)):
        return _empty_result()

    n = len(ohlcv_df)
    if n < left + right + 1:
        return _empty_result()

    highs = ohlcv_df['high'].values
    lows = ohlcv_df['low'].values

    last_pivot_high_idx: Optional[int] = None
    last_pivot_low_idx: Optional[int] = None

    # Iterate from most recent eligible pivot back to the start
    for i in range(n - right - 1, left - 1, -1):
        h = highs[i]
        l = lows[i]
        if pd.isna(h) or pd.isna(l):
            continue

        if last_pivot_high_idx is None:
            left_slice = highs[i - left:i]
            right_slice = highs[i + 1:i + 1 + right]
            if (
                len(left_slice) == left
                and len(right_slice) == right
                and not pd.isna(left_slice).any()
                and not pd.isna(right_slice).any()
                and h > float(pd.Series(left_slice).max())
                and h > float(pd.Series(right_slice).max())
            ):
                last_pivot_high_idx = i

        if last_pivot_low_idx is None:
            left_slice = lows[i - left:i]
            right_slice = lows[i + 1:i + 1 + right]
            if (
                len(left_slice) == left
                and len(right_slice) == right
                and not pd.isna(left_slice).any()
                and not pd.isna(right_slice).any()
                and l < float(pd.Series(left_slice).min())
                and l < float(pd.Series(right_slice).min())
            ):
                last_pivot_low_idx = i

        if last_pivot_high_idx is not None and last_pivot_low_idx is not None:
            break

    if last_pivot_high_idx is None or last_pivot_low_idx is None:
        return _empty_result()

    swing_high = float(highs[last_pivot_high_idx])
    swing_low = float(lows[last_pivot_low_idx])

    if swing_low <= 0 or swing_high <= swing_low:
        return _empty_result()

    range_pct = (swing_high - swing_low) / swing_low

    if last_pivot_high_idx > last_pivot_low_idx:
        direction = 'UP'
    elif last_pivot_low_idx > last_pivot_high_idx:
        direction = 'DOWN'
    else:
        direction = 'RANGE'

    return {
        'swing_high': swing_high,
        'swing_low': swing_low,
        'swing_high_idx': int(last_pivot_high_idx),
        'swing_low_idx': int(last_pivot_low_idx),
        'range_pct': float(range_pct),
        'valid': True,
        'direction': direction,
    }


if __name__ == '__main__':
    import pandas as pd

    # Test 1: clean UP impulse followed by small pullback
    up_df = pd.DataFrame({
        'open':   [100, 101, 102, 103, 104, 105, 106, 107, 106, 105],
        'high':   [101, 102, 103, 104, 105, 106, 107, 108, 107, 106],
        'low':    [ 99, 100, 101, 102, 103, 104, 105, 106, 105, 104],
        'close':  [101, 102, 103, 104, 105, 106, 107, 107, 106, 105],
        'volume': [ 10] * 10,
    })
    print('--- detect_swings on UP impulse ---')
    print(detect_swings(up_df, lookback=10, min_range_pct=0.001))

    # Test 2: DOWN impulse
    down_df = pd.DataFrame({
        'open':   [110, 109, 108, 107, 106, 105, 104, 103, 104, 105],
        'high':   [111, 110, 109, 108, 107, 106, 105, 104, 105, 106],
        'low':    [109, 108, 107, 106, 105, 104, 103, 102, 103, 104],
        'close':  [109, 108, 107, 106, 105, 104, 103, 103, 104, 105],
        'volume': [ 10] * 10,
    })
    print('\n--- detect_swings on DOWN impulse ---')
    print(detect_swings(down_df, lookback=10, min_range_pct=0.001))

    # Test 3: fractal detection on a V-shape with clear pivot high and pivot low
    # (monotonic data has no interior pivots, so we use zigzag data)
    fractal_df = pd.DataFrame({
        'open':   [100, 102, 104, 106, 108, 107, 105, 103, 101, 100,  99,  98, 100, 102, 104],
        'high':   [101, 103, 105, 107, 110, 108, 106, 104, 102, 101, 100,  99, 101, 103, 105],
        'low':    [ 99, 101, 103, 105, 107, 106, 104, 102, 100,  99,  98,  96,  99, 101, 103],
        'close':  [102, 104, 106, 108, 107, 106, 104, 102, 100,  99,  98,  99, 101, 103, 104],
        'volume': [ 10] * 15,
    })
    print('\n--- detect_swings_fractal on V-shape (pivot high then pivot low) ---')
    print(detect_swings_fractal(fractal_df, left=2, right=2))

    # Test 4: empty DataFrame edge case
    empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    print('\n--- detect_swings on empty df ---')
    print(detect_swings(empty_df))

    # Test 5: insufficient range (should be invalid)
    flat_df = pd.DataFrame({
        'open':   [100.00, 100.01, 100.02, 100.01, 100.00],
        'high':   [100.01, 100.02, 100.03, 100.02, 100.01],
        'low':    [ 99.99, 100.00, 100.01, 100.00,  99.99],
        'close':  [100.00, 100.01, 100.02, 100.01, 100.00],
        'volume': [ 10] * 5,
    })
    print('\n--- detect_swings on near-flat df (expect valid=False) ---')
    print(detect_swings(flat_df, lookback=5, min_range_pct=0.003))
