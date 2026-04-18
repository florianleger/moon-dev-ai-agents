"""Break of Structure (BOS) detector.

Detects whether price has broken out of a prior structural level after
a retracement. Used by the Fibonacci OTE scalping strategy to confirm
trade continuation: a bullish position expects price to eventually close
above the prior swing high; a bearish position expects a close below
the prior swing low.

``close`` is used rather than ``high``/``low`` to avoid wick noise.

Examples:
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'open':  [100, 101, 102, 103, 104],
    ...     'high':  [101, 102, 103, 104, 106],
    ...     'low':   [ 99, 100, 101, 102, 103],
    ...     'close': [101, 102, 103, 104, 105],
    ...     'volume':[ 10,  10,  10,  10,  10],
    ... })
    >>> res = detect_bos(df, reference_high=104.5, reference_low=99.0,
    ...                  direction='BUY', lookback=3)
    >>> res['bos_detected']
    True
"""

from typing import Optional, Union

import pandas as pd

Number = Union[int, float]


def _empty_bos_result(df: Optional[pd.DataFrame] = None) -> dict:
    """Return a neutral BOS result for invalid inputs."""
    current_high = 0.0
    current_low = 0.0
    if df is not None and len(df) > 0:
        try:
            current_high = float(df['high'].iloc[-1])
            current_low = float(df['low'].iloc[-1])
        except (KeyError, ValueError, TypeError):
            pass
    return {
        'bos_detected': False,
        'bos_price': 0.0,
        'bos_idx': -1,
        'current_high': current_high,
        'current_low': current_low,
    }


def detect_bos(
    ohlcv_df: pd.DataFrame,
    reference_high: Number,
    reference_low: Number,
    direction: str,
    lookback: int = 10,
) -> dict:
    """Detect a Break of Structure on the last ``lookback`` candles.

    Args:
        ohlcv_df: pandas DataFrame with ``open, high, low, close, volume``.
        reference_high: Structural high to break above for a bullish BOS.
        reference_low: Structural low to break below for a bearish BOS.
        direction: ``'BUY'`` (bullish) or ``'SELL'`` (bearish).
        lookback: How many most-recent candles to scan. Clamped to
            ``len(ohlcv_df)``.

    Returns:
        Dict with:
            - ``bos_detected`` (bool): whether a qualifying close happened.
            - ``bos_price`` (float): close price that triggered the BOS
              (first such candle in the scanned window, 0.0 if none).
            - ``bos_idx`` (int): absolute iloc position of the trigger
              candle (-1 if none).
            - ``current_high`` (float): last candle high.
            - ``current_low`` (float): last candle low.
    """
    if ohlcv_df is None or len(ohlcv_df) == 0:
        return _empty_bos_result(ohlcv_df)

    required_cols = {'high', 'low', 'close'}
    if not required_cols.issubset(set(ohlcv_df.columns)):
        return _empty_bos_result(ohlcv_df)

    direction = (direction or '').upper()
    if direction not in ('BUY', 'SELL'):
        return _empty_bos_result(ohlcv_df)

    try:
        ref_high = float(reference_high)
        ref_low = float(reference_low)
    except (TypeError, ValueError):
        return _empty_bos_result(ohlcv_df)

    # NaN guard
    if ref_high != ref_high or ref_low != ref_low:
        return _empty_bos_result(ohlcv_df)

    n = len(ohlcv_df)
    lb = min(max(int(lookback), 1), n)
    window = ohlcv_df.iloc[-lb:]
    offset = n - lb

    current_high = float(ohlcv_df['high'].iloc[-1]) if not pd.isna(ohlcv_df['high'].iloc[-1]) else 0.0
    current_low = float(ohlcv_df['low'].iloc[-1]) if not pd.isna(ohlcv_df['low'].iloc[-1]) else 0.0

    bos_detected = False
    bos_price = 0.0
    bos_idx = -1

    closes = window['close'].values
    for i, c in enumerate(closes):
        if pd.isna(c):
            continue
        c = float(c)
        if direction == 'BUY' and c > ref_high:
            bos_detected = True
            bos_price = c
            bos_idx = offset + i
            break
        if direction == 'SELL' and c < ref_low:
            bos_detected = True
            bos_price = c
            bos_idx = offset + i
            break

    return {
        'bos_detected': bool(bos_detected),
        'bos_price': float(bos_price),
        'bos_idx': int(bos_idx),
        'current_high': float(current_high),
        'current_low': float(current_low),
    }


if __name__ == '__main__':
    import pandas as pd

    # Test 1: bullish BOS - last close breaks above reference_high
    bull_df = pd.DataFrame({
        'open':   [100, 101, 102, 103, 104, 105],
        'high':   [101, 102, 103, 104, 106, 107],
        'low':    [ 99, 100, 101, 102, 103, 104],
        'close':  [101, 102, 103, 104, 105, 106],
        'volume': [ 10] * 6,
    })
    print('--- Bullish BOS expected (ref_high=104.5) ---')
    print(detect_bos(bull_df, reference_high=104.5, reference_low=99.0,
                    direction='BUY', lookback=3))

    # Test 2: bearish BOS - last close breaks below reference_low
    bear_df = pd.DataFrame({
        'open':   [110, 109, 108, 107, 106, 105],
        'high':   [111, 110, 109, 108, 107, 106],
        'low':    [109, 108, 107, 106, 105, 103],
        'close':  [109, 108, 107, 106, 105, 104],
        'volume': [ 10] * 6,
    })
    print('\n--- Bearish BOS expected (ref_low=104.5) ---')
    print(detect_bos(bear_df, reference_high=111.0, reference_low=104.5,
                    direction='SELL', lookback=3))

    # Test 3: no BOS (price hovers inside range)
    range_df = pd.DataFrame({
        'open':   [100, 101, 102, 101, 100, 101],
        'high':   [101, 102, 103, 102, 101, 102],
        'low':    [ 99, 100, 101, 100,  99, 100],
        'close':  [101, 102, 102, 101, 100, 101],
        'volume': [ 10] * 6,
    })
    print('\n--- No BOS expected ---')
    print(detect_bos(range_df, reference_high=105.0, reference_low=95.0,
                    direction='BUY', lookback=5))

    # Test 4: empty df
    print('\n--- Empty df ---')
    empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    print(detect_bos(empty_df, reference_high=100.0, reference_low=90.0,
                    direction='BUY', lookback=5))

    # Test 5: invalid direction
    print('\n--- Invalid direction ---')
    print(detect_bos(bull_df, reference_high=104.5, reference_low=99.0,
                    direction='HOLD', lookback=3))
