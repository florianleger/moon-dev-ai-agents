"""Tests for the swing_detector module."""
import pandas as pd
import pytest

from src.strategies.modules.swing_detector import detect_swings, detect_swings_fractal


def _make_ohlcv(prices, volume=1000):
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    return pd.DataFrame({
        'open': prices,
        'high': [p * 1.002 for p in prices],
        'low': [p * 0.998 for p in prices],
        'close': prices,
        'volume': [volume] * len(prices),
    })


# ---------------------------------------------------------------------------
# detect_swings (simple lookback extremum)
# ---------------------------------------------------------------------------

class TestDetectSwings:
    """Tests for detect_swings simple lookback extremum method."""

    def test_basic_up_swing(self):
        """Clear uptrend: low at start, high at end -> direction='UP'."""
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        df = _make_ohlcv(prices)

        result = detect_swings(df, lookback=10, min_range_pct=0.001)

        assert result['valid'] is True
        assert result['direction'] == 'UP'
        # Swing high should be the last (highest) bar's high
        assert result['swing_high'] == pytest.approx(109 * 1.002)
        # Swing low should be the first (lowest) bar's low
        assert result['swing_low'] == pytest.approx(100 * 0.998)
        assert result['swing_high_idx'] > result['swing_low_idx']

    def test_basic_down_swing(self):
        """Clear downtrend: high at start, low at end -> direction='DOWN'."""
        prices = [110, 109, 108, 107, 106, 105, 104, 103, 102, 101]
        df = _make_ohlcv(prices)

        result = detect_swings(df, lookback=10, min_range_pct=0.001)

        assert result['valid'] is True
        assert result['direction'] == 'DOWN'
        assert result['swing_high'] == pytest.approx(110 * 1.002)
        assert result['swing_low'] == pytest.approx(101 * 0.998)
        assert result['swing_low_idx'] > result['swing_high_idx']

    def test_invalid_small_range(self):
        """Range < min_range_pct -> valid=False."""
        # Tiny variations, well below 3% default threshold
        prices = [100.00, 100.01, 100.02, 100.01, 100.00]
        df = _make_ohlcv(prices)

        result = detect_swings(df, lookback=5, min_range_pct=0.03)

        assert result['valid'] is False
        # But range_pct should still be populated (computed)
        assert result['range_pct'] >= 0.0

    def test_empty_dataframe(self):
        """Empty df -> valid=False, no crash."""
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        result = detect_swings(empty_df, lookback=10, min_range_pct=0.003)

        assert result['valid'] is False
        assert result['direction'] == 'RANGE'
        assert result['swing_high'] == 0.0
        assert result['swing_low'] == 0.0

    def test_lookback_exceeds_df_size(self):
        """lookback=100 on 20-candle df -> uses all 20, no crash."""
        prices = [100 + i * 0.5 for i in range(20)]  # 100 -> 109.5
        df = _make_ohlcv(prices)

        result = detect_swings(df, lookback=100, min_range_pct=0.001)

        assert result['valid'] is True
        assert result['direction'] == 'UP'
        # All 20 candles used, so high idx and low idx should span them
        assert 0 <= result['swing_high_idx'] < 20
        assert 0 <= result['swing_low_idx'] < 20

    def test_range_calculation(self):
        """Verify range_pct = (swing_high - swing_low) / swing_low."""
        prices = [100, 102, 104, 106, 108, 110]
        df = _make_ohlcv(prices)

        result = detect_swings(df, lookback=6, min_range_pct=0.001)

        expected_high = 110 * 1.002
        expected_low = 100 * 0.998
        expected_range_pct = (expected_high - expected_low) / expected_low

        assert result['range_pct'] == pytest.approx(expected_range_pct, rel=1e-6)

    def test_missing_required_columns(self):
        """DF without high/low columns returns empty/invalid result."""
        bad_df = pd.DataFrame({'close': [100, 101, 102, 103, 104]})

        result = detect_swings(bad_df, lookback=5, min_range_pct=0.001)

        assert result['valid'] is False


# ---------------------------------------------------------------------------
# detect_swings_fractal (pivot-based detection)
# ---------------------------------------------------------------------------

class TestDetectSwingsFractal:
    """Tests for the fractal/pivot swing detection."""

    def test_pivot_high_detection(self):
        """A pivot high surrounded by lower candles should be detected."""
        # Clear V-shape: up, peak, down, trough, up
        df = pd.DataFrame({
            'open':   [100, 102, 104, 106, 108, 107, 105, 103, 101, 100,  99,  98, 100, 102, 104],
            'high':   [101, 103, 105, 107, 110, 108, 106, 104, 102, 101, 100,  99, 101, 103, 105],
            'low':    [ 99, 101, 103, 105, 107, 106, 104, 102, 100,  99,  98,  96,  99, 101, 103],
            'close':  [102, 104, 106, 108, 107, 106, 104, 102, 100,  99,  98,  99, 101, 103, 104],
            'volume': [10] * 15,
        })

        result = detect_swings_fractal(df, left=2, right=2)

        assert result['valid'] is True
        # Pivot high should be the 110 peak (idx 4)
        assert result['swing_high'] == pytest.approx(110.0)
        # Pivot low should be the 96 trough (idx 11)
        assert result['swing_low'] == pytest.approx(96.0)

    def test_no_pivot_in_monotonic_series(self):
        """A strictly increasing series has no confirmed pivot high."""
        # Strictly monotonic: no interior pivot
        prices = [100 + i for i in range(15)]
        df = _make_ohlcv(prices)

        result = detect_swings_fractal(df, left=2, right=2)

        # Monotonic data has no qualifying interior pivot
        assert result['valid'] is False

    def test_minimum_bars_required(self):
        """Need at least left+right+1 bars; below that returns invalid."""
        # Only 3 bars, need left(3)+right(3)+1 = 7 minimum
        prices = [100, 101, 102]
        df = _make_ohlcv(prices)

        result = detect_swings_fractal(df, left=3, right=3)

        assert result['valid'] is False
        assert result['direction'] == 'RANGE'

    def test_empty_df_fractal(self):
        """Empty df returns invalid, no crash."""
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        result = detect_swings_fractal(empty_df, left=3, right=3)

        assert result['valid'] is False
