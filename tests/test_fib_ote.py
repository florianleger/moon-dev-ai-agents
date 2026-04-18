"""Tests for the fib_ote module."""
import math

import pytest

from src.strategies.modules.fib_ote import compute_fib_levels, is_price_in_ote


# ---------------------------------------------------------------------------
# compute_fib_levels
# ---------------------------------------------------------------------------

class TestComputeFibLevels:
    """Tests for the Fibonacci level computation."""

    def test_bullish_fib_levels(self):
        """UP impulse 100->110: ote_mid should be around 102.95."""
        fib = compute_fib_levels(100.0, 110.0, 'UP')

        # level_618 = 110 - (10 * 0.618) = 103.82
        assert fib['level_618'] == pytest.approx(103.82, abs=1e-3)
        # level_786 = 110 - (10 * 0.786) = 102.14
        assert fib['level_786'] == pytest.approx(102.14, abs=1e-3)
        # level_705 = 110 - (10 * 0.705) = 102.95
        assert fib['level_705'] == pytest.approx(102.95, abs=1e-3)

        # OTE band: upper = 61.8% (shallower), lower = 78.6% (deeper)
        assert fib['ote_upper'] == pytest.approx(103.82, abs=1e-3)
        assert fib['ote_lower'] == pytest.approx(102.14, abs=1e-3)
        assert fib['ote_mid'] == pytest.approx(102.95, abs=1e-3)
        assert fib['direction'] == 'UP'
        assert fib['impulse_range'] == pytest.approx(10.0)

    def test_bearish_fib_levels(self):
        """DOWN impulse 110->100: ote_mid should be 107.05."""
        fib = compute_fib_levels(100.0, 110.0, 'DOWN')

        # In DOWN: retracement UP from swing_low
        # level_618 = 100 + (10 * 0.618) = 106.18
        assert fib['level_618'] == pytest.approx(106.18, abs=1e-3)
        # level_786 = 100 + (10 * 0.786) = 107.86
        assert fib['level_786'] == pytest.approx(107.86, abs=1e-3)
        # level_705 = 100 + (10 * 0.705) = 107.05
        assert fib['level_705'] == pytest.approx(107.05, abs=1e-3)

        # OTE band for DOWN: lower = 61.8% (shallower = lower price),
        # upper = 78.6% (deeper = higher price)
        assert fib['ote_lower'] == pytest.approx(106.18, abs=1e-3)
        assert fib['ote_upper'] == pytest.approx(107.86, abs=1e-3)
        assert fib['ote_mid'] == pytest.approx(107.05, abs=1e-3)
        assert fib['direction'] == 'DOWN'

    def test_impulse_range_positive(self):
        """impulse_range should always be swing_high - swing_low (positive)."""
        fib_up = compute_fib_levels(50.0, 75.0, 'UP')
        fib_down = compute_fib_levels(50.0, 75.0, 'DOWN')

        assert fib_up['impulse_range'] == pytest.approx(25.0)
        assert fib_down['impulse_range'] == pytest.approx(25.0)
        assert fib_up['impulse_range'] > 0
        assert fib_down['impulse_range'] > 0

    def test_invalid_direction(self):
        """Invalid direction -> graceful fallback with impulse_range=0."""
        fib = compute_fib_levels(100.0, 110.0, 'SIDEWAYS')

        # All-zero dict returned for unknown direction
        assert fib['impulse_range'] == 0.0
        assert fib['ote_mid'] == 0.0
        assert fib['level_618'] == 0.0

    def test_swing_high_equals_low(self):
        """Equal swings -> impulse_range=0, no crash."""
        fib = compute_fib_levels(100.0, 100.0, 'UP')

        assert fib['impulse_range'] == 0.0
        assert fib['ote_mid'] == 0.0

    def test_swing_high_less_than_low(self):
        """Inverted swings (high < low) -> fallback to empty."""
        fib = compute_fib_levels(110.0, 100.0, 'UP')

        assert fib['impulse_range'] == 0.0

    def test_nan_inputs(self):
        """NaN inputs -> graceful fallback."""
        fib = compute_fib_levels(float('nan'), 110.0, 'UP')

        assert fib['impulse_range'] == 0.0


# ---------------------------------------------------------------------------
# is_price_in_ote
# ---------------------------------------------------------------------------

class TestIsPriceInOte:
    """Tests for the OTE band membership check."""

    def test_price_inside_ote(self):
        """Price at ote_mid should return True."""
        fib = compute_fib_levels(100.0, 110.0, 'UP')

        assert is_price_in_ote(fib['ote_mid'], fib) is True
        # A value clearly inside the 102.14..103.82 band
        assert is_price_in_ote(103.0, fib) is True

    def test_price_below_ote(self):
        """Price below ote_lower should return False."""
        fib = compute_fib_levels(100.0, 110.0, 'UP')

        assert is_price_in_ote(101.0, fib) is False

    def test_price_above_ote(self):
        """Price above ote_upper should return False."""
        fib = compute_fib_levels(100.0, 110.0, 'UP')

        assert is_price_in_ote(108.0, fib) is False

    def test_price_at_boundary(self):
        """Price exactly at ote_lower or ote_upper should be True (inclusive)."""
        fib = compute_fib_levels(100.0, 110.0, 'UP')

        assert is_price_in_ote(fib['ote_lower'], fib) is True
        assert is_price_in_ote(fib['ote_upper'], fib) is True

    def test_down_price_inside(self):
        """For DOWN impulse OTE ~106.18..107.86, price 107.0 inside."""
        fib = compute_fib_levels(100.0, 110.0, 'DOWN')

        assert is_price_in_ote(107.0, fib) is True
        assert is_price_in_ote(102.0, fib) is False
        assert is_price_in_ote(109.0, fib) is False

    def test_edge_cases(self):
        """None, NaN, invalid dict -> False, no crash."""
        fib = compute_fib_levels(100.0, 110.0, 'UP')

        # None fib dict
        assert is_price_in_ote(103.0, None) is False
        # Empty dict (no 'impulse_range' key -> uses default 0)
        assert is_price_in_ote(103.0, {}) is False
        # NaN price
        assert is_price_in_ote(float('nan'), fib) is False
        # Invalid fib (empty / zero range)
        empty_fib = compute_fib_levels(100.0, 100.0, 'UP')  # impulse_range=0
        assert is_price_in_ote(100.0, empty_fib) is False
        # Non-numeric price
        assert is_price_in_ote("not a number", fib) is False
