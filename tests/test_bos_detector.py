"""Tests for the bos_detector module."""
import pandas as pd
import pytest

from src.strategies.modules.bos_detector import detect_bos


def _make_ohlcv(prices, highs=None, lows=None, volume=1000):
    """Build a minimal OHLCV DataFrame.

    If ``highs`` / ``lows`` are provided they are used as-is; otherwise
    a default +/-0.2% wick is applied around each close.
    """
    n = len(prices)
    if highs is None:
        highs = [p * 1.002 for p in prices]
    if lows is None:
        lows = [p * 0.998 for p in prices]
    return pd.DataFrame({
        'open': prices,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': [volume] * n,
    })


class TestDetectBos:
    """Tests for the break-of-structure detector."""

    def test_bos_long_triggered(self):
        """BUY position: a candle close > reference_high -> bos_detected=True."""
        # Last close is 106, reference_high 104.5 -> BOS
        df = pd.DataFrame({
            'open':   [100, 101, 102, 103, 104, 105],
            'high':   [101, 102, 103, 104, 106, 107],
            'low':    [ 99, 100, 101, 102, 103, 104],
            'close':  [101, 102, 103, 104, 105, 106],
            'volume': [10] * 6,
        })

        result = detect_bos(df, reference_high=104.5, reference_low=99.0,
                            direction='BUY', lookback=3)

        assert result['bos_detected'] is True
        assert result['bos_price'] > 104.5
        assert result['bos_idx'] >= 0

    def test_bos_long_not_triggered(self):
        """BUY position: all closes below reference_high -> bos_detected=False."""
        df = pd.DataFrame({
            'open':   [100, 101, 102, 101, 100, 101],
            'high':   [101, 102, 103, 102, 101, 102],
            'low':    [ 99, 100, 101, 100,  99, 100],
            'close':  [101, 102, 102, 101, 100, 101],
            'volume': [10] * 6,
        })

        result = detect_bos(df, reference_high=105.0, reference_low=95.0,
                            direction='BUY', lookback=5)

        assert result['bos_detected'] is False
        assert result['bos_price'] == 0.0
        assert result['bos_idx'] == -1

    def test_bos_short_triggered(self):
        """SELL position: a candle close < reference_low -> bos_detected=True."""
        df = pd.DataFrame({
            'open':   [110, 109, 108, 107, 106, 105],
            'high':   [111, 110, 109, 108, 107, 106],
            'low':    [109, 108, 107, 106, 105, 103],
            'close':  [109, 108, 107, 106, 105, 104],  # last close 104 < 104.5
            'volume': [10] * 6,
        })

        result = detect_bos(df, reference_high=111.0, reference_low=104.5,
                            direction='SELL', lookback=3)

        assert result['bos_detected'] is True
        assert result['bos_price'] < 104.5
        assert result['bos_idx'] >= 0

    def test_bos_uses_close_not_wick(self):
        """A wick piercing reference but close below should NOT trigger (avoid fakeouts)."""
        # Last candle: high=107 pierces 105, but close=104 stays below
        df = pd.DataFrame({
            'open':   [100, 101, 102, 103],
            'high':   [101, 102, 103, 107],  # wick pierces 105
            'low':    [ 99, 100, 101, 102],
            'close':  [101, 102, 103, 104],  # close does NOT pierce
            'volume': [10] * 4,
        })

        result = detect_bos(df, reference_high=105.0, reference_low=98.0,
                            direction='BUY', lookback=3)

        assert result['bos_detected'] is False

    def test_empty_df(self):
        """Empty df -> bos_detected=False, no crash."""
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        result = detect_bos(empty_df, reference_high=100.0, reference_low=90.0,
                            direction='BUY', lookback=5)

        assert result['bos_detected'] is False
        assert result['bos_idx'] == -1

    def test_invalid_direction(self):
        """Invalid direction string -> graceful fallback."""
        df = _make_ohlcv([100, 101, 102, 103, 104, 105])

        result = detect_bos(df, reference_high=104.5, reference_low=99.0,
                            direction='HOLD', lookback=3)

        assert result['bos_detected'] is False
        assert result['bos_idx'] == -1

    def test_lookback_limit(self):
        """BOS only detected within last N candles."""
        # Old BOS (idx 1) but lookback only 2 -> should NOT see it
        df = pd.DataFrame({
            'open':   [100, 101, 102, 103, 104, 100],
            'high':   [101, 106, 103, 104, 105, 101],
            'low':    [ 99, 100, 101, 102, 103,  99],
            # Only idx 1 (close=105) breaks above ref_high=104.5
            'close':  [101, 105, 103, 102, 101, 100],
            'volume': [10] * 6,
        })

        # Lookback=2: only last 2 candles scanned -> no BOS seen
        result_small = detect_bos(df, reference_high=104.5, reference_low=95.0,
                                  direction='BUY', lookback=2)
        assert result_small['bos_detected'] is False

        # Lookback=6: full history scanned -> BOS seen at idx 1
        result_wide = detect_bos(df, reference_high=104.5, reference_low=95.0,
                                 direction='BUY', lookback=6)
        assert result_wide['bos_detected'] is True
        assert result_wide['bos_idx'] == 1

    def test_current_high_low_populated(self):
        """current_high and current_low reflect the last candle."""
        df = _make_ohlcv([100, 101, 102, 103, 104, 105])

        result = detect_bos(df, reference_high=200.0, reference_low=50.0,
                            direction='BUY', lookback=3)

        # Last candle close = 105, high = 105 * 1.002, low = 105 * 0.998
        assert result['current_high'] == pytest.approx(105 * 1.002)
        assert result['current_low'] == pytest.approx(105 * 0.998)
