"""Tests for the funding_composite scoring module."""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ohlcv_50():
    """50-bar OHLCV DataFrame for squeeze detection tests."""
    np.random.seed(77)
    n = 50
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        'open': close + np.random.randn(n) * 0.1,
        'high': close + abs(np.random.randn(n) * 0.5),
        'low': close - abs(np.random.randn(n) * 0.5),
        'close': close,
        'volume': np.random.randint(1000, 10000, n).astype(float),
    })


@pytest.fixture
def base_indicators(sample_ohlcv_50):
    return {
        'close': 100.0, 'rsi': 45.0, 'rsi_oversold': 30, 'rsi_overbought': 70,
        'bb_upper': 102.0, 'bb_lower': 98.0, '_df': sample_ohlcv_50,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFundingComposite:
    """Tests for the merged funding composite module."""

    def test_score_returns_valid_structure(self, base_indicators, sample_ohlcv_50):
        """score() returns dict with direction, score, details keys."""
        from src.strategies.modules.funding_composite import score

        with patch('src.strategies.modules.funding_composite._compute_divergence_signal', return_value=None):
            result = score('BTC', sample_ohlcv_50, indicators=base_indicators, funding_zscore=-2.5)

        assert 'direction' in result
        assert 'score' in result
        assert 'details' in result
        assert result['direction'] in ('BUY', 'SELL', 'NEUTRAL')
        assert 0 <= result['score'] <= 100

    def test_extreme_positive_funding_generates_short(self, base_indicators, sample_ohlcv_50):
        """Very positive funding rate Z-score should generate SHORT signal."""
        from src.strategies.modules.funding_composite import score

        with patch('src.strategies.modules.funding_composite._compute_divergence_signal', return_value=None):
            result = score('BTC', sample_ohlcv_50, indicators=base_indicators, funding_zscore=2.5)

        assert result['direction'] == 'SELL'
        assert result['score'] > 0

    def test_extreme_negative_funding_generates_long(self, base_indicators, sample_ohlcv_50):
        """Very negative funding rate Z-score should generate LONG signal."""
        from src.strategies.modules.funding_composite import score

        with patch('src.strategies.modules.funding_composite._compute_divergence_signal', return_value=None):
            result = score('BTC', sample_ohlcv_50, indicators=base_indicators, funding_zscore=-2.5)

        assert result['direction'] == 'BUY'
        assert result['score'] > 0

    def test_neutral_funding_returns_neutral(self, base_indicators, sample_ohlcv_50):
        """Normal funding rate should return NEUTRAL."""
        from src.strategies.modules.funding_composite import score

        with patch('src.strategies.modules.funding_composite._compute_divergence_signal', return_value=None):
            result = score('BTC', sample_ohlcv_50, indicators=base_indicators, funding_zscore=0.3)

        assert result['direction'] == 'NEUTRAL'
        assert result['score'] == 0

    def test_graceful_degradation_on_data_failure(self, base_indicators, sample_ohlcv_50):
        """If one sub-signal fails, composite still works with remaining signals."""
        from src.strategies.modules.funding_composite import score

        # Divergence fails (returns None), but zscore works
        with patch('src.strategies.modules.funding_composite._compute_divergence_signal', return_value=None):
            result = score('BTC', sample_ohlcv_50, indicators=base_indicators, funding_zscore=-2.0)

        # Should still produce a signal from zscore + squeeze sub-signals
        assert result['direction'] == 'BUY'
        assert result['score'] > 0
        assert result['details']['divergence_signal'] is None

    def test_all_subsignals_fail_returns_neutral(self, base_indicators, sample_ohlcv_50):
        """If all sub-signals fail, returns NEUTRAL with score 0."""
        from src.strategies.modules.funding_composite import score

        with patch('src.strategies.modules.funding_composite._compute_divergence_signal', return_value=None):
            # funding_zscore=None disables zscore and squeeze sub-signals
            result = score('BTC', sample_ohlcv_50, indicators=base_indicators, funding_zscore=None)

        assert result['direction'] == 'NEUTRAL'
        assert result['score'] == 0

    def test_legacy_wrapper_delegates_to_score(self, base_indicators, sample_ohlcv_50):
        """score_funding_composite() backward-compat wrapper produces same result."""
        from src.strategies.modules.funding_composite import score, score_funding_composite

        indicators_with_df = {**base_indicators, '_df': sample_ohlcv_50}

        with patch('src.strategies.modules.funding_composite._compute_divergence_signal', return_value=None):
            result_new = score('BTC', sample_ohlcv_50, indicators=indicators_with_df, funding_zscore=-2.0)
            result_legacy = score_funding_composite('BTC', indicators_with_df, funding_zscore=-2.0)

        assert result_new['direction'] == result_legacy['direction']
        assert result_new['score'] == result_legacy['score']

    def test_squeeze_subsignal_with_oversold_rsi(self, sample_ohlcv_50):
        """Squeeze sub-signal adds bonus when RSI is oversold."""
        from src.strategies.modules.funding_composite import _compute_squeeze_signal

        indicators_oversold = {'rsi': 25, 'rsi_oversold': 30, 'rsi_overbought': 70}
        indicators_neutral = {'rsi': 50, 'rsi_oversold': 30, 'rsi_overbought': 70}

        result_oversold = _compute_squeeze_signal(-2.0, indicators_oversold, sample_ohlcv_50)
        result_neutral = _compute_squeeze_signal(-2.0, indicators_neutral, sample_ohlcv_50)

        # Oversold RSI should produce higher score
        assert result_oversold is not None
        assert result_neutral is not None
        assert result_oversold['score'] >= result_neutral['score']
