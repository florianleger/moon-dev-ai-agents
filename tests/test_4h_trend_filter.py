"""Tests for the 4H trend filter gate."""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def strategy():
    """Create a minimal strategy instance for testing."""
    with patch('src.data_providers.market_data.MarketDataProvider', MagicMock()):
        from src.strategies.custom.adaptive_hybrid_strategy import AdaptiveHybridStrategy
        with patch.object(AdaptiveHybridStrategy, '_preload_funding_history'):
            with patch.object(AdaptiveHybridStrategy, '_load_state_from_csv'):
                s = AdaptiveHybridStrategy()
                s._market_data = None
                s._btc_trend_cache = None
                s._btc_trend_timestamp = None
                s._4h_trend_cache = {}
                yield s


def _make_ohlcv_trending_up(n=100):
    """Create OHLCV DataFrame with clear uptrend."""
    close = np.linspace(100, 150, n)
    return pd.DataFrame({
        'open': close - 0.5,
        'high': close + 2.0,
        'low': close - 1.0,
        'close': close,
        'volume': np.full(n, 5000.0),
    })


def _make_ohlcv_trending_down(n=100):
    """Create OHLCV DataFrame with clear downtrend."""
    close = np.linspace(150, 100, n)
    return pd.DataFrame({
        'open': close + 0.5,
        'high': close + 1.0,
        'low': close - 2.0,
        'close': close,
        'volume': np.full(n, 5000.0),
    })


def _make_ohlcv_ranging(n=100):
    """Create OHLCV DataFrame with no clear trend (oscillating)."""
    np.random.seed(42)
    base = 100.0
    close = base + np.sin(np.linspace(0, 8 * np.pi, n)) * 2.0
    return pd.DataFrame({
        'open': close + np.random.randn(n) * 0.1,
        'high': close + 1.0,
        'low': close - 1.0,
        'close': close,
        'volume': np.full(n, 5000.0),
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrendFilter4H:
    """Tests for the 4H trend filter gate."""

    def test_bullish_trend_detected(self, strategy):
        """Price > EMA20 > EMA50 with ADX > 20 should return BULLISH."""
        import sys
        mock_pta = MagicMock()
        mock_adx = pd.DataFrame({'ADX_14': [35.0] * 100})
        mock_pta.adx.return_value = mock_adx
        df = _make_ohlcv_trending_up(100)

        with patch.object(strategy, '_fetch_candles', return_value=df):
            with patch.dict(sys.modules, {'pandas_ta': mock_pta}):
                result = strategy._get_4h_trend('BTC')

        assert result == 'BULLISH'

    def test_bearish_trend_detected(self, strategy):
        """Price < EMA20 < EMA50 with ADX > 20 should return BEARISH."""
        import sys
        mock_pta = MagicMock()
        mock_adx = pd.DataFrame({'ADX_14': [35.0] * 100})
        mock_pta.adx.return_value = mock_adx
        df = _make_ohlcv_trending_down(100)

        with patch.object(strategy, '_fetch_candles', return_value=df):
            with patch.dict(sys.modules, {'pandas_ta': mock_pta}):
                result = strategy._get_4h_trend('ETH')

        assert result == 'BEARISH'

    def test_neutral_when_low_adx(self, strategy):
        """ADX < 20 should return NEUTRAL regardless of EMA alignment."""
        import sys
        mock_pta = MagicMock()
        mock_adx = pd.DataFrame({'ADX_14': [15.0] * 100})
        mock_pta.adx.return_value = mock_adx
        df = _make_ohlcv_trending_up(100)

        with patch.object(strategy, '_fetch_candles', return_value=df):
            with patch.dict(sys.modules, {'pandas_ta': mock_pta}):
                result = strategy._get_4h_trend('BTC')

        assert result == 'NEUTRAL'

    def test_cache_works(self, strategy):
        """Second call within 30 min returns cached result without fetching."""
        # Pre-populate cache
        strategy._4h_trend_cache['BTC'] = {
            'trend': 'BULLISH',
            'timestamp': datetime.now(),
        }

        with patch.object(strategy, '_fetch_candles') as mock_fetch:
            result = strategy._get_4h_trend('BTC')

        # Should not have fetched new data
        mock_fetch.assert_not_called()
        assert result == 'BULLISH'

    def test_cache_expires(self, strategy):
        """Call after 30 min fetches fresh data."""
        import sys

        # Set cache to 31 minutes ago
        strategy._4h_trend_cache['BTC'] = {
            'trend': 'BULLISH',
            'timestamp': datetime.now() - timedelta(minutes=31),
        }

        df = _make_ohlcv_trending_down(100)
        mock_pta = MagicMock()
        mock_adx = pd.DataFrame({'ADX_14': [35.0] * 100})
        mock_pta.adx.return_value = mock_adx
        with patch.object(strategy, '_fetch_candles', return_value=df) as mock_fetch:
            with patch.dict(sys.modules, {'pandas_ta': mock_pta}):
                result = strategy._get_4h_trend('BTC')

        # Should have fetched fresh data
        mock_fetch.assert_called_once()
        assert result == 'BEARISH'

    def test_error_returns_neutral(self, strategy):
        """On fetch error, returns NEUTRAL gracefully."""
        with patch.object(strategy, '_fetch_candles', side_effect=Exception("API down")):
            result = strategy._get_4h_trend('BTC')

        assert result == 'NEUTRAL'

    def test_insufficient_data_returns_neutral(self, strategy):
        """Less than 50 candles should return NEUTRAL."""
        df = _make_ohlcv_trending_up(30)  # Only 30 candles, need 50

        with patch.object(strategy, '_fetch_candles', return_value=df):
            result = strategy._get_4h_trend('BTC')

        assert result == 'NEUTRAL'

    def test_none_candles_returns_neutral(self, strategy):
        """None from candle fetch should return NEUTRAL."""
        with patch.object(strategy, '_fetch_candles', return_value=None):
            result = strategy._get_4h_trend('BTC')

        assert result == 'NEUTRAL'

    def test_different_symbols_cached_independently(self, strategy):
        """Different symbols should have independent cache entries."""
        strategy._4h_trend_cache['BTC'] = {
            'trend': 'BULLISH',
            'timestamp': datetime.now(),
        }
        strategy._4h_trend_cache['ETH'] = {
            'trend': 'BEARISH',
            'timestamp': datetime.now(),
        }

        assert strategy._get_4h_trend('BTC') == 'BULLISH'
        assert strategy._get_4h_trend('ETH') == 'BEARISH'
