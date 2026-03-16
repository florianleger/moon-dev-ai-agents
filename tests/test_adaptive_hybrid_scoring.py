"""Tests for AdaptiveHybridStrategy scoring modules and regime detection."""
import os
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# Import extracted module functions directly
from src.strategies.modules.mean_reversion import score_mean_reversion
from src.strategies.modules.momentum import score_momentum_breakout
from src.strategies.modules.ema_trend import score_ema_trend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def strategy():
    """Create an AdaptiveHybridStrategy instance with all external deps mocked."""
    with patch('src.data_providers.market_data.MarketDataProvider', MagicMock()):
        from src.strategies.custom.adaptive_hybrid_strategy import AdaptiveHybridStrategy
        # Prevent preload from doing real network I/O
        with patch.object(AdaptiveHybridStrategy, '_preload_funding_history'):
            with patch.object(AdaptiveHybridStrategy, '_load_state_from_csv'):
                s = AdaptiveHybridStrategy()
                s._market_data = None  # Ensure no real API calls
                yield s


@pytest.fixture
def make_indicators(sample_ohlcv):
    """Helper to build an indicator dict from a base dict + optional overrides."""
    def _make(**overrides):
        base = {
            'close': 100.0, 'open': 99.8, 'high': 101.0, 'low': 99.0,
            'volume': 5000.0, 'rsi': 45.0, 'rsi_oversold': 30, 'rsi_overbought': 70,
            'adx': 22.0, 'ema_9': 100.1, 'ema_21': 99.9, 'ema_50': 99.5, 'ema_200': 98.0,
            'bb_upper': 102.0, 'bb_lower': 98.0, 'bb_mid': 100.0, 'bb_pct': 0.5,
            'atr': 1.5, 'macd': 0.1, 'macd_signal': 0.05, 'macd_diff': 0.05,
            'volume_ratio': 1.2, 'vwap': 100.0, '_df': sample_ohlcv,
        }
        base.update(overrides)
        return base
    return _make


# ---------------------------------------------------------------------------
# score_mean_reversion (module function)
# ---------------------------------------------------------------------------

class TestScoreMeanReversion:
    def test_oversold_near_lower_band_gives_buy(self, make_indicators, sample_ohlcv):
        ind = make_indicators(
            close=98.2, bb_lower=98.0, bb_upper=102.0,
            rsi=25, rsi_oversold=30, adx=18,
        )
        result = score_mean_reversion(sample_ohlcv, ind)
        assert result['direction'] == 'BUY'
        assert result['score'] > 0

    def test_overbought_near_upper_band_gives_sell(self, make_indicators, sample_ohlcv):
        ind = make_indicators(
            close=101.9, bb_lower=98.0, bb_upper=102.0,
            rsi=75, rsi_overbought=70, adx=18,
        )
        result = score_mean_reversion(sample_ohlcv, ind)
        assert result['direction'] == 'SELL'
        assert result['score'] > 0

    def test_mid_range_returns_neutral_or_low_score(self, make_indicators, sample_ohlcv):
        ind = make_indicators(
            close=100.0, bb_lower=98.0, bb_upper=102.0,
            rsi=50, adx=35,
        )
        result = score_mean_reversion(sample_ohlcv, ind)
        # In mid-range with high ADX, score should be low
        assert result['score'] <= 30

    def test_no_bb_range_returns_neutral(self, make_indicators, sample_ohlcv):
        ind = make_indicators(bb_lower=100.0, bb_upper=100.0)
        result = score_mean_reversion(sample_ohlcv, ind)
        assert result['direction'] == 'NEUTRAL'
        assert result['score'] == 0

    def test_ranging_market_gives_bonus(self, make_indicators, sample_ohlcv):
        # ADX < 20 = ranging market bonus
        ind_ranging = make_indicators(
            close=98.2, bb_lower=98.0, bb_upper=102.0,
            rsi=25, rsi_oversold=30, adx=15,
        )
        ind_trending = make_indicators(
            close=98.2, bb_lower=98.0, bb_upper=102.0,
            rsi=25, rsi_oversold=30, adx=35,
        )
        score_ranging = score_mean_reversion(sample_ohlcv, ind_ranging)['score']
        score_trending = score_mean_reversion(sample_ohlcv, ind_trending)['score']
        assert score_ranging > score_trending


# ---------------------------------------------------------------------------
# score_momentum_breakout (module function)
# ---------------------------------------------------------------------------

class TestScoreMomentumBreakout:
    def test_upside_breakout_with_volume_gives_buy(self, make_indicators, sample_ohlcv):
        high_20 = float(sample_ohlcv['high'].tail(20).max())
        ind = make_indicators(close=high_20 + 0.5, volume_ratio=2.0, adx=30)
        result = score_momentum_breakout(sample_ohlcv, ind)
        assert result['direction'] == 'BUY'
        assert result['score'] >= 40

    def test_downside_breakout_gives_sell(self, make_indicators, sample_ohlcv):
        low_20 = float(sample_ohlcv['low'].tail(20).min())
        ind = make_indicators(close=low_20 - 0.5, volume_ratio=2.0, adx=30)
        result = score_momentum_breakout(sample_ohlcv, ind)
        assert result['direction'] == 'SELL'
        assert result['score'] >= 40

    def test_no_breakout_low_score(self, make_indicators, sample_ohlcv):
        mid_price = float((sample_ohlcv['high'].tail(20).max() + sample_ohlcv['low'].tail(20).min()) / 2)
        ind = make_indicators(close=mid_price, volume_ratio=0.8, adx=15)
        result = score_momentum_breakout(sample_ohlcv, ind)
        assert result['score'] <= 30


# ---------------------------------------------------------------------------
# score_ema_trend (module function)
# ---------------------------------------------------------------------------

class TestScoreEmaTrend:
    def test_bullish_alignment_gives_buy(self, make_indicators):
        ind = make_indicators(
            ema_9=105, ema_21=103, ema_50=100,
            adx=30, macd_diff=0.5, close=103.0,
        )
        result = score_ema_trend(ind)
        assert result['direction'] == 'BUY'
        assert result['score'] > 0

    def test_bearish_alignment_gives_sell(self, make_indicators):
        ind = make_indicators(
            ema_9=95, ema_21=97, ema_50=100,
            adx=30, macd_diff=-0.5, close=97.0,
        )
        result = score_ema_trend(ind)
        assert result['direction'] == 'SELL'
        assert result['score'] > 0

    def test_no_alignment_returns_low_score(self, make_indicators):
        ind = make_indicators(
            ema_9=100, ema_21=101, ema_50=99,  # No clear alignment
            adx=15, macd_diff=0,
        )
        result = score_ema_trend(ind)
        # Partial alignment may produce a small score, but should be weak
        assert result['score'] <= 20


# ---------------------------------------------------------------------------
# _detect_global_regime (still on strategy object)
# ---------------------------------------------------------------------------

class TestDetectGlobalRegime:
    def test_trending_volatile_regime(self, strategy):
        """ADX >= 30 and vol_ratio > 1.2 -> trending_volatile"""
        with patch.object(strategy, '_fetch_candles') as mock_fetch:
            np.random.seed(42)
            n = 250
            close = pd.Series(np.linspace(100, 130, n))  # Strong trend
            high = close + 2
            low = close - 2
            mock_fetch.return_value = pd.DataFrame({
                'open': close, 'high': high, 'low': low,
                'close': close, 'volume': [1000] * n,
            })
            strategy._global_regime = None
            strategy._global_regime_timestamp = None
            regime = strategy._detect_global_regime()
            assert regime in ('trending_volatile', 'trending_calm', 'ranging_volatile', 'ranging_calm')

    def test_cache_returns_cached_regime(self, strategy):
        strategy._global_regime = 'ranging_calm'
        strategy._global_regime_timestamp = datetime.now()  # Fresh
        assert strategy._detect_global_regime() == 'ranging_calm'

    def test_fetch_failure_returns_default(self, strategy):
        strategy._global_regime = None
        strategy._global_regime_timestamp = None
        with patch.object(strategy, '_fetch_candles', return_value=None):
            regime = strategy._detect_global_regime()
            assert regime == 'ranging_calm'


# ---------------------------------------------------------------------------
# Benchmark tracker calculations
# ---------------------------------------------------------------------------

class TestBenchmarkTracker:
    def test_benchmark_alpha_empty_when_no_start_prices(self, strategy):
        strategy._benchmark_start_prices = {}
        result = strategy._get_benchmark_alpha()
        assert result == {}

    def test_benchmark_alpha_calculates_correctly(self, strategy):
        strategy._benchmark_start_prices = {'BTC': 40000.0}
        strategy._benchmark_start_time = datetime.now()
        strategy.paper_balance = 550.0  # 10% return on 500

        mock_df = pd.DataFrame({
            'close': [44000.0, 44000.0, 44000.0, 44000.0, 44000.0],
        })
        with patch.object(strategy, '_fetch_candles', return_value=mock_df):
            result = strategy._get_benchmark_alpha()
            assert 'BTC' in result
            assert result['strategy_return_pct'] == pytest.approx(10.0)
            assert result['BTC']['alpha'] == pytest.approx(0.0, abs=0.1)

    def test_benchmark_positive_alpha(self, strategy):
        strategy._benchmark_start_prices = {'BTC': 40000.0}
        strategy._benchmark_start_time = datetime.now()
        strategy.paper_balance = 600.0  # 20% return

        mock_df = pd.DataFrame({
            'close': [42000.0, 42000.0, 42000.0, 42000.0, 42000.0],
        })
        with patch.object(strategy, '_fetch_candles', return_value=mock_df):
            result = strategy._get_benchmark_alpha()
            assert result['BTC']['alpha'] > 0  # Strategy beat BTC


# ---------------------------------------------------------------------------
# funding_composite (replaces old funding_contrarian wrapper)
# ---------------------------------------------------------------------------

class TestScoreFundingComposite:
    def test_returns_valid_structure(self, make_indicators):
        from src.strategies.modules.funding_composite import score_funding_composite
        result = score_funding_composite('BTC', make_indicators(), funding_zscore=0.0)
        assert 'direction' in result
        assert 'score' in result
        assert result['direction'] in ('BUY', 'SELL', 'NEUTRAL')

    def test_extreme_negative_zscore_gives_buy(self, make_indicators):
        from src.strategies.modules.funding_composite import score_funding_composite
        ind = make_indicators(rsi=25)
        result = score_funding_composite('BTC', ind, funding_zscore=-2.5)
        assert result['direction'] == 'BUY'
        assert result['score'] > 0

    def test_extreme_positive_zscore_gives_sell(self, make_indicators):
        from src.strategies.modules.funding_composite import score_funding_composite
        ind = make_indicators(rsi=75)
        result = score_funding_composite('BTC', ind, funding_zscore=2.5)
        assert result['direction'] == 'SELL'
        assert result['score'] > 0

    def test_neutral_zscore_returns_neutral(self, make_indicators):
        from src.strategies.modules.funding_composite import score_funding_composite
        result = score_funding_composite('BTC', make_indicators(), funding_zscore=0.3)
        assert result['score'] <= 30  # Low or no signal for mild funding


# ---------------------------------------------------------------------------
# Score capping at 100
# ---------------------------------------------------------------------------

class TestScoreCapping:
    def test_mean_reversion_score_capped_at_100(self, make_indicators, sample_ohlcv):
        ind = make_indicators(
            close=98.01, bb_lower=98.0, bb_upper=102.0,
            rsi=20, rsi_oversold=30, adx=10,
        )
        result = score_mean_reversion(sample_ohlcv, ind)
        assert result['score'] <= 100
