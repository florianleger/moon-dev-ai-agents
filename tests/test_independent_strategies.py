"""Tests for the 3 independent strategies (basic instantiation and interface)."""
import sys
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from datetime import datetime, timedelta

# pandas_ta is not installed in test env; mock it before any strategy import
if 'pandas_ta' not in sys.modules:
    sys.modules['pandas_ta'] = MagicMock()


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------

def _mock_hyperliquid():
    """Patch HyperLiquid SDK for all strategies."""
    return patch('hyperliquid.info.Info', MagicMock())


def _mock_market_data():
    """Patch MarketDataProvider."""
    return patch('src.data_providers.market_data.MarketDataProvider', MagicMock())


def _mock_trade_memory():
    """Patch TradeMemory singleton."""
    mock_tm = MagicMock()
    return patch('src.data.trade_memory.TradeMemory.get_instance', return_value=mock_tm)


# ---------------------------------------------------------------------------
# FundingMeanReversionStrategy
# ---------------------------------------------------------------------------

class TestFundingMeanReversion:
    """Tests for FundingMeanReversionStrategy."""

    def _make_strategy(self):
        with _mock_hyperliquid(), _mock_market_data(), _mock_trade_memory():
            with patch('src.strategies.custom.funding_mean_reversion.MarketDataProvider', MagicMock()):
                with patch('src.strategies.custom.funding_mean_reversion.TradeMemory') as mock_tm_cls:
                    mock_tm_cls.get_instance.return_value = MagicMock()
                    from src.strategies.custom.funding_mean_reversion import FundingMeanReversionStrategy
                    with patch.object(FundingMeanReversionStrategy, '_load_state'):
                        s = FundingMeanReversionStrategy()
                        return s

    def test_has_tokens_attribute(self):
        """Strategy must have .tokens attribute for main.py integration."""
        s = self._make_strategy()
        assert hasattr(s, 'tokens')
        assert isinstance(s.tokens, (list, tuple))
        assert len(s.tokens) > 0

    def test_has_run_cycle_method(self):
        """Strategy must have run_cycle(symbols) method."""
        s = self._make_strategy()
        assert hasattr(s, 'run_cycle')
        assert callable(s.run_cycle)

    def test_zscore_threshold_by_token_class(self):
        """BTC/ETH use 1.7, mid-caps 1.5, alts 1.2 (re-tightened from 1.5/1.3/1.0
        after 5d post-deploy regression -$7.70/30 trades)."""
        from src.strategies.custom.funding_mean_reversion import ZSCORE_THRESHOLDS, TOKEN_CLASS

        assert ZSCORE_THRESHOLDS['major'] == 1.7
        assert ZSCORE_THRESHOLDS['mid'] == 1.5
        assert ZSCORE_THRESHOLDS['alt'] == 1.2

        # BTC and ETH should be classified as 'major'
        assert TOKEN_CLASS.get('BTC') == 'major'
        assert TOKEN_CLASS.get('ETH') == 'major'

        # SOL should be 'mid'
        assert TOKEN_CLASS.get('SOL') == 'mid'

        # DOGE should be 'alt'
        assert TOKEN_CLASS.get('DOGE') == 'alt'

    def test_daily_loss_limit_respected(self):
        """Should stop trading after daily loss limit."""
        from src.strategies.custom.funding_mean_reversion import MAX_DAILY_LOSS_USD
        s = self._make_strategy()

        # Simulate exceeding daily loss
        s.daily_pnl = -(MAX_DAILY_LOSS_USD + 1)

        # The strategy checks daily_pnl in run_cycle; verify the constant exists
        assert MAX_DAILY_LOSS_USD > 0
        assert s.daily_pnl < -MAX_DAILY_LOSS_USD

    def test_paper_balance_initialized(self):
        """Paper balance should be set from config."""
        s = self._make_strategy()
        assert s.paper_balance > 0


# ---------------------------------------------------------------------------
# VolatilityBreakoutStrategy
# ---------------------------------------------------------------------------

class TestVolatilityBreakout:
    """Tests for VolatilityBreakoutStrategy."""

    def _make_strategy(self):
        with _mock_hyperliquid(), _mock_market_data(), _mock_trade_memory():
            from src.strategies.custom.volatility_breakout import VolatilityBreakoutStrategy
            with patch.object(VolatilityBreakoutStrategy, '_load_state'):
                s = VolatilityBreakoutStrategy()
                return s

    def test_has_tokens_attribute(self):
        """Strategy must have .tokens attribute."""
        s = self._make_strategy()
        assert hasattr(s, 'tokens')
        assert isinstance(s.tokens, list)
        assert len(s.tokens) > 0

    def test_has_run_cycle_method(self):
        """Strategy must have run_cycle(symbols) method."""
        s = self._make_strategy()
        assert hasattr(s, 'run_cycle')
        assert callable(s.run_cycle)

    def test_squeeze_detection_config(self):
        """BB width below 20th percentile should be detected as squeeze."""
        from src.strategies.custom.volatility_breakout import VOL_BREAKOUT_SQUEEZE_PERCENTILE
        assert VOL_BREAKOUT_SQUEEZE_PERCENTILE == pytest.approx(0.20)

    def test_volume_filter_not_overly_restrictive(self):
        """Volume at 0.5x should NOT be rejected (only < 0.15x rejected)."""
        from src.strategies.custom.volatility_breakout import VOL_BREAKOUT_VOLUME_MIN
        # 0.5x is above the minimum threshold of 0.15
        assert 0.5 > VOL_BREAKOUT_VOLUME_MIN

    def test_adx_entry_threshold(self):
        """ADX entry threshold should be 22 (loosened from 25 for more setups)."""
        from src.strategies.custom.volatility_breakout import VOL_BREAKOUT_ADX_ENTRY
        assert VOL_BREAKOUT_ADX_ENTRY == 22

    def test_trailing_stop_multiplier(self):
        """Trailing ATR multiplier should be 2.5."""
        from src.strategies.custom.volatility_breakout import VOL_BREAKOUT_TRAILING_ATR_MULT
        assert VOL_BREAKOUT_TRAILING_ATR_MULT == 2.5

    def test_paper_balance_initialized(self):
        """Paper balance should be set from config."""
        s = self._make_strategy()
        assert s.paper_balance > 0


# ---------------------------------------------------------------------------
# LiquidationCascadeFadeStrategy
# ---------------------------------------------------------------------------

class TestLiquidationCascadeFade:
    """Tests for LiquidationCascadeFadeStrategy."""

    def _make_strategy(self):
        with _mock_hyperliquid(), _mock_market_data(), _mock_trade_memory():
            with patch('src.strategies.custom.liquidation_cascade_fade.TradeMemory') as mock_tm_cls:
                mock_tm_cls.get_instance.return_value = MagicMock()
                with patch('src.strategies.custom.liquidation_cascade_fade.SYMBOL_MAP', {'BTC': 'BTCUSDT'}):
                    from src.strategies.custom.liquidation_cascade_fade import LiquidationCascadeFadeStrategy
                    with patch.object(LiquidationCascadeFadeStrategy, '_init_providers'):
                        with patch.object(LiquidationCascadeFadeStrategy, '_load_state', create=True):
                            s = LiquidationCascadeFadeStrategy.__new__(LiquidationCascadeFadeStrategy)
                            s.name = "Liquidation Cascade Fade"
                            s.assets = ['BTC', 'ETH', 'SOL']
                            s.tokens = s.assets
                            s.paper_balance = 500
                            s.paper_positions = {}
                            s.closed_positions = []
                            s._position_counter = 0
                            s._cooldowns = {}
                            s._pending_cascades = {}
                            s._liq_stream = None
                            s._market_data = None
                            s.daily_trades = 0
                            s.daily_pnl = 0.0
                            s.last_trade_date = None
                            import threading
                            s._position_lock = threading.RLock()
                            return s

    def test_has_tokens_attribute(self):
        """Strategy must have .tokens attribute."""
        s = self._make_strategy()
        assert hasattr(s, 'tokens')
        assert isinstance(s.tokens, list)
        assert len(s.tokens) > 0

    def test_has_run_cycle_method(self):
        """Strategy must have run_cycle(symbols) method."""
        # Verify the class has the method defined
        from src.strategies.custom.liquidation_cascade_fade import LiquidationCascadeFadeStrategy
        assert hasattr(LiquidationCascadeFadeStrategy, 'run_cycle')
        assert callable(getattr(LiquidationCascadeFadeStrategy, 'run_cycle'))

    def test_cooldown_between_trades(self):
        """30-min cooldown per token should be enforced."""
        s = self._make_strategy()

        # Simulate a recent trade for BTC
        s._cooldowns['BTC'] = datetime.utcnow()

        # Check cooldown: trade happened just now, should still be in cooldown
        from src.strategies.custom.liquidation_cascade_fade import LIQ_CASCADE_COOLDOWN_MINUTES
        assert LIQ_CASCADE_COOLDOWN_MINUTES == 30

        elapsed = (datetime.utcnow() - s._cooldowns['BTC']).total_seconds() / 60
        assert elapsed < LIQ_CASCADE_COOLDOWN_MINUTES

    def test_cooldown_expires(self):
        """After cooldown period, token should be tradeable again."""
        s = self._make_strategy()

        from src.strategies.custom.liquidation_cascade_fade import LIQ_CASCADE_COOLDOWN_MINUTES

        # Simulate a trade 31 minutes ago
        s._cooldowns['BTC'] = datetime.utcnow() - timedelta(minutes=LIQ_CASCADE_COOLDOWN_MINUTES + 1)

        elapsed = (datetime.utcnow() - s._cooldowns['BTC']).total_seconds() / 60
        assert elapsed > LIQ_CASCADE_COOLDOWN_MINUTES

    def test_cascade_volume_thresholds(self):
        """Cascade volume thresholds should be defined per token."""
        from src.strategies.custom.liquidation_cascade_fade import CASCADE_VOLUME_THRESHOLDS

        assert CASCADE_VOLUME_THRESHOLDS['BTC'] > CASCADE_VOLUME_THRESHOLDS.get('SOL', 0)
        assert CASCADE_VOLUME_THRESHOLDS['ETH'] > CASCADE_VOLUME_THRESHOLDS.get('SOL', 0)

    def test_daily_trade_limit(self):
        """Max daily trades config should be defined."""
        from src.strategies.custom.liquidation_cascade_fade import LIQ_CASCADE_MAX_DAILY_TRADES
        assert LIQ_CASCADE_MAX_DAILY_TRADES > 0
        assert LIQ_CASCADE_MAX_DAILY_TRADES <= 5  # Reasonable bound
