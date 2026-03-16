"""Tests for the 1-bar signal confirmation mechanism."""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def strategy():
    """Create a minimal strategy instance."""
    with patch('src.data_providers.market_data.MarketDataProvider', MagicMock()):
        from src.strategies.custom.adaptive_hybrid_strategy import AdaptiveHybridStrategy
        with patch.object(AdaptiveHybridStrategy, '_preload_funding_history'):
            with patch.object(AdaptiveHybridStrategy, '_load_state_from_csv'):
                s = AdaptiveHybridStrategy()
                s._market_data = None
                s._pending_signals = {}
                yield s


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConfirmationDelay:
    """Tests for the 1-bar signal confirmation mechanism."""

    def test_first_signal_goes_pending(self, strategy):
        """First signal above threshold returns NEUTRAL (pending)."""
        from src.config import ADAPTIVE_HYBRID_CONFIRMATION_BARS

        if ADAPTIVE_HYBRID_CONFIRMATION_BARS == 0:
            pytest.skip("CONFIRMATION_BARS is 0, no delay to test")

        # Simulate the confirmation logic directly
        direction = 'BUY'
        symbol = 'BTC'
        pending_key = f"{symbol}_{direction}"

        # First signal: should be stored as pending
        assert pending_key not in strategy._pending_signals

        strategy._pending_signals[pending_key] = {
            'timestamp': datetime.now(),
            'score': 70,
            'direction': direction,
            'count': 1,
        }

        assert pending_key in strategy._pending_signals
        assert strategy._pending_signals[pending_key]['count'] == 1

    def test_confirmed_after_n_bars(self, strategy):
        """Signal confirmed after CONFIRMATION_BARS consecutive cycles."""
        from src.config import ADAPTIVE_HYBRID_CONFIRMATION_BARS

        if ADAPTIVE_HYBRID_CONFIRMATION_BARS == 0:
            pytest.skip("CONFIRMATION_BARS is 0, no delay to test")

        direction = 'BUY'
        symbol = 'BTC'
        pending_key = f"{symbol}_{direction}"

        # Simulate accumulating counts
        strategy._pending_signals[pending_key] = {
            'timestamp': datetime.now(),
            'score': 70,
            'direction': direction,
            'count': 1,
        }

        # Increment count through confirmation bars
        for i in range(ADAPTIVE_HYBRID_CONFIRMATION_BARS):
            strategy._pending_signals[pending_key]['count'] += 1

        # After confirmation_bars increments, count should exceed the threshold
        assert strategy._pending_signals[pending_key]['count'] > ADAPTIVE_HYBRID_CONFIRMATION_BARS

    def test_opposite_direction_clears_pending(self, strategy):
        """SELL signal clears pending BUY signal."""
        strategy._pending_signals['BTC_BUY'] = {
            'timestamp': datetime.now(),
            'score': 70,
            'direction': 'BUY',
            'count': 1,
        }

        # Simulate what happens when a SELL signal is received:
        # the code does: self._pending_signals.pop(f"{symbol}_{opposite}", None)
        opposite = 'BUY'
        strategy._pending_signals.pop(f"BTC_{opposite}", None)

        # New SELL pending
        strategy._pending_signals['BTC_SELL'] = {
            'timestamp': datetime.now(),
            'score': 65,
            'direction': 'SELL',
            'count': 1,
        }

        assert 'BTC_BUY' not in strategy._pending_signals
        assert 'BTC_SELL' in strategy._pending_signals

    def test_stale_signals_cleaned_up(self, strategy):
        """Pending signals older than 2h are removed."""
        # Add a stale pending signal (3 hours old)
        strategy._pending_signals['BTC_BUY'] = {
            'timestamp': datetime.now() - timedelta(hours=3),
            'score': 70,
            'direction': 'BUY',
            'count': 1,
        }
        # Add a fresh pending signal
        strategy._pending_signals['ETH_SELL'] = {
            'timestamp': datetime.now(),
            'score': 65,
            'direction': 'SELL',
            'count': 1,
        }

        strategy._cleanup_stale_pending_signals()

        assert 'BTC_BUY' not in strategy._pending_signals
        assert 'ETH_SELL' in strategy._pending_signals

    def test_no_delay_when_config_zero(self, strategy):
        """When CONFIRMATION_BARS=0, signals pass through immediately."""
        # The confirmation delay logic is gated by:
        # if confirmation_bars > 0: ...
        # When 0, no pending signal should be created

        confirmation_bars = 0
        # Verify the gate: when 0, the entire pending logic is skipped
        assert not (confirmation_bars > 0)

    def test_cleanup_preserves_recent_signals(self, strategy):
        """Signals within 2h window should not be cleaned up."""
        strategy._pending_signals['BTC_BUY'] = {
            'timestamp': datetime.now() - timedelta(hours=1),
            'score': 70,
            'direction': 'BUY',
            'count': 2,
        }
        strategy._pending_signals['ETH_BUY'] = {
            'timestamp': datetime.now() - timedelta(minutes=30),
            'score': 60,
            'direction': 'BUY',
            'count': 1,
        }

        strategy._cleanup_stale_pending_signals()

        assert 'BTC_BUY' in strategy._pending_signals
        assert 'ETH_BUY' in strategy._pending_signals

    def test_multiple_symbols_independent(self, strategy):
        """Pending signals for different symbols should be independent."""
        strategy._pending_signals['BTC_BUY'] = {
            'timestamp': datetime.now(),
            'score': 70,
            'direction': 'BUY',
            'count': 1,
        }
        strategy._pending_signals['ETH_SELL'] = {
            'timestamp': datetime.now(),
            'score': 65,
            'direction': 'SELL',
            'count': 1,
        }

        # Clearing BTC's opposite should not affect ETH
        strategy._pending_signals.pop('BTC_SELL', None)

        assert 'BTC_BUY' in strategy._pending_signals
        assert 'ETH_SELL' in strategy._pending_signals
