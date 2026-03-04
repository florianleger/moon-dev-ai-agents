"""Tests for RiskAgent: circuit breakers, correlation, recovery, state persistence."""
import json
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest


# Patch config values before importing RiskAgent
_CONFIG_DEFAULTS = {
    'PAPER_TRADING': True,
    'PAPER_TRADING_BALANCE': 500.0,
    'RISK_MAX_DRAWDOWN_PCT': 15,
    'RISK_MAX_DAILY_LOSS_USD': 30,
    'RISK_MAX_POSITIONS': 4,
    'ACTIVE_STRATEGY': 'adaptive_hybrid',
    'RISK_COOLING_OFF_HOURS': 4,
    'RISK_RECOVERY_SIZE_PCT': 50,
    'RISK_RECOVERY_DURATION_HOURS': 24,
    'MINIMUM_BALANCE_USD': 50,
    'CORRELATION_HIGH_THRESHOLD': 0.75,
    'CORRELATION_SIZING_FACTOR': 0.5,
}


@pytest.fixture(autouse=True)
def _patch_config():
    """Patch config imports so RiskAgent can be instantiated without real env."""
    with patch.dict('src.config.__dict__', _CONFIG_DEFAULTS):
        yield


@pytest.fixture
def risk_agent(temp_dir):
    """Create a RiskAgent with a temporary state file and no real strategy."""
    state_file_path = os.path.join(temp_dir, 'risk_state.json')
    with patch('src.agents.risk_agent.RISK_STATE_FILE', type(
            __import__('pathlib').Path())(state_file_path)):
        from src.agents.risk_agent import RiskAgent
        agent = RiskAgent()
        # Don't link any real strategy
        agent._strategy = None
        yield agent


# ---------------------------------------------------------------------------
# Minimum balance check
# ---------------------------------------------------------------------------

class TestMinimumBalance:
    def test_balance_below_minimum_pauses_trading(self, risk_agent):
        """When balance <= MINIMUM_BALANCE_USD, trading should be paused."""
        mock_strategy = MagicMock()
        mock_strategy.get_paper_status.return_value = {
            'paper_balance': 40.0,  # Below $50 minimum
            'open_positions': 0,
        }
        mock_strategy.close_all_paper_positions.return_value = []
        risk_agent._strategy = mock_strategy

        breached = risk_agent.run()
        assert breached is True
        assert risk_agent.trading_paused is True
        assert 'minimum' in risk_agent.pause_reason.lower() or 'below' in risk_agent.pause_reason.lower()

    def test_balance_above_minimum_does_not_pause(self, risk_agent):
        mock_strategy = MagicMock()
        mock_strategy.get_paper_status.return_value = {
            'paper_balance': 490.0,
            'open_positions': 0,
        }
        mock_strategy.paper_positions = {}
        risk_agent._strategy = mock_strategy
        # Set peak_balance = current balance so no HWM drawdown
        risk_agent.peak_balance = 490.0
        risk_agent._daily_start_balance = 490.0
        risk_agent._daily_date = datetime.now().date()

        breached = risk_agent.run()
        assert breached is False
        assert risk_agent.trading_paused is False


# ---------------------------------------------------------------------------
# HWM drawdown detection
# ---------------------------------------------------------------------------

class TestHWMDrawdown:
    def test_hwm_drawdown_breached_pauses_trading(self, risk_agent):
        risk_agent.peak_balance = 500.0
        mock_strategy = MagicMock()
        # 20% drawdown from peak (500 -> 400) exceeds 15% limit
        mock_strategy.get_paper_status.return_value = {
            'paper_balance': 400.0,
            'open_positions': 1,
        }
        mock_strategy.close_all_paper_positions.return_value = ['pos1']
        risk_agent._strategy = mock_strategy
        risk_agent._daily_start_balance = 400.0
        risk_agent._daily_date = datetime.now().date()

        breached = risk_agent.run()
        assert breached is True
        assert risk_agent.trading_paused is True


# ---------------------------------------------------------------------------
# Daily loss limit
# ---------------------------------------------------------------------------

class TestDailyLossLimit:
    def test_daily_loss_breached_pauses_for_day(self, risk_agent):
        mock_strategy = MagicMock()
        mock_strategy.get_paper_status.return_value = {
            'paper_balance': 460.0,
            'open_positions': 0,
        }
        mock_strategy.close_all_paper_positions.return_value = []
        risk_agent._strategy = mock_strategy
        # Simulate: started today at 500, now at 460 -> daily loss = -40 > -30 limit
        risk_agent._daily_date = datetime.now().date()
        risk_agent._daily_start_balance = 500.0

        breached = risk_agent.run()
        assert breached is True
        assert risk_agent.daily_pause is True
        assert risk_agent.daily_pause_date == datetime.now().date()


# ---------------------------------------------------------------------------
# Max positions
# ---------------------------------------------------------------------------

class TestMaxPositions:
    def test_max_positions_blocks_new_trades(self, risk_agent):
        mock_strategy = MagicMock()
        mock_strategy.get_paper_status.return_value = {
            'paper_balance': 500.0,
            'open_positions': 4,  # At max
        }
        risk_agent._strategy = mock_strategy

        assert risk_agent.is_trading_allowed() is False

    def test_under_max_positions_allows_trading(self, risk_agent):
        mock_strategy = MagicMock()
        mock_strategy.get_paper_status.return_value = {
            'paper_balance': 500.0,
            'open_positions': 2,
        }
        risk_agent._strategy = mock_strategy

        assert risk_agent.is_trading_allowed() is True


# ---------------------------------------------------------------------------
# Cooling-off period
# ---------------------------------------------------------------------------

class TestCoolingOff:
    def test_cooling_off_blocks_trading(self, risk_agent):
        risk_agent.trading_paused = True
        risk_agent.pause_timestamp = datetime.now() - timedelta(hours=1)  # Only 1h elapsed, need 4h

        assert risk_agent.is_trading_allowed() is False

    def test_daily_pause_resets_on_new_day(self, risk_agent):
        risk_agent.daily_pause = True
        risk_agent.daily_pause_date = datetime.now().date() - timedelta(days=1)

        # is_trading_allowed should reset the daily pause
        mock_strategy = MagicMock()
        mock_strategy.get_paper_status.return_value = {
            'paper_balance': 500.0,
            'open_positions': 0,
        }
        risk_agent._strategy = mock_strategy

        assert risk_agent.is_trading_allowed() is True
        assert risk_agent.daily_pause is False


# ---------------------------------------------------------------------------
# Recovery mode factor
# ---------------------------------------------------------------------------

class TestRecoveryMode:
    def test_recovery_mode_returns_reduced_factor(self, risk_agent):
        risk_agent.recovery_mode = True
        risk_agent.recovery_start = datetime.now() - timedelta(hours=2)
        factor = risk_agent.get_recovery_size_factor()
        # RISK_RECOVERY_SIZE_PCT = 50 -> factor = 0.5
        assert factor == pytest.approx(0.5)

    def test_recovery_mode_expires_after_duration(self, risk_agent):
        risk_agent.recovery_mode = True
        risk_agent.recovery_start = datetime.now() - timedelta(hours=25)  # > 24h
        factor = risk_agent.get_recovery_size_factor()
        assert factor == 1.0
        assert risk_agent.recovery_mode is False

    def test_no_recovery_mode_returns_full_size(self, risk_agent):
        risk_agent.recovery_mode = False
        assert risk_agent.get_recovery_size_factor() == 1.0


# ---------------------------------------------------------------------------
# Correlation sizing factor
# ---------------------------------------------------------------------------

class TestCorrelationSizing:
    def test_btc_eth_high_correlation_reduces_sizing(self, risk_agent):
        positions = [{'symbol': 'ETH'}]
        factor = risk_agent.get_correlation_sizing_factor('BTC', positions)
        # BTC-ETH correlation = 0.85 >= 0.75 threshold
        assert factor == pytest.approx(0.5)

    def test_uncorrelated_pair_full_sizing(self, risk_agent):
        positions = [{'symbol': 'UNKNOWN_COIN'}]
        factor = risk_agent.get_correlation_sizing_factor('BTC', positions)
        # Default correlation = 0.60 < 0.75 threshold
        assert factor == 1.0

    def test_no_existing_positions_full_sizing(self, risk_agent):
        factor = risk_agent.get_correlation_sizing_factor('BTC', [])
        assert factor == 1.0

    def test_correlation_table_symmetric(self, risk_agent):
        from src.agents.risk_agent import RiskAgent
        assert risk_agent._get_pair_correlation('BTC', 'ETH') == risk_agent._get_pair_correlation('ETH', 'BTC')
        assert risk_agent._get_pair_correlation('BTC', 'SOL') == risk_agent._get_pair_correlation('SOL', 'BTC')

    def test_same_symbol_correlation_is_one(self, risk_agent):
        assert risk_agent._get_pair_correlation('BTC', 'BTC') == 1.0


# ---------------------------------------------------------------------------
# State persistence (JSON save/load)
# ---------------------------------------------------------------------------

class TestStatePersistence:
    def test_save_and_load_roundtrip(self, temp_dir):
        state_file_path = os.path.join(temp_dir, 'risk_state.json')
        from pathlib import Path
        with patch('src.agents.risk_agent.RISK_STATE_FILE', Path(state_file_path)):
            from src.agents.risk_agent import RiskAgent
            agent = RiskAgent()
            agent.peak_balance = 600.0
            agent.trading_paused = True
            agent.pause_reason = "test pause"
            agent.pause_timestamp = datetime(2024, 6, 15, 12, 0, 0)
            agent.recovery_mode = True
            agent.recovery_start = datetime(2024, 6, 15, 10, 0, 0)
            agent._save_state()

            # Verify file exists
            assert os.path.exists(state_file_path)

            # Create new agent that loads state
            agent2 = RiskAgent()
            assert agent2.peak_balance == 600.0
            assert agent2.trading_paused is True
            assert agent2.pause_reason == "test pause"
            assert agent2.recovery_mode is True

    def test_load_missing_state_file_is_safe(self, temp_dir):
        state_file_path = os.path.join(temp_dir, 'nonexistent.json')
        from pathlib import Path
        with patch('src.agents.risk_agent.RISK_STATE_FILE', Path(state_file_path)):
            from src.agents.risk_agent import RiskAgent
            agent = RiskAgent()
            # Should not crash, use defaults
            assert agent.trading_paused is False
