"""Tests for the Jun 2026 strategy tuning (plan Agent B).

Covers:
- B1: strategy enable/disable flags (OTE off, FundingMR off, LiqCascade on)
- B2/B3: vol_breakout prudent extension (cap 4, ADX max 32, widened universe)
- B4: adaptive_hybrid R:R repair (breakeven 2.5 ATR, single scale-out level,
      HOLD_TP_CHECK 12h, stagnation exit 8h / 0.3 ATR)
- B5: simplified sizing (fixed 1.2% risk, leverage-free, no Kelly / score /
      vol-target stacking)
- B6: calibration guardrail reconciliation (max 53 = runtime clamp) and
      AVOID_HOURS minimum sample
- B7: calibration agent corrected PnL (total_pnl + partials - entry fee)
"""
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# pandas_ta is not installed in the test env; mock it before any strategy import
# (same shim as tests/test_independent_strategies.py)
if 'pandas_ta' not in sys.modules:
    sys.modules['pandas_ta'] = MagicMock()

from src import config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def strategy():
    """AdaptiveHybridStrategy with external deps mocked (same pattern as scoring tests)."""
    with patch('src.data_providers.market_data.MarketDataProvider', MagicMock()):
        from src.strategies.custom.adaptive_hybrid_strategy import AdaptiveHybridStrategy
        with patch.object(AdaptiveHybridStrategy, '_preload_funding_history'):
            with patch.object(AdaptiveHybridStrategy, '_load_state_from_csv'):
                s = AdaptiveHybridStrategy()
                s._market_data = None
                s.paper_balance = 500.0
                s.peak_balance = 500.0
                s.paper_positions = {}
                s._risk_agent = None
                yield s


def make_signal(symbol='BTC', direction='BUY', score=70, sl_pct=3.0, price=50000.0):
    return {
        'token': symbol,
        'direction': direction,
        'signal': 0.9,
        'metadata': {
            'current_price': price,
            'stop_loss_pct': sl_pct,
            'take_profit_pct': sl_pct * 2,
            'atr': price * 0.01,
            'score': score,
        },
    }


# ---------------------------------------------------------------------------
# B1 — Strategy flags
# ---------------------------------------------------------------------------

class TestStrategyFlags:
    def test_ote_scalp_disabled(self):
        assert config.OTE_SCALP_ENABLED is False

    def test_funding_mr_disabled(self):
        assert config.FUNDING_MR_ENABLED is False

    def test_liq_cascade_enabled(self):
        assert config.LIQ_CASCADE_ENABLED is True


# ---------------------------------------------------------------------------
# B2/B3 — Volatility Breakout prudent extension
# ---------------------------------------------------------------------------

class TestVolBreakoutTuning:
    def test_daily_cap_raised_to_4(self):
        from src.strategies.custom.volatility_breakout import VOL_BREAKOUT_MAX_DAILY_TRADES
        assert VOL_BREAKOUT_MAX_DAILY_TRADES == 4
        assert config.VOL_BREAKOUT_MAX_DAILY_TRADES == 4

    def test_universe_extended_with_hl_mids(self):
        from src.strategies.custom.volatility_breakout import VOL_BREAKOUT_SYMBOLS
        for sym in ['DOGE', 'ADA', 'LTC', 'ARB', 'OP', 'INJ']:
            assert sym in VOL_BREAKOUT_SYMBOLS
        # Existing universe preserved
        for sym in ['BTC', 'ETH', 'SOL', 'LINK', 'SUI']:
            assert sym in VOL_BREAKOUT_SYMBOLS
        assert VOL_BREAKOUT_SYMBOLS == list(config.VOL_BREAKOUT_TOKENS)

    def test_tuned_params_untouched(self):
        """Edge is fragile (n=29): squeeze/volume/trailing/risk must not change."""
        import src.strategies.custom.volatility_breakout as vb
        assert vb.VOL_BREAKOUT_SQUEEZE_PERCENTILE == pytest.approx(0.20)
        assert vb.VOL_BREAKOUT_VOLUME_MULT == pytest.approx(2.0)
        assert vb.VOL_BREAKOUT_TRAILING_ATR_MULT == pytest.approx(2.5)
        assert vb.VOL_BREAKOUT_RISK_PCT == pytest.approx(0.015)

    def test_adx_max_value(self):
        from src.strategies.custom.volatility_breakout import VOL_BREAKOUT_ADX_MAX
        assert VOL_BREAKOUT_ADX_MAX == 32
        assert config.VOL_BREAKOUT_ADX_MAX == 32

    @pytest.mark.parametrize('adx_value,expect_setup', [(28.0, True), (35.0, False)])
    def test_compute_setup_rejects_extended_adx(self, monkeypatch, adx_value, expect_setup):
        """ADX > VOL_BREAKOUT_ADX_MAX must reject the setup (late entry)."""
        import src.strategies.custom.volatility_breakout as vb

        with patch.object(vb.VolatilityBreakoutStrategy, '_load_state'):
            strat = vb.VolatilityBreakoutStrategy()

        n = 200
        df1h = pd.DataFrame({
            'open': [100.0] * n, 'high': [101.0] * n, 'low': [99.0] * n,
            'close': [100.0] * (n - 1) + [120.0],          # breakout above BBU
            'volume': [100.0] * (n - 1) + [300.0],         # volume spike 3x
        })
        df4h = pd.DataFrame({
            'open': [100.0] * 30, 'high': [101.0] * 30, 'low': [99.0] * 30,
            'close': [100.0] * 30, 'volume': [100.0] * 30,
        })

        def fake_fetch(symbol, interval='1h', candles=200):
            return df1h if interval == '1h' else df4h
        monkeypatch.setattr(strat, '_fetch_candles', fake_fetch)

        # Squeeze on the 2 pre-breakout bars, wide elsewhere; BBU stays at 105
        # so the last close (120) is a breakout.
        width = pd.Series([10.0] * n)
        width.iloc[-3] = width.iloc[-2] = 0.5
        bbm = pd.Series([100.0] * n)
        bbl = bbm - width / 2
        bbu = bbm + width / 2
        fake_bb = pd.DataFrame({'BBU_20_2.0': bbu, 'BBL_20_2.0': bbl, 'BBM_20_2.0': bbm})

        monkeypatch.setattr(vb.ta, 'bbands', lambda *a, **k: fake_bb)
        monkeypatch.setattr(vb.ta, 'adx', lambda *a, **k: pd.DataFrame({'ADX_14': [adx_value] * n}))
        monkeypatch.setattr(vb.ta, 'ema', lambda *a, **k: pd.Series([50.0] * 30))   # below close -> BUY ok
        monkeypatch.setattr(vb.ta, 'atr', lambda *a, **k: pd.Series([2.0] * n))

        setup = strat._compute_setup('BTC')
        if expect_setup:
            assert setup is not None and setup['direction'] == 'BUY'
        else:
            assert setup is None


# ---------------------------------------------------------------------------
# B4 — R:R repair (config) + stagnation exit (behavior)
# ---------------------------------------------------------------------------

class TestRRRepairConfig:
    def test_breakeven_moved_to_2_5_atr(self):
        be_levels = [l for l in config.ADAPTIVE_HYBRID_TRAILING_LEVELS if l.get('breakeven')]
        assert len(be_levels) == 1
        assert be_levels[0]['activate_atr'] == pytest.approx(2.5)

    def test_first_scale_out_level_removed(self):
        levels = config.ADAPTIVE_HYBRID_SCALE_OUT_LEVELS
        assert len(levels) == 1
        assert levels[0]['tp_pct'] == pytest.approx(0.70)

    def test_hold_tp_check_12h(self):
        assert config.ADAPTIVE_HYBRID_HOLD_TP_CHECK_HOURS == 12

    def test_stagnation_config(self):
        assert config.ADAPTIVE_HYBRID_STAGNATION_HOURS == 8
        assert config.ADAPTIVE_HYBRID_STAGNATION_ATR == pytest.approx(0.3)


def _flat_candles(price, rows=20):
    """Flat candles: TR=0 so the periodic ATR refresh in monitoring is a no-op."""
    return pd.DataFrame({
        'open': [price] * rows, 'high': [price] * rows,
        'low': [price] * rows, 'close': [price] * rows,
        'volume': [100.0] * rows,
    })


def _insert_position(strategy, hours_ago, entry=50000.0, atr=500.0):
    import time as _t
    trade = {
        'position_id': 'AH_BTC_TEST_1',
        'symbol': 'BTC', 'direction': 'BUY',
        'entry_price': entry, 'position_size': 100.0, 'leverage': 3,
        'stop_loss': entry * 0.94, 'take_profit': entry * 1.12,
        'sl_pct': 6.0, 'tp_pct': 12.0, 'atr': atr,
        'entry_fee': 0.045, 'entry_time': datetime.now() - timedelta(hours=hours_ago),
        'scale_out_level': 0, 'partial_pnl_realized': 0.0,
        'last_atr_update': _t.time(),
        'status': 'OPEN',
    }
    strategy.paper_positions['AH_BTC_TEST_1'] = trade
    return trade


@pytest.fixture
def quiet_close(monkeypatch):
    """Silence all I/O performed when a paper position closes."""
    import src.strategies.custom.adaptive_hybrid_strategy as ahs
    monkeypatch.setattr(ahs, 'ADAPTIVE_HYBRID_LLM_LEARNER', False)
    monkeypatch.setattr('src.utils.alerting.get_alert_manager', lambda: MagicMock())
    yield


class TestStagnationExit:
    def _run_monitor(self, strategy, monkeypatch, current_price):
        import src.strategies.custom.adaptive_hybrid_strategy as ahs
        monkeypatch.setattr(ahs, 'ADAPTIVE_HYBRID_USE_REALTIME_PRICE', False)
        monkeypatch.setattr(strategy, '_fetch_candles',
                            lambda symbol, interval='15m', candles=5: _flat_candles(current_price))
        monkeypatch.setattr(strategy, '_get_benchmark_alpha', lambda: {})
        monkeypatch.setattr(strategy, '_log_closed_trade', lambda t: None)
        monkeypatch.setattr(strategy, '_update_position_status_in_csv', lambda pid, t: None)
        return strategy.monitor_paper_positions()

    def test_stagnant_position_closed_after_8h(self, strategy, monkeypatch, quiet_close):
        _insert_position(strategy, hours_ago=9, entry=50000.0, atr=500.0)
        # Price within 0.3 ATR (150) of entry after 9h -> stagnation
        closed = self._run_monitor(strategy, monkeypatch, current_price=50050.0)
        assert len(closed) == 1
        assert closed[0]['close_reason'] == 'STAGNATION_EXIT'

    def test_moving_position_not_closed(self, strategy, monkeypatch, quiet_close):
        _insert_position(strategy, hours_ago=9, entry=50000.0, atr=500.0)
        # Price 1 ATR above entry -> NOT stagnant (and below all exit triggers)
        closed = self._run_monitor(strategy, monkeypatch, current_price=50500.0)
        assert closed == []
        assert 'AH_BTC_TEST_1' in strategy.paper_positions

    def test_young_position_not_closed(self, strategy, monkeypatch, quiet_close):
        _insert_position(strategy, hours_ago=5, entry=50000.0, atr=500.0)
        closed = self._run_monitor(strategy, monkeypatch, current_price=50050.0)
        assert closed == []

    def test_tp_check_fires_at_12h(self, strategy, monkeypatch, quiet_close):
        """After 12h with <50% of TP path done, the time exit closes the trade."""
        _insert_position(strategy, hours_ago=13, entry=50000.0, atr=500.0)
        # +1 ATR = far from stagnation but only ~8% of TP path
        closed = self._run_monitor(strategy, monkeypatch, current_price=50500.0)
        assert len(closed) == 1
        assert closed[0]['close_reason'] == 'TIME_EXIT_24H'


# ---------------------------------------------------------------------------
# B5 — Simplified sizing
# ---------------------------------------------------------------------------

class TestSimplifiedSizing:
    def test_fixed_risk_pct_is_1_2(self):
        """Risk is no longer multiplied by leverage: RISK_PCT is the real
        worst-case loss at SL (1.2% = old 0.4% x lev 3, same notionals)."""
        assert config.ADAPTIVE_HYBRID_RISK_PCT == pytest.approx(0.012)

    def test_size_formula_with_cap(self, strategy):
        """risk/sl_fraction (NO leverage), capped at MAX_POSITION_PCT of balance."""
        trade = strategy._prepare_trade(make_signal(sl_pct=3.0))
        assert trade is not None
        # risk = 500*0.012 = $6 ; 6/0.03 = $200 -> capped at 25% of 500 = $125
        assert trade['position_size'] == pytest.approx(125.0, abs=0.5)

    def test_wider_sl_gives_smaller_size(self, strategy):
        trade = strategy._prepare_trade(make_signal(sl_pct=8.0))
        assert trade is not None
        # 6/0.08 = $75, below the cap
        assert trade['position_size'] == pytest.approx(75.0, abs=0.5)

    def test_loss_at_sl_never_exceeds_risk_budget(self, strategy):
        """Worst-case loss = notional x sl_fraction must be <= balance x RISK_PCT
        for ALL stop widths (the old `* leverage` made it 2-3x the budget and
        the 25% cap inverted the risk profile: wide SL = more dollars at risk)."""
        budget = 500.0 * config.ADAPTIVE_HYBRID_RISK_PCT
        for sl in (0.7, 1.8, 2.4, 3.0, 5.0, 8.0, 11.6):
            trade = strategy._prepare_trade(make_signal(sl_pct=sl))
            assert trade is not None
            loss_at_sl = trade['position_size'] * sl / 100
            assert loss_at_sl <= budget + 0.01, (
                f"sl_pct={sl}: loss at SL ${loss_at_sl:.2f} exceeds budget ${budget:.2f}"
            )

    def test_size_independent_of_score(self, strategy):
        low = strategy._prepare_trade(make_signal(score=45))
        high = strategy._prepare_trade(make_signal(score=90))
        assert low['position_size'] == pytest.approx(high['position_size'])

    def test_notional_in_target_range(self, strategy):
        """Median prod notional was $25 — new sizing must land ~$60-125."""
        for sl in (3.0, 5.0, 8.0):
            trade = strategy._prepare_trade(make_signal(sl_pct=sl))
            assert 60 <= trade['position_size'] <= 125.5, (
                f"sl_pct={sl}: notional ${trade['position_size']} outside target range"
            )

    def test_drawdown_scaling_still_applied(self, strategy):
        risk_agent = MagicMock()
        risk_agent.get_recovery_size_factor.return_value = 0.5
        risk_agent.get_correlation_sizing_factor.return_value = 1.0
        strategy._risk_agent = risk_agent
        trade = strategy._prepare_trade(make_signal(sl_pct=3.0))
        assert trade['position_size'] == pytest.approx(62.5, abs=0.5)

    def test_correlation_factor_still_applied(self, strategy):
        risk_agent = MagicMock()
        risk_agent.get_recovery_size_factor.return_value = 1.0
        risk_agent.get_correlation_sizing_factor.return_value = 0.5
        strategy._risk_agent = risk_agent
        trade = strategy._prepare_trade(make_signal(sl_pct=3.0))
        assert trade['position_size'] == pytest.approx(62.5, abs=0.5)

    def test_no_kelly_or_multiplier_stacking_in_source(self):
        path = os.path.join(os.path.dirname(__file__), '..',
                            'src', 'strategies', 'custom', 'adaptive_hybrid_strategy.py')
        with open(path) as f:
            source = f.read()
        for forbidden in ('kelly_size_pct', 'score_exposure', 'strength_multiplier',
                          'vol_target_size', 'WEEKEND_SIZE_REDUCTION'):
            assert forbidden not in source, f"'{forbidden}' still present in sizing"


# ---------------------------------------------------------------------------
# B6 — Calibration guardrails & AVOID_HOURS
# ---------------------------------------------------------------------------

class TestCalibrationGuardrails:
    def test_guardrail_max_is_53(self):
        from src.utils.calibration import GUARDRAILS
        assert GUARDRAILS['ADAPTIVE_HYBRID_BASE_THRESHOLD']['max'] == 53

    def test_apply_guardrail_clamps_above_53(self):
        from src.utils.calibration import apply_guardrail
        # previous=52 -> delta cap allows up to 57.2, absolute max clamps at 53
        val, clamped = apply_guardrail('ADAPTIVE_HYBRID_BASE_THRESHOLD', 65, 52, 48)
        assert val == 53
        assert clamped is True

    def test_apply_guardrail_previous_above_max_cannot_repush_above_max(self):
        """Regression: with the prod override at 65 (> max 53), the +/-10% delta
        clamp used to run AFTER min/max and re-push the value to 58.5 > 53."""
        from src.utils.calibration import apply_guardrail
        val, clamped = apply_guardrail('ADAPTIVE_HYBRID_BASE_THRESHOLD', 70, 65, 55)
        assert val == 53
        assert clamped is True

    def test_apply_guardrail_delta_abs_then_minmax(self):
        """max_delta_abs params also end inside the absolute bounds."""
        from src.utils.calibration import apply_guardrail
        # previous=0.50 (above max 0.40), proposal 0.60: delta clamps to 0.55,
        # then max clamps to 0.40 — never above the absolute max.
        val, clamped = apply_guardrail('ADAPTIVE_HYBRID_VOLUME_FILTER_MIN', 0.60, 0.50, 0.05)
        assert val == pytest.approx(0.40)
        assert clamped is True

    def test_runtime_clamp_uses_guardrail_max(self, strategy, monkeypatch):
        """An override above the guardrail max is clamped to it (single source of truth)."""
        import src.strategies.custom.adaptive_hybrid_strategy as ahs
        from src.utils.calibration import GUARDRAILS
        monkeypatch.setattr(ahs, 'get_calibrated_value', lambda name, default: 65)
        if hasattr(strategy, '_feedback'):
            monkeypatch.delattr(strategy, '_feedback')
        threshold = strategy._get_effective_threshold()
        assert threshold == GUARDRAILS['ADAPTIVE_HYBRID_BASE_THRESHOLD']['max']

    def test_runtime_applies_writable_value(self, strategy, monkeypatch):
        """Any value the agent is allowed to write (<= 53) is applied as-is."""
        import src.strategies.custom.adaptive_hybrid_strategy as ahs
        monkeypatch.setattr(ahs, 'get_calibrated_value', lambda name, default: 52.8)
        if hasattr(strategy, '_feedback'):
            monkeypatch.delattr(strategy, '_feedback')
        threshold = strategy._get_effective_threshold()
        assert threshold == pytest.approx(52.8)


class TestAvoidHoursMinimumSample:
    def _agent(self):
        from src.agents.calibration_agent import CalibrationAgent
        with patch.object(CalibrationAgent, '_load_state'):
            agent = CalibrationAgent()
            agent._state = {}
            return agent

    def _metrics(self, hour_stats):
        return {
            'trades': 30, 'win_rate': 0.50, 'profit_factor': 1.2,
            'sl_pct': 0.30, 'tp_count': 5, 'trades_per_day': 1.0,
            'hour_stats': hour_stats,
        }

    def test_small_sample_hour_ignored(self):
        agent = self._agent()
        # 5 trades, 0% WR -> noise, must NOT be banned
        problems = agent._diagnose_problems(self._metrics({3: {'wins': 0, 'losses': 5}}))
        assert not any(p['type'] == 'BAD_HOURS' for p in problems)

    def test_large_sample_bad_hour_flagged(self):
        agent = self._agent()
        # 12 trades, WR 25% < 35% -> banned
        problems = agent._diagnose_problems(self._metrics({7: {'wins': 3, 'losses': 9}}))
        bad = [p for p in problems if p['type'] == 'BAD_HOURS']
        assert len(bad) == 1
        assert bad[0]['hours'] == [7]

    def test_large_sample_ok_wr_not_flagged(self):
        agent = self._agent()
        # 12 trades, WR 42% >= 35% -> not banned
        problems = agent._diagnose_problems(self._metrics({7: {'wins': 5, 'losses': 7}}))
        assert not any(p['type'] == 'BAD_HOURS' for p in problems)


class TestAvoidHoursRederived:
    """AVOID_HOURS override is RE-DERIVED (config + current window), not merged.

    The old merge (existing + detected) could only grow: hours banned by the
    noise-prone pre-Jun-2026 criterion persisted forever and the override
    replaced the manually configured low-liquidity hours.
    """

    def _agent(self, overrides):
        from src.agents.calibration_agent import CalibrationAgent
        with patch.object(CalibrationAgent, '_load_state'):
            agent = CalibrationAgent()
            agent._state = {}
        agent._load_current_overrides = lambda: overrides
        return agent

    def test_stale_noise_override_is_purged_back_to_config(self):
        # Prod state: 6 hours banned under the old >=3-trades criterion
        agent = self._agent({'ADAPTIVE_HYBRID_AVOID_HOURS': {'value': [0, 2, 5, 9, 12, 18]}})
        adjustments = agent._compute_adjustments(problems=[], metrics={})
        assert adjustments['ADAPTIVE_HYBRID_AVOID_HOURS']['value'] == sorted(
            set(config.ADAPTIVE_HYBRID_AVOID_HOURS))

    def test_detected_bad_hours_added_on_top_of_config(self):
        agent = self._agent({})
        problems = [{'type': 'BAD_HOURS', 'detail': '', 'severity': 'LOW', 'hours': [7]}]
        adjustments = agent._compute_adjustments(problems=problems, metrics={})
        expected = sorted(set(config.ADAPTIVE_HYBRID_AVOID_HOURS) | {7})
        assert adjustments['ADAPTIVE_HYBRID_AVOID_HOURS']['value'] == expected

    def test_no_override_and_no_problem_writes_nothing(self):
        agent = self._agent({})
        adjustments = agent._compute_adjustments(problems=[], metrics={})
        assert 'ADAPTIVE_HYBRID_AVOID_HOURS' not in adjustments

    def test_config_matching_override_not_rewritten(self):
        agent = self._agent({'ADAPTIVE_HYBRID_AVOID_HOURS': {
            'value': sorted(set(config.ADAPTIVE_HYBRID_AVOID_HOURS))}})
        adjustments = agent._compute_adjustments(problems=[], metrics={})
        assert 'ADAPTIVE_HYBRID_AVOID_HOURS' not in adjustments


# ---------------------------------------------------------------------------
# B7 — Calibration agent corrected PnL
# ---------------------------------------------------------------------------

class TestCalibrationCorrectedPnl:
    def _write_csv(self, tmp_path, rows):
        path = tmp_path / 'closed_trades.csv'
        pd.DataFrame(rows).to_csv(path, index=False)
        return str(path)

    def test_total_pnl_and_entry_fee_used(self, tmp_path, monkeypatch):
        from src.agents import calibration_agent as ca
        now = datetime.now().isoformat()
        rows = [
            # AH convention: pnl = final leg (exit fee deducted), total_pnl
            # includes scale-out partials, entry_fee deducted at open.
            {'symbol': 'BTC', 'direction': 'BUY', 'pnl': -5.0, 'total_pnl': 2.0,
             'partial_pnl_realized': 7.0, 'entry_fee': 0.5, 'pnl_pct': 0.5,
             'close_reason': 'TRAILING_STOP', 'entry_time': now, 'exit_time': now,
             'score': 60, 'entry_price': 100, 'close_price': 101},
            {'symbol': 'ETH', 'direction': 'SELL', 'pnl': 3.0, 'total_pnl': 3.0,
             'partial_pnl_realized': 0.0, 'entry_fee': 0.5, 'pnl_pct': 1.0,
             'close_reason': 'TAKE_PROFIT', 'entry_time': now, 'exit_time': now,
             'score': 55, 'entry_price': 200, 'close_price': 198},
        ]
        csv_path = self._write_csv(tmp_path, rows)
        monkeypatch.setattr(ca, '_get_closed_trades_csv', lambda: csv_path)

        with patch.object(ca.CalibrationAgent, '_load_state'):
            agent = ca.CalibrationAgent()
            agent._state = {}
        metrics = agent._collect_performance_data(days=14)

        assert metrics['trades'] == 2
        # Trade 1: 2.0 - 0.5 = 1.5 (win) ; Trade 2: 3.0 - 0.5 = 2.5 (win)
        assert metrics['total_pnl'] == pytest.approx(4.0)
        assert metrics['wins'] == 2
        # Raw `pnl` column would have given -5 + 3 = -2 (a loss): regression guard
        assert metrics['total_pnl'] > 0

    def test_fallback_without_total_pnl_column(self, tmp_path, monkeypatch):
        from src.agents import calibration_agent as ca
        now = datetime.now().isoformat()
        rows = [
            {'symbol': 'BTC', 'direction': 'BUY', 'pnl': -5.0,
             'partial_pnl_realized': 7.0, 'pnl_pct': 0.5,
             'close_reason': 'TRAILING_STOP', 'entry_time': now, 'exit_time': now,
             'score': 60, 'entry_price': 100, 'close_price': 101},
        ]
        csv_path = self._write_csv(tmp_path, rows)
        monkeypatch.setattr(ca, '_get_closed_trades_csv', lambda: csv_path)

        with patch.object(ca.CalibrationAgent, '_load_state'):
            agent = ca.CalibrationAgent()
            agent._state = {}
        metrics = agent._collect_performance_data(days=14)
        # pnl + partial_pnl_realized = 2.0 (no entry_fee column -> nothing deducted)
        assert metrics['total_pnl'] == pytest.approx(2.0)

    def test_corrected_pnl_helper_handles_nan(self):
        from src.agents.calibration_agent import CalibrationAgent
        row = pd.Series({'pnl': 4.0, 'total_pnl': float('nan'),
                         'partial_pnl_realized': 1.0, 'entry_fee': float('nan')})
        assert CalibrationAgent._corrected_pnl(row) == pytest.approx(5.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
