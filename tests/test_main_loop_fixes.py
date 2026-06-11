"""Tests for the main-loop / scheduler / risk fixes (Agent A).

Covers:
- extract_result: get_signals() returns a LIST (old code crashed with
  AttributeError on signal.get), plus dict/empty/None robustness.
- run_token_check: a token whose analysis raises is ALWAYS re-scheduled
  (no more orphan tokens leaking out of the queue).
- heal_orphans: monitored tokens absent from the queue for too long are
  re-enqueued even when the queue is non-empty.
- save_state ordering: the disk snapshot includes the token being processed.
- Risk gate: max positions blocks NEW entries only, never the scan
  (is_trading_allowed vs allows_new_entries).
- StrategyKillSwitch: rolling PF / drawdown auto-pause on CSV fixtures.
- Global daily-loss breaker for independent strategies.
"""

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.scheduling.scheduler import SmartScheduler, extract_result, run_token_check
from src.agents.risk_agent import (
    StrategyKillSwitch,
    corrected_trade_pnls,
    independent_daily_loss_breached,
    KILL_SWITCH_MIN_TRADES,
)
from src.config import RISK_MAX_POSITIONS


@pytest.fixture
def scheduler():
    """Fresh scheduler that never touches the on-disk state file."""
    s = SmartScheduler()
    s.save_state = lambda: None  # type: ignore[assignment]
    return s


def _signal_entry(score=62, threshold=50, regime='MARKUP', atr=150.0, price=50000.0):
    """One approved-signal dict as produced by strategy_agent.get_signals()."""
    return {
        'token': 'BTC',
        'strategy_name': 'adaptive_hybrid',
        'signal': 0.62,
        'direction': 'BUY',
        'metadata': {
            'score': score,
            'threshold': threshold,
            'llm_regime': {'regime': regime},
            'atr': atr,
            'current_price': price,
        },
    }


# ---------------------------------------------------------------------------
# extract_result (A1)
# ---------------------------------------------------------------------------

class TestExtractResult:
    def test_empty_list_falls_back(self):
        from src.config import ADAPTIVE_HYBRID_BASE_THRESHOLD
        res = extract_result('BTC', [])
        # Fallback threshold mirrors the configured base threshold (was a
        # hardcoded 40 that skewed proximity computations).
        assert res == {'score': 0, 'threshold': ADAPTIVE_HYBRID_BASE_THRESHOLD,
                       'regime': '', 'atr_pct': 0}

    def test_none_falls_back(self):
        res = extract_result('BTC', None)
        assert res['score'] == 0

    def test_list_of_signals_extracts_first(self):
        """Regression: get_signals() returns a LIST — old code called
        signal.get() on it and raised AttributeError on every approved signal."""
        res = extract_result('BTC', [_signal_entry()])
        assert res['score'] == 62
        assert res['threshold'] == 50
        assert res['regime'] == 'MARKUP'
        assert res['atr_pct'] == pytest.approx(150.0 / 50000.0)

    def test_dict_signal_still_supported(self):
        res = extract_result('BTC', _signal_entry(score=70))
        assert res['score'] == 70

    def test_regime_as_plain_string(self):
        sig = _signal_entry()
        sig['metadata']['llm_regime'] = 'MARKDOWN'
        res = extract_result('BTC', [sig])
        assert res['regime'] == 'MARKDOWN'

    def test_signal_without_metadata_falls_back(self):
        res = extract_result('BTC', [{'token': 'BTC', 'metadata': {}}])
        assert res['score'] == 0


# ---------------------------------------------------------------------------
# run_token_check (A3): no orphans, ever
# ---------------------------------------------------------------------------

class TestRunTokenCheck:
    def test_nominal_path_records_and_reschedules(self, scheduler):
        scheduler._all_symbols = ['BTC']
        res = run_token_check(scheduler, 'BTC', lambda sym: [_signal_entry()], lambda sym: False)
        assert res['score'] == 62
        assert 'BTC' in scheduler._scheduled_symbols
        assert scheduler._last_check.get('BTC', 0) > 0
        assert scheduler._last_result['BTC']['score'] == 62

    def test_exception_still_reschedules_token(self, scheduler):
        """An analysis crash must not orphan the token (prod bug: 6 tokens
        out of the queue, BTC stale 16h)."""
        scheduler._all_symbols = ['BTC']

        def boom(sym):
            raise RuntimeError("LLM exploded")

        res = run_token_check(scheduler, 'BTC', boom, lambda sym: False)
        assert res['status'] == 'fail'
        assert 'LLM exploded' in res['fail_reason']
        assert 'BTC' in scheduler._scheduled_symbols, "token must be re-enqueued after a crash"
        assert scheduler.queue_size() == 1
        # last_check refreshed so the healthcheck sees the scheduler alive
        assert scheduler._last_check.get('BTC', 0) > 0

    def test_timeout_marked_as_fail(self, scheduler):
        scheduler._all_symbols = ['ETH']

        def timeout(sym):
            raise TimeoutError("watchdog_timeout (120s)")

        res = run_token_check(scheduler, 'ETH', timeout, lambda sym: False)
        assert res['status'] == 'fail'
        assert 'watchdog_timeout' in res['fail_reason']
        assert 'ETH' in scheduler._scheduled_symbols

    def test_has_position_crash_still_reschedules(self, scheduler):
        scheduler._all_symbols = ['SOL']

        def bad_has_pos(sym):
            raise RuntimeError("lock poisoned")

        run_token_check(scheduler, 'SOL', lambda sym: [], bad_has_pos)
        assert 'SOL' in scheduler._scheduled_symbols


# ---------------------------------------------------------------------------
# heal_orphans (A4)
# ---------------------------------------------------------------------------

class TestHealOrphans:
    def test_stale_orphan_is_reenqueued_even_with_nonempty_queue(self, scheduler):
        scheduler._all_symbols = ['BTC', 'ETH', 'SOL']
        # SOL has a queue entry (queue NOT empty — old self-heal required qsize==0)
        scheduler._enqueue('SOL', time.time() + 600, scheduler.PRIORITY_ROUTINE, 'routine')
        # ETH checked recently, not queued -> fresh, not an orphan yet
        scheduler._last_check['ETH'] = time.time()
        # BTC last checked 1h ago, not queued -> orphan
        scheduler._last_check['BTC'] = time.time() - 3600

        healed = scheduler.heal_orphans(stale_s=1800)
        assert healed == ['BTC']
        assert 'BTC' in scheduler._scheduled_symbols
        assert 'ETH' not in scheduler._scheduled_symbols

    def test_never_checked_token_is_healed(self, scheduler):
        scheduler._all_symbols = ['XRP']
        healed = scheduler.heal_orphans(stale_s=1800)
        assert healed == ['XRP']

    def test_queued_tokens_untouched(self, scheduler):
        scheduler._all_symbols = ['BTC']
        scheduler._enqueue('BTC', time.time() + 60, scheduler.PRIORITY_ROUTINE, 'routine')
        assert scheduler.heal_orphans(stale_s=1800) == []
        assert scheduler.queue_size() == 1


# ---------------------------------------------------------------------------
# save_state ordering (A9)
# ---------------------------------------------------------------------------

class TestSaveStateOrdering:
    def test_record_result_does_not_save_schedule_recheck_does(self, scheduler):
        calls = []
        scheduler.save_state = lambda: calls.append('save')  # type: ignore[assignment]
        scheduler.record_result('BTC', {'score': 0, 'threshold': 40})
        assert calls == [], "record_result must not persist (token not yet re-enqueued)"
        scheduler.schedule_recheck('BTC', {'score': 0, 'threshold': 40}, False)
        assert calls == ['save']

    def test_snapshot_includes_token_in_flight(self, scheduler):
        """The persisted snapshot must contain the token being processed."""
        scheduler._all_symbols = ['BTC']
        snapshot = {}

        def fake_save():
            snapshot['queued'] = [r.symbol for r in scheduler._queue]

        scheduler.save_state = fake_save  # type: ignore[assignment]
        run_token_check(scheduler, 'BTC', lambda sym: [], lambda sym: False)
        assert 'BTC' in snapshot.get('queued', [])


# ---------------------------------------------------------------------------
# Risk gate (A2/A5): max positions blocks entries only, never the scan
# ---------------------------------------------------------------------------

@pytest.fixture
def risk_agent(temp_dir):
    state_file = Path(os.path.join(temp_dir, 'risk_state.json'))
    with patch('src.agents.risk_agent.RISK_STATE_FILE', state_file):
        from src.agents.risk_agent import RiskAgent
        agent = RiskAgent()
        agent._strategy = None
        yield agent


def _mock_strategy(open_positions=0, balance=500.0):
    m = MagicMock()
    m.get_paper_status.return_value = {
        'paper_balance': balance,
        'open_positions': open_positions,
    }
    return m


class TestRiskGate:
    def test_max_positions_blocks_entries_but_not_trading(self, risk_agent):
        """4/4 positions: scan must keep running (is_trading_allowed True),
        only new entries are refused (allows_new_entries False)."""
        risk_agent._strategy = _mock_strategy(open_positions=RISK_MAX_POSITIONS)
        assert risk_agent.is_trading_allowed() is True
        assert risk_agent.allows_new_entries() is False
        assert 'max positions' in risk_agent.entries_blocked_reason

    def test_circuit_breaker_blocks_both(self, risk_agent):
        risk_agent._strategy = _mock_strategy(open_positions=0)
        risk_agent.trading_paused = True
        risk_agent.pause_reason = "HWM drawdown 15% reached"
        assert risk_agent.is_trading_allowed() is False
        assert risk_agent.allows_new_entries() is False
        assert risk_agent.entries_blocked_reason == "HWM drawdown 15% reached"

    def test_daily_pause_blocks_both(self, risk_agent):
        from datetime import datetime
        risk_agent._strategy = _mock_strategy(open_positions=0)
        risk_agent.daily_pause = True
        risk_agent.daily_pause_date = datetime.now().date()
        assert risk_agent.is_trading_allowed() is False
        assert risk_agent.allows_new_entries() is False

    def test_all_clear_allows_everything(self, risk_agent):
        risk_agent._strategy = _mock_strategy(open_positions=1)
        assert risk_agent.is_trading_allowed() is True
        assert risk_agent.allows_new_entries() is True
        assert risk_agent.entries_blocked_reason is None


# ---------------------------------------------------------------------------
# StrategyKillSwitch (A6)
# ---------------------------------------------------------------------------

def _write_trades_csv(path, pnls, entry_fees=None, total_pnls=None):
    data = {'symbol': ['BTC'] * len(pnls), 'pnl': pnls}
    if entry_fees is not None:
        data['entry_fee'] = entry_fees
    if total_pnls is not None:
        data['total_pnl'] = total_pnls
    pd.DataFrame(data).to_csv(path, index=False)
    return path


class TestKillSwitch:
    def _ks(self, temp_dir):
        return StrategyKillSwitch(
            state_file=os.path.join(temp_dir, 'kill_switch.json'),
            initial_balance=500.0,
        )

    def test_bad_pf_pauses_strategy(self, temp_dir):
        ks = self._ks(temp_dir)
        # 30 trades: 10 wins of +1, 20 losses of -2 -> PF=0.25, DD=6% (<12%)
        pnls = [1.0] * 10 + [-2.0] * 20
        csv = _write_trades_csv(os.path.join(temp_dir, 'closed_trades.csv'), pnls)
        verdict = ks.evaluate('ote_scalp', csv)
        assert verdict['paused'] is True
        assert 'rolling PF' in verdict['reason']
        assert ks.is_paused('ote_scalp')

    def test_bad_pf_with_few_trades_does_not_pause(self, temp_dir):
        ks = self._ks(temp_dir)
        pnls = [-1.0] * (KILL_SWITCH_MIN_TRADES - 5)  # PF=0 but n too small, DD ~4%
        csv = _write_trades_csv(os.path.join(temp_dir, 'closed_trades.csv'), pnls)
        verdict = ks.evaluate('funding_mr', csv)
        assert verdict['paused'] is False

    def test_drawdown_pauses_even_with_few_trades(self, temp_dir):
        ks = self._ks(temp_dir)
        pnls = [-15.0] * 5  # -75 on 500 = 15% DD > 12%
        csv = _write_trades_csv(os.path.join(temp_dir, 'closed_trades.csv'), pnls)
        verdict = ks.evaluate('funding_mr', csv)
        assert verdict['paused'] is True
        assert 'drawdown' in verdict['reason']

    def test_healthy_strategy_not_paused(self, temp_dir):
        ks = self._ks(temp_dir)
        pnls = [5.0, -2.0] * 20  # PF=2.5
        csv = _write_trades_csv(os.path.join(temp_dir, 'closed_trades.csv'), pnls)
        verdict = ks.evaluate('vol_breakout', csv)
        assert verdict['paused'] is False
        assert verdict['pf'] == pytest.approx(2.5)

    def test_missing_csv_not_paused(self, temp_dir):
        ks = self._ks(temp_dir)
        verdict = ks.evaluate('liq_cascade', os.path.join(temp_dir, 'nope.csv'))
        assert verdict['paused'] is False

    def test_pause_persists_across_instances(self, temp_dir):
        ks = self._ks(temp_dir)
        csv = _write_trades_csv(os.path.join(temp_dir, 'closed_trades.csv'), [-15.0] * 5)
        assert ks.evaluate('funding_mr', csv)['paused'] is True

        ks2 = self._ks(temp_dir)
        assert ks2.is_paused('funding_mr')
        # evaluate() short-circuits on a paused strategy (no CSV re-read)
        assert ks2.evaluate('funding_mr', csv)['paused'] is True

        ks2.resume('funding_mr')
        assert not ks2.is_paused('funding_mr')

    def test_corrected_pnl_prefers_total_pnl(self, temp_dir):
        ks = self._ks(temp_dir)
        # pnl column looks catastrophic but total_pnl (incl. partials) is healthy
        csv = _write_trades_csv(
            os.path.join(temp_dir, 'closed_trades.csv'),
            pnls=[-10.0] * 30, total_pnls=[2.0] * 30)
        verdict = ks.evaluate('adaptive_hybrid', csv)
        assert verdict['paused'] is False

    def test_corrected_pnl_deducts_entry_fee_without_total_pnl(self):
        df = pd.DataFrame({'pnl': [10.0, -5.0], 'entry_fee': [1.0, 1.0]})
        assert corrected_trade_pnls(df).tolist() == [9.0, -6.0]

    def test_corrected_pnl_total_pnl_minus_entry_fee(self):
        # entry fee is deducted from balance at open and never included in
        # total_pnl — same convention as dashboard/calibration_agent
        df = pd.DataFrame({'pnl': [10.0], 'total_pnl': [12.0], 'entry_fee': [1.0]})
        assert corrected_trade_pnls(df).tolist() == [11.0]


# ---------------------------------------------------------------------------
# Global daily-loss breaker (A7)
# ---------------------------------------------------------------------------

class TestGlobalDailyLossBreaker:
    def test_breached_when_total_below_limit(self):
        instances = [SimpleNamespace(daily_pnl=-30.0), SimpleNamespace(daily_pnl=-25.0)]
        breached, total = independent_daily_loss_breached(instances, 50.0)
        assert breached is True
        assert total == pytest.approx(-55.0)

    def test_not_breached_when_above_limit(self):
        instances = [SimpleNamespace(daily_pnl=-10.0), SimpleNamespace(daily_pnl=5.0)]
        breached, total = independent_daily_loss_breached(instances, 50.0)
        assert breached is False
        assert total == pytest.approx(-5.0)

    def test_missing_daily_pnl_counts_as_zero(self):
        instances = [SimpleNamespace(), SimpleNamespace(daily_pnl=-60.0)]
        breached, total = independent_daily_loss_breached(instances, 50.0)
        assert breached is True
        assert total == pytest.approx(-60.0)
