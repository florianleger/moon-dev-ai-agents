"""Tests for SmartScheduler fairness guarantees.

Contract: an over-represented token must be DELAYED (re-enqueued at
+OVER_REPRESENTED_DELAY_S) — NOT full-skipped. A full skip without periodic
re-seeding leaks tokens out of the queue forever (the freeze observed in prod
where 6/14 tokens ended up with priority=null/next_recheck=null after 5 days).

The fairness mechanism still works because:
  1. The runaway is pushed 5 minutes into the future (back of the queue).
  2. The MIN_TOKEN_INTERVAL_S floor still applies for non-runaways.
  3. The main loop has a self-heal that re-seeds if the queue ever empties.
"""

import time

import pytest

from src.scheduling.scheduler import SmartScheduler


@pytest.fixture
def scheduler():
    """Fresh scheduler with no persisted state."""
    s = SmartScheduler()
    # Avoid touching the on-disk state file during tests.
    s.save_state = lambda: None  # type: ignore[assignment]
    return s


def _seed_symbols(s: SmartScheduler, symbols, runaway: str, runaway_count: int, others_count: int):
    """Seed the scheduler so `runaway` is over-represented vs the others."""
    s._all_symbols = list(symbols)
    for sym in symbols:
        s._scan_count[sym] = others_count
    s._scan_count[runaway] = runaway_count


def test_over_represented_token_is_delayed_not_dropped(scheduler):
    """An over-represented token's recheck must be DELAYED (still queued).

    The previous version of this code FULL-SKIPPED the recheck (returned
    without re-enqueueing) under the assumption that enqueue_all_routine()
    would periodically re-seed. It doesn't — enqueue_all_routine is only
    called once at boot. As a result, in prod 6/14 tokens disappeared from
    the queue and were never scanned for 5 days.

    The new contract: delay by OVER_REPRESENTED_DELAY_S (5 min) so the token
    falls to the back of the queue but never leaves it. Other tokens scan
    naturally during that window so the runaway re-equilibrates.
    """
    symbols = [f"T{i}" for i in range(14)]  # 14 tokens like in production
    runaway = "AAVE"
    symbols.append(runaway)
    # AAVE has scanned 100 times, others a median ~5 → AAVE is way over.
    _seed_symbols(scheduler, symbols, runaway=runaway, runaway_count=100, others_count=5)

    assert scheduler._is_over_represented(runaway), "fixture should make AAVE over-represented"
    assert scheduler.queue_size() == 0
    assert runaway not in scheduler._scheduled_symbols

    t0 = time.time()
    scheduler.schedule_recheck(runaway, result={'score': 60, 'threshold': 50, 'regime': 'MARKUP'}, has_position=False)

    # Token must be queued (delayed, not dropped).
    assert scheduler.queue_size() == 1, "over-represented token must remain in the queue"
    assert runaway in scheduler._scheduled_symbols
    req = scheduler._queue[0]
    delay = req.scheduled_at - t0
    # Delayed by ~OVER_REPRESENTED_DELAY_S; allow generous tolerance for clock.
    assert delay >= scheduler.OVER_REPRESENTED_DELAY_S - 5
    # Priority is routine (low) so other tokens get popped first.
    assert req.priority == scheduler.PRIORITY_ROUTINE


def test_over_represented_delay_holds_regardless_of_position(scheduler):
    """An over-represented token must be delayed even if has_position=True.

    The over-rep decision must be evaluated BEFORE the position-driven tight
    interval, otherwise a runaway with an open position keeps coming back at
    the position cadence and re-monopolises the queue.
    """
    symbols = ["A", "B", "C", "D", "E", "F"]
    runaway = "A"
    _seed_symbols(scheduler, symbols, runaway=runaway, runaway_count=50, others_count=2)
    assert scheduler._is_over_represented(runaway)

    t0 = time.time()
    scheduler.schedule_recheck(
        runaway,
        result={'score': 80, 'threshold': 50, 'regime': 'MARKUP', 'atr_pct': 0.05},
        has_position=True,
    )

    assert scheduler.queue_size() == 1
    assert runaway in scheduler._scheduled_symbols
    req = scheduler._queue[0]
    delay = req.scheduled_at - t0
    # Even with a position, the runaway is delayed by OVER_REPRESENTED_DELAY_S
    # (not the tighter position interval).
    assert delay >= scheduler.OVER_REPRESENTED_DELAY_S - 5


def test_non_over_represented_token_is_still_enqueued(scheduler):
    """Sanity: a normally-scanned token must still be re-enqueued (fairness only kicks in for runaways)."""
    symbols = ["X", "Y", "Z", "W", "V"]
    # All tokens evenly scanned ~5 times → none is over-represented.
    for s in symbols:
        scheduler._scan_count[s] = 5
    scheduler._all_symbols = list(symbols)

    assert not scheduler._is_over_represented("X")

    scheduler.schedule_recheck("X", result={'score': 50, 'threshold': 50}, has_position=False)

    assert scheduler.queue_size() == 1
    assert "X" in scheduler._scheduled_symbols


def test_min_token_interval_floor_still_enforced_for_normal_tokens(scheduler):
    """For non-over-represented tokens, the MIN_TOKEN_INTERVAL_S floor must still apply."""
    symbols = ["X", "Y", "Z"]
    for s in symbols:
        scheduler._scan_count[s] = 3
    scheduler._all_symbols = list(symbols)
    assert not scheduler._is_over_represented("X")

    t0 = time.time()
    # Force a result/regime that would normally produce a short interval.
    scheduler.schedule_recheck(
        "X",
        result={'score': 50, 'threshold': 50, 'regime': 'MARKUP', 'atr_pct': 0.05},
        has_position=True,  # tight monitoring
    )

    # The single queued entry must respect MIN_TOKEN_INTERVAL_S as a floor.
    assert scheduler.queue_size() == 1
    req = scheduler._queue[0]
    assert req.scheduled_at - t0 >= scheduler.MIN_TOKEN_INTERVAL_S - 1  # -1 for clock fuzz


# ---------------------------------------------------------------------------
# extract_result — NEUTRAL scans must keep their real score for the
# near-threshold priority (they used to collapse to score=0/threshold=40)
# ---------------------------------------------------------------------------

def test_extract_result_neutral_raw_signal_keeps_real_score():
    from src.scheduling.scheduler import extract_result
    raw = {
        'token': 'BTC', 'strategy_name': 'adaptive_hybrid',
        'signal': 0.0, 'direction': 'NEUTRAL',
        'metadata': {'score': 52.0, 'threshold': 55.0,
                     'atr': 500.0, 'current_price': 50000.0},
    }
    res = extract_result('BTC', [raw])
    assert res['score'] == 52.0
    assert res['threshold'] == 55.0
    assert res['atr_pct'] == pytest.approx(0.01)


def test_empty_result_threshold_mirrors_config_base_threshold():
    from src.scheduling.scheduler import _EMPTY_RESULT
    from src.config import ADAPTIVE_HYBRID_BASE_THRESHOLD
    assert _EMPTY_RESULT['threshold'] == ADAPTIVE_HYBRID_BASE_THRESHOLD
