"""Tests for SmartScheduler fairness guarantees.

Specifically: an over-represented token must be FULL-SKIPPED on schedule_recheck
(not just delayed), and MIN_TOKEN_INTERVAL_S must NOT override the over-rep skip.

Regression test for the production bug where 12/14 tokens were not checked for
6+ days because AAVE monopolised the scan queue (introduced when 21c3c06
partially undid the fairness fix from 42f28c3).
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


def test_over_represented_token_is_full_skipped_not_re_enqueued(scheduler):
    """An over-represented token's recheck must NOT land in the queue at all.

    Previously the code delayed it by OVER_REPRESENTED_DELAY_S=300s and re-enqueued,
    which let the same token monopolise the queue in a 5-minute loop.
    """
    symbols = [f"T{i}" for i in range(14)]  # 14 tokens like in production
    runaway = "AAVE"
    symbols.append(runaway)
    # AAVE has scanned 100 times, others a median ~5 → AAVE is way over.
    _seed_symbols(scheduler, symbols, runaway=runaway, runaway_count=100, others_count=5)

    assert scheduler._is_over_represented(runaway), "fixture should make AAVE over-represented"

    # Pre-condition: queue empty
    assert scheduler.queue_size() == 0
    assert runaway not in scheduler._scheduled_symbols

    # Act: schedule a recheck for the runaway token
    scheduler.schedule_recheck(runaway, result={'score': 60, 'threshold': 50, 'regime': 'MARKUP'}, has_position=False)

    # Assert: full skip — nothing was enqueued
    assert scheduler.queue_size() == 0, "over-represented token must NOT be re-enqueued"
    assert runaway not in scheduler._scheduled_symbols


def test_over_represented_skip_holds_even_with_min_token_interval(scheduler):
    """MIN_TOKEN_INTERVAL_S must NOT override the over-rep full-skip.

    Even if the computed interval would be tiny (volatile token, position open),
    the safety floor must be applied AFTER the over-rep decision, never before.
    """
    symbols = ["A", "B", "C", "D", "E", "F"]
    runaway = "A"
    _seed_symbols(scheduler, symbols, runaway=runaway, runaway_count=50, others_count=2)
    assert scheduler._is_over_represented(runaway)

    # Even with has_position=True (which forces a tighter recheck interval),
    # the over-rep skip must still win.
    scheduler.schedule_recheck(
        runaway,
        result={'score': 80, 'threshold': 50, 'regime': 'MARKUP', 'atr_pct': 0.05},
        has_position=True,
    )

    assert scheduler.queue_size() == 0
    assert runaway not in scheduler._scheduled_symbols


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
