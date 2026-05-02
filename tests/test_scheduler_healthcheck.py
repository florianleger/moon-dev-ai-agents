"""Tests for the scheduler healthcheck (Mission 1).

Covers:
    - Alert fires when min(last_check_ago_s) > 1800
    - No alert when all tokens checked < 1800s ago
    - 1h cooldown anti-spam (2 alerts within window → 1 sent)
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from unittest.mock import MagicMock

import pytest

from src.utils.scheduler_healthcheck import (
    SchedulerHealthcheck,
    compute_scheduler_freshness,
    get_health_snapshot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_state(tmp_path, last_check: dict):
    """Write a minimal scheduler_state.json compatible with the real scheduler."""
    state_path = os.path.join(tmp_path, 'scheduler_state.json')
    payload = {
        'last_check': last_check,
        'last_result': {sym: {'score': 50, 'threshold': 55} for sym in last_check},
        'scan_count': {},
        'queue': [],
    }
    with open(state_path, 'w') as f:
        json.dump(payload, f)
    return state_path


def _make_alert_manager(enabled: bool = True) -> MagicMock:
    am = MagicMock()
    am.is_enabled = enabled
    am.alert = MagicMock()
    return am


# ---------------------------------------------------------------------------
# Freshness computation
# ---------------------------------------------------------------------------

def test_compute_freshness_empty_state(tmp_path):
    """Missing state file → not available, no fake data."""
    snap = compute_scheduler_freshness(path=str(tmp_path / 'nope.json'))
    assert snap['available'] is False
    assert snap['min_age_s'] is None
    assert snap['stale_count'] == 0


def test_compute_freshness_basic(tmp_path):
    now = time.time()
    state_path = _write_state(str(tmp_path), {
        'BTC': now - 60,
        'ETH': now - 120,
        'SOL': now - 7200,  # > 1h, definitely stale
    })
    snap = compute_scheduler_freshness(path=state_path)
    assert snap['available'] is True
    assert snap['token_count'] == 3
    assert snap['min_age_s'] == pytest.approx(60, abs=2)
    assert snap['max_age_s'] == pytest.approx(7200, abs=2)
    assert snap['stale_token'] == 'SOL'
    assert snap['stale_count'] == 1


# ---------------------------------------------------------------------------
# Healthcheck behaviour — alert firing
# ---------------------------------------------------------------------------

def test_alert_fires_when_min_age_above_threshold(tmp_path):
    """min(last_check_ago_s) > 1800s should trigger an alert."""
    now = time.time()
    state_path = _write_state(str(tmp_path), {
        'BTC': now - 7 * 86400,  # 7 days frozen (the real-world incident)
        'ETH': now - 7 * 86400,
        'SOL': now - 7 * 86400,
    })

    am = _make_alert_manager(enabled=True)
    hc = SchedulerHealthcheck(
        state_path=state_path,
        frozen_threshold_s=1800,
        alert_cooldown_s=3600,
        alert_manager=am,
    )

    snap = hc.check_once()

    assert snap['frozen'] is True
    assert am.alert.call_count == 1
    args, kwargs = am.alert.call_args
    assert 'frozen' in args[0].lower() or 'frozen' in kwargs.get('title', '').lower()


def test_no_alert_when_all_recent(tmp_path):
    """All tokens fresh (< 1800s) → no alert."""
    now = time.time()
    state_path = _write_state(str(tmp_path), {
        'BTC': now - 30,
        'ETH': now - 240,
        'SOL': now - 600,
    })

    am = _make_alert_manager(enabled=True)
    hc = SchedulerHealthcheck(
        state_path=state_path,
        frozen_threshold_s=1800,
        alert_manager=am,
    )

    snap = hc.check_once()
    assert snap['frozen'] is False
    am.alert.assert_not_called()


def test_alert_cooldown_anti_spam(tmp_path):
    """Two frozen checks within cooldown window → only one alert sent."""
    now = time.time()
    state_path = _write_state(str(tmp_path), {
        'BTC': now - 4000,
        'ETH': now - 4000,
    })

    am = _make_alert_manager(enabled=True)

    # Inject a fake clock so we control the cooldown window precisely
    fake_now = [10_000.0]
    hc = SchedulerHealthcheck(
        state_path=state_path,
        frozen_threshold_s=1800,
        alert_cooldown_s=3600,
        alert_manager=am,
        clock=lambda: fake_now[0],
    )

    # First check: should alert
    hc.check_once()
    assert am.alert.call_count == 1

    # Second check 30 minutes later (still within 1h cooldown): no alert
    fake_now[0] += 1800
    hc.check_once()
    assert am.alert.call_count == 1, "Cooldown violated: a second alert was sent"

    # Third check 90 minutes after first (outside cooldown): alert again
    fake_now[0] += 3700
    hc.check_once()
    assert am.alert.call_count == 2


def test_silent_when_alert_manager_disabled(tmp_path):
    """No webhook configured → no crash, no exception."""
    now = time.time()
    state_path = _write_state(str(tmp_path), {'BTC': now - 4000})

    am = _make_alert_manager(enabled=False)
    hc = SchedulerHealthcheck(
        state_path=state_path,
        alert_manager=am,
    )
    # Must not raise, must not call alert()
    hc.check_once()
    am.alert.assert_not_called()


# ---------------------------------------------------------------------------
# /api/health snapshot
# ---------------------------------------------------------------------------

def test_health_snapshot_unknown_when_no_state(monkeypatch, tmp_path):
    """If scheduler_state.json doesn't exist → status 'unknown', not 'ok'."""
    monkeypatch.setattr(
        'src.utils.scheduler_healthcheck.DEFAULT_SCHEDULER_STATE',
        str(tmp_path / 'missing.json'),
    )
    snap = get_health_snapshot()
    assert snap['status'] in ('unknown', 'frozen')
    assert 'uptime_s' in snap
    assert snap['uptime_s'] >= 0
    assert 'active_strategies' in snap


def test_health_snapshot_frozen(monkeypatch, tmp_path):
    now = time.time()
    state_path = _write_state(str(tmp_path), {'BTC': now - 7 * 86400})
    monkeypatch.setattr(
        'src.utils.scheduler_healthcheck.DEFAULT_SCHEDULER_STATE',
        state_path,
    )
    snap = get_health_snapshot()
    assert snap['status'] == 'frozen'
    assert snap['scheduler_last_scan_s'] >= 7 * 86400 - 5


def test_health_snapshot_ok(monkeypatch, tmp_path):
    now = time.time()
    state_path = _write_state(str(tmp_path), {'BTC': now - 30})
    monkeypatch.setattr(
        'src.utils.scheduler_healthcheck.DEFAULT_SCHEDULER_STATE',
        state_path,
    )
    snap = get_health_snapshot()
    assert snap['status'] == 'ok'
    assert snap['scheduler_last_scan_s'] < 60
