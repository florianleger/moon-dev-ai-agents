"""Scheduler healthcheck — detects when the scheduler freezes.

Runs as a daemon thread, polling the scheduler state file every CHECK_INTERVAL
seconds. If the most recent token scan is older than FROZEN_THRESHOLD_S, it
fires a Discord/Telegram alert (rate-limited via AlertManager cooldown).

The healthcheck is non-fatal: if the alerting webhook is missing, it simply
logs to console and continues.

Public surface:
    - SchedulerHealthcheck (class): exposes check_once() and start() helpers
    - get_health_snapshot(): returns dict for /api/health endpoint
    - start_healthcheck_thread(): convenience helper used by main.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover — defensive
    psutil = None  # type: ignore

# ---------------------------------------------------------------------------
# Module-level config
# ---------------------------------------------------------------------------

# Default thresholds (seconds)
DEFAULT_FROZEN_THRESHOLD_S = 1800   # 30 min
DEFAULT_DEGRADED_THRESHOLD_S = 600  # 10 min
DEFAULT_CHECK_INTERVAL_S = 300      # 5 min
DEFAULT_ALERT_COOLDOWN_S = 3600     # 1h between alerts

# Path to scheduler state file (matches src/scheduling/scheduler.py)
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
DEFAULT_SCHEDULER_STATE = os.path.join(
    _PROJECT_ROOT, 'src', 'data', 'adaptive_hybrid', 'scheduler_state.json'
)
DEFAULT_HEARTBEAT_FILE = os.path.join(
    _PROJECT_ROOT, 'src', 'data', 'bot_heartbeat.json'
)

# Process start time for uptime calc.
# We avoid capturing time.time() at import (which would yield 0 uptime when this
# module is lazily imported inside the /api/health handler in the web process —
# a different process from the bot itself). Instead, we read the actual PID
# create_time on demand via psutil, with a /proc fallback for environments
# where psutil is unavailable.

def _get_process_start_ts() -> float:
    """Return the current process's start timestamp (epoch seconds)."""
    if psutil is not None:
        try:
            return psutil.Process(os.getpid()).create_time()
        except Exception:
            pass
    # Linux fallback: parse /proc/<pid>/stat field 22 (starttime in clock ticks
    # since boot) and combine with /proc/stat 'btime' (boot time epoch).
    # Account for comm field (field 2) which may contain spaces by finding
    # the last ')' to reliably split it off.
    try:
        with open(f"/proc/{os.getpid()}/stat", 'r') as f:
            data = f.read()
        rparen = data.rfind(')')
        rest = data[rparen + 2:].split()
        # rest[0] is state (field 3); starttime is field 22 → rest index 19
        starttime_ticks = int(rest[19])
        clk_tck = os.sysconf('SC_CLK_TCK')
        with open('/proc/stat', 'r') as f:
            for line in f:
                if line.startswith('btime '):
                    btime = int(line.split()[1])
                    return btime + (starttime_ticks / clk_tck)
    except Exception:
        pass
    # Last-resort fallback: now (uptime will read 0 — better than crashing).
    return time.time()


# ---------------------------------------------------------------------------
# Snapshot helpers (also used by /api/health)
# ---------------------------------------------------------------------------

def _read_scheduler_state(path: str = DEFAULT_SCHEDULER_STATE) -> Optional[Dict]:
    """Read scheduler_state.json. Returns None on missing/parse error."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def compute_scheduler_freshness(
    state: Optional[Dict] = None,
    path: str = DEFAULT_SCHEDULER_STATE,
) -> Dict:
    """Compute freshness stats from scheduler state.

    Returns:
        {
            'available': bool,                 # True if state file exists & parseable
            'min_age_s': int | None,           # min(now - last_check_ts) across tokens
            'max_age_s': int | None,           # max(...) — flags lagging tokens
            'stale_token': str | None,         # token with the largest age
            'stale_count': int,                # tokens with age > frozen_threshold
            'token_count': int,
            'frozen_threshold_s': int,
        }
    """
    if state is None:
        state = _read_scheduler_state(path)

    if not state:
        return {
            'available': False,
            'min_age_s': None,
            'max_age_s': None,
            'stale_token': None,
            'stale_count': 0,
            'token_count': 0,
            'frozen_threshold_s': DEFAULT_FROZEN_THRESHOLD_S,
        }

    now = time.time()
    last_check = state.get('last_check', {}) or {}
    if not last_check:
        return {
            'available': True,
            'min_age_s': None,
            'max_age_s': None,
            'stale_token': None,
            'stale_count': 0,
            'token_count': 0,
            'frozen_threshold_s': DEFAULT_FROZEN_THRESHOLD_S,
        }

    ages: List[tuple] = []
    for sym, ts in last_check.items():
        try:
            age = now - float(ts)
            ages.append((sym, age))
        except (TypeError, ValueError):
            continue

    if not ages:
        return {
            'available': True,
            'min_age_s': None,
            'max_age_s': None,
            'stale_token': None,
            'stale_count': 0,
            'token_count': 0,
            'frozen_threshold_s': DEFAULT_FROZEN_THRESHOLD_S,
        }

    min_sym, min_age = min(ages, key=lambda x: x[1])
    max_sym, max_age = max(ages, key=lambda x: x[1])
    stale_count = sum(1 for _, a in ages if a > DEFAULT_FROZEN_THRESHOLD_S)

    return {
        'available': True,
        'min_age_s': int(min_age),
        'max_age_s': int(max_age),
        'min_token': min_sym,
        'stale_token': max_sym,
        'stale_count': stale_count,
        'token_count': len(ages),
        'frozen_threshold_s': DEFAULT_FROZEN_THRESHOLD_S,
    }


def _count_active_strategies() -> int:
    """Best-effort count of strategy data folders that contain a paper_trades.csv."""
    base = os.path.join(_PROJECT_ROOT, 'src', 'data')
    if not os.path.isdir(base):
        return 0
    candidates = ['adaptive_hybrid', 'funding_mr', 'vol_breakout',
                  'liq_cascade', 'ote_scalp']
    count = 0
    for c in candidates:
        if os.path.exists(os.path.join(base, c, 'paper_trades.csv')):
            count += 1
        elif os.path.isdir(os.path.join(base, c)):
            # Folder exists but no trades yet — still counts as registered
            count += 1
    return count


def get_health_snapshot() -> Dict:
    """Return a snapshot suitable for a /api/health endpoint.

    Status values:
        'ok'        — most recent scan < degraded threshold
        'degraded'  — scan between degraded and frozen
        'frozen'    — scan older than frozen threshold OR scheduler unavailable
        'unknown'   — scheduler state file missing (bot may be in fixed-cycle mode)
    """
    # Read DEFAULT_SCHEDULER_STATE dynamically so monkeypatch / env override works
    snap = compute_scheduler_freshness(path=DEFAULT_SCHEDULER_STATE)
    uptime = int(time.time() - _get_process_start_ts())
    if uptime < 0:
        uptime = 0

    if not snap['available']:
        status = 'unknown'
    elif snap['min_age_s'] is None:
        status = 'unknown'
    elif snap['min_age_s'] > DEFAULT_FROZEN_THRESHOLD_S:
        status = 'frozen'
    elif snap['min_age_s'] > DEFAULT_DEGRADED_THRESHOLD_S:
        status = 'degraded'
    else:
        status = 'ok'

    return {
        'status': status,
        'scheduler_last_scan_s': snap['min_age_s'],
        'scheduler_max_age_s': snap['max_age_s'],
        'scheduler_stale_count': snap['stale_count'],
        'scheduler_token_count': snap['token_count'],
        'most_stale_token': snap.get('stale_token'),
        'active_strategies': _count_active_strategies(),
        'uptime_s': uptime,
        'frozen_threshold_s': DEFAULT_FROZEN_THRESHOLD_S,
        'checked_at': datetime.utcnow().isoformat() + 'Z',
    }


# ---------------------------------------------------------------------------
# Healthcheck class
# ---------------------------------------------------------------------------

class SchedulerHealthcheck:
    """Polls scheduler state and fires alerts when frozen.

    Anti-spam: at most one Discord alert per `alert_cooldown_s` window.
    Recovery alert is sent the first check after a frozen state clears.
    """

    def __init__(
        self,
        state_path: str = DEFAULT_SCHEDULER_STATE,
        frozen_threshold_s: int = DEFAULT_FROZEN_THRESHOLD_S,
        check_interval_s: int = DEFAULT_CHECK_INTERVAL_S,
        alert_cooldown_s: int = DEFAULT_ALERT_COOLDOWN_S,
        alert_manager=None,
        logger=None,
        clock=time.time,  # injectable for tests
    ):
        self.state_path = state_path
        self.frozen_threshold_s = frozen_threshold_s
        self.check_interval_s = check_interval_s
        self.alert_cooldown_s = alert_cooldown_s
        self._alert_manager = alert_manager
        self._logger = logger
        self._clock = clock
        self._last_alert_ts: float = 0.0
        self._was_frozen: bool = False
        self._stop_event = threading.Event()

    # --------------------------------------------------------------- public
    def check_once(self) -> Dict:
        """Run a single check. Returns the snapshot used for the decision.

        Side effects:
            - Fires a Discord alert if frozen and not on cooldown
            - Logs ERROR/INFO to logger if provided
        """
        snap = compute_scheduler_freshness(path=self.state_path)
        snap['frozen'] = False

        if not snap['available']:
            self._log_info("Scheduler state file not yet present; skipping freeze check.")
            return snap

        min_age = snap['min_age_s']
        if min_age is None:
            self._log_info("Scheduler state has no last_check entries yet; skipping.")
            return snap

        if min_age > self.frozen_threshold_s:
            snap['frozen'] = True
            self._handle_frozen(snap)
        else:
            if self._was_frozen:
                self._handle_recovery(snap)
            self._was_frozen = False

        return snap

    def run_forever(self):
        """Blocking loop — call from a daemon thread."""
        while not self._stop_event.is_set():
            try:
                self.check_once()
            except Exception as e:  # pragma: no cover — defensive
                self._log_error(f"Healthcheck iteration crashed: {e}")
            # Use Event.wait so stop() returns quickly
            self._stop_event.wait(self.check_interval_s)

    def stop(self):
        self._stop_event.set()

    # --------------------------------------------------------------- internal
    def _handle_frozen(self, snap: Dict):
        token_count = snap['token_count']
        stale = snap.get('stale_token') or '?'
        max_age = snap.get('max_age_s') or snap['min_age_s']
        stale_count = snap['stale_count']
        min_age = snap['min_age_s']

        msg = (
            f"Scheduler frozen — most recent scan was {min_age}s ago "
            f"(threshold {self.frozen_threshold_s}s). "
            f"Stale tokens: {stale_count}/{token_count}. "
            f"Most lagging: {stale} ({max_age}s)."
        )
        self._log_error(msg)
        self._was_frozen = True

        # Anti-spam cooldown
        now = self._clock()
        if now - self._last_alert_ts < self.alert_cooldown_s:
            self._log_info(
                f"Frozen alert suppressed (cooldown active, "
                f"{int(self.alert_cooldown_s - (now - self._last_alert_ts))}s left)."
            )
            return

        self._send_alert(
            title="Scheduler frozen",
            body=(
                f"**Le scheduler ne tourne plus.**\n\n"
                f"Dernier scan il y a `{_fmt_duration(min_age)}` "
                f"(seuil `{_fmt_duration(self.frozen_threshold_s)}`)\n"
                f"Tokens stale: `{stale_count}/{token_count}`\n"
                f"Plus en retard: `{stale}` (`{_fmt_duration(max_age)}`)\n\n"
                f"Vérifie les logs et redémarre si nécessaire."
            ),
            level='error',
        )
        self._last_alert_ts = now

    def _handle_recovery(self, snap: Dict):
        msg = (
            f"Scheduler recovered — most recent scan {snap['min_age_s']}s ago."
        )
        self._log_info(msg)
        self._send_alert(
            title="Scheduler recovered",
            body=(
                f"Le scheduler a repris normalement.\n"
                f"Dernier scan il y a `{_fmt_duration(snap['min_age_s'])}`."
            ),
            level='info',
        )

    def _send_alert(self, title: str, body: str, level: str):
        am = self._alert_manager
        if am is None:
            try:
                from src.utils.alerting import get_alert_manager
                am = get_alert_manager()
            except Exception:
                am = None
        if am is None or not getattr(am, 'is_enabled', False):
            # Loud-fail: a frozen scheduler with no webhook means we are flying
            # blind. Log at ERROR level AND print to stderr so it shows up in
            # `docker logs` even if the structured logger is misconfigured.
            # 'recovery' alerts (level='info') are expected to be silent — we
            # only escalate when the underlying event was an error.
            if level == 'error':
                msg = (
                    f"[CRITICAL] Healthcheck wanted to send alert '{title}' "
                    f"but no webhook is configured (DISCORD_WEBHOOK_URL / "
                    f"TELEGRAM_BOT_TOKEN+CHAT_ID). The bot is flying blind: "
                    f"any future scheduler freeze will go undetected. "
                    f"Configure alerting now."
                )
                self._log_error(msg)
                try:
                    print(msg, file=sys.stderr, flush=True)
                except Exception:
                    pass
            else:
                self._log_info(
                    f"Alert webhook not configured — skipping '{title}' notification."
                )
            return
        try:
            am.alert(title, body, level=level)
        except Exception as e:  # pragma: no cover — defensive
            self._log_error(f"Failed to send healthcheck alert: {e}")

    def _log_error(self, msg: str):
        if self._logger:
            try:
                self._logger.error(msg)
                return
            except Exception:
                pass
        print(f"[Healthcheck][ERROR] {msg}", flush=True)

    def _log_info(self, msg: str):
        if self._logger:
            try:
                self._logger.info(msg)
                return
            except Exception:
                pass
        print(f"[Healthcheck] {msg}", flush=True)


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}min"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60:02d}"
    return f"{s // 86400}j{(s % 86400) // 3600}h"


# ---------------------------------------------------------------------------
# Convenience launcher
# ---------------------------------------------------------------------------

_running_thread: Optional[threading.Thread] = None
_running_instance: Optional[SchedulerHealthcheck] = None


def start_healthcheck_thread(
    state_path: str = DEFAULT_SCHEDULER_STATE,
    frozen_threshold_s: int = DEFAULT_FROZEN_THRESHOLD_S,
    check_interval_s: int = DEFAULT_CHECK_INTERVAL_S,
    alert_cooldown_s: int = DEFAULT_ALERT_COOLDOWN_S,
) -> SchedulerHealthcheck:
    """Launch the healthcheck as a daemon thread (idempotent)."""
    global _running_thread, _running_instance
    if _running_thread is not None and _running_thread.is_alive():
        return _running_instance  # already running

    instance = SchedulerHealthcheck(
        state_path=state_path,
        frozen_threshold_s=frozen_threshold_s,
        check_interval_s=check_interval_s,
        alert_cooldown_s=alert_cooldown_s,
    )

    # Surface alerting state at boot so we know immediately whether a freeze
    # would be reported (vs. discovering it silently failed days later).
    try:
        from src.utils.alerting import get_alert_manager
        am = get_alert_manager()
        if am is not None and getattr(am, 'is_enabled', False):
            print(
                f"[Healthcheck] Alerting ENABLED — scheduler freezes will be "
                f"reported (frozen_threshold={frozen_threshold_s}s, "
                f"check_interval={check_interval_s}s).",
                flush=True,
            )
        else:
            warn = (
                f"[Healthcheck][WARN] Alerting DISABLED — DISCORD_WEBHOOK_URL "
                f"and TELEGRAM_BOT_TOKEN/CHAT_ID are not configured. Scheduler "
                f"freezes will be logged but NOT sent anywhere."
            )
            print(warn, flush=True)
            try:
                print(warn, file=sys.stderr, flush=True)
            except Exception:
                pass
    except Exception as e:  # pragma: no cover — defensive
        print(f"[Healthcheck] Could not probe alerting state at boot: {e}", flush=True)

    t = threading.Thread(
        target=instance.run_forever,
        daemon=True,
        name="Scheduler_Healthcheck",
    )
    t.start()
    _running_thread = t
    _running_instance = instance
    return instance
