"""
Smart Scheduler: priority-queue based token analysis scheduler.

Replaces the fixed "analyze all tokens every N minutes" cycle with intelligent
per-token scheduling based on positions, score proximity, regime, and spikes.
State is persisted to disk to survive restarts.

Fairness: a per-token scan counter prevents any single token from exceeding
MAX_SCAN_RATIO times the average scan count across all tokens.
"""

import heapq
import json
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from termcolor import cprint

from src.config import (
    FULL_CHECK_COOLDOWN_S,
    FULL_CHECK_BASE_INTERVAL_MIN,
    FULL_CHECK_POSITION_INTERVAL_MIN,
)


@dataclass(order=True)
class FullCheckRequest:
    """A scheduled analysis request for a single token."""
    scheduled_at: float = field(compare=True)       # Unix timestamp (for heap ordering)
    priority: int = field(compare=True)              # 1=spike, 2=position, 3=near threshold, 4=routine
    symbol: str = field(compare=False)
    reason: str = field(compare=False)


class SmartScheduler:
    """Priority-queue based scheduler for token analysis.

    Thread-safe: accessed by the main loop and potentially by LightCheck
    via enqueue_spike().
    """

    PRIORITY_SPIKE = 1
    PRIORITY_POSITION = 2
    PRIORITY_NEAR_THRESHOLD = 3
    PRIORITY_ROUTINE = 4

    _PRIORITY_LABELS = {
        1: 'spike',
        2: 'position',
        3: 'near_threshold',
        4: 'routine',
    }

    # Regimes that warrant more frequent checks
    _ACTIVE_REGIMES = {'MARKUP', 'MARKDOWN', 'CAPITULATION', 'EUPHORIA'}

    # Fairness: max scans for one token = MAX_SCAN_RATIO * average scans across all tokens
    MAX_SCAN_RATIO = 2.5

    # Minimum recheck interval (minutes) regardless of volatility/regime multipliers.
    # This prevents volatile tokens from being scanned every 3 min while others wait 10+.
    MIN_RECHECK_INTERVAL_MIN = 5

    # Hard safety floor: a token cannot be rescheduled to run within this window
    # of its last scan. Protects against any code path (routine, position, spike)
    # monopolising the queue. Positions still need monitoring but even they can
    # wait this long between full LLM-backed analyses.
    MIN_TOKEN_INTERVAL_S = 90

    # Fairness penalty: if a token is over-represented, FULLY skip its recheck (do not
    # re-enqueue it). The token will get a fresh entry on the next natural cycle via
    # enqueue_all_routine(). Previously this was a soft +5min delay, but that allowed
    # the over-represented token to come right back and re-monopolise the queue
    # (concentration loop). Full-skip is the only way to break the loop.
    OVER_REPRESENTED_DELAY_S = 300  # kept for backward compat / metrics; unused for skip

    _STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'adaptive_hybrid')
    _STATE_FILE = os.path.join(_STATE_DIR, 'scheduler_state.json')

    def __init__(self):
        self._queue: list[FullCheckRequest] = []    # heapq
        self._last_check: dict[str, float] = {}     # {symbol: timestamp}
        self._last_result: dict[str, dict] = {}     # {symbol: {score, threshold, regime, ...}}
        self._lock = threading.Lock()
        self._cooldown_s = FULL_CHECK_COOLDOWN_S
        self._scheduled_symbols: set[str] = set()   # symbols currently in the queue
        self._scan_count: dict[str, int] = defaultdict(int)  # {symbol: count} for fairness tracking
        self._all_symbols: list[str] = []            # all monitored symbols (set at enqueue_all_routine)

    def schedule_recheck(self, symbol: str, result: dict, has_position: bool):
        """After a full check, compute and enqueue the next recheckAt.

        Fairness: if this token is over-represented in recent scans, the recheck
        is FULLY SKIPPED (no re-enqueue). The token will re-enter the queue on
        the next enqueue_all_routine() cycle, ensuring all monitored symbols get
        a turn. Otherwise, MIN_TOKEN_INTERVAL_S is enforced as a hard safety
        floor on the recheck delay.
        """
        # Fairness FIRST: if this token is over-represented, FULL-SKIP — do not
        # re-enqueue at all. The token will be picked up again on the next
        # enqueue_all_routine() cycle. Previously we just delayed by +5min which
        # let the same token come back and re-monopolise the queue (the bug).
        with self._lock:
            over = self._is_over_represented(symbol)
        if over:
            cprint(f"  [Scheduler] {symbol} over-represented, SKIPPING recheck (will re-enter on next routine cycle)", "yellow")
            return

        interval_min = self._compute_interval_minutes(symbol, result, has_position)
        delay_s = interval_min * 60

        # Hard safety floor applied AFTER the over-rep decision so it can never
        # override a fairness skip.
        delay_s = max(delay_s, self.MIN_TOKEN_INTERVAL_S)

        scheduled_at = time.time() + delay_s

        if has_position:
            priority = self.PRIORITY_POSITION
        else:
            score = result.get('score', 0)
            threshold = result.get('threshold', 40)
            proximity = abs(score - threshold) / max(threshold, 1)
            if proximity < 0.15:
                priority = self.PRIORITY_NEAR_THRESHOLD
            else:
                priority = self.PRIORITY_ROUTINE

        reason = f"recheck ({self._PRIORITY_LABELS[priority]}, {int(delay_s/60)}min)"
        self._enqueue(symbol, scheduled_at, priority, reason)

    def enqueue_spike(self, symbol: str, reason: str):
        """Enqueue a token detected by the light check (highest priority).

        Respects cooldown: if the token was checked very recently, skip.
        Also respects fairness: if the token is already over-represented, skip.
        """
        now = time.time()
        with self._lock:
            last = self._last_check.get(symbol, 0)
            if now - last < self._cooldown_s:
                return  # Too soon, skip
            if self._is_over_represented(symbol):
                return  # Fairness: this token has had enough scans
            self._enqueue_locked(symbol, now, self.PRIORITY_SPIKE, reason)

    def enqueue_all_routine(self, symbols: list):
        """Seed the queue with all tokens at startup (staggered by 2s each).

        Fairness: guarantees that EVERY monitored symbol has a queue entry scheduled
        within the next ~2 * len(symbols) seconds, even if previous state on disk had
        a symbol scheduled far in the future (or missed entirely). This prevents a
        single token from monopolising the rotation after a restart.
        """
        now = time.time()
        with self._lock:
            self._all_symbols = list(symbols)
            for i, symbol in enumerate(symbols):
                scheduled_at = now + i * 2  # Stagger to avoid burst

                # Force-remove any existing stale entry for this symbol, then
                # insert a fresh "initial" entry. Without this, a restored queue
                # entry scheduled far in the future would be kept and the token
                # would never fire on startup.
                if symbol in self._scheduled_symbols:
                    self._queue = [r for r in self._queue if r.symbol != symbol]
                    heapq.heapify(self._queue)
                    self._scheduled_symbols.discard(symbol)

                self._enqueue_locked(symbol, scheduled_at, self.PRIORITY_ROUTINE, "initial")

    def get_due_symbols(self) -> list[FullCheckRequest]:
        """Return all tokens whose recheckAt has passed, respecting cooldown.

        Pops items from the heap whose scheduled_at <= now.
        Deduplicates: if a symbol appears multiple times, only the first
        (highest priority) is returned. Items in cooldown are re-enqueued.
        """
        now = time.time()
        due: list[FullCheckRequest] = []
        seen: set[str] = set()
        requeue: list[FullCheckRequest] = []

        with self._lock:
            while self._queue and self._queue[0].scheduled_at <= now:
                req = heapq.heappop(self._queue)
                self._scheduled_symbols.discard(req.symbol)

                if req.symbol in seen:
                    continue  # Already in this batch, drop duplicate

                # Cooldown check — re-enqueue after cooldown expires
                last = self._last_check.get(req.symbol, 0)
                if now - last < self._cooldown_s:
                    req = FullCheckRequest(
                        scheduled_at=last + self._cooldown_s,
                        priority=req.priority,
                        symbol=req.symbol,
                        reason=req.reason,
                    )
                    requeue.append(req)
                    continue

                seen.add(req.symbol)
                due.append(req)

            # Re-enqueue cooldown items so they aren't lost
            for req in requeue:
                heapq.heappush(self._queue, req)
                self._scheduled_symbols.add(req.symbol)

        return due

    def record_result(self, symbol: str, result: dict):
        """Record the result of a full check (updates timestamp, last result, and scan count)."""
        with self._lock:
            self._last_check[symbol] = time.time()
            self._last_result[symbol] = result
            self._scan_count[symbol] += 1
        self.save_state()

    def get_last_result(self, symbol: str) -> dict:
        """Thread-safe accessor for a token's last analysis result."""
        with self._lock:
            return dict(self._last_result.get(symbol, {}))

    def queue_size(self) -> int:
        """Return current queue size (for logging)."""
        with self._lock:
            return len(self._queue)

    def get_scan_distribution(self) -> dict:
        """Return scan counts for monitoring (thread-safe)."""
        with self._lock:
            return dict(self._scan_count)

    def save_state(self):
        """Persist scheduler state to disk (atomic write via tmp+rename)."""
        try:
            with self._lock:
                state = {
                    'last_check': dict(self._last_check),
                    'last_result': dict(self._last_result),
                    'scan_count': dict(self._scan_count),
                    'queue': [{'scheduled_at': r.scheduled_at, 'priority': r.priority,
                               'symbol': r.symbol, 'reason': r.reason} for r in self._queue],
                }
            os.makedirs(self._STATE_DIR, exist_ok=True)
            tmp = self._STATE_FILE + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(state, f, indent=2)
            os.rename(tmp, self._STATE_FILE)
        except Exception as e:
            cprint(f"[Scheduler] save_state error: {e}", "yellow")

    def load_state(self):
        """Restore scheduler state from disk (called at startup)."""
        if not os.path.exists(self._STATE_FILE):
            return
        try:
            with open(self._STATE_FILE, 'r') as f:
                state = json.load(f)
            now = time.time()
            with self._lock:
                self._last_check = state.get('last_check', {})
                self._last_result = state.get('last_result', {})
                # Restore scan counts (reset if absent — fresh start)
                for k, v in state.get('scan_count', {}).items():
                    self._scan_count[k] = v
                expired_count = 0
                for item in state.get('queue', []):
                    if item['scheduled_at'] < now:
                        t = now + expired_count * 2  # Stagger expired items
                        expired_count += 1
                    else:
                        t = item['scheduled_at']
                    self._enqueue_locked(item['symbol'], t, item['priority'], item['reason'])
            cprint(f"[Scheduler] State restored: {len(self._last_result)} results, "
                   f"{len(state.get('queue', []))} queued items", "cyan")
        except Exception as e:
            cprint(f"[Scheduler] load_state error (starting fresh): {e}", "yellow")

    def _is_over_represented(self, symbol: str) -> bool:
        """Check if a token has been scanned too many times relative to others.

        Returns True if the token's scan count exceeds MAX_SCAN_RATIO * median.
        Using median instead of average makes the guard robust to a runaway
        token whose own count inflates the average (self-dilution bug).

        Must be called with self._lock held.
        """
        if not self._all_symbols or len(self._all_symbols) < 2:
            return False
        my_count = self._scan_count.get(symbol, 0)
        if my_count < 5:
            return False  # Don't throttle tokens that have barely been scanned

        counts = sorted(self._scan_count.get(s, 0) for s in self._all_symbols)
        n = len(counts)
        median = counts[n // 2] if n % 2 == 1 else (counts[n // 2 - 1] + counts[n // 2]) / 2
        if median < 1:
            return False  # Not enough data yet
        return my_count > self.MAX_SCAN_RATIO * median

    def _compute_interval_minutes(self, symbol: str, result: dict, has_position: bool) -> int:
        """Compute the recheck interval in minutes based on market context."""
        base = FULL_CHECK_BASE_INTERVAL_MIN  # default: 10 min

        # Position ouverte -> monitoring serre
        if has_position:
            base = FULL_CHECK_POSITION_INTERVAL_MIN  # default: 3 min

        # Score proche du threshold -> pourrait basculer
        score = result.get('score', 0)
        threshold = result.get('threshold', 40)
        if threshold > 0:
            proximity = abs(score - threshold) / threshold
            if proximity < 0.15:
                base = min(base, 5)

        # Regime actif -> plus frequent
        regime = result.get('regime', '')
        if regime in self._ACTIVE_REGIMES:
            base *= 0.7  # Was 0.6 — reduced discount to limit concentration bias

        # High ATR % of price -> volatile, check more often
        atr_pct = result.get('atr_pct', 0)
        if atr_pct > 0.02:  # ATR > 2% of price
            base *= 0.7  # Was 0.5 — reduced discount to limit concentration bias

        # Heures creuses UTC 0-5 -> espacer
        hour = datetime.now(timezone.utc).hour
        if 0 <= hour < 6:
            base *= 2.0

        # Fairness floor: don't let volatile tokens dominate the scanner.
        # Positions still get the tighter interval (3 min) since they need monitoring.
        if not has_position:
            base = max(base, self.MIN_RECHECK_INTERVAL_MIN)

        return max(3, min(30, int(base)))

    def _enqueue(self, symbol: str, scheduled_at: float, priority: int, reason: str):
        """Thread-safe enqueue (acquires lock)."""
        with self._lock:
            self._enqueue_locked(symbol, scheduled_at, priority, reason)

    def _enqueue_locked(self, symbol: str, scheduled_at: float, priority: int, reason: str):
        """Enqueue without acquiring lock (caller must hold self._lock).

        If the symbol is already queued, only replace if the new priority
        is higher (lower number). Rebuilds the heap to avoid orphan duplicates.
        """
        if symbol in self._scheduled_symbols:
            if priority >= self.PRIORITY_SPIKE + 1:
                return  # Already scheduled, don't duplicate for non-spike

            # For spike: remove the existing entry first, then re-add with spike priority.
            # This avoids orphan duplicates that cause extra scans.
            self._queue = [r for r in self._queue if r.symbol != symbol]
            heapq.heapify(self._queue)
            self._scheduled_symbols.discard(symbol)

        req = FullCheckRequest(
            scheduled_at=scheduled_at,
            priority=priority,
            symbol=symbol,
            reason=reason,
        )
        heapq.heappush(self._queue, req)
        self._scheduled_symbols.add(symbol)
