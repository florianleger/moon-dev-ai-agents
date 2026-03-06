"""
Lightweight price monitor that detects sudden market movements.

Runs as a daemon thread every 2 minutes, making 1 HTTP call to HyperLiquid
to fetch all mid-prices. Detects spikes and records triggered symbols for
the main trading loop to process with priority.
"""

import json
import os
import threading
import time
from collections import deque

import requests
from termcolor import cprint

from src.config import (
    SNIPER_ASSETS,
    ADAPTIVE_HYBRID_ATR_PROFILES,
    LIGHT_CHECK_ENABLED,
    LIGHT_CHECK_INTERVAL_S,
    LIGHT_CHECK_PRICE_THRESHOLDS,
    LIGHT_CHECK_ROLLING_THRESHOLDS,
    LIGHT_CHECK_ROLLING_WINDOW,
)

# Build token -> volatility class mapping from ATR profiles
# btc/eth -> 'large', mid -> 'mid', alt -> 'small'
_PROFILE_TO_VOL_CLASS = {'btc': 'large', 'eth': 'large', 'mid': 'mid', 'alt': 'small'}
_TOKEN_VOL_CLASS = {}
for profile_key, profile in ADAPTIVE_HYBRID_ATR_PROFILES.items():
    vol_class = _PROFILE_TO_VOL_CLASS.get(profile_key, 'mid')
    for token in profile.get('tokens', []):
        _TOKEN_VOL_CLASS[token] = vol_class


def _fetch_all_mids() -> dict:
    """Fetch all mid-prices from HyperLiquid in 1 call. Returns {symbol: float}."""
    resp = requests.post(
        'https://api.hyperliquid.xyz/info',
        json={"type": "allMids"},
        timeout=5,
    )
    resp.raise_for_status()
    return {k: float(v) for k, v in resp.json().items()}


class LightCheck:
    """Lightweight price monitor that detects sudden market movements."""

    _STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'adaptive_hybrid')
    _STATE_FILE = os.path.join(_STATE_DIR, 'light_check_state.json')

    def __init__(self):
        self.last_prices = {}            # {symbol: float}
        self.price_history = {}          # {symbol: deque(maxlen=rolling_window)}
        self.triggered_symbols = set()   # Symbols that triggered a spike
        self.spike_count = 0             # Total spike detections (persisted)
        self._lock = threading.Lock()
        self._thread = None
        self._stop_event = threading.Event()

        # Pre-create deques for monitored assets
        for symbol in SNIPER_ASSETS:
            self.price_history[symbol] = deque(maxlen=LIGHT_CHECK_ROLLING_WINDOW)

        # Restore last_prices from disk
        self._load_state()

    def run_once(self):
        """Execute one check cycle. Cost: 1 HTTP call."""
        try:
            all_mids = _fetch_all_mids()
        except Exception as e:
            cprint(f"[LightCheck] Fetch error: {e}", "yellow")
            return

        with self._lock:
            for symbol in SNIPER_ASSETS:
                price = all_mids.get(symbol)
                if price is None or price <= 0:
                    continue

                vol_class = _TOKEN_VOL_CLASS.get(symbol, 'mid')
                triggered = False

                # --- Check 2-minute spike (vs last price) ---
                last = self.last_prices.get(symbol)
                if last and last > 0:
                    pct_change = (price - last) / last
                    threshold = LIGHT_CHECK_PRICE_THRESHOLDS.get(vol_class, 0.015)
                    if abs(pct_change) >= threshold:
                        self.triggered_symbols.add(symbol)
                        self.spike_count += 1
                        triggered = True
                        cprint(
                            f"[LightCheck] SPIKE detected: {symbol} {pct_change:+.1%} in 2min",
                            "yellow", attrs=['bold'],
                        )

                # --- Check rolling spike (only if not already triggered) ---
                if not triggered:
                    history = self.price_history[symbol]
                    if len(history) >= 2:
                        oldest = history[0]
                        if oldest > 0:
                            rolling_change = (price - oldest) / oldest
                            rolling_threshold = LIGHT_CHECK_ROLLING_THRESHOLDS.get(vol_class, 0.030)
                            if abs(rolling_change) >= rolling_threshold:
                                self.triggered_symbols.add(symbol)
                                self.spike_count += 1
                                window_min = len(history) * (LIGHT_CHECK_INTERVAL_S / 60)
                                cprint(
                                    f"[LightCheck] SPIKE detected: {symbol} {rolling_change:+.1%} "
                                    f"in {window_min:.0f}min (rolling)",
                                    "yellow", attrs=['bold'],
                                )

                # Update state
                self.last_prices[symbol] = price
                self.price_history[symbol].append(price)

        self._save_state()

    def get_and_clear_triggered(self) -> set:
        """Return triggered symbols and clear the set (thread-safe)."""
        with self._lock:
            triggered = self.triggered_symbols.copy()
            self.triggered_symbols.clear()
            return triggered

    def has_triggers(self) -> bool:
        """Quick check if any triggers are pending."""
        with self._lock:
            return len(self.triggered_symbols) > 0

    def _loop(self):
        """Internal loop for the daemon thread."""
        cprint(
            f"[LightCheck] Started (interval={LIGHT_CHECK_INTERVAL_S}s, "
            f"assets={len(SNIPER_ASSETS)}, rolling={LIGHT_CHECK_ROLLING_WINDOW} checks)",
            "cyan",
        )
        while not self._stop_event.is_set():
            self.run_once()
            self._stop_event.wait(timeout=LIGHT_CHECK_INTERVAL_S)

    def start(self):
        """Start the light check daemon thread."""
        if not LIGHT_CHECK_ENABLED:
            cprint("[LightCheck] Disabled (LIGHT_CHECK_ENABLED=False)", "yellow")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="LightCheck",
        )
        self._thread.start()

    def _save_state(self):
        """Persist last_prices to disk (atomic write). Snapshot under lock, I/O outside."""
        try:
            with self._lock:
                data = {'last_prices': dict(self.last_prices), 'spike_count': self.spike_count}
            os.makedirs(self._STATE_DIR, exist_ok=True)
            tmp_path = self._STATE_FILE + '.tmp'
            with open(tmp_path, 'w') as f:
                json.dump(data, f)
            os.rename(tmp_path, self._STATE_FILE)
        except Exception:
            pass  # Non-critical

    def _load_state(self):
        """Restore last_prices from disk."""
        if not os.path.exists(self._STATE_FILE):
            return
        try:
            with open(self._STATE_FILE, 'r') as f:
                state = json.load(f)
            self.last_prices = state.get('last_prices', {})
            self.spike_count = state.get('spike_count', 0)
            cprint(f"[LightCheck] State restored: {len(self.last_prices)} prices, {self.spike_count} spikes", "cyan")
        except Exception:
            pass  # Start fresh

    def stop(self):
        """Signal the thread to stop (for clean shutdown)."""
        self._stop_event.set()
        self._save_state()
