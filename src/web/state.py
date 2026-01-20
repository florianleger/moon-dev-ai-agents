"""
Shared state management for trading system and web dashboard.
Uses a JSON file for persistence across restarts.

Key behaviors:
- AUTO_START env var controls whether strategy starts on fresh deployment
- Existing state is always respected (user's last choice)
- State persists via Docker volume at /app/data/web_state.json
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from termcolor import cprint

# State file path - use /app/data in Docker, or project/data locally
_data_dir = os.getenv("DATA_DIR", str(Path(__file__).parent.parent.parent / "data"))
STATE_FILE = Path(_data_dir) / "web_state.json"


def _get_auto_start() -> bool:
    """Check AUTO_START env var. Default is True for trading bots."""
    return os.getenv("AUTO_START", "true").lower() in ("true", "1", "yes")


def _get_initial_balance() -> float:
    """Get initial paper trading balance from env."""
    return float(os.getenv("PAPER_TRADING_BALANCE", "500.0"))


def _default_state() -> Dict:
    """
    Default state structure for NEW deployments.
    Uses AUTO_START env var to determine initial running state.
    """
    auto_start = _get_auto_start()
    initial_balance = _get_initial_balance()

    return {
        "running": auto_start,
        "last_updated": datetime.now().isoformat(),
        "signals_history": [],
        "paper_positions": [],
        "paper_balance": initial_balance,
        "initial_balance": initial_balance,
        "daily_pnl": 0.0,
        "total_pnl": 0.0,
        "trades_today": 0,
    }


def _ensure_dir():
    """Ensure state directory exists."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load_state() -> Dict:
    """Load state from file, or create default if not exists."""
    _ensure_dir()
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return _default_state()


def ensure_state_initialized() -> Dict:
    """
    Ensure state file exists and is valid.
    Creates with defaults if needed.
    Returns the current state and logs initialization status.

    Call this at startup to ensure predictable behavior.
    """
    _ensure_dir()
    state_existed = STATE_FILE.exists()

    if state_existed:
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
                cprint(f"[State] Restored from {STATE_FILE}", "green")
                cprint(f"[State] Strategy running: {state.get('running', False)}", "cyan")
                return state
        except json.JSONDecodeError:
            cprint(f"[State] Warning: Corrupted state file, recreating...", "yellow")
            state_existed = False

    # Create new state file with defaults
    state = _default_state()
    _save_state(state)

    auto_start = _get_auto_start()
    cprint(f"[State] Created new state file at {STATE_FILE}", "green")
    cprint(f"[State] AUTO_START={auto_start} → Strategy running: {state['running']}", "cyan")

    if auto_start:
        cprint("[State] Strategy will auto-start (set AUTO_START=false to disable)", "yellow")

    return state


def _save_state(state: Dict):
    """Save state to file."""
    _ensure_dir()
    state["last_updated"] = datetime.now().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def get_strategy_state() -> Dict:
    """Get current strategy state."""
    return _load_state()


def set_strategy_running(running: bool):
    """Set strategy running state."""
    state = _load_state()
    state["running"] = running
    _save_state(state)


def is_strategy_running() -> bool:
    """Check if strategy should be running."""
    return _load_state().get("running", False)


def add_signal(signal: Dict):
    """Add a signal to history."""
    state = _load_state()
    signal["timestamp"] = datetime.now().isoformat()
    state["signals_history"].insert(0, signal)
    # Keep last 100 signals
    state["signals_history"] = state["signals_history"][:100]
    _save_state(state)


def get_signals_history(limit: int = 50) -> List[Dict]:
    """Get recent signals."""
    state = _load_state()
    return state.get("signals_history", [])[:limit]


def update_paper_status(
    balance: float,
    positions: List[Dict],
    daily_pnl: float,
    total_pnl: float,
    trades_today: int
):
    """Update paper trading status from RAMF strategy."""
    state = _load_state()
    state["paper_balance"] = balance
    state["paper_positions"] = positions
    state["daily_pnl"] = daily_pnl
    state["total_pnl"] = total_pnl
    state["trades_today"] = trades_today
    _save_state(state)


def get_paper_positions() -> List[Dict]:
    """Get current paper positions."""
    return _load_state().get("paper_positions", [])


def get_dashboard_stats() -> Dict:
    """Get all stats for dashboard."""
    state = _load_state()
    return {
        "balance": state.get("paper_balance", 500.0),
        "initial_balance": state.get("initial_balance", 500.0),
        "daily_pnl": state.get("daily_pnl", 0.0),
        "total_pnl": state.get("total_pnl", 0.0),
        "open_positions": len(state.get("paper_positions", [])),
        "trades_today": state.get("trades_today", 0),
        "running": state.get("running", False),
    }
