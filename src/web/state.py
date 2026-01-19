"""
Shared state management for trading system and web dashboard.
Uses a JSON file for persistence across restarts.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# State file path - use /app/data in Docker, or project/data locally
_data_dir = os.getenv("DATA_DIR", str(Path(__file__).parent.parent.parent / "data"))
STATE_FILE = Path(_data_dir) / "web_state.json"


def _default_state() -> Dict:
    """Default state structure."""
    return {
        "running": False,
        "last_updated": datetime.now().isoformat(),
        "signals_history": [],
        "paper_positions": [],
        "paper_balance": 500.0,
        "initial_balance": 500.0,
        "daily_pnl": 0.0,
        "total_pnl": 0.0,
        "trades_today": 0,
    }


def _ensure_dir():
    """Ensure state directory exists."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load_state() -> Dict:
    """Load state from file."""
    _ensure_dir()
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return _default_state()


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
