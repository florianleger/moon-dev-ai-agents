"""
Strategy control API endpoints
"""

from typing import Dict

from fastapi import APIRouter, Depends, HTTPException

from src.web.auth import verify_credentials_or_cookie as verify_credentials
from src.web.state import is_strategy_running, set_strategy_running, get_signals_history

router = APIRouter()


@router.get("/status")
async def get_status(username: str = Depends(verify_credentials)) -> Dict:
    """Get strategy status."""
    running = is_strategy_running()

    # Try to get more details from strategy instance
    details = {
        "running": running,
        "strategy": "ADAPTIVE_HYBRID",
        "mode": "paper",
    }

    try:
        from src.config import PAPER_TRADING, ACTIVE_STRATEGY, SNIPER_ASSETS

        details["mode"] = "paper" if PAPER_TRADING else "live"
        details["strategy"] = ACTIVE_STRATEGY.upper()
        details["assets"] = SNIPER_ASSETS
    except Exception:
        pass

    return details


@router.post("/start")
async def start_strategy(username: str = Depends(verify_credentials)) -> Dict:
    """Start the trading strategy."""
    if is_strategy_running():
        return {"status": "already_running", "message": "Strategy is already running"}

    set_strategy_running(True)

    return {
        "status": "started",
        "message": "Strategy marked as running. Main loop will pick up on next cycle.",
    }


@router.post("/stop")
async def stop_strategy(username: str = Depends(verify_credentials)) -> Dict:
    """Stop the trading strategy."""
    if not is_strategy_running():
        return {"status": "already_stopped", "message": "Strategy is not running"}

    set_strategy_running(False)

    return {
        "status": "stopped",
        "message": "Strategy marked as stopped. Will stop after current cycle.",
    }


@router.get("/signals")
async def get_signals(
    limit: int = 50,
    username: str = Depends(verify_credentials)
) -> Dict:
    """Get recent signals."""
    signals = get_signals_history(limit=limit)

    # Signals are stored in web_state.json by strategy_agent - no need for
    # strategy instance fallback (was referencing obsolete RAMFStrategy)

    return {
        "count": len(signals),
        "signals": signals,
    }
