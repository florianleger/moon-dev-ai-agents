"""
Strategy control API endpoints
"""

from typing import Dict

from fastapi import APIRouter, Depends, HTTPException

from src.web.auth import verify_credentials
from src.web.state import is_strategy_running, set_strategy_running, get_signals_history

router = APIRouter()


@router.get("/status")
async def get_status(username: str = Depends(verify_credentials)) -> Dict:
    """Get strategy status."""
    running = is_strategy_running()

    # Try to get more details from strategy instance
    details = {
        "running": running,
        "strategy": "RAMF",
        "mode": "paper",
    }

    try:
        from src.config import PAPER_TRADING, RAMF_ASSETS
        from src.strategies.custom.ramf_strategy import RAMFStrategy

        details["mode"] = "paper" if PAPER_TRADING else "live"
        details["assets"] = RAMF_ASSETS

        strategy = RAMFStrategy._instance if hasattr(RAMFStrategy, '_instance') else None
        if strategy:
            paper_status = strategy.get_paper_status()
            details.update({
                "open_positions": paper_status.get("open_positions", 0),
                "trades_today": paper_status.get("trades_today", 0),
                "daily_pnl": paper_status.get("daily_pnl", 0),
            })
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

    # Try to get latest from strategy
    try:
        from src.strategies.custom.ramf_strategy import RAMFStrategy

        strategy = RAMFStrategy._instance if hasattr(RAMFStrategy, '_instance') else None
        if strategy and hasattr(strategy, 'signals_history'):
            # Merge with stored signals
            live_signals = list(strategy.signals_history)[:limit]
            # Combine, preferring live signals
            all_signals = live_signals + [s for s in signals if s not in live_signals]
            signals = all_signals[:limit]
    except Exception:
        pass

    return {
        "count": len(signals),
        "signals": signals,
    }
