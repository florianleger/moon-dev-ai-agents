"""
Dashboard API endpoints
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.web.auth import verify_credentials
from src.web.state import get_dashboard_stats, get_signals_history, get_paper_positions

router = APIRouter()


@router.get("/stats")
async def get_stats(username: str = Depends(verify_credentials)) -> Dict:
    """Get dashboard statistics."""
    from src.web.state import is_strategy_running

    stats = get_dashboard_stats()
    stats["running"] = is_strategy_running()

    # Try to get live data from RAMF strategy if available
    try:
        from src.strategies.custom.ramf_strategy import RAMFStrategy
        from src.config import RAMF_MAX_DAILY_TRADES

        # Get singleton instance if it exists
        strategy = RAMFStrategy._instance if hasattr(RAMFStrategy, '_instance') else None
        if strategy:
            paper_status = strategy.get_paper_status()
            stats.update({
                "balance": paper_status.get("paper_balance", stats["balance"]),
                "daily_pnl": paper_status.get("daily_pnl", stats["daily_pnl"]),
                "total_pnl": paper_status.get("total_pnl", stats["total_pnl"]),
                "open_positions": paper_status.get("open_positions", stats["open_positions"]),
                "max_daily_trades": RAMF_MAX_DAILY_TRADES,
            })
    except Exception:
        pass

    return stats


@router.get("/pnl")
async def get_pnl_history(
    days: int = 7,
    username: str = Depends(verify_credentials)
) -> Dict:
    """Get PnL history for chart."""
    # Generate sample data based on signals history
    signals = get_signals_history(limit=100)

    # Group by date and calculate cumulative PnL
    pnl_by_date = {}
    cumulative = 0.0

    for signal in reversed(signals):
        if "pnl" in signal:
            ts = signal.get("timestamp", "")
            if ts:
                date = ts[:10]  # YYYY-MM-DD
                if date not in pnl_by_date:
                    pnl_by_date[date] = 0.0
                pnl_by_date[date] += signal.get("pnl", 0)

    # Build chart data
    today = datetime.now()
    labels = []
    data = []

    for i in range(days - 1, -1, -1):
        date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        labels.append(date[-5:])  # MM-DD format
        cumulative += pnl_by_date.get(date, 0)
        data.append(round(cumulative, 2))

    return {
        "labels": labels,
        "data": data,
    }


@router.get("/sse")
async def sse_updates(username: str = Depends(verify_credentials)):
    """Server-Sent Events for real-time updates."""

    async def event_generator():
        while True:
            stats = get_dashboard_stats()
            positions = get_paper_positions()
            signals = get_signals_history(limit=5)

            # Send stats update
            yield f"event: stats\ndata: {_json_dumps(stats)}\n\n"

            # Send positions update
            yield f"event: positions\ndata: {_json_dumps(positions)}\n\n"

            # Send latest signals
            yield f"event: signals\ndata: {_json_dumps(signals)}\n\n"

            # Wait before next update
            await asyncio.sleep(5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


def _json_dumps(obj) -> str:
    """Convert object to JSON string."""
    import json
    return json.dumps(obj, default=str)
