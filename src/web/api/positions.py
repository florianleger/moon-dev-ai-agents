"""
Positions API endpoints
"""

from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException

from src.web.auth import verify_credentials
from src.web.state import get_paper_positions

router = APIRouter()


@router.get("")
async def get_positions(username: str = Depends(verify_credentials)) -> List[Dict]:
    """Get all open positions."""
    positions = []

    # Try to get live positions from RAMF strategy
    try:
        from src.strategies.custom.ramf_strategy import RAMFStrategy

        strategy = RAMFStrategy._instance if hasattr(RAMFStrategy, '_instance') else None
        if strategy and hasattr(strategy, 'paper_positions'):
            for pos in strategy.paper_positions:
                # Get current price for PnL calculation
                current_price = pos.get("current_price", pos.get("entry_price", 0))

                positions.append({
                    "symbol": pos.get("symbol", "?"),
                    "side": pos.get("side", "LONG"),
                    "size_usd": pos.get("size_usd", 0),
                    "entry_price": pos.get("entry_price", 0),
                    "current_price": current_price,
                    "unrealized_pnl": pos.get("unrealized_pnl", 0),
                    "stop_loss": pos.get("stop_loss", 0),
                    "take_profit": pos.get("take_profit", 0),
                    "opened_at": pos.get("opened_at", ""),
                })
    except Exception:
        # Fall back to state file
        positions = get_paper_positions()

    return positions


@router.post("/{symbol}/close")
async def close_position(
    symbol: str,
    username: str = Depends(verify_credentials)
) -> Dict:
    """Close a specific position."""
    try:
        from src.strategies.custom.ramf_strategy import RAMFStrategy

        strategy = RAMFStrategy._instance if hasattr(RAMFStrategy, '_instance') else None
        if not strategy:
            raise HTTPException(status_code=503, detail="Strategy not initialized")

        # Find and close the position
        if hasattr(strategy, 'paper_positions'):
            for i, pos in enumerate(strategy.paper_positions):
                if pos.get("symbol") == symbol.upper():
                    # Calculate PnL
                    from src.nice_funcs_hyperliquid import get_current_price
                    current_price = get_current_price(symbol.upper())

                    if pos.get("side") == "LONG":
                        pnl = (current_price - pos["entry_price"]) / pos["entry_price"] * pos["size_usd"]
                    else:
                        pnl = (pos["entry_price"] - current_price) / pos["entry_price"] * pos["size_usd"]

                    # Update balance
                    strategy.paper_balance += pos["size_usd"] + pnl
                    strategy.total_pnl += pnl
                    strategy.daily_pnl += pnl

                    # Remove position
                    strategy.paper_positions.pop(i)

                    return {
                        "status": "closed",
                        "symbol": symbol.upper(),
                        "pnl": round(pnl, 2),
                    }

            raise HTTPException(status_code=404, detail=f"Position {symbol} not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/close-all")
async def close_all_positions(username: str = Depends(verify_credentials)) -> Dict:
    """Close all open positions."""
    try:
        from src.strategies.custom.ramf_strategy import RAMFStrategy

        strategy = RAMFStrategy._instance if hasattr(RAMFStrategy, '_instance') else None
        if not strategy:
            raise HTTPException(status_code=503, detail="Strategy not initialized")

        closed_count = 0
        total_pnl = 0.0

        if hasattr(strategy, 'paper_positions'):
            from src.nice_funcs_hyperliquid import get_current_price

            for pos in list(strategy.paper_positions):
                symbol = pos.get("symbol")
                try:
                    current_price = get_current_price(symbol)

                    if pos.get("side") == "LONG":
                        pnl = (current_price - pos["entry_price"]) / pos["entry_price"] * pos["size_usd"]
                    else:
                        pnl = (pos["entry_price"] - current_price) / pos["entry_price"] * pos["size_usd"]

                    strategy.paper_balance += pos["size_usd"] + pnl
                    strategy.total_pnl += pnl
                    strategy.daily_pnl += pnl
                    total_pnl += pnl
                    closed_count += 1
                except Exception:
                    continue

            strategy.paper_positions.clear()

        return {
            "status": "closed",
            "count": closed_count,
            "total_pnl": round(total_pnl, 2),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
