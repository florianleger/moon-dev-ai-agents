"""
Positions API endpoints
"""

import os
from typing import Dict, List

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from src.web.auth import verify_credentials
from src.web.state import get_paper_positions

router = APIRouter()

# Path to paper trades CSV file
PAPER_TRADES_CSV = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ramf', 'paper_trades.csv')


def _get_positions_from_csv() -> List[Dict]:
    """Read open positions from paper_trades.csv file."""
    positions = []

    if not os.path.exists(PAPER_TRADES_CSV):
        return positions

    try:
        df = pd.read_csv(PAPER_TRADES_CSV)
        if df.empty:
            return positions

        # Filter for OPEN positions only
        open_positions = df[df['status'] == 'OPEN']

        for _, row in open_positions.iterrows():
            direction = row.get('direction', 'BUY')
            side = "LONG" if direction == "BUY" else "SHORT"
            entry_price = float(row.get('entry_price', 0))

            positions.append({
                "symbol": row.get('symbol', '?'),
                "side": side,
                "size_usd": float(row.get('position_size', 0)),
                "entry_price": entry_price,
                "current_price": entry_price,  # Will be updated by frontend or separate call
                "unrealized_pnl": 0,  # Calculated dynamically
                "stop_loss": float(row.get('stop_loss', 0)),
                "take_profit": float(row.get('take_profit', 0)),
                "opened_at": row.get('timestamp', ''),
                "position_id": row.get('position_id', ''),
            })

    except Exception as e:
        print(f"[Positions API] Error reading CSV: {e}")

    return positions


@router.get("")
async def get_positions(username: str = Depends(verify_credentials)) -> List[Dict]:
    """Get all open positions."""
    # Read positions from CSV file (shared between processes)
    positions = _get_positions_from_csv()

    if positions:
        return positions

    # Fallback to in-memory singleton (works when API and strategy in same process)
    try:
        from src.strategies.custom.ramf_strategy import RAMFStrategy

        strategy = RAMFStrategy._instance if hasattr(RAMFStrategy, '_instance') else None
        if strategy and hasattr(strategy, 'paper_positions') and strategy.paper_positions:
            for position_id, pos in strategy.paper_positions.items():
                direction = pos.get("direction", "BUY")
                side = "LONG" if direction == "BUY" else "SHORT"
                entry_price = pos.get("entry_price", 0)

                positions.append({
                    "symbol": pos.get("symbol", "?"),
                    "side": side,
                    "size_usd": pos.get("position_size", 0),
                    "entry_price": entry_price,
                    "current_price": entry_price,
                    "unrealized_pnl": 0,
                    "stop_loss": pos.get("stop_loss", 0),
                    "take_profit": pos.get("take_profit", 0),
                    "opened_at": pos.get("timestamp", ""),
                    "position_id": position_id,
                })
    except Exception:
        pass

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

        # Find and close the position (paper_positions is a dict)
        if hasattr(strategy, 'paper_positions') and strategy.paper_positions:
            # Find position by symbol
            position_id_to_close = None
            for position_id, pos in strategy.paper_positions.items():
                if pos.get("symbol") == symbol.upper():
                    position_id_to_close = position_id
                    break

            if position_id_to_close:
                # Get current price
                current_price = 0
                try:
                    df = strategy._fetch_candles(symbol.upper(), interval='15m', candles=5)
                    if df is not None and len(df) > 0:
                        current_price = float(df['close'].iloc[-1])
                except Exception:
                    pass

                # Use strategy's close method
                closed = strategy._close_paper_position(position_id_to_close, current_price, 'MANUAL')
                if closed:
                    return {
                        "status": "closed",
                        "symbol": symbol.upper(),
                        "pnl": round(closed.get('pnl', 0), 2),
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

        # Use strategy's built-in close all method
        if hasattr(strategy, 'close_all_paper_positions'):
            closed = strategy.close_all_paper_positions()
            total_pnl = sum(pos.get('pnl', 0) for pos in closed)
            return {
                "status": "closed",
                "count": len(closed),
                "total_pnl": round(total_pnl, 2),
            }

        return {
            "status": "closed",
            "count": 0,
            "total_pnl": 0.0,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
