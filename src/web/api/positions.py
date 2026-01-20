"""
Positions API endpoints
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from src.web.auth import verify_credentials
from src.web.state import get_paper_positions

router = APIRouter()

# Path to paper trades CSV file
PAPER_TRADES_CSV = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ramf', 'paper_trades.csv')


def _get_positions_from_csv() -> List[Dict]:
    """Read open positions from paper_trades.csv file with real-time PnL."""
    positions = []

    if not os.path.exists(PAPER_TRADES_CSV):
        return positions

    try:
        df = pd.read_csv(PAPER_TRADES_CSV)
        if df.empty:
            return positions

        # Filter for OPEN positions only
        open_positions = df[df['status'] == 'OPEN']

        # Get market data provider singleton (uses 30s cache)
        provider = None
        try:
            from src.data_providers.market_data import get_market_data_provider
            provider = get_market_data_provider()
        except Exception:
            pass

        for _, row in open_positions.iterrows():
            symbol = row.get('symbol', '?')
            direction = row.get('direction', 'BUY')
            side = "LONG" if direction == "BUY" else "SHORT"
            entry_price = float(row.get('entry_price', 0))
            position_size = float(row.get('position_size', 0))

            # Get current price from HyperLiquid
            current_price = entry_price
            if provider:
                try:
                    price = provider.get_current_price(symbol)
                    if price:
                        current_price = price
                except Exception:
                    pass

            # Calculate unrealized PnL
            unrealized_pnl = 0.0
            if entry_price > 0:
                if direction == "BUY":
                    unrealized_pnl = position_size * (current_price - entry_price) / entry_price
                else:
                    unrealized_pnl = position_size * (entry_price - current_price) / entry_price

            positions.append({
                "symbol": symbol,
                "side": side,
                "size_usd": position_size,
                "entry_price": entry_price,
                "current_price": current_price,
                "unrealized_pnl": round(unrealized_pnl, 2),
                "stop_loss": float(row.get('stop_loss', 0)),
                "take_profit": float(row.get('take_profit', 0)),
                "opened_at": row.get('timestamp', ''),
                "position_id": row.get('position_id', ''),
            })

    except Exception as e:
        print(f"[Positions API] Error reading CSV: {e}")

    return positions


def _close_position_in_csv(symbol: str, exit_price: float) -> Optional[Dict]:
    """Close a position directly in the CSV file. Returns closed position info or None."""
    if not os.path.exists(PAPER_TRADES_CSV):
        return None

    try:
        df = pd.read_csv(PAPER_TRADES_CSV)
        if df.empty:
            return None

        # Find the OPEN position for this symbol
        mask = (df['status'] == 'OPEN') & (df['symbol'].str.upper() == symbol.upper())
        if not mask.any():
            return None

        idx = df[mask].index[0]
        row = df.loc[idx]

        # Calculate PnL
        entry_price = float(row.get('entry_price', 0))
        position_size = float(row.get('position_size', 0))
        direction = row.get('direction', 'BUY')

        if direction == 'BUY':
            pnl = position_size * (exit_price - entry_price) / entry_price
        else:
            pnl = position_size * (entry_price - exit_price) / entry_price

        # Update the row
        df.loc[idx, 'status'] = 'CLOSED_MANUAL'
        df.loc[idx, 'exit_price'] = exit_price
        df.loc[idx, 'exit_time'] = datetime.now().isoformat()
        df.loc[idx, 'pnl'] = round(pnl, 2)

        # Save back to CSV
        df.to_csv(PAPER_TRADES_CSV, index=False)

        return {
            'symbol': symbol.upper(),
            'pnl': round(pnl, 2),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
        }

    except Exception as e:
        print(f"[Positions API] Error closing position in CSV: {e}")
        return None


def _close_all_positions_in_csv() -> List[Dict]:
    """Close all open positions directly in the CSV file. Returns list of closed positions."""
    closed = []

    if not os.path.exists(PAPER_TRADES_CSV):
        return closed

    try:
        df = pd.read_csv(PAPER_TRADES_CSV)
        if df.empty:
            return closed

        # Find all OPEN positions
        mask = df['status'] == 'OPEN'
        if not mask.any():
            return closed

        # Get market data provider singleton (uses 30s cache)
        provider = None
        try:
            from src.data_providers.market_data import get_market_data_provider
            provider = get_market_data_provider()
        except Exception:
            pass

        # Get current prices for all symbols (use entry price as fallback)
        for idx in df[mask].index:
            row = df.loc[idx]
            symbol = row.get('symbol', '')
            entry_price = float(row.get('entry_price', 0))
            position_size = float(row.get('position_size', 0))
            direction = row.get('direction', 'BUY')

            # Try to get current price, fallback to entry price (no loss assumed)
            exit_price = entry_price
            if provider:
                try:
                    current = provider.get_current_price(symbol)
                    if current:
                        exit_price = current
                except Exception:
                    pass

            # Calculate PnL
            if direction == 'BUY':
                pnl = position_size * (exit_price - entry_price) / entry_price if entry_price > 0 else 0
            else:
                pnl = position_size * (entry_price - exit_price) / entry_price if entry_price > 0 else 0

            # Update the row
            df.loc[idx, 'status'] = 'CLOSED_MANUAL'
            df.loc[idx, 'exit_price'] = exit_price
            df.loc[idx, 'exit_time'] = datetime.now().isoformat()
            df.loc[idx, 'pnl'] = round(pnl, 2)

            closed.append({
                'symbol': symbol,
                'pnl': round(pnl, 2),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position_size': position_size,
            })

        # Save back to CSV
        df.to_csv(PAPER_TRADES_CSV, index=False)

    except Exception as e:
        print(f"[Positions API] Error closing all positions in CSV: {e}")

    return closed


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

        # Try strategy-based close first
        if strategy and hasattr(strategy, 'paper_positions') and strategy.paper_positions:
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

        # Fallback: Close directly via CSV when strategy not running
        # Try to get current price
        exit_price = 0
        try:
            from src.data_providers.market_data import get_market_data_provider
            provider = get_market_data_provider()
            exit_price = provider.get_current_price(symbol.upper()) or 0
        except Exception:
            pass

        # If no price available, read entry price from CSV
        if exit_price == 0:
            positions = _get_positions_from_csv()
            for pos in positions:
                if pos.get('symbol', '').upper() == symbol.upper():
                    exit_price = pos.get('entry_price', 0)
                    break

        result = _close_position_in_csv(symbol, exit_price)
        if result:
            return {
                "status": "closed",
                "symbol": result['symbol'],
                "pnl": result['pnl'],
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

        # Try strategy-based close first
        if strategy and hasattr(strategy, 'close_all_paper_positions'):
            closed = strategy.close_all_paper_positions()
            total_pnl = sum(pos.get('pnl', 0) for pos in closed)
            return {
                "status": "closed",
                "count": len(closed),
                "total_pnl": round(total_pnl, 2),
            }

        # Fallback: Close directly via CSV when strategy not running
        closed = _close_all_positions_in_csv()
        total_pnl = sum(pos.get('pnl', 0) for pos in closed)
        return {
            "status": "closed",
            "count": len(closed),
            "total_pnl": round(total_pnl, 2),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
