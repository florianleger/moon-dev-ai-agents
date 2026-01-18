"""
Data Providers Module

Provides unified access to market data from multiple free sources:
- Binance Futures: Real-time liquidations via WebSocket
- HyperLiquid: Funding rates and open interest
"""

from .binance_futures import BinanceLiquidationStream
from .market_data import MarketDataProvider

__all__ = ['BinanceLiquidationStream', 'MarketDataProvider']
