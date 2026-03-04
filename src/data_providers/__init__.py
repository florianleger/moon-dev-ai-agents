"""
Data Providers Module

Provides unified access to market data from multiple free sources:
- Binance Futures: Real-time liquidations via WebSocket
- HyperLiquid: Funding rates and open interest
- Alternative.me: Fear & Greed Index
- DefiLlama: TVL and DEX volumes
- Binance Sentiment: Long/Short ratio + Taker Buy/Sell volume
- CoinGecko Social: Trending coins + global market macro
- Cross-Exchange Funding: HL vs Binance funding rate divergence
"""

from .binance_futures import BinanceLiquidationStream
from .market_data import MarketDataProvider
from .fear_greed import FearGreedProvider
from .defi_llama import DefiLlamaProvider
from .binance_sentiment import BinanceSentimentProvider
from .coingecko_social import CoinGeckoSocialProvider
from .cross_exchange_funding import CrossExchangeFundingProvider

__all__ = [
    'BinanceLiquidationStream', 'MarketDataProvider',
    'FearGreedProvider', 'DefiLlamaProvider',
    'BinanceSentimentProvider', 'CoinGeckoSocialProvider',
    'CrossExchangeFundingProvider',
]
