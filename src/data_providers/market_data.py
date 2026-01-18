"""
Market Data Provider

Unified interface for market data from multiple free sources:
- HyperLiquid: Funding rates and open interest
- Binance Futures: Real-time liquidations

This module eliminates the dependency on Moon Dev API.
"""

import time
from typing import Dict, Optional
from termcolor import cprint

from .binance_futures import get_liquidation_stream, get_liquidation_ratio


class MarketDataProvider:
    """
    Unified market data provider using free APIs.

    Data Sources:
    - Funding Rates: HyperLiquid (metaAndAssetCtxs endpoint)
    - Open Interest: HyperLiquid (metaAndAssetCtxs endpoint)
    - Liquidations: Binance Futures WebSocket

    All data is real-time and free of charge.
    """

    def __init__(self, start_liquidation_stream: bool = True):
        """
        Initialize the market data provider.

        Args:
            start_liquidation_stream: Whether to start Binance WebSocket immediately
        """
        self._liquidation_stream = None
        self._hl_cache = {}
        self._hl_cache_time = {}
        self._cache_ttl = 30  # Cache HyperLiquid data for 30 seconds

        if start_liquidation_stream:
            self._init_liquidation_stream()

    def _init_liquidation_stream(self):
        """Initialize the Binance liquidation stream."""
        try:
            self._liquidation_stream = get_liquidation_stream()
            if not self._liquidation_stream.is_connected:
                self._liquidation_stream.start_stream()
                # Give it time to connect and collect initial data
                time.sleep(2)
        except Exception as e:
            cprint(f"[MarketData] Warning: Could not start liquidation stream: {e}", "yellow")

    def get_funding_rate(self, symbol: str) -> Optional[Dict]:
        """
        Get current funding rate from HyperLiquid.

        Args:
            symbol: Asset symbol (e.g., 'BTC', 'ETH', 'SOL')

        Returns:
            Dict with funding_rate, mark_price, open_interest
            Or None if unavailable
        """
        # Check cache first
        cache_key = f"funding_{symbol}"
        if cache_key in self._hl_cache:
            age = time.time() - self._hl_cache_time.get(cache_key, 0)
            if age < self._cache_ttl:
                return self._hl_cache[cache_key]

        try:
            # Import here to avoid circular imports
            from src.nice_funcs_hyperliquid import get_funding_rates

            data = get_funding_rates(symbol)
            if data:
                # Cache the result
                self._hl_cache[cache_key] = data
                self._hl_cache_time[cache_key] = time.time()
                return data

            return None

        except Exception as e:
            cprint(f"[MarketData] Error getting funding rate for {symbol}: {e}", "yellow")
            return None

    def get_open_interest(self, symbol: str) -> Optional[Dict]:
        """
        Get current open interest from HyperLiquid.

        Args:
            symbol: Asset symbol (e.g., 'BTC', 'ETH', 'SOL')

        Returns:
            Dict with open_interest and mark_price
            Or None if unavailable
        """
        # OI is included in funding rate response
        data = self.get_funding_rate(symbol)
        if data:
            return {
                'open_interest': data.get('open_interest', 0),
                'mark_price': data.get('mark_price', 0)
            }
        return None

    def get_funding_zscore(self, symbol: str) -> float:
        """
        Calculate funding rate Z-score approximation.

        Uses typical perpetual funding distribution:
        - Mean: ~10% annual (slight long bias)
        - Std: ~15% annual

        Args:
            symbol: Asset symbol

        Returns:
            float: Z-score (-3 to +3 typical range), 0 if unavailable
        """
        try:
            data = self.get_funding_rate(symbol)
            if data and 'funding_rate' in data:
                # HyperLiquid returns hourly funding rate
                hourly_rate = data['funding_rate']
                annual_rate = hourly_rate * 24 * 365 * 100  # Convert to annual %

                # Typical funding parameters
                mean_funding = 10.0  # 10% annual mean
                std_funding = 15.0   # 15% annual std

                zscore = (annual_rate - mean_funding) / std_funding
                return round(zscore, 2)

            return 0.0

        except Exception as e:
            cprint(f"[MarketData] Error calculating funding Z-score: {e}", "yellow")
            return 0.0

    def get_liquidation_ratio(self, minutes: int = 15) -> float:
        """
        Get long/short liquidation ratio from Binance.

        Ratio > 1.0 = More longs liquidated (bearish pressure)
        Ratio < 1.0 = More shorts liquidated (bullish pressure)
        Ratio = 1.0 = Balanced (or no data)

        Args:
            minutes: Lookback period in minutes

        Returns:
            float: Long/Short liquidation ratio (1.0 if no data)
        """
        try:
            if self._liquidation_stream is None:
                self._init_liquidation_stream()

            if self._liquidation_stream and self._liquidation_stream.is_connected:
                return self._liquidation_stream.get_liquidation_ratio(minutes)
            else:
                # Fallback: try REST API
                return get_liquidation_ratio(minutes)

        except Exception as e:
            cprint(f"[MarketData] Error getting liquidation ratio: {e}", "yellow")
            return 1.0  # Neutral fallback

    def get_liquidation_summary(self, minutes: int = 15) -> Dict:
        """
        Get detailed liquidation summary.

        Args:
            minutes: Lookback period in minutes

        Returns:
            Dict with liquidation statistics
        """
        try:
            if self._liquidation_stream is None:
                self._init_liquidation_stream()

            if self._liquidation_stream:
                return self._liquidation_stream.get_liquidation_summary(minutes)

            return {
                'total_count': 0,
                'total_usd': 0.0,
                'long_usd': 0.0,
                'short_usd': 0.0,
                'ratio': 1.0,
                'top_symbols': []
            }

        except Exception as e:
            cprint(f"[MarketData] Error getting liquidation summary: {e}", "yellow")
            return {
                'total_count': 0,
                'total_usd': 0.0,
                'long_usd': 0.0,
                'short_usd': 0.0,
                'ratio': 1.0,
                'top_symbols': []
            }

    def get_market_snapshot(self, symbol: str) -> Dict:
        """
        Get a complete market snapshot for an asset.

        Combines funding, OI, and liquidation data.

        Args:
            symbol: Asset symbol (e.g., 'BTC')

        Returns:
            Dict with comprehensive market data
        """
        funding = self.get_funding_rate(symbol)
        liq_summary = self.get_liquidation_summary(minutes=15)

        return {
            'symbol': symbol,
            'funding_rate': funding.get('funding_rate', 0) if funding else 0,
            'funding_zscore': self.get_funding_zscore(symbol),
            'open_interest': funding.get('open_interest', 0) if funding else 0,
            'mark_price': funding.get('mark_price', 0) if funding else 0,
            'liquidation_ratio': liq_summary.get('ratio', 1.0),
            'total_liquidations_usd': liq_summary.get('total_usd', 0),
            'long_liquidations_usd': liq_summary.get('long_usd', 0),
            'short_liquidations_usd': liq_summary.get('short_usd', 0),
        }

    def cleanup(self):
        """Cleanup resources (stop WebSocket, etc.)."""
        if self._liquidation_stream:
            self._liquidation_stream.stop_stream()


# Singleton instance
_provider: Optional[MarketDataProvider] = None


def get_market_data_provider() -> MarketDataProvider:
    """
    Get or create the singleton market data provider.

    Returns:
        MarketDataProvider: Shared instance
    """
    global _provider
    if _provider is None:
        _provider = MarketDataProvider()
    return _provider


# Convenience functions for direct access
def get_funding_rate(symbol: str) -> Optional[Dict]:
    """Get funding rate for a symbol."""
    return get_market_data_provider().get_funding_rate(symbol)


def get_open_interest(symbol: str) -> Optional[Dict]:
    """Get open interest for a symbol."""
    return get_market_data_provider().get_open_interest(symbol)


def get_funding_zscore(symbol: str) -> float:
    """Get funding rate Z-score for a symbol."""
    return get_market_data_provider().get_funding_zscore(symbol)


# For standalone testing
if __name__ == "__main__":
    cprint("\n" + "=" * 60, "cyan")
    cprint("  Testing Market Data Provider", "cyan", attrs=['bold'])
    cprint("=" * 60, "cyan")

    provider = MarketDataProvider(start_liquidation_stream=True)

    # Test HyperLiquid data
    for symbol in ['BTC', 'ETH', 'SOL']:
        cprint(f"\n{symbol} Market Data:", "white", attrs=['bold'])

        funding = provider.get_funding_rate(symbol)
        if funding:
            hourly = funding['funding_rate'] * 100
            annual = hourly * 24 * 365
            cprint(f"  Funding Rate: {hourly:.4f}% hourly ({annual:.2f}% annual)", "white")
            cprint(f"  Open Interest: {funding['open_interest']:,.2f}", "white")
            cprint(f"  Mark Price: ${funding['mark_price']:,.2f}", "white")

        zscore = provider.get_funding_zscore(symbol)
        cprint(f"  Funding Z-Score: {zscore:.2f}", "yellow" if abs(zscore) > 1.5 else "white")

    # Test liquidation data
    cprint("\nLiquidation Data (last 15 min):", "white", attrs=['bold'])
    liq_summary = provider.get_liquidation_summary(minutes=15)
    cprint(f"  Total liquidations: {liq_summary['total_count']}", "white")
    cprint(f"  Total USD: ${liq_summary['total_usd']:,.2f}", "white")
    cprint(f"  Long liquidations: ${liq_summary['long_usd']:,.2f}", "red")
    cprint(f"  Short liquidations: ${liq_summary['short_usd']:,.2f}", "green")
    cprint(f"  Ratio (long/short): {liq_summary['ratio']:.2f}", "yellow")

    # Cleanup
    provider.cleanup()
    cprint("\nTest completed!", "green")
