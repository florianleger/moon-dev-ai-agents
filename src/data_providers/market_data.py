"""
Market Data Provider

Unified interface for market data from multiple free sources:
- HyperLiquid: Funding rates and open interest
- Bybit: Real-time liquidations (allLiquidation WS — the Binance forceOrder
  WS is blocked from the production IP and its REST fallback was removed)

This module eliminates the dependency on Moon Dev API.
"""

import time
import requests
from typing import Dict, Optional
from termcolor import cprint
from src.utils.alerting import alert_service_down

from .bybit_liquidations import get_liquidation_stream, get_liquidation_ratio

# HyperLiquid API endpoint
HYPERLIQUID_API_URL = 'https://api.hyperliquid.xyz/info'


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
        self._all_prices_cache = {}  # Cache for all prices from single API call
        self._all_prices_cache_time = 0
        self._cache_ttl = 60  # Cache HyperLiquid data for 60 seconds (rate-limit mitigation)

        if start_liquidation_stream:
            self._init_liquidation_stream()

    def _fetch_with_retry(self, url: str, method: str = 'get', json_body=None,
                          headers=None, max_retries: int = 3, timeout: int = 10):
        """
        Fetch URL with exponential backoff retry.

        Args:
            url: URL to fetch
            method: HTTP method ('get' or 'post')
            json_body: JSON body for POST requests
            headers: HTTP headers
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds

        Returns:
            Response JSON or None on failure
        """
        for attempt in range(max_retries):
            try:
                if method == 'post':
                    response = requests.post(url, headers=headers, json=json_body, timeout=timeout)
                else:
                    response = requests.get(url, headers=headers, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt == max_retries - 1:
                    cprint(f"[MarketData] API call failed after {max_retries} attempts: {e}", "red")
                    alert_service_down("HyperLiquid", e)
                    return None
                wait_time = 2 ** attempt
                cprint(f"[MarketData] Retry {attempt + 1}/{max_retries} in {wait_time}s: {e}", "yellow")
                time.sleep(wait_time)

    def _init_liquidation_stream(self):
        """Initialize the Bybit liquidation stream."""
        try:
            self._liquidation_stream = get_liquidation_stream()
            if not self._liquidation_stream.is_connected:
                self._liquidation_stream.start_stream()
                # Give it time to connect and collect initial data
                time.sleep(2)
        except Exception as e:
            cprint(f"[MarketData] Warning: Could not start liquidation stream: {e}", "yellow")

    def _fetch_all_prices(self) -> bool:
        """
        Fetch all prices from HyperLiquid in a single API call.
        Caches all symbol data for subsequent lookups.
        Uses retry with exponential backoff.

        Returns:
            bool: True if successful, False otherwise
        """
        # Check if cache is still valid
        if self._all_prices_cache and (time.time() - self._all_prices_cache_time) < self._cache_ttl:
            return True

        data = self._fetch_with_retry(
            HYPERLIQUID_API_URL,
            method='post',
            headers={'Content-Type': 'application/json'},
            json_body={"type": "metaAndAssetCtxs"},
        )

        if data and len(data) >= 2 and isinstance(data[0], dict) and isinstance(data[1], list):
            # Build symbol -> index mapping
            universe = {coin['name']: i for i, coin in enumerate(data[0]['universe'])}
            funding_data = data[1]

            # Cache all prices
            self._all_prices_cache = {}
            for symbol, idx in universe.items():
                if idx < len(funding_data):
                    asset_data = funding_data[idx]
                    self._all_prices_cache[symbol] = {
                        'funding_rate': float(asset_data['funding']),
                        'mark_price': float(asset_data['markPx']),
                        'open_interest': float(asset_data['openInterest'])
                    }

            self._all_prices_cache_time = time.time()
            return True

        return False

    def get_funding_rate(self, symbol: str) -> Optional[Dict]:
        """
        Get current funding rate from HyperLiquid.
        Uses bulk cache to avoid multiple API calls.

        Args:
            symbol: Asset symbol (e.g., 'BTC', 'ETH', 'SOL')

        Returns:
            Dict with funding_rate, mark_price, open_interest
            Or None if unavailable
        """
        # Ensure we have fresh data (single API call caches all symbols)
        self._fetch_all_prices()

        # Return from cache
        return self._all_prices_cache.get(symbol)

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

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price (mark price) from HyperLiquid.

        Args:
            symbol: Asset symbol (e.g., 'BTC', 'ETH', 'SOL')

        Returns:
            float: Current mark price or None if unavailable
        """
        data = self.get_funding_rate(symbol)
        if data and 'mark_price' in data:
            return data.get('mark_price')
        return None

    def get_funding_zscore(self, symbol: str) -> Optional[float]:
        """
        Calculate funding rate Z-score using adaptive parameters from cache.

        If enough symbols are cached, calculates mean/std from actual market data.
        Falls back to typical perpetual funding distribution if insufficient data.

        Args:
            symbol: Asset symbol

        Returns:
            float: Z-score (-3 to +3 typical range), 0 if unavailable
        """
        try:
            data = self.get_funding_rate(symbol)
            if data and 'funding_rate' in data:
                hourly_rate = data['funding_rate']
                annual_rate = hourly_rate * 24 * 365 * 100  # Convert to annual %

                # Adaptive: compute mean/std from all cached funding rates
                if len(self._all_prices_cache) >= 5:
                    all_annual_rates = [
                        d['funding_rate'] * 24 * 365 * 100
                        for d in self._all_prices_cache.values()
                        if 'funding_rate' in d
                    ]
                    if len(all_annual_rates) >= 5:
                        import statistics
                        mean_funding = statistics.mean(all_annual_rates)
                        std_funding = statistics.stdev(all_annual_rates)
                        if std_funding < 1.0:
                            std_funding = 1.0  # Floor to avoid division by near-zero
                        zscore = (annual_rate - mean_funding) / std_funding
                        return round(zscore, 2)

                # Fallback: typical funding parameters
                mean_funding = 10.0  # 10% annual mean
                std_funding = 15.0   # 15% annual std

                zscore = (annual_rate - mean_funding) / std_funding
                return round(zscore, 2)

            cprint(f"[MarketData] No funding data for {symbol}, returning None (not Z=0)", "yellow")
            return None

        except Exception as e:
            cprint(f"[MarketData] Error calculating funding Z-score for {symbol}: {e}", "yellow")
            return None

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

    def get_l2_book(self, symbol: str, depth: int = 10) -> Optional[Dict]:
        """Get L2 order book from HyperLiquid. Uses retry with exponential backoff."""
        data = self._fetch_with_retry(
            HYPERLIQUID_API_URL,
            method='post',
            headers={'Content-Type': 'application/json'},
            json_body={"type": "l2Book", "coin": symbol},
        )

        if not data or 'levels' not in data:
            return None

        bids = data['levels'][0][:depth]
        asks = data['levels'][1][:depth]

        bid_depth = sum(float(b['sz']) for b in bids)
        ask_depth = sum(float(a['sz']) for a in asks)
        total = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total if total > 0 else 0

        return {
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'imbalance': imbalance,
            'spread': float(asks[0]['px']) - float(bids[0]['px']) if bids and asks else 0,
            'best_bid': float(bids[0]['px']) if bids else 0,
            'best_ask': float(asks[0]['px']) if asks else 0,
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
            'funding_zscore': self.get_funding_zscore(symbol) or 0.0,
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


def get_funding_zscore(symbol: str) -> Optional[float]:
    """Get funding rate Z-score for a symbol. Returns None if data unavailable."""
    return get_market_data_provider().get_funding_zscore(symbol)


def get_current_price(symbol: str) -> Optional[float]:
    """Get current price for a symbol."""
    return get_market_data_provider().get_current_price(symbol)


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
        if zscore is not None:
            cprint(f"  Funding Z-Score: {zscore:.2f}", "yellow" if abs(zscore) > 1.5 else "white")
        else:
            cprint(f"  Funding Z-Score: N/A (no data)", "red")

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
