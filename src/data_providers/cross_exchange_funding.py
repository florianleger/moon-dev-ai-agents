"""Cross-Exchange Funding Rate Comparison provider.

Compares funding rates between HyperLiquid and Binance to detect:
- Funding divergence (different rates = arbitrage opportunity / directional signal)
- Extreme funding on one exchange but not the other

Both APIs are free, no keys required.
"""
import requests
import time
from termcolor import cprint


class CrossExchangeFundingProvider:
    """Singleton provider for cross-exchange funding rate comparison."""
    _instance = None
    _hl_cache = None
    _hl_cache_time = 0
    _binance_cache = {}
    _binance_cache_time = {}
    _cache_ttl = 60  # 1 min cache

    HL_API_URL = "https://api.hyperliquid.xyz/info"
    BINANCE_API_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"

    # Map HL symbols to Binance pairs
    SYMBOL_MAP = {
        'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'SOL': 'SOLUSDT',
        'XRP': 'XRPUSDT', 'DOGE': 'DOGEUSDT', 'ADA': 'ADAUSDT',
        'AVAX': 'AVAXUSDT', 'LINK': 'LINKUSDT', 'DOT': 'DOTUSDT',
        'SUI': 'SUIUSDT', 'NEAR': 'NEARUSDT', 'TAO': 'TAOUSDT',
        'AAVE': 'AAVEUSDT', 'ENA': 'ENAUSDT',
        'kPEPE': '1000PEPEUSDT',
    }

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _fetch_hl_funding(self):
        """Fetch all funding rates from HyperLiquid (single call)."""
        if self._hl_cache and time.time() - self._hl_cache_time < self._cache_ttl:
            return self._hl_cache

        try:
            resp = requests.post(self.HL_API_URL,
                headers={'Content-Type': 'application/json'},
                json={"type": "metaAndAssetCtxs"},
                timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data and len(data) >= 2:
                universe = {coin['name']: i for i, coin in enumerate(data[0]['universe'])}
                result = {}
                for symbol, idx in universe.items():
                    if idx < len(data[1]):
                        result[symbol] = float(data[1][idx]['funding'])
                self._hl_cache = result
                self._hl_cache_time = time.time()
                return result
        except Exception as e:
            cprint(f"[CrossFunding] HL API error: {e}", "red")

        return self._hl_cache or {}

    def _fetch_binance_funding(self, symbol):
        """Fetch funding rate for a Binance symbol."""
        binance_sym = self.SYMBOL_MAP.get(symbol)
        if not binance_sym:
            return None

        cache_key = f"binance_{binance_sym}"
        if cache_key in self._binance_cache and time.time() - self._binance_cache_time.get(cache_key, 0) < self._cache_ttl:
            return self._binance_cache[cache_key]

        try:
            resp = requests.get(self.BINANCE_API_URL,
                params={'symbol': binance_sym}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            rate = float(data.get('lastFundingRate', 0))
            self._binance_cache[cache_key] = rate
            self._binance_cache_time[cache_key] = time.time()
            return rate
        except Exception as e:
            cprint(f"[CrossFunding] Binance API error ({binance_sym}): {e}", "red")
            return self._binance_cache.get(cache_key)

    def get_funding_comparison(self, symbol):
        """Compare funding rates between HyperLiquid and Binance.

        Args:
            symbol: HL symbol (e.g. 'BTC', 'ETH')

        Returns:
            dict with hl_rate, binance_rate, spread, divergence_signal
        """
        hl_rates = self._fetch_hl_funding()
        hl_rate = hl_rates.get(symbol)
        binance_rate = self._fetch_binance_funding(symbol)

        if hl_rate is None or binance_rate is None:
            return None

        # Both are hourly rates (HL) vs 8h rates (Binance)
        # Normalize: convert Binance 8h rate to hourly
        binance_hourly = binance_rate / 8

        spread = hl_rate - binance_hourly
        avg_rate = (hl_rate + binance_hourly) / 2

        # Annualize for human readability
        hl_annual = hl_rate * 24 * 365 * 100
        binance_annual = binance_hourly * 24 * 365 * 100
        spread_annual = spread * 24 * 365 * 100

        return {
            'hl_rate_hourly': hl_rate,
            'binance_rate_hourly': binance_hourly,
            'hl_rate_annual_pct': round(hl_annual, 2),
            'binance_rate_annual_pct': round(binance_annual, 2),
            'spread_hourly': spread,
            'spread_annual_pct': round(spread_annual, 2),
            'avg_rate_hourly': avg_rate,
        }

    def get_all_comparisons(self, symbols=None):
        """Get funding comparison for all tracked symbols."""
        if symbols is None:
            symbols = list(self.SYMBOL_MAP.keys())

        results = {}
        for sym in symbols:
            comp = self.get_funding_comparison(sym)
            if comp:
                results[sym] = comp
        return results

    def get_divergence_signal(self, symbol):
        """Generate trading signal from funding rate divergence.

        Divergence logic:
        - If HL funding >> Binance: HL longs are overcrowded -> short bias
        - If Binance funding >> HL: Binance longs overcrowded -> short bias
        - If both extremely positive: overall market too leveraged long -> contrarian short
        - If both extremely negative: market too leveraged short -> contrarian long
        - If one positive, one negative: true divergence = strong signal

        Returns:
            dict with 'direction', 'confidence', 'reason'
        """
        comp = self.get_funding_comparison(symbol)
        if not comp:
            return {'direction': 'NEUTRAL', 'confidence': 0, 'reason': 'No data'}

        hl = comp['hl_rate_annual_pct']
        binance = comp['binance_rate_annual_pct']
        spread = comp['spread_annual_pct']

        long_score = 0
        short_score = 0
        reasons = []

        # Check for extreme unified funding (both exchanges agree)
        avg = (hl + binance) / 2
        if avg > 50:  # > 50% annualized = very high
            short_score += 35
            reasons.append(f'both_high_funding(avg={avg:.0f}%)')
        elif avg > 25:
            short_score += 15
            reasons.append(f'elevated_funding(avg={avg:.0f}%)')
        elif avg < -50:
            long_score += 35
            reasons.append(f'both_neg_funding(avg={avg:.0f}%)')
        elif avg < -25:
            long_score += 15
            reasons.append(f'low_funding(avg={avg:.0f}%)')

        # Check for cross-exchange divergence
        if abs(spread) > 30:  # > 30% annualized spread
            # Significant divergence
            if spread > 0:
                # HL higher than Binance: HL is more leveraged long
                short_score += 25
                reasons.append(f'HL_premium(spread={spread:+.0f}%)')
            else:
                # Binance higher than HL
                short_score += 20
                reasons.append(f'Binance_premium(spread={spread:+.0f}%)')
        elif abs(spread) > 15:
            if spread > 0:
                short_score += 10
            else:
                short_score += 8
            reasons.append(f'mild_divergence(spread={spread:+.0f}%)')

        # True divergence: one positive, one negative
        if (hl > 5 and binance < -5) or (hl < -5 and binance > 5):
            # This is a very strong signal
            if hl > binance:
                short_score += 30
            else:
                long_score += 30
            reasons.append(f'true_divergence(HL={hl:.0f}% BN={binance:.0f}%)')

        best = max(long_score, short_score)
        if long_score > short_score:
            direction = 'BUY'
        elif short_score > long_score:
            direction = 'SELL'
        else:
            direction = 'NEUTRAL'

        return {
            'direction': direction,
            'confidence': min(100, best),
            'reason': ' | '.join(reasons) if reasons else 'funding_neutral',
            'hl_annual_pct': hl,
            'binance_annual_pct': binance,
            'spread_annual_pct': spread,
        }


if __name__ == "__main__":
    cprint("Testing Cross-Exchange Funding Provider...", "cyan")
    provider = CrossExchangeFundingProvider.get_instance()

    for symbol in ['BTC', 'ETH', 'SOL', 'DOGE']:
        comp = provider.get_funding_comparison(symbol)
        if comp:
            cprint(f"\n{symbol}:", "white", attrs=['bold'])
            cprint(f"  HyperLiquid: {comp['hl_rate_annual_pct']:+.2f}% annual", "white")
            cprint(f"  Binance:     {comp['binance_rate_annual_pct']:+.2f}% annual", "white")
            cprint(f"  Spread:      {comp['spread_annual_pct']:+.2f}% annual", "yellow")

            signal = provider.get_divergence_signal(symbol)
            cprint(f"  Signal: {signal['direction']} "
                   f"(confidence={signal['confidence']}) - {signal['reason']}", "yellow")
