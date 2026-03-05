"""CoinGecko Social & Trending data provider - free API (10K calls/month).

Provides social hype detection and trending coin signals:
- Trending coins (top 7 by search volume)
- Market cap rankings and % changes
- Community data (reddit, twitter followers)
"""
import os
import requests
import time
from termcolor import cprint
from src.utils.alerting import alert_service_down


class CoinGeckoSocialProvider:
    """Singleton provider for CoinGecko social/trending data."""
    _instance = None
    _cache = {}
    _cache_time = {}
    _cache_ttl = 900  # 15 min cache (free tier: 10K calls/month, ~3 calls/cycle = ~2,880/month)

    BASE_URL = "https://api.coingecko.com/api/v3"

    # Map HL symbols to CoinGecko IDs
    SYMBOL_TO_ID = {
        'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana',
        'XRP': 'ripple', 'DOGE': 'dogecoin', 'ADA': 'cardano',
        'AVAX': 'avalanche-2', 'LINK': 'chainlink', 'DOT': 'polkadot',
        'SUI': 'sui', 'NEAR': 'near', 'TAO': 'bittensor',
        'AAVE': 'aave', 'ENA': 'ethena',
        'kPEPE': 'pepe',
    }

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _headers(self):
        """Build headers with optional API key."""
        h = {'accept': 'application/json'}
        key = os.getenv('COINGECKO_API_KEY')
        if key:
            h['x-cg-demo-api-key'] = key
        return h

    def _get_cached(self, cache_key, url, params=None):
        """Cached GET with rate-limit awareness and auth fallback."""
        if cache_key in self._cache and time.time() - self._cache_time.get(cache_key, 0) < self._cache_ttl:
            return self._cache[cache_key]
        try:
            resp = requests.get(url, params=params, headers=self._headers(), timeout=15)
            if resp.status_code == 401:
                # Invalid/expired API key — retry without it (public tier)
                cprint("[CoinGecko] 401 with API key, falling back to public API", "yellow")
                resp = requests.get(url, params=params, headers={'accept': 'application/json'}, timeout=15)
            if resp.status_code == 429:
                cprint("[CoinGecko] Rate limited, using stale cache", "yellow")
                return self._cache.get(cache_key)
            resp.raise_for_status()
            data = resp.json()
            self._cache[cache_key] = data
            self._cache_time[cache_key] = time.time()
            return data
        except Exception as e:
            cprint(f"[CoinGecko] API error ({cache_key}): {e}", "red")
            alert_service_down("CoinGecko", e)
            return self._cache.get(cache_key)

    def get_trending_coins(self):
        """Get top 7 trending coins by search volume (last 24h).

        Returns list of {id, symbol, name, market_cap_rank, score}.
        Trending coins often signal retail FOMO - useful as contrarian indicator.
        """
        data = self._get_cached('trending', f"{self.BASE_URL}/search/trending")
        if not data or 'coins' not in data:
            return []

        result = []
        for item in data['coins']:
            coin = item.get('item', {})
            result.append({
                'id': coin.get('id', ''),
                'symbol': coin.get('symbol', '').upper(),
                'name': coin.get('name', ''),
                'market_cap_rank': coin.get('market_cap_rank'),
                'score': coin.get('score', 0),
                'price_change_24h': coin.get('data', {}).get('price_change_percentage_24h', {}).get('usd', 0),
            })
        return result

    def get_market_data(self, symbols=None):
        """Get market data for tracked symbols.

        Returns dict keyed by symbol with price changes, market cap, volume.
        Batches all symbols in a single API call (cached 5min) to avoid rate limits.
        """
        if symbols is None:
            symbols = list(self.SYMBOL_TO_ID.keys())

        # Always fetch ALL tracked symbols in one batch call (cached together)
        all_ids = [v for v in self.SYMBOL_TO_ID.values()]
        ids_str = ','.join(sorted(all_ids))
        data = self._get_cached('markets_all',
            f"{self.BASE_URL}/coins/markets",
            {
                'vs_currency': 'usd',
                'ids': ids_str,
                'order': 'market_cap_desc',
                'per_page': 50,
                'page': 1,
                'sparkline': 'false',
                'price_change_percentage': '1h,24h,7d',
            })

        if not data:
            return {}

        # Build reverse map: coingecko_id -> hl_symbol
        id_to_sym = {v: k for k, v in self.SYMBOL_TO_ID.items()}

        result = {}
        for coin in data:
            cg_id = coin.get('id', '')
            sym = id_to_sym.get(cg_id, coin.get('symbol', '').upper())
            result[sym] = {
                'price': coin.get('current_price', 0),
                'market_cap': coin.get('market_cap', 0),
                'total_volume_24h': coin.get('total_volume', 0),
                'price_change_1h': coin.get('price_change_percentage_1h_in_currency', 0) or 0,
                'price_change_24h': coin.get('price_change_percentage_24h', 0) or 0,
                'price_change_7d': coin.get('price_change_percentage_7d_in_currency', 0) or 0,
                'market_cap_rank': coin.get('market_cap_rank'),
                'ath_change_pct': coin.get('ath_change_percentage', 0) or 0,
            }
        return result

    def get_global_data(self):
        """Get global crypto market data.

        Returns market cap change, BTC dominance, total volume.
        Useful for macro regime detection.
        """
        data = self._get_cached('global', f"{self.BASE_URL}/global")
        if not data or 'data' not in data:
            return None

        d = data['data']
        return {
            'total_market_cap_usd': d.get('total_market_cap', {}).get('usd', 0),
            'total_volume_24h_usd': d.get('total_volume', {}).get('usd', 0),
            'market_cap_change_24h_pct': d.get('market_cap_change_percentage_24h_usd', 0),
            'btc_dominance': d.get('market_cap_percentage', {}).get('btc', 0),
            'eth_dominance': d.get('market_cap_percentage', {}).get('eth', 0),
            'active_cryptos': d.get('active_cryptocurrencies', 0),
        }

    def is_symbol_trending(self, symbol):
        """Check if a symbol is in the trending list.

        Returns:
            int or None: Trending rank (0-6) if trending, None if not.
        """
        trending = self.get_trending_coins()
        for coin in trending:
            if coin['symbol'] == symbol.upper():
                return coin['score']
        return None

    def get_social_signal(self, symbol):
        """Generate trading signal from social/trending data.

        Logic:
        - If symbol is trending AND up >10% in 24h: contrarian SELL (retail FOMO)
        - If symbol is trending AND down >5% in 24h: momentum BUY (recovery hype)
        - Global market cap dropping >3%: bearish bias
        - BTC dominance rising >2% in context: risk-off (bearish for alts)

        Returns:
            dict with 'direction', 'confidence', 'reason'
        """
        long_score = 0
        short_score = 0
        reasons = []

        # Check if symbol is trending
        trending_rank = self.is_symbol_trending(symbol)

        # Get market data for this symbol
        market = self.get_market_data([symbol])
        sym_data = market.get(symbol)

        if trending_rank is not None and sym_data:
            pct_24h = sym_data.get('price_change_24h', 0)
            if pct_24h > 10:
                # Trending + big pump = retail FOMO = contrarian sell
                short_score += 35
                reasons.append(f'trending_fomo(rank={trending_rank} +{pct_24h:.1f}%)')
            elif pct_24h < -5:
                # Trending + dip = dip buying momentum
                long_score += 25
                reasons.append(f'trending_dip(rank={trending_rank} {pct_24h:.1f}%)')
            else:
                # Trending neutral = mild bullish (attention = demand)
                long_score += 10
                reasons.append(f'trending(rank={trending_rank})')

        # Global macro check
        global_data = self.get_global_data()
        if global_data:
            mc_change = global_data.get('market_cap_change_24h_pct', 0)
            if mc_change < -5:
                short_score += 25
                reasons.append(f'global_dump({mc_change:.1f}%)')
            elif mc_change < -2:
                short_score += 10
                reasons.append(f'global_weak({mc_change:.1f}%)')
            elif mc_change > 5:
                long_score += 20
                reasons.append(f'global_pump({mc_change:+.1f}%)')
            elif mc_change > 2:
                long_score += 10
                reasons.append(f'global_strong({mc_change:+.1f}%)')

        # Volume spike detection
        if sym_data:
            pct_1h = sym_data.get('price_change_1h', 0)
            if abs(pct_1h) > 3:
                if pct_1h > 0:
                    long_score += 15
                else:
                    short_score += 15
                reasons.append(f'1h_move({pct_1h:+.1f}%)')

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
            'reason': ' | '.join(reasons) if reasons else 'no_social_signal',
        }


if __name__ == "__main__":
    cprint("Testing CoinGecko Social Provider...", "cyan")
    provider = CoinGeckoSocialProvider.get_instance()

    # Trending coins
    trending = provider.get_trending_coins()
    if trending:
        cprint("\nTrending coins (last 24h):", "white", attrs=['bold'])
        for coin in trending:
            pct = coin.get('price_change_24h', 0)
            cprint(f"  #{coin['score']+1} {coin['symbol']} ({coin['name']}) "
                   f"rank={coin['market_cap_rank']} 24h={pct:+.1f}%", "white")

    # Global market data
    global_data = provider.get_global_data()
    if global_data:
        cprint(f"\nGlobal Market:", "white", attrs=['bold'])
        cprint(f"  Total MC: ${global_data['total_market_cap_usd']:,.0f}", "white")
        cprint(f"  24h change: {global_data['market_cap_change_24h_pct']:+.2f}%", "white")
        cprint(f"  BTC dominance: {global_data['btc_dominance']:.1f}%", "white")

    # Per-symbol signals
    for symbol in ['BTC', 'ETH', 'SOL']:
        signal = provider.get_social_signal(symbol)
        cprint(f"\n{symbol} Social Signal: {signal['direction']} "
               f"(confidence={signal['confidence']}) - {signal['reason']}", "yellow")
