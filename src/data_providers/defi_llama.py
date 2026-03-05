"""DefiLlama TVL data provider - free API, no key required."""
import requests
import time
from termcolor import cprint
from src.utils.alerting import alert_service_down


class DefiLlamaProvider:
    """Singleton provider for DefiLlama TVL and DEX volume data."""
    _instance = None
    _protocols_cache = None
    _protocols_cache_time = 0
    _chains_cache = None
    _chains_cache_time = 0
    _dex_cache = None
    _dex_cache_time = 0
    _cache_ttl = 300  # 5 min cache

    BASE_URL = "https://api.llama.fi"

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get(self, url, cache_attr, cache_time_attr):
        """Generic cached GET request."""
        cached = getattr(self, cache_attr)
        cached_time = getattr(self, cache_time_attr)
        if cached and time.time() - cached_time < self._cache_ttl:
            return cached
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            setattr(self, cache_attr, data)
            setattr(self, cache_time_attr, time.time())
            return data
        except Exception as e:
            cprint(f"[DefiLlama] API error ({url}): {e}", "red")
            alert_service_down("DefiLlama", e)
            return cached  # Return stale cache if available

    def _fetch_protocols(self):
        return self._get(f"{self.BASE_URL}/v2/protocols", '_protocols_cache', '_protocols_cache_time')

    def _fetch_chains(self):
        return self._get(f"{self.BASE_URL}/v2/chains", '_chains_cache', '_chains_cache_time')

    def _fetch_dex_volumes(self):
        return self._get(f"{self.BASE_URL}/overview/dexs", '_dex_cache', '_dex_cache_time')

    def get_top_protocols(self, chain='Solana', limit=20):
        """Get top protocols by TVL for a given chain."""
        protocols = self._fetch_protocols()
        if not protocols:
            return []
        filtered = []
        for p in protocols:
            chain_tvls = p.get('chainTvls', {})
            tvl = chain_tvls.get(chain, 0)
            if tvl > 0:
                filtered.append({
                    'name': p.get('name', ''),
                    'tvl': tvl,
                    'category': p.get('category', ''),
                    'change_1d': p.get('change_1d', 0),
                    'change_7d': p.get('change_7d', 0),
                })
        filtered.sort(key=lambda x: x['tvl'], reverse=True)
        return filtered[:limit]

    def get_chain_tvl(self, chain='Solana'):
        """Get total TVL for a chain."""
        chains = self._fetch_chains()
        if not chains:
            return None
        for c in chains:
            if c.get('name', '').lower() == chain.lower():
                return {
                    'name': c['name'],
                    'tvl': c.get('tvl', 0),
                }
        return None

    def get_tvl_changes(self, chain='Solana', period='1d'):
        """Get TVL change for a chain. period: '1d' or '7d'."""
        protocols = self._fetch_protocols()
        if not protocols:
            return None

        change_key = 'change_1d' if period == '1d' else 'change_7d'
        total_tvl = 0
        weighted_change = 0

        for p in protocols:
            chain_tvls = p.get('chainTvls', {})
            tvl = chain_tvls.get(chain, 0)
            change = p.get(change_key)
            if tvl > 0 and change is not None:
                total_tvl += tvl
                weighted_change += tvl * change

        if total_tvl == 0:
            return None

        avg_change = weighted_change / total_tvl
        return {
            'chain': chain,
            'period': period,
            'total_tvl': total_tvl,
            'avg_change_pct': round(avg_change, 2),
        }

    def get_dex_volumes(self):
        """Get aggregated DEX volumes."""
        data = self._fetch_dex_volumes()
        if not data:
            return None
        protocols = data.get('protocols', [])
        total_24h = sum(p.get('total24h', 0) or 0 for p in protocols)
        total_7d = sum(p.get('total7d', 0) or 0 for p in protocols)
        top = sorted(protocols, key=lambda x: x.get('total24h', 0) or 0, reverse=True)[:10]
        return {
            'total_24h': total_24h,
            'total_7d': total_7d,
            'top_dexes': [{'name': p.get('name', ''), 'volume_24h': p.get('total24h', 0)} for p in top],
        }

    def get_signal(self, chain='Solana'):
        """Convert TVL change to trading signal.
        TVL drop > 5% in 24h = bearish signal
        TVL rise > 5% in 24h = bullish signal
        """
        changes = self.get_tvl_changes(chain, '1d')
        if not changes:
            return {'direction': 'NEUTRAL', 'confidence': 0, 'tvl_change': None}

        pct = changes['avg_change_pct']
        if pct <= -10:
            return {'direction': 'SELL', 'confidence': 75, 'tvl_change': pct, 'reason': f'TVL crash {pct:.1f}% on {chain}'}
        elif pct <= -5:
            return {'direction': 'SELL', 'confidence': 55, 'tvl_change': pct, 'reason': f'TVL declining {pct:.1f}% on {chain}'}
        elif pct >= 10:
            return {'direction': 'BUY', 'confidence': 70, 'tvl_change': pct, 'reason': f'TVL surge +{pct:.1f}% on {chain}'}
        elif pct >= 5:
            return {'direction': 'BUY', 'confidence': 50, 'tvl_change': pct, 'reason': f'TVL rising +{pct:.1f}% on {chain}'}
        else:
            return {'direction': 'NEUTRAL', 'confidence': 25, 'tvl_change': pct, 'reason': f'TVL stable ({pct:+.1f}%)'}


if __name__ == "__main__":
    cprint("Testing DefiLlama Provider...", "cyan")
    provider = DefiLlamaProvider.get_instance()

    chain_tvl = provider.get_chain_tvl('Solana')
    if chain_tvl:
        cprint(f"  Solana TVL: ${chain_tvl['tvl']:,.0f}", "white")

    changes = provider.get_tvl_changes('Solana', '1d')
    if changes:
        cprint(f"  24h TVL change: {changes['avg_change_pct']:+.2f}%", "white")

    top = provider.get_top_protocols('Solana', limit=5)
    if top:
        cprint("  Top protocols:", "white")
        for p in top:
            cprint(f"    {p['name']}: ${p['tvl']:,.0f}", "white")

    dex = provider.get_dex_volumes()
    if dex:
        cprint(f"  DEX 24h volume: ${dex['total_24h']:,.0f}", "white")

    signal = provider.get_signal('Solana')
    cprint(f"  Signal: {signal['direction']} (confidence: {signal['confidence']})", "yellow")
