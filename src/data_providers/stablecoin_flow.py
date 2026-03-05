"""Stablecoin supply flow provider - DefiLlama Stablecoins API (free, no key)."""
import requests
import time
import threading
from termcolor import cprint
from src.utils.alerting import alert_service_down


class StablecoinFlowProvider:
    """Singleton provider for stablecoin supply flow data (USDT + USDC)."""
    _instance = None
    _instance_lock = threading.Lock()
    _cache_ttl = 300  # 5 min

    API_URL = "https://stablecoins.llama.fi/stablecoins?includePrices=true"

    def __init__(self):
        self._cache = None
        self._cache_time = 0
        self._data_lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _fetch(self):
        with self._data_lock:
            if self._cache and time.time() - self._cache_time < self._cache_ttl:
                return self._cache
        try:
            resp = requests.get(self.API_URL, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            with self._data_lock:
                self._cache = data
                self._cache_time = time.time()
            return data
        except Exception as e:
            cprint(f"[StablecoinFlow] API error: {e}", "red")
            alert_service_down("StablecoinFlow", e)
            return self._cache

    def get_supply_changes(self):
        """Get USDT+USDC supply changes (day, week, month)."""
        data = self._fetch()
        if not data or 'peggedAssets' not in data:
            return None

        result = {}
        for asset in data['peggedAssets']:
            if asset.get('symbol') in ('USDT', 'USDC'):
                circ = asset.get('circulating', {}).get('peggedUSD', 0)
                prev_day = asset.get('circulatingPrevDay', {}).get('peggedUSD', 0)
                prev_week = asset.get('circulatingPrevWeek', {}).get('peggedUSD', 0)
                prev_month = asset.get('circulatingPrevMonth', {}).get('peggedUSD', 0)
                result[asset['symbol']] = {
                    'circulating': circ,
                    'change_1d': circ - prev_day if prev_day else 0,
                    'change_7d': circ - prev_week if prev_week else 0,
                    'change_30d': circ - prev_month if prev_month else 0,
                    'change_7d_pct': ((circ - prev_week) / prev_week * 100) if prev_week else 0,
                }

        total_circ = sum(v['circulating'] for v in result.values()) if result else 0
        total_7d_change = sum(v['change_7d'] for v in result.values()) if result else 0
        total_1d_change = sum(v['change_1d'] for v in result.values()) if result else 0
        base = total_circ - total_7d_change
        result['TOTAL'] = {
            'circulating': total_circ,
            'change_1d': total_1d_change,
            'change_7d': total_7d_change,
            'change_7d_pct': (total_7d_change / base * 100) if base else 0,
        }
        return result

    def get_signal(self):
        """Stablecoin inflow/outflow signal."""
        changes = self.get_supply_changes()
        if not changes or 'TOTAL' not in changes:
            return {'direction': 'NEUTRAL', 'confidence': 0, 'reason': 'No stablecoin data'}

        total = changes['TOTAL']
        pct_7d = total.get('change_7d_pct', 0)
        change_1d = total.get('change_1d', 0)

        if pct_7d > 2.0:
            return {'direction': 'BUY', 'confidence': 70, 'reason': f'Stablecoin inflow +{pct_7d:.1f}% 7d'}
        elif pct_7d > 0.5:
            return {'direction': 'BUY', 'confidence': 40, 'reason': f'Stablecoin mild inflow +{pct_7d:.1f}% 7d'}
        elif pct_7d < -2.0:
            return {'direction': 'SELL', 'confidence': 70, 'reason': f'Stablecoin outflow {pct_7d:.1f}% 7d'}
        elif pct_7d < -0.5:
            return {'direction': 'SELL', 'confidence': 40, 'reason': f'Stablecoin mild outflow {pct_7d:.1f}% 7d'}

        if abs(change_1d) > 500_000_000:
            direction = 'BUY' if change_1d > 0 else 'SELL'
            return {'direction': direction, 'confidence': 60,
                    'reason': f'Stablecoin daily spike ${change_1d/1e9:+.1f}B'}

        return {'direction': 'NEUTRAL', 'confidence': 20, 'reason': f'Stablecoin stable ({pct_7d:+.1f}% 7d)'}


if __name__ == "__main__":
    cprint("Testing Stablecoin Flow Provider...", "cyan")
    provider = StablecoinFlowProvider.get_instance()

    changes = provider.get_supply_changes()
    if changes:
        for symbol in ('USDT', 'USDC', 'TOTAL'):
            if symbol in changes:
                c = changes[symbol]
                cprint(f"  {symbol}: ${c['circulating']:,.0f} (7d: {c.get('change_7d_pct', 0):+.2f}%)", "white")

    signal = provider.get_signal()
    cprint(f"  Signal: {signal['direction']} (confidence: {signal['confidence']})", "yellow")
    if signal.get('reason'):
        cprint(f"  Reason: {signal['reason']}", "white")
