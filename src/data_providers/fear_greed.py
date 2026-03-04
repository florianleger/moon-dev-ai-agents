"""Fear & Greed Index data provider - Alternative.me API (gratuit)"""
import requests
import time
from termcolor import cprint


class FearGreedProvider:
    """Singleton provider for Crypto Fear & Greed Index."""
    _instance = None
    _cache = None
    _cache_time = 0
    _cache_ttl = 300  # 5 min cache

    API_URL = "https://api.alternative.me/fng/"

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_current(self):
        """Get current Fear & Greed value (0-100). 0=Extreme Fear, 100=Extreme Greed."""
        data = self._fetch(limit=1)
        if data:
            return {
                'value': int(data[0]['value']),
                'classification': data[0]['value_classification'],
                'timestamp': int(data[0]['timestamp'])
            }
        return None

    def get_history(self, days=30):
        """Get historical Fear & Greed values."""
        return self._fetch(limit=days)

    def get_signal(self):
        """Convert Fear & Greed to trading signal.
        Extreme Fear (<25) = contrarian BUY signal
        Extreme Greed (>75) = contrarian SELL signal
        """
        current = self.get_current()
        if not current:
            return {'direction': 'NEUTRAL', 'confidence': 0, 'value': None}

        value = current['value']
        if value < 25:
            return {'direction': 'BUY', 'confidence': 80, 'value': value, 'reason': 'Extreme Fear - contrarian buy'}
        elif value < 35:
            return {'direction': 'BUY', 'confidence': 55, 'value': value, 'reason': 'Fear - mild buy signal'}
        elif value > 80:
            return {'direction': 'SELL', 'confidence': 75, 'value': value, 'reason': 'Extreme Greed - contrarian sell'}
        elif value > 65:
            return {'direction': 'SELL', 'confidence': 50, 'value': value, 'reason': 'Greed - mild sell signal'}
        else:
            return {'direction': 'NEUTRAL', 'confidence': 30, 'value': value, 'reason': 'Neutral zone'}

    def _fetch(self, limit=1):
        if self._cache and time.time() - self._cache_time < self._cache_ttl and limit <= len(self._cache):
            return self._cache[:limit]
        try:
            resp = requests.get(f"{self.API_URL}?limit={limit}", timeout=10)
            resp.raise_for_status()
            data = resp.json().get('data', [])
            self._cache = data
            self._cache_time = time.time()
            return data
        except Exception as e:
            cprint(f"Fear & Greed API error: {e}", "red")
            return None


if __name__ == "__main__":
    cprint("Testing Fear & Greed Index...", "cyan")
    provider = FearGreedProvider.get_instance()

    current = provider.get_current()
    if current:
        cprint(f"  Value: {current['value']}", "white")
        cprint(f"  Classification: {current['classification']}", "white")

    signal = provider.get_signal()
    cprint(f"  Signal: {signal['direction']} (confidence: {signal['confidence']})", "yellow")
    if signal.get('reason'):
        cprint(f"  Reason: {signal['reason']}", "white")
