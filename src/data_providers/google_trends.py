"""Google Trends data provider via pytrends (free, rate-limited)."""
import time
import threading
from termcolor import cprint
from src.utils.alerting import alert_service_down


class GoogleTrendsProvider:
    """Singleton provider for Google Trends interest data."""
    _instance = None
    _instance_lock = threading.Lock()
    _cache_ttl = 3600  # 1h (slow data + pytrends rate limits)

    DEFAULT_KEYWORDS = ['buy bitcoin', 'bitcoin crash', 'crypto']

    def __init__(self):
        self._cache = {}
        self._cache_time = {}
        self._data_lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def get_interest(self, keywords=None, timeframe='now 7-d'):
        """Get search interest for keywords.

        Returns dict per keyword with current, mean, max, zscore.
        """
        if keywords is None:
            keywords = self.DEFAULT_KEYWORDS

        cache_key = f"{','.join(sorted(keywords))}_{timeframe}"
        with self._data_lock:
            if cache_key in self._cache and time.time() - self._cache_time.get(cache_key, 0) < self._cache_ttl:
                return self._cache[cache_key]

        try:
            from pytrends.request import TrendReq

            pytrends = TrendReq(hl='en-US', tz=0, timeout=(5, 10))
            pytrends.build_payload(keywords, cat=0, timeframe=timeframe)
            df = pytrends.interest_over_time()

            if df is None or df.empty:
                return None

            result = {}
            for kw in keywords:
                if kw not in df.columns:
                    continue
                series = df[kw]
                current = float(series.iloc[-1])
                mean = float(series.mean())
                std = float(series.std())
                max_val = float(series.max())
                zscore = (current - mean) / std if std > 0 else 0

                result[kw] = {
                    'current': current,
                    'mean': round(mean, 2),
                    'max': max_val,
                    'zscore': round(zscore, 2),
                }

            with self._data_lock:
                self._cache[cache_key] = result
                self._cache_time[cache_key] = time.time()
            return result

        except Exception as e:
            cprint(f"[GoogleTrends] Error: {e}", "red")
            alert_service_down("Google Trends", e)
            return self._cache.get(cache_key)

    def get_signal(self):
        """Retail sentiment signal from Google Trends (contrarian)."""
        data = self.get_interest()
        if not data:
            return {'direction': 'NEUTRAL', 'confidence': 0, 'reason': 'No Google Trends data'}

        buy_z = data.get('buy bitcoin', {}).get('zscore', 0)
        crash_z = data.get('bitcoin crash', {}).get('zscore', 0)

        reasons = []

        # "buy bitcoin" spikes = retail FOMO = contrarian sell
        buy_direction = 'NEUTRAL'
        buy_confidence = 0
        if buy_z > 2.0:
            buy_direction = 'SELL'
            buy_confidence = 70
            reasons.append(f'"buy bitcoin" zscore {buy_z:.1f} (retail FOMO)')
        elif buy_z > 1.0:
            buy_direction = 'SELL'
            buy_confidence = 40
            reasons.append(f'"buy bitcoin" zscore {buy_z:.1f} (mild FOMO)')

        # "bitcoin crash" spikes = retail panic = contrarian buy
        crash_direction = 'NEUTRAL'
        crash_confidence = 0
        if crash_z > 2.0:
            crash_direction = 'BUY'
            crash_confidence = 70
            reasons.append(f'"bitcoin crash" zscore {crash_z:.1f} (retail panic)')
        elif crash_z > 1.0:
            crash_direction = 'BUY'
            crash_confidence = 40
            reasons.append(f'"bitcoin crash" zscore {crash_z:.1f} (mild panic)')

        # Combine: if both fire in opposite directions, take the stronger one
        if buy_confidence > 0 and crash_confidence > 0:
            # Conflicting signals - take higher confidence
            if buy_confidence >= crash_confidence:
                return {'direction': buy_direction, 'confidence': buy_confidence,
                        'reason': ' | '.join(reasons), 'buy_zscore': buy_z, 'crash_zscore': crash_z}
            else:
                return {'direction': crash_direction, 'confidence': crash_confidence,
                        'reason': ' | '.join(reasons), 'buy_zscore': buy_z, 'crash_zscore': crash_z}
        elif buy_confidence > 0:
            return {'direction': buy_direction, 'confidence': buy_confidence,
                    'reason': ' | '.join(reasons), 'buy_zscore': buy_z, 'crash_zscore': crash_z}
        elif crash_confidence > 0:
            return {'direction': crash_direction, 'confidence': crash_confidence,
                    'reason': ' | '.join(reasons), 'buy_zscore': buy_z, 'crash_zscore': crash_z}

        if not reasons:
            reasons.append(f'Trends normal (buy z={buy_z:.1f}, crash z={crash_z:.1f})')

        return {'direction': 'NEUTRAL', 'confidence': 20,
                'reason': ' | '.join(reasons), 'buy_zscore': buy_z, 'crash_zscore': crash_z}


if __name__ == "__main__":
    cprint("Testing Google Trends Provider...", "cyan")
    provider = GoogleTrendsProvider.get_instance()

    data = provider.get_interest()
    if data:
        for kw, vals in data.items():
            cprint(f"  {kw}: current={vals['current']}, mean={vals['mean']}, zscore={vals['zscore']}", "white")
    else:
        cprint("  No data returned (pytrends may be rate-limited)", "yellow")

    signal = provider.get_signal()
    cprint(f"  Signal: {signal['direction']} (confidence: {signal['confidence']})", "yellow")
    if signal.get('reason'):
        cprint(f"  Reason: {signal['reason']}", "white")
