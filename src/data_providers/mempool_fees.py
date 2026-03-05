"""Bitcoin mempool fees provider - mempool.space API (free, no key)."""
import requests
import time
import threading
from collections import deque
from termcolor import cprint
from src.utils.alerting import alert_service_down


class MempoolFeesProvider:
    """Singleton provider for Bitcoin mempool fee data."""
    _instance = None
    _instance_lock = threading.Lock()
    _cache_ttl = 60  # 1 min

    API_URL = "https://mempool.space/api/v1/fees/recommended"

    # Rolling history for spike detection (1h at 60s intervals = 60 entries)
    _history_maxlen = 60

    def __init__(self):
        self._cache = None
        self._cache_time = 0
        self._data_lock = threading.Lock()
        self._fee_history = deque(maxlen=self._history_maxlen)

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
            resp = requests.get(self.API_URL, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            with self._data_lock:
                self._cache = data
                self._cache_time = time.time()
                self._fee_history.append({
                    'time': time.time(),
                    'fastestFee': data.get('fastestFee', 0),
                })
            return data
        except Exception as e:
            cprint(f"[MempoolFees] API error: {e}", "red")
            alert_service_down("Mempool Fees", e)
            return self._cache

    def get_fees(self):
        """Get current recommended fees (sat/vB)."""
        data = self._fetch()
        if not data:
            return None
        return {
            'fastestFee': data.get('fastestFee', 0),
            'halfHourFee': data.get('halfHourFee', 0),
            'hourFee': data.get('hourFee', 0),
            'economyFee': data.get('economyFee', 0),
        }

    def _get_history_avg(self):
        """Average fastestFee over rolling 1h window."""
        if not self._fee_history or len(self._fee_history) < 2:
            return None
        total = sum(h['fastestFee'] for h in self._fee_history)
        return total / len(self._fee_history)

    def get_signal(self):
        """Fee-based activity signal. Always NEUTRAL direction (non-directional)
        but provides volatility_signal for other modules."""
        fees = self.get_fees()
        if not fees:
            return {'direction': 'NEUTRAL', 'confidence': 0, 'volatility_signal': False,
                    'reason': 'No mempool data'}

        fastest = fees['fastestFee']
        economy = fees['economyFee']
        urgency_ratio = fastest / economy if economy > 0 else 1

        # Spike detection vs rolling average
        avg = self._get_history_avg()
        is_spike = False
        if avg and avg > 0:
            is_spike = fastest > (avg * 3)

        # Classify fee environment
        if fastest > 150:
            level = 'extreme'
            confidence = 80
            volatility = True
            reason = f'Extreme fees {fastest} sat/vB (urgency ratio {urgency_ratio:.1f}x)'
        elif fastest > 80:
            level = 'high'
            confidence = 60
            volatility = True
            reason = f'High fees {fastest} sat/vB (urgency ratio {urgency_ratio:.1f}x)'
        elif urgency_ratio > 5:
            level = 'panic'
            confidence = 70
            volatility = True
            reason = f'Fee panic: urgency ratio {urgency_ratio:.1f}x ({fastest}/{economy} sat/vB)'
        elif is_spike:
            level = 'spike'
            confidence = 65
            volatility = True
            reason = f'Fee spike {fastest} sat/vB (avg {avg:.0f}, {fastest/avg:.1f}x)'
        else:
            level = 'normal'
            confidence = 20
            volatility = False
            reason = f'Normal fees {fastest} sat/vB'

        return {
            'direction': 'NEUTRAL',
            'confidence': confidence,
            'volatility_signal': volatility,
            'fee_level': level,
            'fastest_fee': fastest,
            'urgency_ratio': round(urgency_ratio, 2),
            'reason': reason,
        }


if __name__ == "__main__":
    cprint("Testing Mempool Fees Provider...", "cyan")
    provider = MempoolFeesProvider.get_instance()

    fees = provider.get_fees()
    if fees:
        cprint(f"  Fastest: {fees['fastestFee']} sat/vB", "white")
        cprint(f"  Half hour: {fees['halfHourFee']} sat/vB", "white")
        cprint(f"  Hour: {fees['hourFee']} sat/vB", "white")
        cprint(f"  Economy: {fees['economyFee']} sat/vB", "white")

    signal = provider.get_signal()
    cprint(f"  Signal: {signal['direction']} (confidence: {signal['confidence']})", "yellow")
    cprint(f"  Volatility: {signal.get('volatility_signal', False)}", "white")
    if signal.get('reason'):
        cprint(f"  Reason: {signal['reason']}", "white")
