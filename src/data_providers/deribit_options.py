"""Deribit options data provider - public API, no auth required."""
import requests
import time
import threading
from datetime import datetime, timezone
from termcolor import cprint
from src.utils.alerting import alert_service_down


class DeribitOptionsProvider:
    """Singleton provider for Deribit options data (put/call ratio, max pain)."""
    _instance = None
    _instance_lock = threading.Lock()
    _cache_ttl = 120  # 2 min

    BASE_URL = "https://www.deribit.com/api/v2/public"

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

    def _fetch_book_summary(self, currency='BTC'):
        cache_key = f"book_{currency}"
        with self._data_lock:
            if cache_key in self._cache and time.time() - self._cache_time.get(cache_key, 0) < self._cache_ttl:
                return self._cache[cache_key]
        try:
            url = f"{self.BASE_URL}/get_book_summary_by_currency?currency={currency}&kind=option"
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json().get('result', [])
            with self._data_lock:
                self._cache[cache_key] = data
                self._cache_time[cache_key] = time.time()
            return data
        except Exception as e:
            cprint(f"[DeribitOptions] API error: {e}", "red")
            alert_service_down("Deribit Options", e)
            return self._cache.get(cache_key)

    def _fetch_ticker(self, currency='BTC'):
        """Fetch underlying index price."""
        cache_key = f"ticker_{currency}"
        with self._data_lock:
            if cache_key in self._cache and time.time() - self._cache_time.get(cache_key, 0) < self._cache_ttl:
                return self._cache[cache_key]
        try:
            url = f"{self.BASE_URL}/ticker?instrument_name={currency}-PERPETUAL"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json().get('result', {})
            with self._data_lock:
                self._cache[cache_key] = data
                self._cache_time[cache_key] = time.time()
            return data
        except Exception as e:
            cprint(f"[DeribitOptions] Ticker error: {e}", "red")
            return self._cache.get(cache_key)

    def get_put_call_ratio(self, currency='BTC'):
        """Get put/call ratio by open interest."""
        data = self._fetch_book_summary(currency)
        if not data:
            return None

        call_oi = 0
        put_oi = 0
        for item in data:
            name = item.get('instrument_name', '')
            oi = item.get('open_interest', 0) or 0
            if '-C' in name:
                call_oi += oi
            elif '-P' in name:
                put_oi += oi

        ratio = put_oi / call_oi if call_oi > 0 else 0
        return {
            'put_call_ratio': round(ratio, 3),
            'call_oi': call_oi,
            'put_oi': put_oi,
        }

    def get_max_pain(self, currency='BTC'):
        """Get max pain strike for nearest expiry."""
        data = self._fetch_book_summary(currency)
        if not data:
            return None

        # Group OI by expiry and strike
        expiries = {}
        for item in data:
            name = item.get('instrument_name', '')
            parts = name.split('-')
            if len(parts) < 4:
                continue
            expiry = parts[1]
            try:
                strike = float(parts[2])
            except (ValueError, IndexError):
                continue
            option_type = parts[3]  # C or P
            oi = item.get('open_interest', 0) or 0

            if expiry not in expiries:
                expiries[expiry] = {}
            if strike not in expiries[expiry]:
                expiries[expiry][strike] = {'C': 0, 'P': 0}
            expiries[expiry][strike][option_type] += oi

        if not expiries:
            return None

        # Pick nearest future expiry by parsing Deribit date format (DDMMMYY)
        now = datetime.now(timezone.utc).replace(tzinfo=None)  # naive UTC for strptime comparison
        best_expiry = None
        best_expiry_date = None
        for exp_label in expiries.keys():
            try:
                exp_date = datetime.strptime(exp_label, "%d%b%y")
                if exp_date >= now:
                    if best_expiry_date is None or exp_date < best_expiry_date:
                        best_expiry_date = exp_date
                        best_expiry = exp_label
            except (ValueError, TypeError):
                continue

        # Fallback: if no future expiry parsed, pick the one with most OI
        if best_expiry is None:
            best_expiry = max(expiries.keys(), key=lambda e: sum(
                s['C'] + s['P'] for s in expiries[e].values()
            ))
        strikes_data = expiries[best_expiry]
        all_strikes = sorted(strikes_data.keys())

        if not all_strikes:
            return None

        # Calculate max pain: strike where total pain (losses) for option holders is maximized
        min_pain = float('inf')
        max_pain_strike = all_strikes[0]

        for test_strike in all_strikes:
            total_pain = 0
            for strike, oi_data in strikes_data.items():
                # Max pain = strike where total payout to option holders is minimized
                call_value = max(0, test_strike - strike) * oi_data['C']
                put_value = max(0, strike - test_strike) * oi_data['P']
                total_pain += call_value + put_value

            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = test_strike

        # Get underlying price for distance calc
        ticker = self._fetch_ticker(currency)
        underlying = ticker.get('last_price', 0) if ticker else 0
        distance_pct = ((underlying - max_pain_strike) / max_pain_strike * 100) if max_pain_strike else 0

        return {
            'max_pain': max_pain_strike,
            'expiry': best_expiry,
            'underlying': underlying,
            'distance_pct': round(distance_pct, 2),
        }

    def get_signal(self, currency='BTC'):
        """Combined options signal from P/C ratio and max pain."""
        pcr = self.get_put_call_ratio(currency)
        mp = self.get_max_pain(currency)

        if not pcr:
            return {'direction': 'NEUTRAL', 'confidence': 0, 'reason': 'No options data'}

        ratio = pcr['put_call_ratio']
        reasons = []

        # P/C ratio signal (contrarian)
        pcr_direction = 'NEUTRAL'
        pcr_confidence = 0
        if ratio > 1.2:
            pcr_direction = 'BUY'
            pcr_confidence = 65
            reasons.append(f'P/C ratio {ratio:.2f} (high hedging = fear)')
        elif ratio > 0.9:
            pcr_direction = 'BUY'
            pcr_confidence = 35
            reasons.append(f'P/C ratio {ratio:.2f} (elevated puts)')
        elif ratio < 0.5:
            pcr_direction = 'SELL'
            pcr_confidence = 65
            reasons.append(f'P/C ratio {ratio:.2f} (complacency)')
        elif ratio < 0.7:
            pcr_direction = 'SELL'
            pcr_confidence = 35
            reasons.append(f'P/C ratio {ratio:.2f} (low hedging)')
        else:
            pcr_confidence = 20
            reasons.append(f'P/C ratio {ratio:.2f} (neutral)')

        # Max pain signal
        mp_direction = 'NEUTRAL'
        mp_confidence = 0
        if mp and mp['underlying'] > 0:
            dist = mp['distance_pct']
            if dist > 5:
                mp_direction = 'SELL'
                mp_confidence = 50
                reasons.append(f'Price {dist:+.1f}% above max pain ${mp["max_pain"]:,.0f}')
            elif dist < -5:
                mp_direction = 'BUY'
                mp_confidence = 50
                reasons.append(f'Price {dist:+.1f}% below max pain ${mp["max_pain"]:,.0f}')
            else:
                reasons.append(f'Near max pain ${mp["max_pain"]:,.0f} ({dist:+.1f}%)')

        # Combine signals
        if pcr_direction == mp_direction and pcr_direction != 'NEUTRAL':
            return {'direction': pcr_direction, 'confidence': min(85, pcr_confidence + mp_confidence),
                    'reason': ' | '.join(reasons), 'put_call_ratio': ratio}
        elif pcr_confidence >= mp_confidence:
            return {'direction': pcr_direction, 'confidence': pcr_confidence,
                    'reason': ' | '.join(reasons), 'put_call_ratio': ratio}
        else:
            return {'direction': mp_direction, 'confidence': mp_confidence,
                    'reason': ' | '.join(reasons), 'put_call_ratio': ratio}


if __name__ == "__main__":
    cprint("Testing Deribit Options Provider...", "cyan")
    provider = DeribitOptionsProvider.get_instance()

    pcr = provider.get_put_call_ratio('BTC')
    if pcr:
        cprint(f"  P/C Ratio: {pcr['put_call_ratio']:.3f} (calls: {pcr['call_oi']:.0f}, puts: {pcr['put_oi']:.0f})", "white")

    mp = provider.get_max_pain('BTC')
    if mp:
        cprint(f"  Max Pain: ${mp['max_pain']:,.0f} (expiry: {mp['expiry']}, dist: {mp['distance_pct']:+.1f}%)", "white")

    signal = provider.get_signal('BTC')
    cprint(f"  Signal: {signal['direction']} (confidence: {signal['confidence']})", "yellow")
    if signal.get('reason'):
        cprint(f"  Reason: {signal['reason']}", "white")
