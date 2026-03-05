"""Binance Futures Sentiment data provider - Long/Short ratio + Taker Buy/Sell volume.

Free API, no key required. Provides crowd positioning data for contrarian signals.
Endpoints:
- Global Long/Short Account Ratio
- Top Trader Long/Short Account Ratio
- Top Trader Long/Short Position Ratio
- Taker Buy/Sell Volume
"""
import requests
import time
from termcolor import cprint
from src.utils.alerting import alert_service_down


class BinanceSentimentProvider:
    """Singleton provider for Binance Futures positioning data."""
    _instance = None
    _cache = {}
    _cache_time = {}
    _cache_ttl = 120  # 2 min cache (data updates every 5 min on Binance)

    BASE_URL = "https://fapi.binance.com"

    # Map HyperLiquid symbols to Binance USDT pairs
    SYMBOL_MAP = {
        'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'SOL': 'SOLUSDT',
        'XRP': 'XRPUSDT', 'DOGE': 'DOGEUSDT', 'ADA': 'ADAUSDT',
        'AVAX': 'AVAXUSDT', 'LINK': 'LINKUSDT', 'DOT': 'DOTUSDT',
        'SUI': 'SUIUSDT', 'NEAR': 'NEARUSDT', 'TAO': 'TAOUSDT',
        'AAVE': 'AAVEUSDT', 'ENA': 'ENAUSDT',
        'kPEPE': 'PEPEUSDT',
    }

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_cached(self, cache_key, url, params, timeout=10):
        """Generic cached GET request."""
        if cache_key in self._cache and time.time() - self._cache_time.get(cache_key, 0) < self._cache_ttl:
            return self._cache[cache_key]
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            self._cache[cache_key] = data
            self._cache_time[cache_key] = time.time()
            return data
        except Exception as e:
            cprint(f"[BinanceSentiment] API error ({cache_key}): {e}", "red")
            alert_service_down("Binance Sentiment", e)
            return self._cache.get(cache_key)  # stale cache

    def get_long_short_ratio(self, symbol, period='5m', limit=1):
        """Get global Long/Short Account Ratio.

        Ratio > 1 = more accounts long than short (crowd is long)
        Ratio < 1 = more accounts short than long (crowd is short)

        Args:
            symbol: HL symbol (e.g. 'BTC')
            period: '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'
            limit: Number of data points (max 500)
        """
        binance_sym = self.SYMBOL_MAP.get(symbol)
        if not binance_sym:
            return None

        key = f"ls_ratio_{binance_sym}_{period}"
        data = self._get_cached(key,
            f"{self.BASE_URL}/futures/data/globalLongShortAccountRatio",
            {'symbol': binance_sym, 'period': period, 'limit': limit})

        if data and len(data) > 0:
            entry = data[-1]
            return {
                'long_account': float(entry.get('longAccount', 0.5)),
                'short_account': float(entry.get('shortAccount', 0.5)),
                'long_short_ratio': float(entry.get('longShortRatio', 1.0)),
                'timestamp': int(entry.get('timestamp', 0)),
            }
        return None

    def get_top_trader_ratio(self, symbol, period='5m', limit=1):
        """Get Top Trader Long/Short Position Ratio (top 20% by margin).

        More informative than global ratio - tracks smart money positioning.
        """
        binance_sym = self.SYMBOL_MAP.get(symbol)
        if not binance_sym:
            return None

        key = f"top_ratio_{binance_sym}_{period}"
        data = self._get_cached(key,
            f"{self.BASE_URL}/futures/data/topLongShortPositionRatio",
            {'symbol': binance_sym, 'period': period, 'limit': limit})

        if data and len(data) > 0:
            entry = data[-1]
            return {
                'long_account': float(entry.get('longAccount', 0.5)),
                'short_account': float(entry.get('shortAccount', 0.5)),
                'long_short_ratio': float(entry.get('longShortRatio', 1.0)),
                'timestamp': int(entry.get('timestamp', 0)),
            }
        return None

    def get_taker_buy_sell_volume(self, symbol, period='5m', limit=1):
        """Get Taker Buy/Sell Volume ratio.

        buy_vol > sell_vol = aggressive buying pressure
        sell_vol > buy_vol = aggressive selling pressure

        This is one of the strongest short-term signals: shows actual market
        order flow direction, not just passive limit orders.
        """
        binance_sym = self.SYMBOL_MAP.get(symbol)
        if not binance_sym:
            return None

        key = f"taker_vol_{binance_sym}_{period}"
        data = self._get_cached(key,
            f"{self.BASE_URL}/futures/data/takerlongshortRatio",
            {'symbol': binance_sym, 'period': period, 'limit': limit})

        if data and len(data) > 0:
            entry = data[-1]
            return {
                'buy_sell_ratio': float(entry.get('buySellRatio', 1.0)),
                'buy_vol': float(entry.get('buyVol', 0)),
                'sell_vol': float(entry.get('sellVol', 0)),
                'timestamp': int(entry.get('timestamp', 0)),
            }
        return None

    def get_taker_volume_history(self, symbol, period='5m', limit=12):
        """Get historical taker volume for trend detection."""
        binance_sym = self.SYMBOL_MAP.get(symbol)
        if not binance_sym:
            return []

        key = f"taker_hist_{binance_sym}_{period}_{limit}"
        data = self._get_cached(key,
            f"{self.BASE_URL}/futures/data/takerlongshortRatio",
            {'symbol': binance_sym, 'period': period, 'limit': limit})

        if data:
            return [{
                'buy_sell_ratio': float(d.get('buySellRatio', 1.0)),
                'buy_vol': float(d.get('buyVol', 0)),
                'sell_vol': float(d.get('sellVol', 0)),
                'timestamp': int(d.get('timestamp', 0)),
            } for d in data]
        return []

    def get_composite_signal(self, symbol):
        """Get composite positioning signal for a symbol.

        Combines global L/S ratio, top trader ratio, and taker volume.

        Returns:
            dict with 'direction', 'confidence', 'reason', and raw data
        """
        global_ls = self.get_long_short_ratio(symbol, period='15m')
        top_ls = self.get_top_trader_ratio(symbol, period='15m')
        taker = self.get_taker_buy_sell_volume(symbol, period='15m')

        if not any([global_ls, top_ls, taker]):
            return {'direction': 'NEUTRAL', 'confidence': 0, 'reason': 'No data'}

        long_pressure = 0
        short_pressure = 0
        reasons = []

        # Global L/S ratio (contrarian): crowd is usually wrong at extremes
        if global_ls:
            ratio = global_ls['long_short_ratio']
            if ratio > 2.0:
                short_pressure += 30  # Too many longs = contrarian short
                reasons.append(f'crowd_long={ratio:.2f}')
            elif ratio > 1.5:
                short_pressure += 15
                reasons.append(f'crowd_lean_long={ratio:.2f}')
            elif ratio < 0.5:
                long_pressure += 30  # Too many shorts = contrarian long
                reasons.append(f'crowd_short={ratio:.2f}')
            elif ratio < 0.7:
                long_pressure += 15
                reasons.append(f'crowd_lean_short={ratio:.2f}')

        # Top trader ratio (follow smart money)
        if top_ls:
            ratio = top_ls['long_short_ratio']
            if ratio > 1.5:
                long_pressure += 20  # Smart money is long
                reasons.append(f'smart_long={ratio:.2f}')
            elif ratio < 0.7:
                short_pressure += 20  # Smart money is short
                reasons.append(f'smart_short={ratio:.2f}')

        # Taker volume (momentum signal)
        if taker:
            ratio = taker['buy_sell_ratio']
            if ratio > 1.3:
                long_pressure += 25  # Aggressive buying
                reasons.append(f'taker_buy={ratio:.2f}')
            elif ratio < 0.7:
                short_pressure += 25  # Aggressive selling
                reasons.append(f'taker_sell={ratio:.2f}')

        best = max(long_pressure, short_pressure)
        if long_pressure > short_pressure:
            direction = 'BUY'
        elif short_pressure > long_pressure:
            direction = 'SELL'
        else:
            direction = 'NEUTRAL'

        return {
            'direction': direction,
            'confidence': min(100, best),
            'reason': ' | '.join(reasons) if reasons else 'neutral',
            'global_ls': global_ls,
            'top_ls': top_ls,
            'taker': taker,
        }


if __name__ == "__main__":
    cprint("Testing Binance Sentiment Provider...", "cyan")
    provider = BinanceSentimentProvider.get_instance()

    for symbol in ['BTC', 'ETH', 'SOL']:
        cprint(f"\n{symbol}:", "white", attrs=['bold'])

        ls = provider.get_long_short_ratio(symbol, period='15m')
        if ls:
            cprint(f"  Global L/S ratio: {ls['long_short_ratio']:.3f} "
                   f"(long={ls['long_account']:.1%} short={ls['short_account']:.1%})", "white")

        top = provider.get_top_trader_ratio(symbol, period='15m')
        if top:
            cprint(f"  Top Trader ratio: {top['long_short_ratio']:.3f}", "white")

        taker = provider.get_taker_buy_sell_volume(symbol, period='15m')
        if taker:
            cprint(f"  Taker Buy/Sell: {taker['buy_sell_ratio']:.3f} "
                   f"(buy={taker['buy_vol']:,.0f} sell={taker['sell_vol']:,.0f})", "white")

        signal = provider.get_composite_signal(symbol)
        cprint(f"  Signal: {signal['direction']} (confidence={signal['confidence']}) "
               f"- {signal['reason']}", "yellow")
