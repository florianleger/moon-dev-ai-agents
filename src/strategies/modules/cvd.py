"""CVD (Cumulative Volume Delta) scoring module."""
import requests
import json
import os
from collections import deque
from datetime import datetime
import threading
import time

HYPERLIQUID_API_URL = 'https://api.hyperliquid.xyz/info'
_cvd_lock = threading.Lock()
_trade_cache = {}  # {symbol: (timestamp, trades)}
_CACHE_TTL = 30  # seconds
_PERSIST_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'adaptive_hybrid', 'cvd_history.json')


def _load_history() -> dict:
    """Load persisted CVD history from disk."""
    try:
        with open(_PERSIST_PATH, 'r') as f:
            raw = json.load(f)
        result = {}
        for sym, entries in raw.items():
            dq = deque(maxlen=50)
            for ts_str, ratio, price in entries:
                dq.append((datetime.fromisoformat(ts_str), ratio, price))
            result[sym] = dq
        return result
    except Exception:
        return {}


def _save_history(snapshot: dict):
    """Persist CVD history to disk (atomic write). Caller passes a snapshot."""
    try:
        os.makedirs(os.path.dirname(_PERSIST_PATH), exist_ok=True)
        raw = {}
        for sym, entries in snapshot.items():
            raw[sym] = [(ts.isoformat(), ratio, price) for ts, ratio, price in entries]
        tmp = _PERSIST_PATH + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(raw, f)
        os.rename(tmp, _PERSIST_PATH)
    except Exception:
        pass


_cvd_history = _load_history()


def _fetch_recent_trades(symbol: str) -> list:
    now = time.monotonic()
    with _cvd_lock:
        cached = _trade_cache.get(symbol)
    if cached and (now - cached[0]) < _CACHE_TTL:
        return cached[1]
    try:
        resp = requests.post(HYPERLIQUID_API_URL,
                             headers={'Content-Type': 'application/json'},
                             json={"type": "recentTrades", "coin": symbol},
                             timeout=10)
        resp.raise_for_status()
        trades = resp.json()
        with _cvd_lock:
            _trade_cache[symbol] = (now, trades)
        return trades
    except Exception:
        return cached[1] if cached else []


def score_cvd(symbol: str, indicators: dict, config: dict = None) -> dict:
    """CVD divergence + momentum scoring.

    Args:
        symbol: Token symbol (e.g. 'BTC').
        indicators: Dict of last-row indicator values (must include 'close').
        config: Optional config overrides.

    Returns:
        dict with 'score' (0-100), 'direction', and 'reason'.
    """
    try:
        trades = _fetch_recent_trades(symbol)
        if not trades or len(trades) < 20:
            return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'Insufficient trades'}

        buy_vol = sum(float(t['sz']) * float(t['px']) for t in trades if t['side'] == 'B')
        sell_vol = sum(float(t['sz']) * float(t['px']) for t in trades if t['side'] == 'A')
        cvd = buy_vol - sell_vol
        total_vol = buy_vol + sell_vol
        if total_vol == 0:
            return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'No volume'}

        cvd_ratio = cvd / total_vol  # -1 to +1

        now = datetime.now()
        with _cvd_lock:
            if symbol not in _cvd_history:
                _cvd_history[symbol] = deque(maxlen=50)
            _cvd_history[symbol].append((now, cvd_ratio, indicators.get('close', 0)))
            history = list(_cvd_history[symbol])
            snapshot = {s: list(dq) for s, dq in _cvd_history.items()}
        _save_history(snapshot)

        long_score = 0
        short_score = 0

        # Signal 1: CVD momentum (threshold lowered from 0.15 to 0.05)
        if cvd_ratio > 0.05:
            long_score += 20 + int(min(cvd_ratio, 0.5) * 60)
        elif cvd_ratio < -0.05:
            short_score += 20 + int(min(abs(cvd_ratio), 0.5) * 60)

        # Signal 2: CVD divergence (requires history)
        if len(history) >= 5:
            old_cvd = history[-5][1]
            old_price = history[-5][2]
            if old_price > 0:
                price_change = (indicators.get('close', 0) - old_price) / old_price
                # Bullish divergence: price down, CVD up
                if price_change < -0.005 and cvd_ratio > old_cvd + 0.1:
                    long_score += 35
                # Bearish divergence: price up, CVD down
                if price_change > 0.005 and cvd_ratio < old_cvd - 0.1:
                    short_score += 35

        best = max(long_score, short_score)
        direction = 'BUY' if long_score > short_score else ('SELL' if short_score > long_score else 'NEUTRAL')
        return {'score': min(100, best), 'direction': direction,
                'reason': f'CVD={cvd_ratio:+.3f} buy={buy_vol:,.0f} sell={sell_vol:,.0f}'}
    except Exception as e:
        return {'score': 0, 'direction': 'NEUTRAL', 'reason': f'CVD error: {e}'}
