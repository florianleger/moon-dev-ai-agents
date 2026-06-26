"""Mechanical market-regime gate (no LLM).

A single pure function used identically by the backtest harness and the live
Adaptive Hybrid strategy, so what is validated is exactly what runs. It filters
entries by direction according to a regime derived purely from closed-bar OHLCV:

  - Kaufman Efficiency Ratio (ER) over `er_period` bars distinguishes trend from
    chop. ER = |close[t] - close[t-n]| / sum_{i}|close[i] - close[i-1]|, in [0,1].
    Low ER = choppy/range; high ER = directional.
  - EMA200 position + slope gives trend direction.

Regimes: TRENDING_UP / TRENDING_DOWN / RANGE.

Entry rule (conservative, literature thresholds, no per-trade tuning):
  - TRENDING_DOWN : block BUYs (counter-trend longs in a downtrend bleed),
                    allow SELLs.
  - TRENDING_UP   : block SELLs, allow BUYs.
  - RANGE         : raise the score threshold both ways (harder to enter chop).

In soft mode (hard_block=False) a blocked direction is not refused outright but
penalised by `block_threshold_delta` instead.
"""
from __future__ import annotations

# Literature-based defaults (NOT optimised on the live sample).
ER_PERIOD = 10
ER_CHOP_THRESHOLD = 0.30      # ER below this = chop/range
EMA_SLOPE_LOOKBACK = 10       # bars used to measure EMA200 slope
RANGE_THRESHOLD_DELTA = 5.0   # extra score required to enter in RANGE
BLOCK_THRESHOLD_DELTA = 8.0   # extra score required for a counter-trend entry (soft mode)


def efficiency_ratio(closes, period: int = ER_PERIOD) -> float:
    """Kaufman Efficiency Ratio over the last `period` closed bars."""
    if closes is None or len(closes) < period + 1:
        return 0.0
    window = [float(c) for c in closes[-(period + 1):]]
    direction = abs(window[-1] - window[0])
    volatility = sum(abs(window[i] - window[i - 1]) for i in range(1, len(window)))
    if volatility <= 0:
        return 0.0
    return direction / volatility


def _is_nan(x) -> bool:
    return x is None or x != x  # NaN != NaN


def classify_regime(close: float, ema_200: float, ema_200_prev: float, er: float,
                    er_chop: float = ER_CHOP_THRESHOLD) -> str:
    """Return 'TRENDING_UP', 'TRENDING_DOWN' or 'RANGE'.

    Trend direction needs EMA200 and its slope. If either is unavailable
    (NaN/None — e.g. EMA200 not yet converged), the trend is undetermined and
    we return RANGE: that only stiffens the threshold, it never blocks a
    direction on missing data.
    """
    if er < er_chop:
        return 'RANGE'
    if _is_nan(close) or _is_nan(ema_200) or _is_nan(ema_200_prev):
        return 'RANGE'
    rising = ema_200 > ema_200_prev
    falling = ema_200 < ema_200_prev
    if close > ema_200 and rising:
        return 'TRENDING_UP'
    if close < ema_200 and falling:
        return 'TRENDING_DOWN'
    return 'RANGE'


def _decision(regime: str, direction: str, hard_block: bool):
    """Return (blocked: bool, threshold_delta: float) for a candidate entry."""
    d = (direction or '').upper()
    if regime == 'TRENDING_DOWN' and d == 'BUY':
        return (hard_block, 0.0 if hard_block else BLOCK_THRESHOLD_DELTA)
    if regime == 'TRENDING_UP' and d == 'SELL':
        return (hard_block, 0.0 if hard_block else BLOCK_THRESHOLD_DELTA)
    if regime == 'RANGE':
        return (False, RANGE_THRESHOLD_DELTA)
    return (False, 0.0)


def regime_gate(closes, ema_200: float, ema_200_prev: float, direction: str,
                *, hard_block: bool = True, er_period: int = ER_PERIOD,
                er_chop: float = ER_CHOP_THRESHOLD) -> dict:
    """Evaluate the gate for a candidate entry.

    Returns {regime, er, blocked, threshold_delta}. `closes` is the sequence of
    closed-bar closes up to and including the signal bar (no look-ahead).
    """
    er = efficiency_ratio(closes, er_period)
    close = float(closes[-1]) if closes is not None and len(closes) else 0.0
    regime = classify_regime(close, ema_200, ema_200_prev, er, er_chop)
    blocked, delta = _decision(regime, direction, hard_block)
    return {'regime': regime, 'er': round(er, 4), 'blocked': blocked,
            'threshold_delta': delta}
