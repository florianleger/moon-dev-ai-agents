"""Funding Composite scoring module.

Merges three funding-related signals into a single weighted composite score:
  1. Per-token funding Z-score contrarian signal (50% weight)
  2. Cross-exchange funding divergence HL vs Binance (30% weight)
  3. Funding squeeze detection with volatility compression (20% weight)

Each sub-signal is computed independently and can fail gracefully; the
composite is re-weighted from whichever sub-signals succeed.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module metadata
# ---------------------------------------------------------------------------
family = 'derivatives'

# Internal weights (must sum to 1.0)
W_ZSCORE = 0.50
W_DIVERGENCE = 0.30
W_SQUEEZE = 0.20


def score(symbol: str, ohlcv_df, market_data=None, **kwargs) -> dict:
    """Composite funding signal combining contrarian Z-score, cross-exchange
    divergence, and squeeze detection.

    Args:
        symbol: Token symbol (e.g. 'BTC').
        ohlcv_df: OHLCV DataFrame (used for BB-width squeeze detection).
        market_data: MarketDataProvider instance (for funding rate fetch).
        **kwargs: Must include 'indicators' dict and optionally
            'funding_zscore' (float). If funding_zscore is not provided,
            the Z-score sub-signal is skipped.

    Returns:
        dict: {
            'direction': 'BUY' | 'SELL' | 'NEUTRAL',
            'score': 0-100,
            'reason': str,
            'details': {
                'zscore_signal': {...} | None,
                'divergence_signal': {...} | None,
                'squeeze_signal': {...} | None,
                'composite_reasoning': str,
            }
        }
    """
    indicators = kwargs.get('indicators', {})
    funding_zscore = kwargs.get('funding_zscore')

    sub_results = {}
    weights = {}

    # ------------------------------------------------------------------
    # 1. Funding Z-score contrarian (50%)
    # ------------------------------------------------------------------
    zscore_sig = _compute_zscore_signal(funding_zscore)
    if zscore_sig is not None:
        sub_results['zscore'] = zscore_sig
        weights['zscore'] = W_ZSCORE

    # ------------------------------------------------------------------
    # 2. Cross-exchange divergence (30%)
    # ------------------------------------------------------------------
    div_sig = _compute_divergence_signal(symbol)
    if div_sig is not None:
        sub_results['divergence'] = div_sig
        weights['divergence'] = W_DIVERGENCE

    # ------------------------------------------------------------------
    # 3. Funding squeeze (20%)
    # ------------------------------------------------------------------
    squeeze_sig = _compute_squeeze_signal(funding_zscore, indicators, ohlcv_df)
    if squeeze_sig is not None:
        sub_results['squeeze'] = squeeze_sig
        weights['squeeze'] = W_SQUEEZE

    # ------------------------------------------------------------------
    # Combine
    # ------------------------------------------------------------------
    if not sub_results:
        return _neutral('All funding sub-signals unavailable')

    # Rescale weights so they sum to 1.0
    total_w = sum(weights.values())
    norm_weights = {k: v / total_w for k, v in weights.items()}

    # Weighted BUY / SELL tallies
    buy_score = 0.0
    sell_score = 0.0
    parts = []

    for key, sig in sub_results.items():
        w = norm_weights[key]
        if sig['direction'] == 'BUY':
            buy_score += sig['score'] * w
        elif sig['direction'] == 'SELL':
            sell_score += sig['score'] * w
        parts.append(f"{key}={sig['direction'][0]}{sig['score']}")

    composite_score = max(buy_score, sell_score)
    if buy_score > sell_score:
        direction = 'BUY'
    elif sell_score > buy_score:
        direction = 'SELL'
    else:
        direction = 'NEUTRAL'

    composite_score = int(min(100, round(composite_score)))
    reasoning = ' | '.join(parts)

    return {
        'score': composite_score,
        'direction': direction,
        'reason': f'FundingComp: {reasoning}',
        'details': {
            'zscore_signal': sub_results.get('zscore'),
            'divergence_signal': sub_results.get('divergence'),
            'squeeze_signal': sub_results.get('squeeze'),
            'composite_reasoning': reasoning,
        },
    }


# -------------------------------------------------------------------------
# Legacy wrapper — keeps backward compatibility with existing callers that
# use score_funding_composite(symbol, indicators, funding_zscore, config).
# -------------------------------------------------------------------------
def score_funding_composite(
    symbol: str,
    indicators: dict,
    funding_zscore: float = 0.0,
    config: dict = None,
) -> dict:
    """Backward-compatible entry point.

    Delegates to :func:`score` so that callers importing the old function
    name continue to work without changes.
    """
    return score(
        symbol=symbol,
        ohlcv_df=indicators.get('_df'),
        indicators=indicators,
        funding_zscore=funding_zscore,
    )


# =========================================================================
# Sub-signal helpers
# =========================================================================

def _compute_zscore_signal(funding_zscore):
    """Contrarian signal from per-token funding Z-score.

    Mirrors logic from funding.py:score_funding_contrarian.
    Extreme negative funding (shorts paying longs) -> contrarian long.
    Extreme positive funding (longs paying shorts) -> contrarian short.
    """
    if funding_zscore is None:
        return None

    long_score = 0
    short_score = 0

    # Extreme negative funding -> contrarian long
    if funding_zscore <= -2.0:
        long_score += 80
    elif funding_zscore <= -1.5:
        long_score += 55
    elif funding_zscore <= -1.0:
        long_score += 30

    # Extreme positive funding -> contrarian short
    if funding_zscore >= 2.0:
        short_score += 80
    elif funding_zscore >= 1.5:
        short_score += 55
    elif funding_zscore >= 1.0:
        short_score += 30

    best = max(long_score, short_score)
    if best == 0:
        return None

    direction = 'BUY' if long_score > short_score else (
        'SELL' if short_score > long_score else 'NEUTRAL')

    return {
        'score': min(100, best),
        'direction': direction,
        'reason': f'Z={funding_zscore:.2f}',
    }


def _compute_divergence_signal(symbol: str):
    """Cross-exchange funding divergence (HL vs Binance).

    Delegates to CrossExchangeFundingProvider.get_divergence_signal and
    normalises the output.  Returns None on any failure so the composite
    can degrade gracefully.
    """
    try:
        from src.data_providers.cross_exchange_funding import (
            CrossExchangeFundingProvider,
        )
        provider = CrossExchangeFundingProvider.get_instance()
        signal = provider.get_divergence_signal(symbol)

        if not signal or signal.get('confidence', 0) == 0:
            return None

        return {
            'score': signal['confidence'],
            'direction': signal['direction'],
            'reason': signal.get('reason', ''),
            'hl_annual_pct': signal.get('hl_annual_pct'),
            'binance_annual_pct': signal.get('binance_annual_pct'),
            'spread_annual_pct': signal.get('spread_annual_pct'),
        }
    except Exception:
        return None


def _compute_squeeze_signal(funding_zscore, indicators: dict, ohlcv_df=None):
    """Funding squeeze component: extreme funding + volatility compression.

    Mirrors the funding-related logic from squeeze.py:score_squeeze_detector.
    Returns None when funding_zscore is unavailable or when no squeeze
    condition is detected.
    """
    if funding_zscore is None:
        return None

    long_score = 0
    short_score = 0

    # Bollinger squeeze detection (BB width compression)
    bb_squeeze = False
    df = ohlcv_df if ohlcv_df is not None else indicators.get('_df')
    if df is not None and len(df) >= 20:
        bb_width_series = (
            df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        )
        bb_width_percentile = bb_width_series.rank(pct=True).iloc[-1]
        bb_squeeze = bb_width_percentile < 0.25  # Bottom 25% = squeeze

    rsi = indicators.get('rsi', 50)
    rsi_oversold = indicators.get('rsi_oversold', 30)
    rsi_overbought = indicators.get('rsi_overbought', 70)

    # Short squeeze: very negative funding + compression -> long
    if funding_zscore < -1.5:
        long_score += 40
        if bb_squeeze:
            long_score += 20
        if rsi < rsi_oversold + 10:
            long_score += 15

    # Long squeeze: very positive funding + compression -> short
    if funding_zscore > 1.5:
        short_score += 40
        if bb_squeeze:
            short_score += 20
        if rsi > rsi_overbought - 10:
            short_score += 15

    best = max(long_score, short_score)
    if best == 0:
        return None

    direction = 'BUY' if long_score > short_score else (
        'SELL' if short_score > long_score else 'NEUTRAL')

    return {
        'score': min(100, best),
        'direction': direction,
        'reason': f'squeeze(z={funding_zscore:.2f} bb_sq={bb_squeeze})',
    }


# =========================================================================
# Utilities
# =========================================================================

def _neutral(reason: str) -> dict:
    """Return a neutral / zero-score result."""
    return {
        'score': 0,
        'direction': 'NEUTRAL',
        'reason': reason,
        'details': {
            'zscore_signal': None,
            'divergence_signal': None,
            'squeeze_signal': None,
            'composite_reasoning': reason,
        },
    }
