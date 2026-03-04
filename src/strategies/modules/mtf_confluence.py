"""Multi-Timeframe Confluence scoring module.

Fetches 4h and 1d candles alongside the primary 1h timeframe.
Scores trend alignment across timeframes:
- Higher timeframe alignment = score bonus
- Misalignment = score penalty

This is a pure technical module (no LLM needed).
"""

import pandas as pd
import numpy as np
from termcolor import cprint
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator


def _detect_trend(df: pd.DataFrame) -> dict:
    """Detect trend direction and strength from a DataFrame.

    Returns:
        dict with 'direction' ('BULLISH', 'BEARISH', 'NEUTRAL'),
              'strength' (0-100), and 'details' string.
    """
    if df is None or len(df) < 20:
        return {'direction': 'NEUTRAL', 'strength': 0, 'details': 'Insufficient data'}

    close = df['close']
    high = df['high']
    low = df['low']

    # EMAs
    ema_9 = EMAIndicator(close=close, window=min(9, len(df) - 1)).ema_indicator()
    ema_21 = EMAIndicator(close=close, window=min(21, len(df) - 1)).ema_indicator()

    # ADX
    adx_window = min(14, len(df) - 1)
    if adx_window >= 2:
        adx_ind = ADXIndicator(high=high, low=low, close=close, window=adx_window)
        adx_val = float(adx_ind.adx().iloc[-1]) if pd.notna(adx_ind.adx().iloc[-1]) else 20
    else:
        adx_val = 20

    # RSI
    rsi_window = min(14, len(df) - 1)
    if rsi_window >= 2:
        rsi_ind = RSIIndicator(close=close, window=rsi_window)
        rsi_val = float(rsi_ind.rsi().iloc[-1]) if pd.notna(rsi_ind.rsi().iloc[-1]) else 50
    else:
        rsi_val = 50

    current_price = float(close.iloc[-1])
    current_ema9 = float(ema_9.iloc[-1]) if pd.notna(ema_9.iloc[-1]) else current_price
    current_ema21 = float(ema_21.iloc[-1]) if pd.notna(ema_21.iloc[-1]) else current_price

    # Determine direction
    bullish_signals = 0
    bearish_signals = 0

    # Price vs EMAs
    if current_price > current_ema9:
        bullish_signals += 1
    else:
        bearish_signals += 1

    if current_price > current_ema21:
        bullish_signals += 1
    else:
        bearish_signals += 1

    # EMA alignment
    if current_ema9 > current_ema21:
        bullish_signals += 1
    else:
        bearish_signals += 1

    # RSI bias
    if rsi_val > 55:
        bullish_signals += 1
    elif rsi_val < 45:
        bearish_signals += 1

    # Determine direction
    if bullish_signals >= 3:
        direction = 'BULLISH'
    elif bearish_signals >= 3:
        direction = 'BEARISH'
    else:
        direction = 'NEUTRAL'

    # Strength: based on ADX and signal agreement
    agreement = max(bullish_signals, bearish_signals) / 4.0
    strength = min(100, int(adx_val * agreement))

    return {
        'direction': direction,
        'strength': strength,
        'details': f'EMA9={current_ema9:.2f} EMA21={current_ema21:.2f} ADX={adx_val:.0f} RSI={rsi_val:.0f}',
    }


def score_mtf_confluence(
    symbol: str,
    primary_direction: str,
    fetch_candles_fn,
    primary_indicators: dict = None,
) -> dict:
    """Score multi-timeframe confluence for a given signal direction.

    Args:
        symbol: Trading symbol (e.g. 'BTC')
        primary_direction: 'BUY' or 'SELL' from the primary timeframe signal
        fetch_candles_fn: Callable(symbol, interval, candles) -> DataFrame
        primary_indicators: Optional dict of primary timeframe indicators

    Returns:
        dict with:
            'score': -30 to +30 (bonus/penalty to apply to aggregated score)
            'aligned_count': number of aligned timeframes
            'total_checked': total timeframes checked
            'details': human-readable description
            'timeframes': {tf: direction} mapping
    """
    if primary_direction not in ('BUY', 'SELL'):
        return {
            'score': 0,
            'aligned_count': 0,
            'total_checked': 0,
            'details': 'No directional signal to check',
            'timeframes': {},
        }

    expected_trend = 'BULLISH' if primary_direction == 'BUY' else 'BEARISH'

    # Higher timeframes to check
    htf_configs = [
        ('4h', 100),   # 4-hour candles, fetch 100
        ('1d', 50),    # Daily candles, fetch 50
    ]

    timeframe_results = {}
    aligned_count = 0
    total_checked = 0

    for interval, candles in htf_configs:
        try:
            df = fetch_candles_fn(symbol, interval, candles)
            if df is None or len(df) < 10:
                continue

            trend = _detect_trend(df)
            timeframe_results[interval] = trend
            total_checked += 1

            if trend['direction'] == expected_trend:
                aligned_count += 1

        except Exception as e:
            cprint(f"  [MTF] Error fetching {interval} for {symbol}: {e}", "yellow")
            continue

    if total_checked == 0:
        return {
            'score': 0,
            'aligned_count': 0,
            'total_checked': 0,
            'details': 'No higher timeframe data available',
            'timeframes': {},
        }

    # Calculate score bonus/penalty
    alignment_ratio = aligned_count / total_checked

    if alignment_ratio >= 1.0:
        # All higher timeframes agree
        score = 15
        detail_msg = "All HTFs aligned"
    elif alignment_ratio >= 0.5:
        # Partial agreement
        score = 5
        detail_msg = "Partial HTF alignment"
    elif alignment_ratio == 0:
        # No agreement (all against)
        # Check if they're NEUTRAL (not opposed)
        opposed = sum(
            1 for tf, trend in timeframe_results.items()
            if trend['direction'] != 'NEUTRAL' and trend['direction'] != expected_trend
        )
        if opposed > 0:
            score = -15
            detail_msg = "HTFs oppose signal"
        else:
            score = 0
            detail_msg = "HTFs neutral"
    else:
        score = 0
        detail_msg = "Mixed HTF signals"

    # Build details
    tf_details = []
    for tf, trend in timeframe_results.items():
        emoji = '+' if trend['direction'] == expected_trend else '-' if trend['direction'] != 'NEUTRAL' else '~'
        tf_details.append(f"{tf}={trend['direction'][0]}({trend['strength']})")

    details = f"{detail_msg}: {', '.join(tf_details)}"

    return {
        'score': score,
        'aligned_count': aligned_count,
        'total_checked': total_checked,
        'details': details,
        'timeframes': {tf: trend['direction'] for tf, trend in timeframe_results.items()},
    }
