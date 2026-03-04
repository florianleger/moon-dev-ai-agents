"""RSI Divergence scoring module (pivot-based price vs RSI divergence)."""

import numpy as np


def score_rsi_divergence(indicators: dict, config: dict = None) -> dict:
    """RSI divergence detection using swing pivot points.

    Args:
        indicators: Dict of last-row indicator values. Must include '_df' key with full DataFrame.
        config: Optional config overrides.

    Returns:
        dict with 'score' (0-100), 'direction', and 'reason'.
    """
    lookback = 30
    df = indicators.get('_df')
    if df is None or len(df) < lookback:
        return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'Insufficient data'}

    prices = df['close'].values[-lookback:]
    rsis = df['rsi'].values[-lookback:]

    # Filter NaN from RSI
    valid = ~np.isnan(rsis)
    if valid.sum() < lookback // 2:
        return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'Not enough RSI data'}

    def find_swing_lows(arr, order=3):
        lows = []
        for i in range(order, len(arr) - order):
            if all(arr[i] <= arr[i-j] for j in range(1, order+1)) and all(arr[i] <= arr[i+j] for j in range(1, order+1)):
                lows.append(i)
        return lows

    def find_swing_highs(arr, order=3):
        highs = []
        for i in range(order, len(arr) - order):
            if all(arr[i] >= arr[i-j] for j in range(1, order+1)) and all(arr[i] >= arr[i+j] for j in range(1, order+1)):
                highs.append(i)
        return highs

    long_score = 0
    short_score = 0

    # Bullish divergence: price lower lows, RSI higher lows
    price_lows = find_swing_lows(prices)
    if len(price_lows) >= 2:
        i, j = price_lows[-2], price_lows[-1]
        if prices[j] < prices[i] and rsis[j] > rsis[i]:
            long_score += 60
            if indicators['rsi'] < indicators['rsi_oversold'] + 10:
                long_score += 25

    # Bearish divergence: price higher highs, RSI lower highs
    price_highs = find_swing_highs(prices)
    if len(price_highs) >= 2:
        i, j = price_highs[-2], price_highs[-1]
        if prices[j] > prices[i] and rsis[j] < rsis[i]:
            short_score += 60
            if indicators['rsi'] > indicators['rsi_overbought'] - 10:
                short_score += 25

    best = max(long_score, short_score)
    direction = 'BUY' if long_score > short_score else ('SELL' if short_score > long_score else 'NEUTRAL')
    return {'score': min(100, best), 'direction': direction,
            'reason': f'RSI divergence (pivot-based, lookback={lookback})'}
