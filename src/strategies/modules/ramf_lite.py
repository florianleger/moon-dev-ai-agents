"""RAMF Lite scoring module (volatility regime + momentum exhaustion)."""

import pandas as pd


def score_ramf_lite(df: pd.DataFrame, indicators: dict, config: dict = None) -> dict:
    """Volatility regime + momentum exhaustion (no dead zone).

    Args:
        df: OHLCV DataFrame with 'atr' column computed.
        indicators: Dict of last-row indicator values.
        config: Optional config overrides.

    Returns:
        dict with 'score' (0-100), 'direction', and 'reason'.
    """
    long_score = 0
    short_score = 0

    if 'atr' not in df.columns or len(df) < 50:
        return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'Insufficient data'}

    # Volatility regime (NO dead zone - always classify)
    atr_values = df['atr'].dropna().tail(50)
    if len(atr_values) < 20:
        return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'Not enough ATR data'}

    current_atr = indicators['atr']
    atr_percentile = (atr_values < current_atr).sum() / len(atr_values) * 100

    is_high_vol = atr_percentile >= 50
    regime = 'HIGH' if is_high_vol else 'LOW'

    if is_high_vol:
        # High vol: mean reversion / exhaustion
        vwap_dist = abs(indicators['close'] - indicators['vwap']) / current_atr if current_atr > 0 else 0

        if vwap_dist >= 1.0:
            if indicators['close'] < indicators['vwap']:
                long_score += 40  # Extended below VWAP
            else:
                short_score += 40  # Extended above VWAP

        # Consecutive bars check (relaxed: 2 instead of 3)
        recent = df.tail(3)
        up_bars = (recent['close'] > recent['open']).sum()
        down_bars = (recent['close'] < recent['open']).sum()

        if down_bars >= 2 and long_score > 0:
            long_score += 30
        if up_bars >= 2 and short_score > 0:
            short_score += 30

        # RSI confirmation
        if indicators['rsi'] < 35:
            long_score += 30
        elif indicators['rsi'] < 45:
            long_score += 15
        if indicators['rsi'] > 65:
            short_score += 30
        elif indicators['rsi'] > 55:
            short_score += 15
    else:
        # Low vol: trend following
        if indicators['ema_9'] > indicators['ema_21']:
            long_score += 35
            if indicators['macd_diff'] > 0:
                long_score += 25
            if indicators['rsi'] > 50:
                long_score += 20
        elif indicators['ema_9'] < indicators['ema_21']:
            short_score += 35
            if indicators['macd_diff'] < 0:
                short_score += 25
            if indicators['rsi'] < 50:
                short_score += 20

    best = max(long_score, short_score)
    direction = 'BUY' if long_score > short_score else 'SELL' if short_score > long_score else 'NEUTRAL'
    return {'score': min(100, best), 'direction': direction,
            'reason': f'Regime={regime} ATR%={atr_percentile:.0f} VWAP_dist={abs(indicators["close"] - indicators["vwap"]):.2f}'}
