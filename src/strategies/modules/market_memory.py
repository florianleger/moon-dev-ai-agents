"""Market Memory scoring module (Hurst Exponent + ACF Decay)."""

import numpy as np
import pandas as pd


def rolling_hurst(prices: np.ndarray, window: int = 100) -> float:
    """Compute Hurst exponent via R/S analysis on a rolling window.

    Args:
        prices: Array of close prices.
        window: Lookback window size.

    Returns:
        Hurst exponent (0-1). >0.6 trending, <0.4 mean-reverting, ~0.5 random walk.
    """
    if len(prices) < window:
        return 0.5  # default to random walk

    series = prices[-window:]
    series = series[series > 0]
    if len(series) < 10:
        return 0.5
    returns = np.diff(np.log(series))
    if len(returns) < 10:
        return 0.5

    # R/S analysis over multiple sub-periods
    max_k = min(len(returns) // 2, 50)
    if max_k < 4:
        return 0.5

    rs_list = []
    ns = []
    for n in range(4, max_k + 1):
        num_segments = len(returns) // n
        if num_segments == 0:
            continue
        rs_vals = []
        for i in range(num_segments):
            segment = returns[i * n:(i + 1) * n]
            mean_seg = segment.mean()
            deviate = np.cumsum(segment - mean_seg)
            r = deviate.max() - deviate.min()
            s = segment.std(ddof=1)
            if s > 0:
                rs_vals.append(r / s)
        if rs_vals:
            rs_list.append(np.mean(rs_vals))
            ns.append(n)

    if len(ns) < 3:
        return 0.5

    log_ns = np.log(ns)
    log_rs = np.log(rs_list)
    # Linear regression: log(R/S) = H * log(n) + c
    coeffs = np.polyfit(log_ns, log_rs, 1)
    hurst = float(np.clip(coeffs[0], 0.0, 1.0))
    return hurst


def autocorrelation_decay(returns: np.ndarray, max_lag: int = 20) -> float:
    """Compute average absolute autocorrelation over lags.

    Args:
        returns: Array of log returns.
        max_lag: Maximum lag to compute.

    Returns:
        Mean |ACF| across lags 1..max_lag (higher = more persistent memory).
    """
    if max_lag < 1:
        return 0.0

    if len(returns) < max_lag + 5:
        return 0.0

    mean_r = returns.mean()
    var_r = returns.var()
    if var_r == 0:
        return 0.0

    acf_abs = []
    for lag in range(1, max_lag + 1):
        cov = np.mean((returns[lag:] - mean_r) * (returns[:-lag] - mean_r))
        acf_abs.append(abs(cov / var_r))

    return float(np.mean(acf_abs))


def score_market_memory(df: pd.DataFrame, indicators: dict, config: dict = None) -> dict:
    """Score based on Hurst exponent and autocorrelation decay.

    Args:
        df: OHLCV DataFrame with indicators computed.
        indicators: Dict of last-row indicator values.
        config: Optional config overrides.

    Returns:
        dict with 'score' (0-100), 'direction', and 'reason'.
    """
    if len(df) < 50:
        return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'Insufficient data for Hurst'}

    closes = df['close'].values
    hurst = rolling_hurst(closes, window=min(100, len(closes)))

    returns = np.diff(np.log(closes))
    acf = autocorrelation_decay(returns, max_lag=min(20, len(returns) // 3))

    long_score = 0
    short_score = 0

    rsi = indicators.get('rsi', 50)
    # Proxy momentum from recent price change
    if len(closes) >= 10:
        recent_momentum = (closes[-1] - closes[-10]) / closes[-10]
    else:
        recent_momentum = 0.0

    if hurst > 0.6:
        # Trending regime: follow the trend
        trend_strength = int((hurst - 0.5) * 200)  # 0-100 scale
        if recent_momentum > 0:
            long_score += min(60, 20 + trend_strength)
        elif recent_momentum < 0:
            short_score += min(60, 20 + trend_strength)

        # ACF confirmation: high ACF = stronger trend persistence
        if acf > 0.15:
            if recent_momentum > 0:
                long_score += 20
            elif recent_momentum < 0:
                short_score += 20

    elif hurst < 0.4:
        # Mean-reverting regime: fade extremes
        mr_strength = int((0.5 - hurst) * 200)  # 0-100 scale
        rsi_oversold = indicators.get('rsi_oversold', 30)
        rsi_overbought = indicators.get('rsi_overbought', 70)

        if rsi < rsi_oversold:
            long_score += min(60, 25 + mr_strength)
        elif rsi < 40:
            long_score += min(30, mr_strength)

        if rsi > rsi_overbought:
            short_score += min(60, 25 + mr_strength)
        elif rsi > 60:
            short_score += min(30, mr_strength)

    # Hurst ~0.5 = random walk, low conviction either way
    best = max(long_score, short_score)
    direction = 'BUY' if long_score > short_score else ('SELL' if short_score > long_score else 'NEUTRAL')
    return {'score': min(100, best), 'direction': direction,
            'reason': f'Hurst={hurst:.3f} ACF={acf:.3f} momentum={recent_momentum:+.4f}'}
