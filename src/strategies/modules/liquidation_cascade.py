"""Liquidation Cascade Detector scoring module.

Detects cascading liquidations from Binance Futures WebSocket data.
Uses as contrarian signal: cascade liquidations = forced exits = potential reversal.
Also provides a 'suppress' flag to warn strategy that current price action is
driven by forced liquidations, not organic order flow.
"""

# Symbol mapping from strategy symbols to Binance futures symbols
SYMBOL_MAP = {
    'BTC': 'BTCUSDT',
    'ETH': 'ETHUSDT',
    'SOL': 'SOLUSDT',
    'DOGE': 'DOGEUSDT',
    'XRP': 'XRPUSDT',
    'ADA': 'ADAUSDT',
    'AVAX': 'AVAXUSDT',
    'LINK': 'LINKUSDT',
    'DOT': 'DOTUSDT',
    'MATIC': 'MATICUSDT',
    'ARB': 'ARBUSDT',
    'OP': 'OPUSDT',
    'SUI': 'SUIUSDT',
    'APT': 'APTUSDT',
    'NEAR': 'NEARUSDT',
    'FIL': 'FILUSDT',
    'ATOM': 'ATOMUSDT',
    'UNI': 'UNIUSDT',
    'LTC': 'LTCUSDT',
    'BCH': 'BCHUSDT',
    'PEPE': 'PEPEUSDT',
    'kPEPE': '1000PEPEUSDT',
    'ENA': 'ENAUSDT',
    'AAVE': 'AAVEUSDT',
    'TAO': 'TAOUSDT',
    'WIF': 'WIFUSDT',
    'BONK': 'BONKUSDT',
    'INJ': 'INJUSDT',
    'TIA': 'TIAUSDT',
    'SEI': 'SEIUSDT',
    'JUP': 'JUPUSDT',
    'RENDER': 'RENDERUSDT',
    'FET': 'FETUSDT',
    'WLD': 'WLDUSDT',
}


def score_liquidation_cascade(symbol: str, indicators: dict, config: dict = None) -> dict:
    """Detect liquidation cascades and return a contrarian signal.

    If large cascading LONG liquidations detected -> BUY (bottom forming).
    If large cascading SHORT liquidations detected -> SELL (top forming).
    No cascade -> score=0 NEUTRAL.

    Args:
        symbol: Token symbol (e.g. 'BTC').
        indicators: Dict of last-row indicator values.
        config: Optional config overrides.

    Returns:
        dict with 'score' (0-100), 'direction', 'reason', and 'suppress_breakout'.
        None if data provider is unavailable.
    """
    _neutral = {'score': 0, 'direction': 'NEUTRAL', 'reason': 'API unavailable', 'data_quality': 0.0, 'suppress_breakout': False}

    try:
        from src.data_providers.binance_futures import get_liquidation_stream
    except Exception:
        return _neutral

    cfg = config or {}
    lookback_minutes = cfg.get('cascade_lookback_minutes', 5)
    intensity_threshold = cfg.get('cascade_intensity_threshold', 2.0)
    suppress_threshold = cfg.get('cascade_suppress_threshold', 3.0)
    # Typical 5-minute liquidation volume in USD (baseline for normalization)
    typical_5min_volume = cfg.get('cascade_typical_volume', 500_000)

    # Map strategy symbol to Binance symbol
    binance_symbol = SYMBOL_MAP.get(symbol.upper(), f'{symbol.upper()}USDT')

    try:
        stream = get_liquidation_stream()
    except Exception:
        return _neutral

    # Get recent liquidations from the stream buffer
    try:
        df = stream.get_recent_liquidations(minutes=lookback_minutes)
    except Exception:
        return _neutral

    if df is None or df.empty:
        return {
            'score': 0,
            'direction': 'NEUTRAL',
            'reason': 'No liquidation data available',
            'suppress_breakout': False,
        }

    # Filter for this symbol
    symbol_df = df[df['symbol'] == binance_symbol]

    if symbol_df.empty:
        return {
            'score': 0,
            'direction': 'NEUTRAL',
            'reason': f'No liquidation data for {symbol}',
            'suppress_breakout': False,
        }

    # SELL side = long positions liquidated (forced to sell)
    # BUY side = short positions liquidated (forced to buy)
    long_liqs = symbol_df[symbol_df['side'] == 'SELL']
    short_liqs = symbol_df[symbol_df['side'] == 'BUY']

    liq_count_buy = len(long_liqs)
    liq_count_sell = len(short_liqs)
    liq_volume_buy = float(long_liqs['usd_value'].sum()) if not long_liqs.empty else 0.0
    liq_volume_sell = float(short_liqs['usd_value'].sum()) if not short_liqs.empty else 0.0

    total_volume = liq_volume_buy + liq_volume_sell
    total_count = liq_count_buy + liq_count_sell

    # Liquidation volume as ratio of total market volume
    if total_volume > 0:
        cascade_ratio = total_volume / (typical_5min_volume + 1)
    else:
        cascade_ratio = 0

    # How extreme vs normal activity
    cascade_intensity = total_volume / (typical_5min_volume + 1)

    # No cascade detected
    if cascade_intensity < intensity_threshold:
        return {
            'score': 0,
            'direction': 'NEUTRAL',
            'reason': f'Normal liq activity ({total_count} liqs, ${total_volume:,.0f}, intensity={cascade_intensity:.1f}x)',
            'suppress_breakout': False,
        }

    # Cascade detected - generate contrarian signal
    score = min(100, int(cascade_intensity * 15))

    # Determine dominant liquidation side and go contrarian
    if liq_volume_buy > liq_volume_sell:
        # Longs getting liquidated -> contrarian BUY (bottom forming)
        direction = 'BUY'
        dominant = 'LONG'
        dominant_pct = liq_volume_buy / total_volume * 100
    else:
        # Shorts getting liquidated -> contrarian SELL (top forming)
        direction = 'SELL'
        dominant = 'SHORT'
        dominant_pct = liq_volume_sell / total_volume * 100

    suppress_breakout = cascade_intensity >= suppress_threshold

    reason = (
        f'Cascade: {total_count} liqs, ${total_volume:,.0f} '
        f'({dominant} {dominant_pct:.0f}%, intensity={cascade_intensity:.1f}x)'
    )
    if suppress_breakout:
        reason += ' [SUPPRESS: forced liquidation move]'

    return {
        'score': score,
        'direction': direction,
        'reason': reason,
        'suppress_breakout': suppress_breakout,
    }
