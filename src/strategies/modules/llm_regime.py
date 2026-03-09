"""LLM Market Regime Classifier module.

Classifies the market into one of 6 Wyckoff-inspired regimes:
  accumulation, distribution, markup, markdown, capitulation, euphoria

Uses multiple data points: price action, volume, funding, sentiment.
Calls LLM periodically (every 15-30min, not every candle) and caches.
Adjusts strategy weights based on the classified regime.
"""

import json
import time
from termcolor import cprint


# Regime cache: {symbol: (timestamp, regime_result)}
_regime_cache = {}
_REGIME_TTL_SECONDS = 900  # 15 minutes


SYSTEM_PROMPT = (
    "You are a professional crypto market regime classifier. "
    "You classify the current market phase using Wyckoff methodology. "
    "You always respond with valid JSON."
)

USER_PROMPT_TEMPLATE = """Classify the current market regime for {symbol}.

## Price Action (1h candles)
- Current Price: ${price:,.2f}
- 24h Change: {change_24h:+.2f}%
- RSI(14): {rsi:.1f}
- ADX(14): {adx:.1f}
- EMA9: ${ema_9:,.2f} | EMA21: ${ema_21:,.2f} | EMA50: ${ema_50:,.2f}
- Bollinger %B: {bb_pct:.2f}

## Volume & Volatility
- Volume Ratio (vs 20MA): {volume_ratio:.2f}x
- ATR: {atr:.4f}
- MACD: {macd:.4f} (signal: {macd_signal:.4f})

## On-Chain / Derivatives
- Funding Rate Z-Score: {funding_zscore:+.2f}
{extra_context}

Classify into exactly ONE regime:
1. ACCUMULATION - Low volatility, range-bound, smart money buying (ADX low, volume declining)
2. DISTRIBUTION - Smart money selling into strength (high volume on up, declining on down)
3. MARKUP - Strong uptrend, increasing volume (ADX rising, EMAs aligned bullish)
4. MARKDOWN - Strong downtrend, panic selling (ADX rising, EMAs aligned bearish)
5. CAPITULATION - Extreme selling, panic, potential bottom (very low RSI, volume spike)
6. EUPHORIA - Extreme buying, FOMO, potential top (very high RSI, extreme funding)

Respond ONLY with valid JSON:
{{
  "regime": "ACCUMULATION" | "DISTRIBUTION" | "MARKUP" | "MARKDOWN" | "CAPITULATION" | "EUPHORIA",
  "confidence": 0-100,
  "reasoning": "one-sentence explanation",
  "bias": "LONG" | "SHORT" | "NEUTRAL"
}}
"""

# Weight adjustments per regime
# These multiply the base weights to favor appropriate strategies
REGIME_WEIGHT_ADJUSTMENTS = {
    'ACCUMULATION': {
        'mean_reversion': 1.4,
        'momentum_breakout': 0.6,
        'ema_trend': 0.7,
        'funding_contrarian': 1.2,
        'rsi_divergence': 1.3,
        'sniper_lite': 0.8,
        'ramf_lite': 1.2,
        'oi_delta': 1.0,
        'sentiment': 0.8,
        'squeeze_detector': 1.5,
        'order_imbalance': 1.2,
        'cvd': 1.0, 'vwap_deviation': 1.3, 'market_memory': 1.2,
        'stablecoin_flow': 1.0, 'options_sentiment': 1.0, 'liquidation_cascade': 1.0,
    },
    'DISTRIBUTION': {
        'mean_reversion': 1.2,
        'momentum_breakout': 0.7,
        'ema_trend': 0.8,
        'funding_contrarian': 1.3,
        'rsi_divergence': 1.3,
        'sniper_lite': 1.2,
        'ramf_lite': 1.0,
        'oi_delta': 1.3,
        'sentiment': 1.2,
        'squeeze_detector': 1.0,
        'order_imbalance': 1.3,
        'cvd': 1.3, 'vwap_deviation': 1.0, 'market_memory': 1.0,
        'stablecoin_flow': 1.2, 'options_sentiment': 1.3, 'liquidation_cascade': 1.2,
    },
    'MARKUP': {
        'mean_reversion': 0.5,
        'momentum_breakout': 1.5,
        'ema_trend': 1.5,
        'funding_contrarian': 0.7,
        'rsi_divergence': 0.8,
        'sniper_lite': 0.8,
        'ramf_lite': 0.7,
        'oi_delta': 1.2,
        'sentiment': 1.0,
        'squeeze_detector': 0.8,
        'order_imbalance': 1.0,
        'cvd': 1.3, 'vwap_deviation': 0.8, 'market_memory': 0.8,
        'stablecoin_flow': 1.0, 'options_sentiment': 0.8, 'liquidation_cascade': 0.7,
    },
    'MARKDOWN': {
        'mean_reversion': 0.5,
        'momentum_breakout': 1.4,
        'ema_trend': 1.4,
        'funding_contrarian': 0.8,
        'rsi_divergence': 0.8,
        'sniper_lite': 1.0,
        'ramf_lite': 0.7,
        'oi_delta': 1.3,
        'sentiment': 1.0,
        'squeeze_detector': 0.8,
        'order_imbalance': 1.0,
        'cvd': 1.3, 'vwap_deviation': 0.8, 'market_memory': 0.8,
        'stablecoin_flow': 1.0, 'options_sentiment': 0.8, 'liquidation_cascade': 1.3,
    },
    'CAPITULATION': {
        'mean_reversion': 1.5,
        'momentum_breakout': 0.5,
        'ema_trend': 0.5,
        'funding_contrarian': 1.5,
        'rsi_divergence': 1.5,
        'sniper_lite': 1.5,
        'ramf_lite': 1.3,
        'oi_delta': 1.0,
        'sentiment': 1.3,
        'squeeze_detector': 1.2,
        'order_imbalance': 1.0,
        'cvd': 1.5, 'vwap_deviation': 1.0, 'market_memory': 0.7,
        'stablecoin_flow': 1.3, 'options_sentiment': 1.2, 'liquidation_cascade': 1.5,
    },
    'EUPHORIA': {
        'mean_reversion': 1.3,
        'momentum_breakout': 0.5,
        'ema_trend': 0.6,
        'funding_contrarian': 1.5,
        'rsi_divergence': 1.5,
        'sniper_lite': 1.5,
        'ramf_lite': 1.3,
        'oi_delta': 1.0,
        'sentiment': 1.5,
        'squeeze_detector': 1.0,
        'order_imbalance': 1.0,
        'cvd': 1.5, 'vwap_deviation': 1.0, 'market_memory': 0.7,
        'stablecoin_flow': 1.3, 'options_sentiment': 1.5, 'liquidation_cascade': 1.5,
    },
}


def _parse_regime_response(response_text):
    """Extract JSON regime classification from LLM response."""
    text = response_text.strip()
    if '```json' in text:
        text = text.split('```json')[1].split('```')[0].strip()
    elif '```' in text:
        text = text.split('```')[1].split('```')[0].strip()

    result = json.loads(text)

    regime = result.get('regime', 'ACCUMULATION').upper()
    valid_regimes = {'ACCUMULATION', 'DISTRIBUTION', 'MARKUP', 'MARKDOWN', 'CAPITULATION', 'EUPHORIA'}
    if regime not in valid_regimes:
        regime = 'ACCUMULATION'

    bias = result.get('bias', 'NEUTRAL').upper()
    if bias not in ('LONG', 'SHORT', 'NEUTRAL'):
        bias = 'NEUTRAL'

    return {
        'regime': regime,
        'confidence': max(0, min(100, int(result.get('confidence', 50)))),
        'reasoning': result.get('reasoning', ''),
        'bias': bias,
    }


def classify_regime(
    symbol: str,
    indicators: dict,
    funding_zscore: float = 0.0,
    extra_context: str = "",
    model=None,
    bypass: bool = False,
) -> dict:
    """Classify market regime for the given symbol.

    Args:
        symbol: Trading symbol (e.g. 'BTC')
        indicators: Dict of technical indicators from _compute_indicators
        funding_zscore: Funding rate Z-score for this symbol
        extra_context: Additional context string (sentiment, OI, etc.)
        model: LLM model instance. If None, uses rule-based fallback.
        bypass: If True, use rule-based classification only (no LLM).

    Returns:
        dict with 'regime', 'confidence', 'reasoning', 'bias'
    """
    # Check cache first
    now = time.time()
    if symbol in _regime_cache:
        cached_ts, cached_result = _regime_cache[symbol]
        if now - cached_ts < _REGIME_TTL_SECONDS:
            return cached_result

    # Rule-based fallback (used in bypass mode or when LLM unavailable)
    if bypass or model is None:
        result = _rule_based_regime(indicators, funding_zscore)
        _regime_cache[symbol] = (now, result)
        return result

    # Calculate 24h change from EMA
    price = indicators.get('close', 0)
    ema_21 = indicators.get('ema_21', price)
    change_24h = ((price - ema_21) / ema_21 * 100) if ema_21 > 0 else 0

    user_content = USER_PROMPT_TEMPLATE.format(
        symbol=symbol,
        price=price,
        change_24h=change_24h,
        rsi=indicators.get('rsi', 50),
        adx=indicators.get('adx', 20),
        ema_9=indicators.get('ema_9', price),
        ema_21=ema_21,
        ema_50=indicators.get('ema_50', price),
        bb_pct=indicators.get('bb_pct', 0.5),
        volume_ratio=indicators.get('volume_ratio', 1.0),
        atr=indicators.get('atr', 0),
        macd=indicators.get('macd', 0),
        macd_signal=indicators.get('macd_signal', 0),
        funding_zscore=funding_zscore,
        extra_context=extra_context or "No additional data",
    )

    try:
        start_time = time.time()
        response = model.generate_response(
            system_prompt=SYSTEM_PROMPT,
            user_content=user_content,
            temperature=0.2,
            max_tokens=256,
        )
        latency_ms = int((time.time() - start_time) * 1000)

        # Log stats
        try:
            from src.models.model_factory import ModelFactory
            ModelFactory.log_call(getattr(model, 'model_type', 'unknown'), True, latency_ms)
        except Exception:
            pass

        if response is None:
            result = _rule_based_regime(indicators, funding_zscore)
            _regime_cache[symbol] = (now, result)
            return result

        response_text = response.content if hasattr(response, 'content') else str(response)
        result = _parse_regime_response(response_text)

        _regime_cache[symbol] = (now, result)

        cprint(f"  [LLM Regime] {symbol}: {result['regime']} "
               f"(conf={result['confidence']}%, bias={result['bias']}, {latency_ms}ms)", "cyan")

        return result

    except Exception as e:
        cprint(f"  [LLM Regime] Error for {symbol}: {e}", "yellow")
        try:
            from src.models.model_factory import ModelFactory
            ModelFactory.log_call(getattr(model, 'model_type', 'unknown'), False)
        except Exception:
            pass
        result = _rule_based_regime(indicators, funding_zscore)
        _regime_cache[symbol] = (now, result)
        return result


def _rule_based_regime(indicators: dict, funding_zscore: float = 0.0) -> dict:
    """Fallback rule-based regime classification (no LLM needed)."""
    rsi = indicators.get('rsi', 50)
    adx = indicators.get('adx', 20)
    volume_ratio = indicators.get('volume_ratio', 1.0)
    close = indicators.get('close', 0)
    ema_50 = indicators.get('ema_50', close)
    bb_pct = indicators.get('bb_pct', 0.5)

    bullish_trend = close > ema_50
    strong_trend = adx > 30

    # Capitulation: extreme oversold + high volume
    if rsi < 25 and volume_ratio > 2.0:
        return {'regime': 'CAPITULATION', 'confidence': 75, 'reasoning': 'Extreme RSI + volume spike', 'bias': 'LONG'}

    # Euphoria: extreme overbought + extreme funding
    if rsi > 75 and (funding_zscore > 2.0 or volume_ratio > 2.5):
        return {'regime': 'EUPHORIA', 'confidence': 70, 'reasoning': 'Extreme RSI + funding/volume', 'bias': 'SHORT'}

    # Markup: strong bullish trend
    if bullish_trend and strong_trend and rsi > 55:
        return {'regime': 'MARKUP', 'confidence': 65, 'reasoning': 'Strong uptrend with momentum', 'bias': 'LONG'}

    # Markdown: strong bearish trend
    if not bullish_trend and strong_trend and rsi < 45:
        return {'regime': 'MARKDOWN', 'confidence': 65, 'reasoning': 'Strong downtrend', 'bias': 'SHORT'}

    # Distribution: price high but losing momentum
    if bullish_trend and not strong_trend and rsi > 60 and volume_ratio < 0.8:
        return {'regime': 'DISTRIBUTION', 'confidence': 55, 'reasoning': 'High price, declining volume', 'bias': 'SHORT'}

    # Default: Accumulation (range-bound, low volatility)
    return {'regime': 'ACCUMULATION', 'confidence': 50, 'reasoning': 'Range-bound, low directional bias', 'bias': 'NEUTRAL'}


def adjust_weights_for_regime(base_weights: dict, regime: str) -> dict:
    """Apply regime-based weight adjustments to module weights.

    Args:
        base_weights: Original module weight dict (must sum to ~1.0)
        regime: Regime string from classify_regime

    Returns:
        Adjusted weights dict (re-normalized to sum to 1.0)
    """
    adjustments = REGIME_WEIGHT_ADJUSTMENTS.get(regime.upper(), {})
    if not adjustments:
        return base_weights

    adjusted = {}
    for module, weight in base_weights.items():
        factor = adjustments.get(module, 1.0)
        adjusted[module] = weight * factor

    # Re-normalize to sum to 1.0
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}

    return adjusted
