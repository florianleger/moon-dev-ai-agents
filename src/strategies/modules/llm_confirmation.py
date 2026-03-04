"""LLM Trade Confirmation module.

Takes the aggregated signal + all module scores + market context,
calls a fast LLM (Groq or Haiku) to confirm/reject/adjust the trade.
Returns adjusted score and optional SL/TP modifications.

Features:
- Chain-of-Thought structured prompts
- Short TTL cache (~5min) for identical market states
- BYPASS mode for backtesting (no LLM calls)
"""

import json
import time
import hashlib
from termcolor import cprint


# Cache: {hash -> (timestamp, result)}
_confirmation_cache = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes


SYSTEM_PROMPT = (
    "You are a professional crypto trading risk filter. "
    "You analyze aggregated trading signals and decide whether to CONFIRM, REJECT, or ADJUST a trade. "
    "You always respond with valid JSON."
)

USER_PROMPT_TEMPLATE = """Evaluate this proposed trade:

## Signal
- Symbol: {symbol}
- Direction: {direction}
- Aggregated Score: {score}/100 (threshold: {threshold})
- Signal Strength: {strength:.0%}

## Module Scores (who voted for this direction)
{module_scores}

## Market Context
- Price: ${price:,.2f}
- RSI: {rsi:.1f}
- ADX: {adx:.1f}
- ATR: {atr:.4f}
- Volume Ratio: {volume_ratio:.2f}x
- Proposed SL: {sl_pct:.2f}%
- Proposed TP: {tp_pct:.2f}%

## Trade Memory
{memory_context}

## Recent Lessons
{recent_lessons}

Think step by step:
1. SIGNAL QUALITY: Are enough independent modules agreeing? Any red flags?
2. MARKET CONDITIONS: Does the market regime support this trade type?
3. RISK/REWARD: Is the SL/TP appropriate given current volatility?
4. HISTORICAL: Do past trades in similar conditions suggest caution?
5. VERDICT: Should we take this trade?

Respond ONLY with valid JSON:
{{
  "decision": "CONFIRM" | "REJECT" | "ADJUST",
  "confidence": 0-100,
  "reasoning": "one-sentence summary",
  "adjusted_score": null | 0-100,
  "sl_adjustment": null | float,
  "tp_adjustment": null | float
}}

Rules:
- CONFIRM: Take the trade as-is
- REJECT: Skip this trade (set adjusted_score to 0)
- ADJUST: Modify score or SL/TP (provide adjusted values)
- If in doubt, REJECT (protecting capital is priority)
"""


def _build_cache_key(symbol, direction, score, rsi, adx):
    """Build a cache key from the main signal parameters."""
    # Round values to reduce cache misses from tiny fluctuations
    key_str = f"{symbol}|{direction}|{round(score, 0)}|{round(rsi, 0)}|{round(adx, 0)}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _clean_cache():
    """Remove expired cache entries."""
    now = time.time()
    expired = [k for k, (ts, _) in _confirmation_cache.items() if now - ts > _CACHE_TTL_SECONDS]
    for k in expired:
        del _confirmation_cache[k]


def _parse_llm_response(response_text):
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = response_text.strip()
    if '```json' in text:
        text = text.split('```json')[1].split('```')[0].strip()
    elif '```' in text:
        text = text.split('```')[1].split('```')[0].strip()

    result = json.loads(text)

    # Validate required fields
    decision = result.get('decision', 'REJECT').upper()
    if decision not in ('CONFIRM', 'REJECT', 'ADJUST'):
        decision = 'REJECT'

    return {
        'decision': decision,
        'confidence': max(0, min(100, int(result.get('confidence', 50)))),
        'reasoning': result.get('reasoning', ''),
        'adjusted_score': result.get('adjusted_score'),
        'sl_adjustment': result.get('sl_adjustment'),
        'tp_adjustment': result.get('tp_adjustment'),
    }


def llm_confirm_trade(
    symbol: str,
    direction: str,
    aggregated: dict,
    indicators: dict,
    metadata: dict,
    trade_memory=None,
    model=None,
    bypass: bool = False,
) -> dict:
    """Call LLM to confirm/reject/adjust a trade signal.

    Args:
        symbol: Trading symbol (e.g. 'BTC')
        direction: 'BUY' or 'SELL'
        aggregated: Result from _aggregate_scores
        indicators: Dict of technical indicators
        metadata: Signal metadata (score, threshold, sl_pct, tp_pct, etc.)
        trade_memory: TradeMemory instance (optional)
        model: LLM model instance (BaseModel). If None, returns CONFIRM by default.
        bypass: If True, skip LLM call and auto-confirm (for backtesting).

    Returns:
        dict with:
            'decision': 'CONFIRM' | 'REJECT' | 'ADJUST'
            'confidence': 0-100
            'reasoning': str
            'adjusted_score': float or None
            'sl_adjustment': float or None (new SL %)
            'tp_adjustment': float or None (new TP %)
    """
    # BYPASS mode: auto-confirm without LLM call
    if bypass:
        return {
            'decision': 'CONFIRM',
            'confidence': 100,
            'reasoning': 'Bypass mode (backtesting)',
            'adjusted_score': None,
            'sl_adjustment': None,
            'tp_adjustment': None,
        }

    # No model available: auto-confirm (graceful degradation)
    if model is None:
        return {
            'decision': 'CONFIRM',
            'confidence': 75,
            'reasoning': 'No LLM available, auto-confirmed',
            'adjusted_score': None,
            'sl_adjustment': None,
            'tp_adjustment': None,
        }

    score = metadata.get('score', aggregated.get('score', 0))
    rsi = indicators.get('rsi', 50)
    adx = indicators.get('adx', 20)

    # Check cache
    _clean_cache()
    cache_key = _build_cache_key(symbol, direction, score, rsi, adx)
    if cache_key in _confirmation_cache:
        cached_ts, cached_result = _confirmation_cache[cache_key]
        if time.time() - cached_ts < _CACHE_TTL_SECONDS:
            cprint(f"  [LLM Confirm] Cache hit for {symbol} {direction}", "cyan")
            return cached_result

    # Build prompt
    module_scores = aggregated.get('module_scores', metadata.get('module_scores', {}))
    module_lines = "\n".join(f"- {name}: {score_val}" for name, score_val in module_scores.items())
    if not module_lines:
        module_lines = "No module data available"

    memory_context = ""
    recent_lessons = ""
    if trade_memory:
        memory_context = trade_memory.build_context_prompt(symbol, direction) or "No historical data"
        # Get recent lessons
        try:
            mistakes = trade_memory.get_common_mistakes(days=14, min_occurrences=2)
            if mistakes:
                recent_lessons = "\n".join(
                    f"- {m['direction']} in {m['market_regime']}: {m['count']} losses, avg ${m['avg_loss']:.2f}"
                    for m in mistakes[:3]
                )
        except Exception:
            pass
    if not memory_context:
        memory_context = "No historical data"
    if not recent_lessons:
        recent_lessons = "No lessons yet"

    user_content = USER_PROMPT_TEMPLATE.format(
        symbol=symbol,
        direction=direction,
        score=score,
        threshold=metadata.get('threshold', 45),
        strength=metadata.get('signal_strength', 0.7),
        module_scores=module_lines,
        price=indicators.get('close', 0),
        rsi=rsi,
        adx=adx,
        atr=indicators.get('atr', 0),
        volume_ratio=indicators.get('volume_ratio', 1.0),
        sl_pct=metadata.get('stop_loss_pct', 1.5),
        tp_pct=metadata.get('take_profit_pct', 2.5),
        memory_context=memory_context,
        recent_lessons=recent_lessons,
    )

    # Call LLM with timeout protection
    try:
        start_time = time.time()
        response = model.generate_response(
            system_prompt=SYSTEM_PROMPT,
            user_content=user_content,
            temperature=0.2,
            max_tokens=512,
        )
        latency_ms = int((time.time() - start_time) * 1000)

        # Log call stats
        try:
            from src.models.model_factory import ModelFactory
            ModelFactory.log_call(getattr(model, 'model_type', 'unknown'), True, latency_ms)
        except Exception:
            pass

        if response is None:
            cprint(f"  [LLM Confirm] No response for {symbol}", "yellow")
            return {
                'decision': 'CONFIRM',
                'confidence': 60,
                'reasoning': 'LLM returned no response, auto-confirmed',
                'adjusted_score': None,
                'sl_adjustment': None,
                'tp_adjustment': None,
            }

        response_text = response.content if hasattr(response, 'content') else str(response)
        result = _parse_llm_response(response_text)

        # Cache the result
        _confirmation_cache[cache_key] = (time.time(), result)

        color = 'green' if result['decision'] == 'CONFIRM' else 'red' if result['decision'] == 'REJECT' else 'yellow'
        cprint(f"  [LLM Confirm] {symbol} {direction}: {result['decision']} "
               f"(conf={result['confidence']}%, {latency_ms}ms) - {result['reasoning']}", color)

        return result

    except json.JSONDecodeError as e:
        cprint(f"  [LLM Confirm] JSON parse error for {symbol}: {e}", "yellow")
        return {
            'decision': 'CONFIRM',
            'confidence': 50,
            'reasoning': f'LLM response unparseable: {e}',
            'adjusted_score': None,
            'sl_adjustment': None,
            'tp_adjustment': None,
        }
    except Exception as e:
        cprint(f"  [LLM Confirm] Error for {symbol}: {e}", "yellow")
        # Log failure
        try:
            from src.models.model_factory import ModelFactory
            ModelFactory.log_call(getattr(model, 'model_type', 'unknown'), False)
        except Exception:
            pass
        return {
            'decision': 'CONFIRM',
            'confidence': 50,
            'reasoning': f'LLM error: {e}, auto-confirmed',
            'adjusted_score': None,
            'sl_adjustment': None,
            'tp_adjustment': None,
        }
