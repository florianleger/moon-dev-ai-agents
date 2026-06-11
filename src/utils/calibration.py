"""Runtime calibration overrides -- read by strategy, written by CalibrationAgent."""
import json
import os
import time
import threading

_OVERRIDES_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'calibration_overrides.json')
_cache = {}
_cache_ts = 0
_CACHE_TTL = 60  # seconds
_lock = threading.RLock()

# Hard guardrails: absolute min/max and max per-adjustment delta
# SINGLE SOURCE OF TRUTH: _get_effective_threshold (adaptive_hybrid_strategy.py)
# clamps the runtime override with THIS max — any value writable here is applied.
# (Was 65 while runtime clamped at configured*1.10=52.8 -> CalibrationAgent wrote
# values that were never applied, looping no-op.)
GUARDRAILS = {
    'ADAPTIVE_HYBRID_BASE_THRESHOLD': {'min': 35, 'max': 53, 'max_delta_pct': 0.10},
    'ADAPTIVE_HYBRID_VOLUME_FILTER_MIN': {'min': 0.02, 'max': 0.40, 'max_delta_abs': 0.05},
    'ADAPTIVE_HYBRID_4H_TREND_PENALTY': {'min': 0.05, 'max': 0.50, 'max_delta_abs': 0.05},
    # ATR profiles are handled specially inside the agent
}


def get_calibrated_value(param_name: str, default):
    """Return override if exists, else default. Cached 60s."""
    global _cache, _cache_ts
    with _lock:
        now = time.time()
        if now - _cache_ts > _CACHE_TTL:
            _cache = _load_overrides()
            _cache_ts = now
        entry = _cache.get(param_name)
        if entry and 'value' in entry:
            return entry['value']
        return default


def _load_overrides() -> dict:
    try:
        path = os.path.normpath(_OVERRIDES_PATH)
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            return data.get('overrides', {})
    except Exception:
        pass
    return {}


def apply_guardrail(param_name, new_value, previous_value, default_value):
    """Clamp value to guardrails. Returns (clamped_value, was_clamped).

    Order matters: the delta clamps run FIRST and the absolute min/max LAST.
    If min/max ran first, a previous_value outside the bounds (e.g. a stale
    override at 65 with max=53) would let the delta clamp re-push the result
    above the max, breaking the "any value writable here is applied" invariant.
    """
    g = GUARDRAILS.get(param_name)
    if not g:
        return new_value, False
    clamped = new_value
    if 'max_delta_pct' in g and previous_value:
        max_delta = abs(previous_value * g['max_delta_pct'])
        clamped = max(previous_value - max_delta, min(previous_value + max_delta, clamped))
    if 'max_delta_abs' in g and previous_value is not None:
        clamped = max(previous_value - g['max_delta_abs'], min(previous_value + g['max_delta_abs'], clamped))
    if 'min' in g:
        clamped = max(g['min'], clamped)
    if 'max' in g:
        clamped = min(g['max'], clamped)
    return clamped, clamped != new_value
