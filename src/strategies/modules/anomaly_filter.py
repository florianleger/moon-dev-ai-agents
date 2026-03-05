"""Market anomaly detection filter using Isolation Forest."""

import numpy as np
from collections import deque
import threading

try:
    from sklearn.ensemble import IsolationForest
    _sklearn_available = True
except ImportError:
    IsolationForest = None
    _sklearn_available = False

_model = None

_buffer = deque(maxlen=1000)
_is_fitted = False
_obs_since_train = 0
_retrain_every = 200
_lock = threading.Lock()


def _to_vector(indicators: dict) -> np.ndarray:
    """Convert indicator dict to a feature vector for anomaly detection."""
    close = indicators.get('close', 1)
    return np.array([
        indicators.get('rsi', 50) / 100,
        indicators.get('adx', 20) / 100,
        indicators.get('volume_ratio', 1.0),
        indicators.get('bb_pct', 0.5),
        indicators.get('atr', 0) / close * 100 if close > 0 else 0,
    ])


def observe(indicators: dict):
    """Store a market observation and retrain periodically.

    Should be called every cycle to build up the baseline distribution.

    Args:
        indicators: Dict of last-row indicator values.
    """
    global _obs_since_train, _is_fitted, _model
    if not _sklearn_available:
        return
    vec = _to_vector(indicators)
    need_retrain = False
    with _lock:
        _buffer.append(vec)
        _obs_since_train += 1
        if len(_buffer) >= 200 and _obs_since_train >= _retrain_every:
            need_retrain = True
            X = np.array(list(_buffer))
            _obs_since_train = 0
    if need_retrain:
        new_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        new_model.fit(X)
        with _lock:
            _model = new_model
            _is_fitted = True


def is_anomalous(indicators: dict) -> tuple:
    """Check if current market conditions are anomalous.

    Args:
        indicators: Dict of last-row indicator values.

    Returns:
        Tuple of (is_anomaly: bool, anomaly_score: float).
        anomaly_score < 0 means more anomalous; the threshold is ~0.
        Returns (False, 0.0) if model is not yet fitted.
    """
    with _lock:
        if not _is_fitted or not _sklearn_available:
            return False, 0.0
        model = _model  # local ref
    vec = _to_vector(indicators).reshape(1, -1)
    score = model.decision_function(vec)[0]
    prediction = model.predict(vec)[0]
    return prediction == -1, float(score)
