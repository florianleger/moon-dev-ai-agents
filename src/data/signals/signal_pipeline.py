"""
Standardized signal pipeline for agent -> strategy communication.
Agents write signals, strategies read them.
"""
import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock

SIGNALS_DIR = os.path.join(os.path.dirname(__file__))

# Configurable weights per signal source
SOURCE_WEIGHTS = {
    'adaptive_hybrid': 1.0,
    'sentiment': 0.6,
    'sentiment_agent': 0.6,
    'whale': 0.8,
    'whale_agent': 0.8,
    'funding': 0.7,
    'funding_agent': 0.7,
    'liquidation': 0.7,
    'liquidation_agent': 0.7,
    'fear_greed': 0.5,
    'defi_llama': 0.5,
    'default': 0.5,
}

# Recency decay brackets (age in minutes -> weight multiplier)
RECENCY_BRACKETS = [
    (5, 1.0),    # < 5 min
    (30, 0.8),   # 5-30 min
    (60, 0.5),   # 30-60 min
]
RECENCY_STALE = 0.3  # > 60 min


def _recency_weight(signal_time: datetime) -> float:
    """Calculate recency weight for a signal based on its age."""
    age_minutes = (datetime.now() - signal_time).total_seconds() / 60
    for max_age, weight in RECENCY_BRACKETS:
        if age_minutes < max_age:
            return weight
    return RECENCY_STALE


def _source_weight(source: str) -> float:
    """Get configurable weight for a signal source."""
    return SOURCE_WEIGHTS.get(source, SOURCE_WEIGHTS['default'])


class SignalPipeline:
    _lock = Lock()

    @staticmethod
    def write_signal(source: str, symbol: str, direction: str, confidence: float, reasoning: str, metadata: dict = None):
        """Write a signal from an agent."""
        signal = {
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'symbol': symbol,
            'direction': direction,  # BUY, SELL, NOTHING/NEUTRAL
            'confidence': confidence,  # 0-100
            'reasoning': reasoning,
            'metadata': metadata or {},
        }
        filename = f"{source}_{symbol}.json"
        filepath = os.path.join(SIGNALS_DIR, filename)

        with SignalPipeline._lock:
            with open(filepath, 'w') as f:
                json.dump(signal, f, indent=2)

        return signal

    @staticmethod
    def read_signal(source: str, symbol: str, max_age_minutes: int = 60) -> dict:
        """Read the latest signal from a source for a symbol."""
        filename = f"{source}_{symbol}.json"
        filepath = os.path.join(SIGNALS_DIR, filename)

        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, 'r') as f:
                signal = json.load(f)

            # Check age
            signal_time = datetime.fromisoformat(signal['timestamp'])
            if datetime.now() - signal_time > timedelta(minutes=max_age_minutes):
                return None  # Stale signal

            return signal
        except Exception:
            return None

    @staticmethod
    def read_all_signals(symbol: str = None, max_age_minutes: int = 60) -> list:
        """Read all recent signals, optionally filtered by symbol."""
        signals = []
        if not os.path.exists(SIGNALS_DIR):
            return signals

        for filename in os.listdir(SIGNALS_DIR):
            if not filename.endswith('.json') or filename == '__init__.py':
                continue
            filepath = os.path.join(SIGNALS_DIR, filename)
            try:
                with open(filepath, 'r') as f:
                    signal = json.load(f)
                signal_time = datetime.fromisoformat(signal['timestamp'])
                if datetime.now() - signal_time > timedelta(minutes=max_age_minutes):
                    continue
                if symbol and signal.get('symbol') != symbol:
                    continue
                signals.append(signal)
            except Exception:
                continue
        return signals

    @staticmethod
    def cleanup(max_age_minutes: int = 120):
        """Remove stale signal files."""
        if not os.path.exists(SIGNALS_DIR):
            return
        cutoff = datetime.now() - timedelta(minutes=max_age_minutes)
        for filename in os.listdir(SIGNALS_DIR):
            if not filename.endswith('.json'):
                continue
            filepath = os.path.join(SIGNALS_DIR, filename)
            try:
                with open(filepath, 'r') as f:
                    signal = json.load(f)
                signal_time = datetime.fromisoformat(signal['timestamp'])
                if signal_time < cutoff:
                    os.remove(filepath)
            except Exception:
                pass

    @staticmethod
    def get_consensus(symbol: str, max_age_minutes: int = 60) -> dict:
        """Get weighted consensus from all agent signals for a symbol.

        Each signal is weighted by:
        1. Source weight (configurable per agent)
        2. Recency weight (newer signals count more)

        Returns dict with direction, weighted confidence, source count, and convergence score.
        """
        signals = SignalPipeline.read_all_signals(symbol, max_age_minutes)
        if not signals:
            return {'direction': 'NEUTRAL', 'avg_confidence': 0, 'sources': [],
                    'signal_count': 0, 'convergence': 0.0}

        weighted_buy = 0.0
        weighted_sell = 0.0
        total_weight = 0.0

        source_details = []
        for s in signals:
            src_w = _source_weight(s.get('source', ''))
            try:
                sig_time = datetime.fromisoformat(s['timestamp'])
            except Exception:
                sig_time = datetime.now()
            rec_w = _recency_weight(sig_time)
            combined_w = src_w * rec_w
            weighted_conf = s['confidence'] * combined_w

            if s['direction'] == 'BUY':
                weighted_buy += weighted_conf
            elif s['direction'] == 'SELL':
                weighted_sell += weighted_conf

            total_weight += combined_w
            source_details.append({
                'source': s['source'],
                'direction': s['direction'],
                'confidence': s['confidence'],
                'source_weight': src_w,
                'recency_weight': rec_w,
                'weighted_confidence': round(weighted_conf, 2),
            })

        total = len(signals)

        if weighted_buy > weighted_sell:
            direction = 'BUY'
            avg_conf = weighted_buy / total_weight if total_weight > 0 else 0
        elif weighted_sell > weighted_buy:
            direction = 'SELL'
            avg_conf = weighted_sell / total_weight if total_weight > 0 else 0
        else:
            direction = 'NEUTRAL'
            avg_conf = 0

        # Convergence: how many sources agree with the consensus direction
        if direction != 'NEUTRAL' and total > 0:
            agreeing = sum(1 for s in signals if s['direction'] == direction)
            convergence = agreeing / total
        else:
            convergence = 0.0

        return {
            'direction': direction,
            'avg_confidence': round(avg_conf, 2),
            'buy_confidence': round(weighted_buy, 2),
            'sell_confidence': round(weighted_sell, 2),
            'sources': source_details,
            'signal_count': total,
            'convergence': round(convergence, 2),
        }
