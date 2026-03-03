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
        """Get consensus from all agent signals for a symbol."""
        signals = SignalPipeline.read_all_signals(symbol, max_age_minutes)
        if not signals:
            return {'direction': 'NEUTRAL', 'avg_confidence': 0, 'sources': []}

        buy_conf = sum(s['confidence'] for s in signals if s['direction'] == 'BUY')
        sell_conf = sum(s['confidence'] for s in signals if s['direction'] == 'SELL')
        total = len(signals)

        if buy_conf > sell_conf:
            direction = 'BUY'
            avg_conf = buy_conf / total
        elif sell_conf > buy_conf:
            direction = 'SELL'
            avg_conf = sell_conf / total
        else:
            direction = 'NEUTRAL'
            avg_conf = 0

        return {
            'direction': direction,
            'avg_confidence': avg_conf,
            'buy_confidence': buy_conf,
            'sell_confidence': sell_conf,
            'sources': [{'source': s['source'], 'direction': s['direction'], 'confidence': s['confidence']} for s in signals],
            'signal_count': total,
        }
