"""Tests for the signal pipeline: write/read signals, consensus, recency, cleanup."""
import json
import os
import time
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from src.data.signals.signal_pipeline import (
    RECENCY_BRACKETS,
    RECENCY_STALE,
    SOURCE_WEIGHTS,
    SignalPipeline,
    _recency_weight,
    _source_weight,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_signal_file(signals_dir, source, symbol, direction, confidence,
                       timestamp=None, reasoning="test"):
    """Write a signal JSON directly to *signals_dir*."""
    sig = {
        'timestamp': (timestamp or datetime.now()).isoformat(),
        'source': source,
        'symbol': symbol,
        'direction': direction,
        'confidence': confidence,
        'reasoning': reasoning,
        'metadata': {},
    }
    filepath = os.path.join(signals_dir, f"{source}_{symbol}.json")
    with open(filepath, 'w') as f:
        json.dump(sig, f)
    return sig


# ---------------------------------------------------------------------------
# write_signal / read_signal roundtrip
# ---------------------------------------------------------------------------

class TestWriteReadRoundtrip:
    def test_write_then_read_returns_same_data(self, temp_dir):
        with patch("src.data.signals.signal_pipeline.SIGNALS_DIR", temp_dir):
            written = SignalPipeline.write_signal(
                source='sentiment', symbol='BTC', direction='BUY',
                confidence=75.0, reasoning='fear index low',
            )
            read = SignalPipeline.read_signal('sentiment', 'BTC', max_age_minutes=5)
            assert read is not None
            assert read['source'] == 'sentiment'
            assert read['symbol'] == 'BTC'
            assert read['direction'] == 'BUY'
            assert read['confidence'] == 75.0

    def test_read_returns_none_for_missing_signal(self, temp_dir):
        with patch("src.data.signals.signal_pipeline.SIGNALS_DIR", temp_dir):
            assert SignalPipeline.read_signal('missing', 'ETH') is None

    def test_read_returns_none_for_stale_signal(self, temp_dir):
        old_ts = datetime.now() - timedelta(minutes=120)
        with patch("src.data.signals.signal_pipeline.SIGNALS_DIR", temp_dir):
            _write_signal_file(temp_dir, 'whale', 'BTC', 'SELL', 60, timestamp=old_ts)
            result = SignalPipeline.read_signal('whale', 'BTC', max_age_minutes=60)
            assert result is None

    def test_write_includes_metadata(self, temp_dir):
        with patch("src.data.signals.signal_pipeline.SIGNALS_DIR", temp_dir):
            SignalPipeline.write_signal(
                source='test', symbol='SOL', direction='BUY',
                confidence=80, reasoning='r', metadata={'key': 'val'},
            )
            sig = SignalPipeline.read_signal('test', 'SOL')
            assert sig['metadata'] == {'key': 'val'}


# ---------------------------------------------------------------------------
# get_consensus
# ---------------------------------------------------------------------------

class TestGetConsensus:
    def test_empty_signals_returns_neutral(self, temp_dir):
        with patch("src.data.signals.signal_pipeline.SIGNALS_DIR", temp_dir):
            result = SignalPipeline.get_consensus('XRP')
            assert result['direction'] == 'NEUTRAL'
            assert result['signal_count'] == 0
            assert result['convergence'] == 0.0

    def test_single_buy_signal(self, temp_dir):
        with patch("src.data.signals.signal_pipeline.SIGNALS_DIR", temp_dir):
            SignalPipeline.write_signal('sentiment', 'BTC', 'BUY', 80, 'bullish')
            result = SignalPipeline.get_consensus('BTC')
            assert result['direction'] == 'BUY'
            assert result['signal_count'] == 1
            assert result['convergence'] == 1.0

    def test_majority_buy_wins(self, temp_dir):
        with patch("src.data.signals.signal_pipeline.SIGNALS_DIR", temp_dir):
            SignalPipeline.write_signal('sentiment', 'ETH', 'BUY', 70, 'r1')
            SignalPipeline.write_signal('whale', 'ETH', 'BUY', 80, 'r2')
            SignalPipeline.write_signal('funding', 'ETH', 'SELL', 60, 'r3')
            result = SignalPipeline.get_consensus('ETH')
            assert result['direction'] == 'BUY'
            assert result['convergence'] == pytest.approx(2 / 3, abs=0.01)

    def test_sell_wins_when_higher_weighted(self, temp_dir):
        with patch("src.data.signals.signal_pipeline.SIGNALS_DIR", temp_dir):
            # adaptive_hybrid weight=1.0, fear_greed weight=0.5
            SignalPipeline.write_signal('adaptive_hybrid', 'SOL', 'SELL', 90, 'r1')
            SignalPipeline.write_signal('fear_greed', 'SOL', 'BUY', 60, 'r2')
            result = SignalPipeline.get_consensus('SOL')
            assert result['direction'] == 'SELL'

    def test_consensus_returns_source_details(self, temp_dir):
        with patch("src.data.signals.signal_pipeline.SIGNALS_DIR", temp_dir):
            SignalPipeline.write_signal('whale', 'BTC', 'BUY', 70, 'r')
            result = SignalPipeline.get_consensus('BTC')
            assert len(result['sources']) == 1
            detail = result['sources'][0]
            assert detail['source'] == 'whale'
            assert 'source_weight' in detail
            assert 'recency_weight' in detail
            assert 'weighted_confidence' in detail


# ---------------------------------------------------------------------------
# Recency decay
# ---------------------------------------------------------------------------

class TestRecencyDecay:
    def test_fresh_signal_has_full_weight(self):
        now = datetime.now()
        assert _recency_weight(now) == 1.0

    def test_30min_old_signal_decayed(self):
        old = datetime.now() - timedelta(minutes=15)
        weight = _recency_weight(old)
        assert weight == 0.8

    def test_45min_old_signal_decayed(self):
        old = datetime.now() - timedelta(minutes=45)
        weight = _recency_weight(old)
        assert weight == 0.5

    def test_very_old_signal_gets_stale_weight(self):
        old = datetime.now() - timedelta(minutes=120)
        weight = _recency_weight(old)
        assert weight == RECENCY_STALE


# ---------------------------------------------------------------------------
# Source weights
# ---------------------------------------------------------------------------

class TestSourceWeights:
    def test_known_source_returns_configured_weight(self):
        assert _source_weight('adaptive_hybrid') == 1.0
        assert _source_weight('sentiment') == 0.6
        assert _source_weight('whale') == 0.8

    def test_unknown_source_returns_default(self):
        assert _source_weight('unknown_agent') == SOURCE_WEIGHTS['default']


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_cleanup_removes_stale_signals(self, temp_dir):
        old_ts = datetime.now() - timedelta(minutes=200)
        _write_signal_file(temp_dir, 'old_source', 'BTC', 'BUY', 50, timestamp=old_ts)
        # Fresh signal should survive
        _write_signal_file(temp_dir, 'fresh_source', 'BTC', 'SELL', 60)

        with patch("src.data.signals.signal_pipeline.SIGNALS_DIR", temp_dir):
            SignalPipeline.cleanup(max_age_minutes=120)

        assert not os.path.exists(os.path.join(temp_dir, 'old_source_BTC.json'))
        assert os.path.exists(os.path.join(temp_dir, 'fresh_source_BTC.json'))

    def test_cleanup_ignores_non_json_files(self, temp_dir):
        # Create a non-JSON file
        txt_path = os.path.join(temp_dir, 'notes.txt')
        with open(txt_path, 'w') as f:
            f.write('keep me')
        with patch("src.data.signals.signal_pipeline.SIGNALS_DIR", temp_dir):
            SignalPipeline.cleanup(max_age_minutes=0)
        assert os.path.exists(txt_path)


# ---------------------------------------------------------------------------
# read_all_signals
# ---------------------------------------------------------------------------

class TestReadAllSignals:
    def test_read_all_filters_by_symbol(self, temp_dir):
        with patch("src.data.signals.signal_pipeline.SIGNALS_DIR", temp_dir):
            SignalPipeline.write_signal('s1', 'BTC', 'BUY', 70, 'r')
            SignalPipeline.write_signal('s2', 'ETH', 'SELL', 60, 'r')
            signals = SignalPipeline.read_all_signals(symbol='BTC')
            assert len(signals) == 1
            assert signals[0]['symbol'] == 'BTC'

    def test_read_all_without_symbol_returns_all(self, temp_dir):
        with patch("src.data.signals.signal_pipeline.SIGNALS_DIR", temp_dir):
            SignalPipeline.write_signal('s1', 'BTC', 'BUY', 70, 'r')
            SignalPipeline.write_signal('s2', 'ETH', 'SELL', 60, 'r')
            signals = SignalPipeline.read_all_signals()
            assert len(signals) == 2

    def test_read_all_excludes_stale(self, temp_dir):
        old_ts = datetime.now() - timedelta(minutes=200)
        _write_signal_file(temp_dir, 'old', 'BTC', 'BUY', 50, timestamp=old_ts)
        with patch("src.data.signals.signal_pipeline.SIGNALS_DIR", temp_dir):
            signals = SignalPipeline.read_all_signals(max_age_minutes=60)
            assert len(signals) == 0


# ---------------------------------------------------------------------------
# Convergence calculation
# ---------------------------------------------------------------------------

class TestConvergence:
    def test_all_agree_convergence_is_one(self, temp_dir):
        with patch("src.data.signals.signal_pipeline.SIGNALS_DIR", temp_dir):
            SignalPipeline.write_signal('s1', 'BTC', 'BUY', 80, 'r1')
            SignalPipeline.write_signal('s2', 'BTC', 'BUY', 70, 'r2')
            SignalPipeline.write_signal('s3', 'BTC', 'BUY', 90, 'r3')
            result = SignalPipeline.get_consensus('BTC')
            assert result['convergence'] == 1.0

    def test_split_convergence_is_partial(self, temp_dir):
        with patch("src.data.signals.signal_pipeline.SIGNALS_DIR", temp_dir):
            SignalPipeline.write_signal('s1', 'BTC', 'BUY', 80, 'r1')
            SignalPipeline.write_signal('s2', 'BTC', 'SELL', 70, 'r2')
            result = SignalPipeline.get_consensus('BTC')
            assert 0 < result['convergence'] < 1.0
