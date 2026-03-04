import pytest
import os
import sys
import tempfile
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def temp_dir():
    """Temporary directory for test data."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_ohlcv():
    """Sample OHLCV DataFrame for testing (100 bars, 15min)."""
    import pandas as pd
    import numpy as np

    dates = pd.date_range('2024-01-01', periods=100, freq='15min')
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    return pd.DataFrame({
        'open': close + np.random.randn(100) * 0.1,
        'high': close + abs(np.random.randn(100) * 0.3),
        'low': close - abs(np.random.randn(100) * 0.3),
        'close': close,
        'volume': np.random.randint(1000, 10000, 100).astype(float),
    }, index=dates)


@pytest.fixture
def sample_ohlcv_250():
    """Larger OHLCV DataFrame (250 bars) for indicator calculations requiring more data."""
    import pandas as pd
    import numpy as np

    dates = pd.date_range('2024-01-01', periods=250, freq='1h')
    np.random.seed(123)
    close = 40000 + np.cumsum(np.random.randn(250) * 50)
    return pd.DataFrame({
        'open': close + np.random.randn(250) * 10,
        'high': close + abs(np.random.randn(250) * 30),
        'low': close - abs(np.random.randn(250) * 30),
        'close': close,
        'volume': np.random.randint(100, 5000, 250).astype(float),
    }, index=dates)


@pytest.fixture
def sample_indicators():
    """Pre-built indicator dict matching _compute_indicators output."""
    return {
        'close': 100.0,
        'open': 99.8,
        'high': 101.0,
        'low': 99.0,
        'volume': 5000.0,
        'rsi': 45.0,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'adx': 22.0,
        'ema_9': 100.1,
        'ema_21': 99.9,
        'ema_50': 99.5,
        'ema_200': 98.0,
        'bb_upper': 102.0,
        'bb_lower': 98.0,
        'bb_mid': 100.0,
        'bb_pct': 0.5,
        'atr': 1.5,
        'macd': 0.1,
        'macd_signal': 0.05,
        'macd_diff': 0.05,
        'volume_ratio': 1.2,
        'vwap': 100.0,
    }
