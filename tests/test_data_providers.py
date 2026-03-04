"""Tests for data providers: FearGreedProvider and DefiLlamaProvider."""
import time
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# FearGreedProvider
# ---------------------------------------------------------------------------

class TestFearGreedSignal:
    """Test FearGreedProvider.get_signal() with various mocked values."""

    def _make_provider(self):
        from src.data_providers.fear_greed import FearGreedProvider
        p = FearGreedProvider()
        # Reset cache
        p._cache = None
        p._cache_time = 0
        return p

    def _mock_current(self, provider, value):
        """Mock get_current() to return a specific F&G value."""
        provider.get_current = MagicMock(return_value={
            'value': value,
            'classification': 'test',
            'timestamp': int(time.time()),
        })

    def test_extreme_fear_gives_buy_high_confidence(self):
        p = self._make_provider()
        self._mock_current(p, 15)
        signal = p.get_signal()
        assert signal['direction'] == 'BUY'
        assert signal['confidence'] == 80
        assert signal['value'] == 15

    def test_fear_gives_buy_mild_confidence(self):
        p = self._make_provider()
        self._mock_current(p, 30)
        signal = p.get_signal()
        assert signal['direction'] == 'BUY'
        assert signal['confidence'] == 55

    def test_extreme_greed_gives_sell_high_confidence(self):
        p = self._make_provider()
        self._mock_current(p, 85)
        signal = p.get_signal()
        assert signal['direction'] == 'SELL'
        assert signal['confidence'] == 75

    def test_greed_gives_sell_mild_confidence(self):
        p = self._make_provider()
        self._mock_current(p, 70)
        signal = p.get_signal()
        assert signal['direction'] == 'SELL'
        assert signal['confidence'] == 50

    def test_neutral_zone_gives_neutral(self):
        p = self._make_provider()
        self._mock_current(p, 50)
        signal = p.get_signal()
        assert signal['direction'] == 'NEUTRAL'
        assert signal['confidence'] == 30

    def test_api_failure_gives_neutral_zero_confidence(self):
        p = self._make_provider()
        p.get_current = MagicMock(return_value=None)
        signal = p.get_signal()
        assert signal['direction'] == 'NEUTRAL'
        assert signal['confidence'] == 0


class TestFearGreedCache:
    """Test FearGreedProvider cache TTL behavior."""

    def test_cache_is_used_within_ttl(self):
        from src.data_providers.fear_greed import FearGreedProvider
        p = FearGreedProvider()

        cached_data = [{'value': '42', 'value_classification': 'Fear', 'timestamp': str(int(time.time()))}]
        p._cache = cached_data
        p._cache_time = time.time()  # Fresh cache

        # Should use cache, not call API
        with patch('requests.get') as mock_get:
            result = p._fetch(limit=1)
            mock_get.assert_not_called()
            assert result == cached_data[:1]

    def test_cache_expired_triggers_api_call(self):
        from src.data_providers.fear_greed import FearGreedProvider
        p = FearGreedProvider()

        p._cache = [{'value': '42', 'value_classification': 'Fear', 'timestamp': str(int(time.time()))}]
        p._cache_time = time.time() - 600  # Expired (>300s TTL)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {'data': [{'value': '55', 'value_classification': 'Neutral', 'timestamp': str(int(time.time()))}]}
        mock_resp.raise_for_status = MagicMock()

        with patch('requests.get', return_value=mock_resp) as mock_get:
            result = p._fetch(limit=1)
            mock_get.assert_called_once()
            assert result[0]['value'] == '55'


# ---------------------------------------------------------------------------
# DefiLlamaProvider
# ---------------------------------------------------------------------------

class TestDefiLlamaSignal:
    """Test DefiLlamaProvider.get_signal() with mocked TVL changes."""

    def _make_provider(self):
        from src.data_providers.defi_llama import DefiLlamaProvider
        p = DefiLlamaProvider()
        # Reset caches
        p._protocols_cache = None
        p._protocols_cache_time = 0
        p._chains_cache = None
        p._chains_cache_time = 0
        return p

    def test_tvl_crash_gives_sell_high_confidence(self):
        p = self._make_provider()
        p.get_tvl_changes = MagicMock(return_value={
            'chain': 'Solana', 'period': '1d', 'total_tvl': 1e9, 'avg_change_pct': -12.0,
        })
        signal = p.get_signal('Solana')
        assert signal['direction'] == 'SELL'
        assert signal['confidence'] == 75
        assert signal['tvl_change'] == -12.0

    def test_tvl_decline_gives_sell_mild_confidence(self):
        p = self._make_provider()
        p.get_tvl_changes = MagicMock(return_value={
            'chain': 'Solana', 'period': '1d', 'total_tvl': 1e9, 'avg_change_pct': -7.0,
        })
        signal = p.get_signal('Solana')
        assert signal['direction'] == 'SELL'
        assert signal['confidence'] == 55

    def test_tvl_surge_gives_buy_high_confidence(self):
        p = self._make_provider()
        p.get_tvl_changes = MagicMock(return_value={
            'chain': 'Solana', 'period': '1d', 'total_tvl': 1e9, 'avg_change_pct': 15.0,
        })
        signal = p.get_signal('Solana')
        assert signal['direction'] == 'BUY'
        assert signal['confidence'] == 70

    def test_tvl_rise_gives_buy_mild_confidence(self):
        p = self._make_provider()
        p.get_tvl_changes = MagicMock(return_value={
            'chain': 'Solana', 'period': '1d', 'total_tvl': 1e9, 'avg_change_pct': 6.0,
        })
        signal = p.get_signal('Solana')
        assert signal['direction'] == 'BUY'
        assert signal['confidence'] == 50

    def test_tvl_stable_gives_neutral(self):
        p = self._make_provider()
        p.get_tvl_changes = MagicMock(return_value={
            'chain': 'Solana', 'period': '1d', 'total_tvl': 1e9, 'avg_change_pct': 1.5,
        })
        signal = p.get_signal('Solana')
        assert signal['direction'] == 'NEUTRAL'
        assert signal['confidence'] == 25

    def test_no_data_gives_neutral(self):
        p = self._make_provider()
        p.get_tvl_changes = MagicMock(return_value=None)
        signal = p.get_signal('Solana')
        assert signal['direction'] == 'NEUTRAL'
        assert signal['confidence'] == 0


class TestDefiLlamaCache:
    """Test DefiLlamaProvider cache behavior."""

    def test_cached_chains_used_within_ttl(self):
        from src.data_providers.defi_llama import DefiLlamaProvider
        p = DefiLlamaProvider()

        p._chains_cache = [{'name': 'Solana', 'tvl': 5e9}]
        p._chains_cache_time = time.time()  # Fresh

        with patch('requests.get') as mock_get:
            result = p.get_chain_tvl('Solana')
            mock_get.assert_not_called()
            assert result['tvl'] == 5e9

    def test_expired_cache_triggers_api_call(self):
        from src.data_providers.defi_llama import DefiLlamaProvider
        p = DefiLlamaProvider()

        p._chains_cache = [{'name': 'Solana', 'tvl': 5e9}]
        p._chains_cache_time = time.time() - 600  # Expired

        mock_resp = MagicMock()
        mock_resp.json.return_value = [{'name': 'Solana', 'tvl': 6e9}]
        mock_resp.raise_for_status = MagicMock()

        with patch('requests.get', return_value=mock_resp) as mock_get:
            result = p.get_chain_tvl('Solana')
            mock_get.assert_called_once()
            assert result['tvl'] == 6e9


class TestDefiLlamaTopProtocols:
    """Test get_top_protocols with mocked data."""

    def test_filters_by_chain_and_sorts_by_tvl(self):
        from src.data_providers.defi_llama import DefiLlamaProvider
        p = DefiLlamaProvider()

        mock_protocols = [
            {'name': 'Raydium', 'chainTvls': {'Solana': 500e6}, 'category': 'DEX', 'change_1d': 2.0, 'change_7d': 5.0},
            {'name': 'Marinade', 'chainTvls': {'Solana': 800e6}, 'category': 'Staking', 'change_1d': 1.0, 'change_7d': 3.0},
            {'name': 'Aave', 'chainTvls': {'Ethereum': 10e9}, 'category': 'Lending', 'change_1d': 0.5, 'change_7d': 2.0},
        ]
        p._fetch_protocols = MagicMock(return_value=mock_protocols)

        result = p.get_top_protocols('Solana', limit=5)
        assert len(result) == 2
        assert result[0]['name'] == 'Marinade'  # Higher TVL first
        assert result[1]['name'] == 'Raydium'


class TestDefiLlamaDexVolumes:
    """Test get_dex_volumes with mocked data."""

    def test_aggregates_volumes(self):
        from src.data_providers.defi_llama import DefiLlamaProvider
        p = DefiLlamaProvider()

        p._fetch_dex_volumes = MagicMock(return_value={
            'protocols': [
                {'name': 'Uniswap', 'total24h': 1e9, 'total7d': 5e9},
                {'name': 'Raydium', 'total24h': 500e6, 'total7d': 2e9},
            ]
        })

        result = p.get_dex_volumes()
        assert result['total_24h'] == 1.5e9
        assert result['total_7d'] == 7e9
        assert len(result['top_dexes']) == 2
