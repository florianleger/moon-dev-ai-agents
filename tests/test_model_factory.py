"""Tests for ModelFactory: provider creation, aliases, fallback, retry, call stats."""
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_call_stats():
    """Reset call stats between tests."""
    from src.models.model_factory import ModelFactory
    ModelFactory._call_stats = {'total': 0, 'errors': 0, 'by_provider': {}}
    yield


@pytest.fixture
def factory():
    """Create a ModelFactory with no real API keys (all models unavailable)."""
    with patch.dict('os.environ', {}, clear=True):
        with patch('src.models.model_factory.ModelFactory._initialize_models'):
            from src.models.model_factory import ModelFactory
            f = ModelFactory()
            f._models = {}
            return f


# ---------------------------------------------------------------------------
# _PROVIDER_ALIASES
# ---------------------------------------------------------------------------

class TestProviderAliases:
    def test_anthropic_maps_to_claude(self):
        from src.models.model_factory import ModelFactory
        assert ModelFactory._PROVIDER_ALIASES['anthropic'] == 'claude'
        assert ModelFactory._PROVIDER_ALIASES['claude'] == 'claude'

    def test_gpt_maps_to_openai(self):
        from src.models.model_factory import ModelFactory
        assert ModelFactory._PROVIDER_ALIASES['gpt'] == 'openai'
        assert ModelFactory._PROVIDER_ALIASES['openai'] == 'openai'

    def test_grok_maps_to_xai(self):
        from src.models.model_factory import ModelFactory
        assert ModelFactory._PROVIDER_ALIASES['grok'] == 'xai'
        assert ModelFactory._PROVIDER_ALIASES['xai'] == 'xai'

    def test_all_aliases_resolve_to_known_model_types(self):
        from src.models.model_factory import ModelFactory
        known_types = set(ModelFactory.MODEL_IMPLEMENTATIONS.keys())
        for alias, model_type in ModelFactory._PROVIDER_ALIASES.items():
            assert model_type in known_types, f"Alias '{alias}' maps to unknown type '{model_type}'"


# ---------------------------------------------------------------------------
# FALLBACK_CHAIN
# ---------------------------------------------------------------------------

class TestFallbackChain:
    def test_fallback_chain_has_expected_providers(self):
        from src.models.model_factory import ModelFactory
        chain = ModelFactory.FALLBACK_CHAIN
        assert 'anthropic' in chain
        assert 'openai' in chain
        assert len(chain) >= 3

    def test_create_model_with_fallback_tries_chain(self, factory):
        from src.models.model_factory import ModelFactory

        mock_model = MagicMock()
        mock_model.is_available.return_value = True

        # Patch the singleton model_factory to use our fixture
        with patch('src.models.model_factory.model_factory', factory):
            # First provider fails, second succeeds
            call_count = 0
            def side_effect(model_type, model_name=None):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return None  # First provider unavailable
                return mock_model

            factory.get_model = MagicMock(side_effect=side_effect)
            result = ModelFactory.create_model_with_fallback('anthropic')
            assert result is mock_model
            assert call_count >= 2


# ---------------------------------------------------------------------------
# create_model
# ---------------------------------------------------------------------------

class TestCreateModel:
    def test_create_model_resolves_alias(self, factory):
        from src.models.model_factory import ModelFactory

        mock_model = MagicMock()
        factory.get_model = MagicMock(return_value=mock_model)

        with patch('src.models.model_factory.model_factory', factory):
            result = ModelFactory.create_model('anthropic')
            factory.get_model.assert_called_with('claude', model_name=None)
            assert result is mock_model

    def test_create_model_with_unknown_provider(self, factory):
        from src.models.model_factory import ModelFactory

        factory.get_model = MagicMock(return_value=None)

        with patch('src.models.model_factory.model_factory', factory):
            result = ModelFactory.create_model('nonexistent_provider')
            assert result is None


# ---------------------------------------------------------------------------
# generate_response_with_retry
# ---------------------------------------------------------------------------

class TestGenerateResponseWithRetry:
    def test_success_on_first_try(self):
        from src.models.base_model import BaseModel

        mock_model = MagicMock(spec=BaseModel)
        mock_model.generate_response = MagicMock(return_value="success response")
        # Call the real retry method
        result = BaseModel.generate_response_with_retry(mock_model, "sys", "user")
        assert result == "success response"
        assert mock_model.generate_response.call_count == 1

    def test_retries_on_failure_then_succeeds(self):
        from src.models.base_model import BaseModel

        mock_model = MagicMock(spec=BaseModel)
        mock_model.generate_response = MagicMock(
            side_effect=[Exception("fail"), Exception("fail"), "ok"]
        )
        with patch('time.sleep'):  # Don't actually sleep in tests
            result = BaseModel.generate_response_with_retry(mock_model, "sys", "user")
        assert result == "ok"
        assert mock_model.generate_response.call_count == 3

    def test_returns_none_after_all_retries_exhausted(self):
        from src.models.base_model import BaseModel

        mock_model = MagicMock(spec=BaseModel)
        mock_model.generate_response = MagicMock(
            side_effect=Exception("always fails")
        )
        with patch('time.sleep'):
            result = BaseModel.generate_response_with_retry(mock_model, "sys", "user")
        assert result is None
        assert mock_model.generate_response.call_count == 3

    def test_retry_delays_are_exponential(self):
        from src.models.base_model import BaseModel

        mock_model = MagicMock(spec=BaseModel)
        mock_model.generate_response = MagicMock(
            side_effect=Exception("always fails")
        )
        sleep_calls = []
        with patch('time.sleep', side_effect=lambda s: sleep_calls.append(s)):
            BaseModel.generate_response_with_retry(mock_model, "sys", "user")
        # Code sleeps after every failed attempt (including the last one)
        # Delays: 3^0=1, 3^1=3, 3^2=9
        assert len(sleep_calls) == 3
        assert sleep_calls[0] == 1   # 3^0
        assert sleep_calls[1] == 3   # 3^1
        assert sleep_calls[2] == 9   # 3^2


# ---------------------------------------------------------------------------
# call_stats tracking
# ---------------------------------------------------------------------------

class TestCallStats:
    def test_log_call_increments_total(self):
        from src.models.model_factory import ModelFactory
        ModelFactory.log_call('claude', True, 100)
        stats = ModelFactory.get_stats()
        assert stats['total'] == 1
        assert stats['errors'] == 0

    def test_log_call_tracks_errors(self):
        from src.models.model_factory import ModelFactory
        ModelFactory.log_call('openai', False, 50)
        stats = ModelFactory.get_stats()
        assert stats['total'] == 1
        assert stats['errors'] == 1
        assert stats['by_provider']['openai']['errors'] == 1

    def test_log_call_tracks_latency(self):
        from src.models.model_factory import ModelFactory
        ModelFactory.log_call('claude', True, 150)
        ModelFactory.log_call('claude', True, 250)
        stats = ModelFactory.get_stats()
        assert stats['by_provider']['claude']['total_latency_ms'] == 400
        assert stats['by_provider']['claude']['calls'] == 2

    def test_stats_by_provider_isolated(self):
        from src.models.model_factory import ModelFactory
        ModelFactory.log_call('claude', True, 100)
        ModelFactory.log_call('openai', True, 200)
        stats = ModelFactory.get_stats()
        assert 'claude' in stats['by_provider']
        assert 'openai' in stats['by_provider']
        assert stats['by_provider']['claude']['calls'] == 1
        assert stats['by_provider']['openai']['calls'] == 1


# ---------------------------------------------------------------------------
# DEFAULT_MODELS
# ---------------------------------------------------------------------------

class TestDefaultModels:
    def test_all_implementations_have_default_model(self):
        from src.models.model_factory import ModelFactory
        for model_type in ModelFactory.MODEL_IMPLEMENTATIONS:
            assert model_type in ModelFactory.DEFAULT_MODELS, (
                f"Model type '{model_type}' has no default model"
            )
