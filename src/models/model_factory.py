"""
🌙 Moon Dev's Model Factory
Built with love by Moon Dev 🚀

This module manages all available AI models and provides a unified interface.
"""

import os
from typing import Dict, Optional, Type
from termcolor import cprint
from dotenv import load_dotenv
from pathlib import Path
from .base_model import BaseModel
from .claude_model import ClaudeModel
from .groq_model import GroqModel
from .openai_model import OpenAIModel
from .gemini_model import GeminiModel  # Re-enabled with Gemini 2.5 models
from .deepseek_model import DeepSeekModel
from .ollama_model import OllamaModel
from .xai_model import XAIModel
from .openrouter_model import OpenRouterModel  # 🌙 Moon Dev: OpenRouter - access to 200+ models!

class ModelFactory:
    """Factory for creating and managing AI models"""

    # Fallback chain for provider resilience
    FALLBACK_CHAIN = ['anthropic', 'openai', 'deepseek', 'groq']

    # Provider name aliases (maps common names to internal model type keys)
    _PROVIDER_ALIASES = {
        'anthropic': 'claude',
        'claude': 'claude',
        'openai': 'openai',
        'gpt': 'openai',
        'deepseek': 'deepseek',
        'groq': 'groq',
        'gemini': 'gemini',
        'ollama': 'ollama',
        'xai': 'xai',
        'grok': 'xai',
        'openrouter': 'openrouter',
    }

    # Simple call tracking
    _call_stats = {'total': 0, 'errors': 0, 'by_provider': {}}

    @classmethod
    def log_call(cls, provider, success, latency_ms=0):
        """Track LLM call statistics."""
        cls._call_stats['total'] += 1
        if not success:
            cls._call_stats['errors'] += 1
        if provider not in cls._call_stats['by_provider']:
            cls._call_stats['by_provider'][provider] = {'calls': 0, 'errors': 0, 'total_latency_ms': 0}
        stats = cls._call_stats['by_provider'][provider]
        stats['calls'] += 1
        if not success:
            stats['errors'] += 1
        stats['total_latency_ms'] += latency_ms

    @classmethod
    def get_stats(cls):
        """Return current call statistics."""
        return cls._call_stats

    # Map model types to their implementations
    MODEL_IMPLEMENTATIONS = {
        "claude": ClaudeModel,
        "groq": GroqModel,
        "openai": OpenAIModel,
        "gemini": GeminiModel,  # Re-enabled with Gemini 2.5 models
        "deepseek": DeepSeekModel,
        "ollama": OllamaModel,  # Add Ollama implementation
        "xai": XAIModel,  # xAI Grok models
        "openrouter": OpenRouterModel  # 🌙 Moon Dev: OpenRouter - 200+ models!
    }
    
    # Default models for each type
    DEFAULT_MODELS = {
        "claude": "claude-haiku-4-5-20251001",  # Latest fast Claude model
        "groq": "mixtral-8x7b-32768",        # Fast Mixtral model
        "openai": "gpt-4o",                  # Latest GPT-4 Optimized
        "gemini": "gemini-2.5-flash",        # Fast Gemini 2.5 model
        "deepseek": "deepseek-reasoner",     # Enhanced reasoning model
        "ollama": "llama3.2",                # Meta's Llama 3.2 - balanced performance
        "xai": "grok-4-fast-reasoning",      # xAI's Grok 4 Fast with reasoning (best value: 2M context, cheap!)
        "openrouter": "google/gemini-2.5-flash"  # 🌙 Moon Dev: OpenRouter default - fast & cheap Gemini!
    }
    
    def __init__(self):
        # Load environment variables
        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / '.env'
        load_dotenv(dotenv_path=env_path)

        self._models: Dict[str, BaseModel] = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all available models"""
        # Try to initialize each model type silently
        for model_type, key_name in self._get_api_key_mapping().items():
            if api_key := os.getenv(key_name):
                try:
                    if model_type in self.MODEL_IMPLEMENTATIONS:
                        model_class = self.MODEL_IMPLEMENTATIONS[model_type]
                        model_instance = model_class(api_key)

                        if model_instance.is_available():
                            self._models[model_type] = model_instance
                            # Just show the ready message
                            cprint(f"✅ {model_instance.model_name} ready", "green")
                except Exception as e:
                    cprint(f"⚠️ Failed to initialize {model_type}: {e}", "yellow")

        # Initialize Ollama separately (no API key needed)
        try:
            model_class = self.MODEL_IMPLEMENTATIONS["ollama"]
            model_instance = model_class(model_name=self.DEFAULT_MODELS["ollama"])

            if model_instance.is_available():
                self._models["ollama"] = model_instance
                cprint(f"✅ {model_instance.model_name} ready", "green")
        except Exception as e:
            cprint(f"⚠️ Ollama not available: {e}", "yellow")

        if not self._models:
            cprint("⚠️ No AI models available - check API keys in .env", "yellow")
    
    def get_model(self, model_type: str, model_name: Optional[str] = None) -> Optional[BaseModel]:
        """Get a specific model instance"""
        if model_type not in self.MODEL_IMPLEMENTATIONS or model_type not in self._models:
            return None

        model = self._models[model_type]
        if model_name and model.model_name != model_name:
            try:
                # Special handling for Ollama models
                if model_type == "ollama":
                    model = self.MODEL_IMPLEMENTATIONS[model_type](model_name=model_name)
                else:
                    # For API-based models that need a key
                    if api_key := os.getenv(self._get_api_key_mapping()[model_type]):
                        model = self.MODEL_IMPLEMENTATIONS[model_type](api_key, model_name=model_name)
                    else:
                        return None

                self._models[model_type] = model
            except Exception as e:
                cprint(f"⚠️ Failed to switch model {model_type}/{model_name}: {e}", "yellow")
                return None

        return model
    
    def _get_api_key_mapping(self) -> Dict[str, str]:
        """Get mapping of model types to their API key environment variable names"""
        return {
            "claude": "ANTHROPIC_KEY",
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_KEY",
            "gemini": "GEMINI_KEY",  # Re-enabled with Gemini 2.5 models
            "deepseek": "DEEPSEEK_KEY",
            "xai": "GROK_API_KEY",  # Grok/xAI uses GROK_API_KEY
            "openrouter": "OPENROUTER_API_KEY",  # 🌙 Moon Dev: OpenRouter - 200+ models!
            # Ollama doesn't need an API key as it runs locally
        }
    
    @property
    def available_models(self) -> Dict[str, list]:
        """Get all available models and their configurations"""
        return {
            model_type: model.AVAILABLE_MODELS
            for model_type, model in self._models.items()
        }
    
    def is_model_available(self, model_type: str) -> bool:
        """Check if a specific model type is available"""
        return model_type in self._models and self._models[model_type].is_available()

    @classmethod
    def create_model(cls, provider='anthropic', model_name=None):
        """Convenience class method to get a model from the singleton factory.

        Args:
            provider: Provider name (e.g. 'anthropic', 'claude', 'openai', 'deepseek', 'groq')
            model_name: Optional specific model name override
        Returns:
            BaseModel instance or None
        """
        model_type = cls._PROVIDER_ALIASES.get(provider, provider)
        return model_factory.get_model(model_type, model_name=model_name)

    @classmethod
    def create_model_with_fallback(cls, preferred_provider='anthropic'):
        """Try preferred provider, fall back to alternatives if unavailable."""
        preferred_type = cls._PROVIDER_ALIASES.get(preferred_provider, preferred_provider)
        providers = [preferred_type] + [
            cls._PROVIDER_ALIASES.get(p, p) for p in cls.FALLBACK_CHAIN
            if cls._PROVIDER_ALIASES.get(p, p) != preferred_type
        ]
        for provider in providers:
            try:
                model = model_factory.get_model(provider)
                if model:
                    return model
            except Exception:
                continue
        cprint("All LLM providers unavailable", "red")
        return None

# Create a singleton instance
model_factory = ModelFactory() 