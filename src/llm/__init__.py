"""LLM provider factory and package exports."""

from src.config.settings import Settings
from src.llm.anthropic_provider import AnthropicProvider
from src.llm.base import LLMProvider
from src.llm.deepseek_provider import DeepSeekProvider
from src.llm.openai_provider import OpenAIProvider


def create_llm_provider(settings: Settings) -> LLMProvider:
    """Factory: return the configured LLM provider based on settings.

    Args:
        settings: Application settings specifying which provider to use.

    Returns:
        An LLMProvider instance (DeepSeekProvider, OpenAIProvider, or
        AnthropicProvider).

    Raises:
        ValueError: If the configured provider is not supported.
    """
    if settings.llm_provider == "deepseek":
        return DeepSeekProvider(settings)
    elif settings.llm_provider == "openai":
        return OpenAIProvider(settings)
    elif settings.llm_provider == "anthropic":
        return AnthropicProvider(settings)
    else:
        raise ValueError(
            f"Unsupported LLM provider: {settings.llm_provider}. "
            f"Use 'deepseek', 'openai', or 'anthropic'."
        )


__all__ = [
    "LLMProvider",
    "DeepSeekProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "create_llm_provider",
]
