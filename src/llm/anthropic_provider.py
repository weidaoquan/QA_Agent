"""Anthropic LLM provider implementation."""

from langchain_anthropic import ChatAnthropic

from src.config.settings import Settings


class AnthropicProvider:
    """Wraps ChatAnthropic as an LLM provider for the Knowledge Q&A Agent.

    Uses a low temperature for factual, grounded responses.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def get_chat_model(self) -> ChatAnthropic:
        """Return a configured ChatAnthropic instance."""
        return ChatAnthropic(
            model=self._settings.anthropic_model,
            temperature=0.3,
            api_key=self._settings.anthropic_api_key,
        )

    def get_model_name(self) -> str:
        """Return the model name."""
        return self._settings.anthropic_model
