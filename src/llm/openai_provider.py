"""OpenAI LLM provider implementation."""

from langchain_openai import ChatOpenAI

from src.config.settings import Settings


class OpenAIProvider:
    """Wraps ChatOpenAI as an LLM provider for the Knowledge Q&A Agent.

    Uses a low temperature for factual, grounded responses.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def get_chat_model(self) -> ChatOpenAI:
        """Return a configured ChatOpenAI instance."""
        return ChatOpenAI(
            model=self._settings.openai_model,
            temperature=0.3,
            api_key=self._settings.openai_api_key,
        )

    def get_model_name(self) -> str:
        """Return the model name."""
        return self._settings.openai_model
