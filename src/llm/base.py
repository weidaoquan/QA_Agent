"""Protocol defining the LLM provider interface."""

from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel


class LLMProvider(Protocol):
    """Protocol that all LLM providers must implement.

    Defines the contract for obtaining a configured chat model
    and retrieving its name.
    """

    def get_chat_model(self) -> BaseChatModel:
        """Return a configured chat model instance ready for use."""
        ...

    def get_model_name(self) -> str:
        """Return the model name/identifier string."""
        ...
