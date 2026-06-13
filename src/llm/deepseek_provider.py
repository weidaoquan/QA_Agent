"""DeepSeek LLM provider implementation.

DeepSeek provides an OpenAI-compatible API, so we use ChatOpenAI
with a custom base_url pointing to the DeepSeek API endpoint.
"""

from langchain_openai import ChatOpenAI

from src.config.settings import Settings


class DeepSeekProvider:
    """Wraps ChatOpenAI configured for the DeepSeek API.

    DeepSeek's API is fully OpenAI-compatible. We reuse ChatOpenAI
    with the DeepSeek base URL and API key.

    Supported models:
    - deepseek-v4-pro (latest flagship)
    - deepseek-chat (DeepSeek-V3)
    - deepseek-reasoner (DeepSeek-R1)
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the DeepSeek provider.

        Args:
            settings: Application settings with DeepSeek configuration.
        """
        self._settings = settings

    def get_chat_model(self) -> ChatOpenAI:
        """Return a ChatOpenAI instance configured for DeepSeek.

        Returns:
            ChatOpenAI pointed at the DeepSeek API endpoint.
        """
        return ChatOpenAI(
            model=self._settings.deepseek_model,
            base_url=self._settings.deepseek_base_url,
            api_key=self._settings.deepseek_api_key,
            temperature=0.3,
        )

    def get_model_name(self) -> str:
        """Return the DeepSeek model name."""
        return self._settings.deepseek_model
