"""Application settings loaded from environment variables / .env file."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed configuration for the Knowledge Q&A Agent.

    All values can be set via environment variables or a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM Provider ---
    llm_provider: Literal["deepseek", "openai", "anthropic"] = "deepseek"

    # --- DeepSeek ---
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-v4-pro"
    deepseek_base_url: str = "https://api.deepseek.com"

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # --- Anthropic ---
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

    # --- Embedding ---
    embedding_provider: Literal["local", "openai"] = "local"
    # Model name for HuggingFace (BGE-M3) or OpenAI embedding model
    embedding_model: str = "BAAI/bge-m3"

    # --- Local Model ---
    local_model_dir: Path = Path("models")

    # --- Paths ---
    docs_dir: Path = Path("docs")
    chroma_persist_dir: Path = Path("chroma_db")
    memory_file: Path = Path("conversation_history.json")

    # --- Chunking ---
    chunk_size: int = 500
    chunk_overlap: int = 100

    # --- Retrieval ---
    retrieval_k: int = 4


# Singleton instance
settings = Settings()
