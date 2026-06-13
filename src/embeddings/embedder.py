"""Embedding model factory with local and remote provider support."""

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from src.config.settings import Settings
from src.embeddings.model_downloader import get_model_path


def create_embeddings(settings: Settings) -> Embeddings:
    """Factory: create an embedding model instance based on settings.

    Supports two embedding providers:
    - 'local': Uses a HuggingFace model (default: BGE-M3) loaded from disk.
      The model is downloaded via ModelScope if not already present.
    - 'openai': Uses OpenAI's embedding API (requires openai_api_key).

    BGE-M3 produces 1024-dimensional normalized embeddings and supports
    100+ languages including Chinese and English.

    Args:
        settings: Application settings with embedding configuration.

    Returns:
        An Embeddings instance ready for document embedding.

    Raises:
        ValueError: If the configured embedding provider is not supported.
    """
    if settings.embedding_provider == "local":
        model_path = get_model_path(settings.local_model_dir)
        print(f"Loading local embedding model from: {model_path}")
        return HuggingFaceEmbeddings(
            model_name=str(model_path),
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    elif settings.embedding_provider == "openai":
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )

    else:
        raise ValueError(
            f"Unsupported embedding provider: {settings.embedding_provider}. "
            f"Use 'local' or 'openai'."
        )
