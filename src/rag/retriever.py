"""Retrieval strategy configuration."""

from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma


def create_retriever(vector_store: Chroma, k: int = 4) -> BaseRetriever:
    """Create a similarity-search retriever from a Chroma vector store.

    Args:
        vector_store: Chroma vector store instance.
        k: Number of top documents to retrieve (default: 4).

    Returns:
        A BaseRetriever configured for similarity search.
    """
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
