"""ChromaDB vector store wrapper for persistent document storage."""

import shutil
from pathlib import Path
from typing import Optional

from chromadb import PersistentClient
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config.settings import Settings


class VectorStore:
    """Wraps ChromaDB with persistence for the Knowledge Q&A Agent.

    Provides create, load, add, and query operations on the vector store.
    Uses a singleton-like pattern — the underlying Chroma collection is
    loaded once and reused for the lifetime of the instance.
    """

    def __init__(self, app_settings: Settings, embeddings: Embeddings) -> None:
        """Initialize the vector store wrapper.

        Args:
            app_settings: Application configuration.
            embeddings: Embedding model to use for vectorization.
        """
        self._settings = app_settings
        self._embeddings = embeddings
        self._persist_dir = str(app_settings.chroma_persist_dir.resolve())
        self._client: Optional[PersistentClient] = None
        self._store: Optional[Chroma] = None

    def _get_or_create_client(self) -> PersistentClient:
        """Lazily initialize and return the persistent ChromaDB client."""
        if self._client is None:
            # Ensure the persist directory exists
            persist_path = Path(self._persist_dir)
            persist_path.mkdir(parents=True, exist_ok=True)

            self._client = PersistentClient(
                path=self._persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    def create_or_load(self) -> Chroma:
        """Load an existing vector store or create a new one.

        If a persisted ChromaDB exists on disk, it is loaded.
        Otherwise a new empty collection is created.

        Returns:
            A Chroma vector store instance.
        """
        if self._store is not None:
            return self._store

        client = self._get_or_create_client()

        # Check if the collection already exists
        existing = client.list_collections()
        collection_names = [c.name for c in existing]

        if "knowledge_base" in collection_names:
            self._store = Chroma(
                client=client,
                collection_name="knowledge_base",
                embedding_function=self._embeddings,
            )
        else:
            self._store = Chroma(
                client=client,
                collection_name="knowledge_base",
                embedding_function=self._embeddings,
            )

        return self._store

    def add_documents(self, documents: list[Document]) -> None:
        """Add document chunks to the vector store.

        Args:
            documents: List of Document chunks to embed and store.
        """
        if not documents:
            return
        store = self.create_or_load()
        store.add_documents(documents)

    def similarity_search(
        self, query: str, k: Optional[int] = None
    ) -> list[Document]:
        """Perform similarity search against the vector store.

        Args:
            query: The search query string.
            k: Number of results to return. Defaults to settings.retrieval_k.

        Returns:
            List of the k most similar Documents.
        """
        if k is None:
            k = self._settings.retrieval_k
        store = self.create_or_load()
        return store.similarity_search(query, k=k)

    def get_stats(self) -> dict:
        """Return collection statistics.

        Returns:
            Dictionary with collection name and document count.
        """
        store = self.create_or_load()
        collection = store._collection  # Access underlying Chroma collection
        return {
            "collection_name": collection.name,
            "document_count": collection.count(),
            "persist_dir": self._persist_dir,
        }

    def clear(self) -> None:
        """Delete the entire vector store (collection + persisted data)."""
        if self._client is not None:
            try:
                self._client.delete_collection("knowledge_base")
            except Exception:
                pass
        self._store = None
        self._client = None

        # Remove persisted data
        shutil.rmtree(self._persist_dir, ignore_errors=True)
