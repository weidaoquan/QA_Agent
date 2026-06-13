"""Ingestion orchestrator: load → split → embed → store."""

from pathlib import Path

from src.config.settings import Settings
from src.embeddings.embedder import create_embeddings
from src.ingestion.loader import load_all_documents, load_document
from src.ingestion.splitter import DocumentSplitter
from src.vectordb.store import VectorStore


class IngestionPipeline:
    """Orchestrates the full document ingestion pipeline.

    Flow: Load documents → Split into chunks → Embed → Store in vector DB.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the ingestion pipeline.

        Args:
            settings: Application configuration.
        """
        self._settings = settings
        self._embeddings = create_embeddings(settings)
        self._splitter = DocumentSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self._vector_store = VectorStore(settings, self._embeddings)

    def ingest_directory(self, docs_dir: Path | None = None) -> int:
        """Ingest all supported documents from a directory.

        Args:
            docs_dir: Directory to scan. Defaults to settings.docs_dir.

        Returns:
            Total number of chunks ingested.
        """
        if docs_dir is None:
            docs_dir = self._settings.docs_dir

        # Load
        documents = load_all_documents(docs_dir)
        if not documents:
            print(f"No supported documents found in {docs_dir}")
            return 0

        print(f"Loaded {len(documents)} page(s) from {docs_dir}")

        # Split
        chunks = self._splitter.split(documents)
        print(f"Split into {len(chunks)} chunk(s)")

        # Embed & store
        self._vector_store.add_documents(chunks)
        print(f"Ingested {len(chunks)} chunk(s) into the knowledge base")

        return len(chunks)

    def ingest_file(self, file_path: Path) -> int:
        """Ingest a single document file.

        Args:
            file_path: Path to the document file.

        Returns:
            Number of chunks ingested.
        """
        # Load
        documents = load_document(file_path)
        print(f"Loaded {file_path.name} ({len(documents)} page(s))")

        # Split
        chunks = self._splitter.split(documents)
        print(f"Split into {len(chunks)} chunk(s)")

        # Embed & store
        self._vector_store.add_documents(chunks)
        print(f"Ingested {len(chunks)} chunk(s) into the knowledge base")

        return len(chunks)

    @property
    def vector_store(self) -> VectorStore:
        """Expose the vector store for querying."""
        return self._vector_store
