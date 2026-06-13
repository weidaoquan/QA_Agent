"""Document loader dispatcher for supported file types."""

from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyMuPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document


# Supported file extensions and their corresponding loader factories
LOADER_MAP = {
    ".pdf": PyMuPDFLoader,
    ".txt": lambda path: TextLoader(str(path), encoding="utf-8"),
    ".md": lambda path: TextLoader(str(path), encoding="utf-8"),
    ".markdown": lambda path: TextLoader(str(path), encoding="utf-8"),
    ".docx": Docx2txtLoader,
}


def load_document(file_path: Path) -> list[Document]:
    """Load a single document file.

    Dispatches to the correct LangChain loader based on file extension.
    Supported formats: PDF (.pdf), Word (.docx), Markdown (.md, .markdown), Text (.txt).

    Args:
        file_path: Absolute or relative path to the document.

    Returns:
        List of Document objects with source metadata attached.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    ext = file_path.suffix.lower()
    if ext not in LOADER_MAP:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported types: {', '.join(sorted(LOADER_MAP))}"
        )

    loader_factory = LOADER_MAP[ext]
    loader = loader_factory(str(file_path))
    documents = loader.load()

    # Enrich metadata with source filename for citation
    for doc in documents:
        doc.metadata["source"] = file_path.name
        doc.metadata["file_path"] = str(file_path.resolve())

    return documents


def load_all_documents(docs_dir: Path) -> list[Document]:
    """Scan the docs directory recursively and load all supported files.

    Args:
        docs_dir: Directory containing documents to ingest.

    Returns:
        Combined list of Document objects from all files.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    all_docs: list[Document] = []
    supported_extensions = set(LOADER_MAP.keys())

    for file_path in sorted(docs_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() in supported_extensions:
            try:
                docs = load_document(file_path)
                all_docs.extend(docs)
            except Exception as e:
                print(f"Warning: Failed to load {file_path.name}: {e}")
                continue

    return all_docs
