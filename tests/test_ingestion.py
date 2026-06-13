"""Tests for document loading and splitting."""

import tempfile
from pathlib import Path

import pytest

from src.ingestion.loader import load_all_documents, load_document
from src.ingestion.splitter import DocumentSplitter


class TestDocumentLoader:
    """Tests for the document loader module."""

    def test_load_txt_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("This is a test document.\nIt has multiple lines.\n")
            tmp_path = Path(f.name)

        try:
            docs = load_document(tmp_path)
            assert len(docs) == 1
            assert "test document" in docs[0].page_content
            assert docs[0].metadata["source"] == tmp_path.name
        finally:
            tmp_path.unlink()

    def test_load_md_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("# Heading\n\nSome content here.\n")
            tmp_path = Path(f.name)

        try:
            docs = load_document(tmp_path)
            assert len(docs) == 1
            assert "Heading" in docs[0].page_content
        finally:
            tmp_path.unlink()

    def test_load_unsupported_extension(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xyz", delete=False, encoding="utf-8"
        ) as f:
            f.write("test")
            tmp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                load_document(tmp_path)
        finally:
            tmp_path.unlink()

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_document(Path("/nonexistent/file.pdf"))

    def test_load_all_from_directory(self):
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            # Create a couple of text files
            (tmp_dir / "doc1.txt").write_text("Document one content.", encoding="utf-8")
            (tmp_dir / "doc2.md").write_text("# Doc 2\nContent here.", encoding="utf-8")

            docs = load_all_documents(tmp_dir)
            assert len(docs) == 2
            sources = {d.metadata["source"] for d in docs}
            assert sources == {"doc1.txt", "doc2.md"}
        finally:
            import shutil

            shutil.rmtree(tmp_dir)

    def test_load_empty_directory(self):
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            docs = load_all_documents(tmp_dir)
            assert docs == []
        finally:
            import shutil

            shutil.rmtree(tmp_dir)


class TestDocumentSplitter:
    """Tests for the document splitter module."""

    def test_splits_long_document(self):
        from langchain_core.documents import Document

        splitter = DocumentSplitter(chunk_size=50, chunk_overlap=10)
        # Create a document longer than 50 tokens
        long_text = "This is a sentence. " * 50
        doc = Document(page_content=long_text, metadata={"source": "test.txt"})

        chunks = splitter.split([doc])
        assert len(chunks) > 1
        # Each chunk should have chunk_index and chunk_total metadata
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
            assert "chunk_total" in chunk.metadata

    def test_short_document_not_split(self):
        from langchain_core.documents import Document

        splitter = DocumentSplitter(chunk_size=500, chunk_overlap=100)
        short_text = "A short document."
        doc = Document(page_content=short_text, metadata={"source": "test.txt"})

        chunks = splitter.split([doc])
        assert len(chunks) == 1
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[0].metadata["chunk_total"] == 1

    def test_empty_input(self):
        splitter = DocumentSplitter()
        chunks = splitter.split([])
        assert chunks == []

    def test_preserves_source_metadata(self):
        from langchain_core.documents import Document

        splitter = DocumentSplitter(chunk_size=50, chunk_overlap=10)
        doc = Document(
            page_content="Long text. " * 30,
            metadata={"source": "policy.pdf", "author": "HR"},
        )

        chunks = splitter.split([doc])
        for chunk in chunks:
            assert chunk.metadata["source"] == "policy.pdf"
