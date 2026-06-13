"""Document chunking strategy for the knowledge base."""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentSplitter:
    """Token-aware recursive character text splitter.

    Chunks documents using the cl100k_base tokenizer (compatible with
    GPT-4/3.5 and Claude models). Splits respect natural boundaries
    (paragraphs, sentences) before falling back to hard cuts.

    Default chunk size of 500 tokens with 100-token overlap provides
    a good balance between retrieval precision and context preservation
    for company knowledge base documents.
    """

    # Separators ordered by priority: prefer natural boundaries
    SEPARATORS = ["\n\n", "\n", ". ", "。", " ", ""]

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100) -> None:
        """Initialize the splitter.

        Args:
            chunk_size: Maximum tokens per chunk.
            chunk_overlap: Number of tokens to overlap between chunks.
        """
        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.SEPARATORS,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks with enriched metadata.

        Each chunk receives a chunk_index in its metadata for traceability.

        Args:
            documents: List of Documents to split.

        Returns:
            List of chunked Documents.
        """
        if not documents:
            return []

        chunks = self._splitter.split_documents(documents)

        # Enrich with chunk index for traceability
        # Group by source to number chunks per-document
        source_chunks: dict[str, list[Document]] = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            source_chunks.setdefault(source, []).append(chunk)

        for source, doc_chunks in source_chunks.items():
            for i, chunk in enumerate(doc_chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunk_total"] = len(doc_chunks)

        return chunks
