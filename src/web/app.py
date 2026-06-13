"""FastAPI web application for the Knowledge Q&A Agent.

Provides a REST API and serves the chat web UI.
"""

import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.config.settings import settings
from src.embeddings.embedder import create_embeddings
from src.ingestion.pipeline import IngestionPipeline
from src.llm import create_llm_provider
from src.memory.history import create_memory_manager
from src.rag.chain import create_conversational_rag_chain
from src.rag.retriever import create_retriever
from src.vectordb.store import VectorStore

# ---------------------------------------------------------------------------
# Global chain state (lazy init, thread-safe)
# ---------------------------------------------------------------------------
_chain_lock = threading.Lock()
_chain = None
_llm_provider = None
_vector_store = None


def _get_chain():
    """Lazily build and cache the RAG chain (thread-safe)."""
    global _chain, _llm_provider, _vector_store

    if _chain is not None:
        return _chain, _llm_provider, _vector_store

    with _chain_lock:
        if _chain is not None:
            return _chain, _llm_provider, _vector_store

        llm_provider = create_llm_provider(settings)
        llm = llm_provider.get_chat_model()
        embeddings = create_embeddings(settings)

        vector_store = VectorStore(settings, embeddings)
        chroma = vector_store.create_or_load()
        retriever = create_retriever(chroma, k=settings.retrieval_k)

        chain = create_conversational_rag_chain(
            retriever=retriever,
            llm=llm,
            memory_file=str(settings.memory_file),
        )

        _chain = chain
        _llm_provider = llm_provider
        _vector_store = vector_store

        return _chain, _llm_provider, _vector_store


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    session_id: str = "default"


class Source(BaseModel):
    document: str
    preview: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    model: str
    session_id: str


class ClearRequest(BaseModel):
    session_id: str = "default"


class StatusResponse(BaseModel):
    llm_provider: str
    llm_model: str
    embedding_provider: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    collection_name: str
    document_count: int
    persist_dir: str


class MessageResponse(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Knowledge Q&A Agent",
        description="Company Internal Knowledge Q&A Agent — Web UI",
        version="0.1.0",
    )

    # ---- Static file: favicon (prevent 404) ----
    @app.get("/favicon.ico")
    async def favicon():
        return HTMLResponse("")

    # ---- Serve frontend ----
    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve the chat web UI."""
        templates_dir = Path(__file__).parent / "templates"
        html_path = templates_dir / "index.html"
        if not html_path.exists():
            raise HTTPException(status_code=404, detail="Frontend not found")
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    # ---- API: Ask question ----
    @app.post("/api/ask", response_model=AskResponse)
    async def api_ask(req: AskRequest):
        """Ask a question against the knowledge base."""
        try:
            chain, llm_provider, _ = _get_chain()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize chain: {e}")

        # run_in_executor would be ideal, but chain.invoke is sync and
        # uses requests under the hood. For simplicity, run it directly.
        try:
            result = chain.invoke(
                {"input": req.question},
                config={"configurable": {"session_id": req.session_id}},
            )
        except Exception as e:
            error_msg = str(e)
            if "sk-" in error_msg.lower():
                error_msg = "An API error occurred. Check your API key configuration."
            raise HTTPException(status_code=500, detail=error_msg)

        answer = result.get("answer", "No answer generated.")
        context_docs = result.get("context", [])

        # Deduplicate sources
        seen = set()
        sources: list[Source] = []
        for doc in context_docs:
            source = doc.metadata.get("source", "unknown")
            if source not in seen:
                seen.add(source)
                preview = doc.page_content[:120].replace("\n", " ") + "..."
                sources.append(Source(document=source, preview=preview))

        return AskResponse(
            answer=answer,
            sources=sources,
            model=llm_provider.get_model_name(),
            session_id=req.session_id,
        )

    # ---- API: Knowledge base status ----
    @app.get("/api/status", response_model=StatusResponse)
    async def api_status():
        """Get knowledge base statistics and configuration."""
        try:
            _, llm_provider, vector_store = _get_chain()
            stats = vector_store.get_stats()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return StatusResponse(
            llm_provider=settings.llm_provider,
            llm_model=llm_provider.get_model_name(),
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            retrieval_k=settings.retrieval_k,
            collection_name=stats["collection_name"],
            document_count=stats["document_count"],
            persist_dir=stats["persist_dir"],
        )

    # ---- API: Clear session ----
    @app.post("/api/clear", response_model=MessageResponse)
    async def api_clear(req: ClearRequest):
        """Clear conversation history for a session."""
        memory_manager = create_memory_manager(str(settings.memory_file))
        memory_manager.clear_session(req.session_id)
        return MessageResponse(
            message=f"Cleared conversation history for session: {req.session_id}"
        )

    # ---- API: Ingest documents ----
    @app.post("/api/ingest", response_model=MessageResponse)
    async def api_ingest():
        """Re-ingest documents from the docs directory."""
        try:
            pipeline = IngestionPipeline(settings)
            docs_dir = settings.docs_dir.resolve()
            if not docs_dir.exists():
                raise HTTPException(status_code=400, detail=f"Docs directory not found: {docs_dir}")

            supported = {".pdf", ".txt", ".md", ".markdown", ".docx"}
            files = [f for f in docs_dir.rglob("*") if f.is_file() and f.suffix.lower() in supported]
            if not files:
                raise HTTPException(
                    status_code=400,
                    detail=f"No supported files found in {docs_dir}",
                )

            chunk_count = pipeline.ingest_directory(docs_dir)
            return MessageResponse(
                message=f"Ingested {len(files)} files ({chunk_count} chunks) into the knowledge base."
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app
