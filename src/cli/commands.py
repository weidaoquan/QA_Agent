"""CLI command implementations for the Knowledge Q&A Agent."""

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config.settings import settings
from src.embeddings.embedder import create_embeddings
from src.ingestion.pipeline import IngestionPipeline
from src.llm import create_llm_provider
from src.memory.history import create_memory_manager
from src.rag.chain import create_conversational_rag_chain
from src.rag.retriever import create_retriever
from src.vectordb.store import VectorStore

console = Console()


def _build_chain():
    """Build and return a conversational RAG chain.

    This is the common setup used by the ask command.
    """
    # Provider & embeddings
    llm_provider = create_llm_provider(settings)
    llm = llm_provider.get_chat_model()
    embeddings = create_embeddings(settings)

    # Vector store & retriever
    vector_store = VectorStore(settings, embeddings)
    chroma = vector_store.create_or_load()
    retriever = create_retriever(chroma, k=settings.retrieval_k)

    # Conversational RAG chain
    chain = create_conversational_rag_chain(
        retriever=retriever,
        llm=llm,
        memory_file=str(settings.memory_file),
    )

    return chain, llm_provider


def ingest_command(
    docs_dir: Optional[str] = None,
    file: Optional[str] = None,
) -> None:
    """Ingest documents into the knowledge base.

    Scans the docs/ directory (or a custom directory) for supported
    files and indexes them into the vector store.

    Usage examples:
        knowledge-qa ingest
        knowledge-qa ingest --docs-dir ./my_docs
        knowledge-qa ingest --file ./docs/policy.pdf
    """
    pipeline = IngestionPipeline(settings)
    file_count = 0
    chunk_count = 0

    if file:
        file_path = Path(file).resolve()
        if not file_path.exists():
            console.print(f"[red]Error:[/red] File not found: {file_path}")
            return
        chunk_count = pipeline.ingest_file(file_path)
    else:
        docs_path = Path(docs_dir).resolve() if docs_dir else settings.docs_dir.resolve()
        if not docs_path.exists():
            console.print(f"[red]Error:[/red] Docs directory not found: {docs_path}")
            return

        # Count files before ingesting
        supported = {".pdf", ".txt", ".md", ".markdown", ".docx"}
        files = [f for f in docs_path.rglob("*") if f.is_file() and f.suffix.lower() in supported]
        file_count = len(files)

        if file_count == 0:
            console.print(f"[yellow]Warning:[/yellow] No supported files found in {docs_path}")
            console.print("Supported formats: PDF, Markdown (.md), Text (.txt), Word (.docx)")
            return

        chunk_count = pipeline.ingest_directory(docs_path)

    # Print summary
    panel_content = (
        f"Files processed: {file_count or 1}\n"
        f"Chunks ingested: {chunk_count}\n"
        f"Status: [green]OK Complete[/green]"
    )
    console.print(Panel(panel_content, title="Ingestion Summary", border_style="green"))


def ask_command(
    question: str,
    session_id: str = "default",
) -> None:
    """Ask a question against the knowledge base.

    The agent retrieves relevant documents and generates an answer
    based on the company's internal knowledge. Conversation history
    is maintained across questions within the same session.

    Usage examples:
        knowledge-qa ask "What is the remote work policy?"
        knowledge-qa ask "What holidays do we have?" --session-id hr
    """
    chain, llm_provider = _build_chain()

    with console.status("[bold green]Searching knowledge base...[/bold green]"):
        try:
            result = chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": session_id}},
            )
        except Exception as e:
            error_msg = str(e)
            # Redact API keys that might appear in error messages
            if "sk-" in error_msg.lower():
                error_msg = "An API error occurred. Check your API key configuration."
            console.print(f"[red]Error:[/red] {error_msg}")
            return

    answer = result.get("answer", "No answer generated.")
    context_docs = result.get("context", [])

    # Print the answer
    console.print()
    console.print(Panel(answer, title="Answer", border_style="blue"))

    # Print source references
    if context_docs:
        table = Table(title="Sources", show_header=True, header_style="bold cyan")
        table.add_column("#", style="dim", width=4)
        table.add_column("Document", style="green")
        table.add_column("Preview", style="dim", max_width=80)

        seen_sources = set()
        source_num = 0
        for doc in context_docs:
            source = doc.metadata.get("source", "unknown")
            if source not in seen_sources:
                seen_sources.add(source)
                source_num += 1
                preview = doc.page_content[:100].replace("\n", " ") + "..."
                table.add_row(str(source_num), source, preview)

        console.print(table)

    console.print(f"[dim]Model: {llm_provider.get_model_name()} | Session: {session_id}[/dim]")
    console.print()


def status_command() -> None:
    """Show knowledge base statistics.

    Displays the number of indexed documents, chunks,
    and the current configuration.
    """
    # Show config section first (no API key needed)
    # Determine the actual LLM model in use
    if settings.llm_provider == "openai":
        llm_model = settings.openai_model
    elif settings.llm_provider == "anthropic":
        llm_model = settings.anthropic_model
    else:
        llm_model = settings.deepseek_model

    config_table = Table(title="Configuration", show_header=False)
    config_table.add_column("Setting", style="bold cyan")
    config_table.add_column("Value", style="green")
    config_table.add_row("LLM Provider", settings.llm_provider)
    config_table.add_row("LLM Model", llm_model)
    if settings.llm_provider == "deepseek":
        config_table.add_row("DeepSeek API", settings.deepseek_base_url)
    config_table.add_row("Embedding Provider", settings.embedding_provider)
    config_table.add_row("Embedding Model", settings.embedding_model)
    config_table.add_row("Chunk Size", str(settings.chunk_size))
    config_table.add_row("Chunk Overlap", str(settings.chunk_overlap))
    config_table.add_row("Retrieval Top-K", str(settings.retrieval_k))
    config_table.add_row("Docs Directory", str(settings.docs_dir.resolve()))
    console.print(config_table)

    # Try to connect to the vector store
    try:
        embeddings = create_embeddings(settings)
        vector_store = VectorStore(settings, embeddings)
        stats = vector_store.get_stats()

        kb_table = Table(title="Knowledge Base", show_header=False)
        kb_table.add_column("Metric", style="bold cyan")
        kb_table.add_column("Value", style="green")
        kb_table.add_row("Collection", stats["collection_name"])
        kb_table.add_row("Indexed Chunks", str(stats["document_count"]))
        kb_table.add_row("Storage", stats["persist_dir"])
        console.print(kb_table)
    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "credentials" in error_msg.lower():
            console.print(
                "\n[yellow]API key not configured.[/yellow] "
                "Create a [bold].env[/bold] file with your API keys to connect to the vector store.\n"
                "Run [bold]cp .env.example .env[/bold] and edit the file."
            )
        else:
            console.print(f"[yellow]Knowledge base status unavailable: {error_msg}[/yellow]")


def clear_command(session_id: str = "default") -> None:
    """Clear conversation history for a session.

    Resets the chat memory so the next question starts fresh.

    Usage examples:
        knowledge-qa clear
        knowledge-qa clear --session-id hr
    """
    memory_manager = create_memory_manager(str(settings.memory_file))
    memory_manager.clear_session(session_id)
    console.print(f"[green]OK[/green] Cleared conversation history for session: [bold]{session_id}[/bold]")
