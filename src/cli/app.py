"""CLI application entry point for the Knowledge Q&A Agent."""

import typer

from src.cli.commands import (
    ask_command,
    clear_command,
    ingest_command,
    status_command,
)

app = typer.Typer(
    name="knowledge-qa",
    help="Company Internal Knowledge Q&A Agent — ask questions against your document base.",
    add_completion=False,
)


@app.command()
def web(
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        "-h",
        help="Host to bind the web server to.",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Port to bind the web server to.",
    ),
) -> None:
    """Launch the web UI for the Knowledge Q&A Agent.

    Starts a FastAPI server with a chat interface at http://{host}:{port}.
    """
    import uvicorn

    from src.web.app import create_app

    fastapi_app = create_app()

    print(f"\n  ╔══════════════════════════════════════════════════╗")
    print(f"  ║   Knowledge Q&A — Web UI                        ║")
    print(f"  ║                                                  ║")
    print(f"  ║   Open: http://{host}:{port}" + " " * (36 - len(f"{host}:{port}")) + "║")
    print(f"  ╚══════════════════════════════════════════════════╝\n")

    uvicorn.run(fastapi_app, host=host, port=port, log_level="info")


@app.command()
def ingest(
    docs_dir: str = typer.Option(
        None,
        "--docs-dir",
        "-d",
        help="Directory containing documents to ingest. Defaults to docs/",
    ),
    file: str = typer.Option(
        None,
        "--file",
        "-f",
        help="Ingest a single file instead of a directory.",
    ),
) -> None:
    """Index documents into the knowledge base.

    Scans the docs/ directory for PDF, Markdown, and text files,
    splits them into chunks, creates embeddings, and stores them
    in the vector database for later retrieval.
    """
    ingest_command(docs_dir=docs_dir, file=file)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question about company knowledge."),
    session_id: str = typer.Option(
        "default",
        "--session-id",
        "-s",
        help="Session identifier for conversation memory.",
    ),
) -> None:
    """Ask a question against the knowledge base.

    The agent retrieves relevant documents and generates an answer.
    Conversation history is maintained per session.
    """
    ask_command(question=question, session_id=session_id)


@app.command()
def status() -> None:
    """Show knowledge base statistics and configuration."""
    status_command()


@app.command()
def clear(
    session_id: str = typer.Option(
        "default",
        "--session-id",
        "-s",
        help="Session to clear history for.",
    ),
) -> None:
    """Clear conversation history for a session."""
    clear_command(session_id=session_id)


def main() -> None:
    """Entry point: launch the CLI application."""
    app()


if __name__ == "__main__":
    main()
