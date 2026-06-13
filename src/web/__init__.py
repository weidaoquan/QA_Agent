"""Web server module for the Knowledge Q&A Agent.

Provides a FastAPI application with REST API endpoints and a chat web UI.
"""

from src.web.app import create_app

__all__ = ["create_app"]
