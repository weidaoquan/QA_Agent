"""Chat message history with JSON file persistence."""

import json
from pathlib import Path
from typing import Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict


class JsonChatMessageHistory(BaseChatMessageHistory):
    """Chat message history backed by a JSON file.

    Each session's messages are stored as a JSON array of message dicts.
    Supports multiple sessions via session_id — each session gets its
    own file: <prefix>_<session_id>.json
    """

    def __init__(self, file_path: Path) -> None:
        """Initialize the history store.

        Args:
            file_path: Path to the JSON file for storing messages.
        """
        self._file_path = Path(file_path)
        self._messages: list[BaseMessage] = []
        self._load()

    def _load(self) -> None:
        """Load messages from the JSON file if it exists."""
        if self._file_path.exists():
            try:
                raw = self._file_path.read_text(encoding="utf-8")
                data = json.loads(raw)
                self._messages = messages_from_dict(data)
            except (json.JSONDecodeError, KeyError):
                # Corrupted or empty file — start fresh
                self._messages = []

    def _save(self) -> None:
        """Persist messages to the JSON file."""
        data = messages_to_dict(self.messages)
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @property
    def messages(self) -> list[BaseMessage]:
        """Return all messages in the history."""
        return self._messages

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add messages to the history and persist.

        Args:
            messages: List of BaseMessage objects to add.
        """
        self._messages.extend(messages)
        self._save()

    def clear(self) -> None:
        """Remove all messages and delete the backing file."""
        self._messages.clear()
        if self._file_path.exists():
            self._file_path.unlink()


class MemoryManager:
    """Manages chat history sessions across the application.

    Each session is identified by a session_id and backed by a
    separate JSON file.
    """

    def __init__(self, base_path: str = "conversation_history.json") -> None:
        """Initialize the memory manager.

        Args:
            base_path: Base filename for history storage. Session files
                       are derived as <stem>_<session_id>.json
        """
        base = Path(base_path)
        self._base_stem = base.stem
        self._base_dir = base.parent
        self._sessions: dict[str, JsonChatMessageHistory] = {}

    def _session_path(self, session_id: str) -> Path:
        """Build the file path for a given session."""
        filename = f"{self._base_stem}_{session_id}.json"
        return (self._base_dir / filename).resolve()

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create a chat history for the given session.

        Args:
            session_id: Unique identifier for the conversation session.

        Returns:
            JsonChatMessageHistory instance for the session.
        """
        if session_id not in self._sessions:
            file_path = self._session_path(session_id)
            self._sessions[session_id] = JsonChatMessageHistory(file_path)
        return self._sessions[session_id]

    def clear_session(self, session_id: str) -> None:
        """Clear the chat history for a specific session.

        Args:
            session_id: The session to clear.
        """
        if session_id in self._sessions:
            self._sessions[session_id].clear()
            del self._sessions[session_id]


# Module-level singleton factory for convenience
_memory_manager: Optional[MemoryManager] = None


def create_memory_manager(memory_file: str = "conversation_history.json") -> MemoryManager:
    """Factory: create or return the singleton MemoryManager.

    Args:
        memory_file: Path to the conversation history JSON file.

    Returns:
        MemoryManager instance.
    """
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(memory_file)
    return _memory_manager
