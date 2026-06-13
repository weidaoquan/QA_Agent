"""Tests for chat history memory management."""

import json
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from src.memory.history import JsonChatMessageHistory, MemoryManager


class TestJsonChatMessageHistory:
    """Tests for JSON-backed chat message history."""

    def test_add_and_retrieve_messages(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = Path(f.name)

        try:
            history = JsonChatMessageHistory(tmp_path)
            history.add_messages([HumanMessage(content="Hello")])
            history.add_messages([AIMessage(content="Hi there!")])

            assert len(history.messages) == 2
            assert isinstance(history.messages[0], HumanMessage)
            assert history.messages[0].content == "Hello"
            assert isinstance(history.messages[1], AIMessage)
            assert history.messages[1].content == "Hi there!"
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_persistence_across_instances(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = Path(f.name)

        try:
            # Write messages with first instance
            history1 = JsonChatMessageHistory(tmp_path)
            history1.add_messages([HumanMessage(content="Persistent?")])

            # Load with second instance
            history2 = JsonChatMessageHistory(tmp_path)
            assert len(history2.messages) == 1
            assert history2.messages[0].content == "Persistent?"
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_clear_removes_messages(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = Path(f.name)

        try:
            history = JsonChatMessageHistory(tmp_path)
            history.add_messages([HumanMessage(content="Temp message")])
            assert len(history.messages) == 1

            history.clear()
            assert len(history.messages) == 0
            assert not tmp_path.exists()
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_corrupted_file_starts_fresh(self):
        tmp_path = Path(tempfile.mktemp(suffix=".json"))
        tmp_path.write_text("this is not valid json{", encoding="utf-8")

        try:
            history = JsonChatMessageHistory(tmp_path)
            assert history.messages == []
            # After adding, the corrupted file should be overwritten
            history.add_messages([HumanMessage(content="Fresh start")])
            assert len(history.messages) == 1
        finally:
            tmp_path.unlink(missing_ok=True)


class TestMemoryManager:
    """Tests for the multi-session memory manager."""

    def test_session_isolation(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            base_path = f.name
            # Manager will use this as the base filename stem pattern

        try:
            # Use a custom base path in a temp dir
            tmp_dir = Path(tempfile.mkdtemp())
            mgr = MemoryManager(str(tmp_dir / "chat_history.json"))

            # Get two different sessions
            session_a = mgr.get_session_history("hr")
            session_b = mgr.get_session_history("engineering")

            session_a.add_messages([HumanMessage(content="HR question")])
            session_b.add_messages([HumanMessage(content="Tech question")])

            assert len(session_a.messages) == 1
            assert len(session_b.messages) == 1
            assert session_a.messages[0].content == "HR question"
            assert session_b.messages[0].content == "Tech question"

            # Session files should be different
            assert mgr._session_path("hr") != mgr._session_path("engineering")
        finally:
            import shutil

            shutil.rmtree(tmp_dir)

    def test_clear_session(self):
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            mgr = MemoryManager(str(tmp_dir / "chat_history.json"))
            session = mgr.get_session_history("test")
            session.add_messages([HumanMessage(content="Will be cleared")])
            assert len(session.messages) == 1

            mgr.clear_session("test")
            # After clearing, getting the session again should have no messages
            new_session = mgr.get_session_history("test")
            assert len(new_session.messages) == 0
        finally:
            import shutil

            shutil.rmtree(tmp_dir)
