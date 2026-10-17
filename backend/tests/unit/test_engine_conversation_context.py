"""Tests for the context engine's conversation-context helper.

The helper previously read a key that the conversation service never
returns ("older_message_count"), so every assembled context reported
zero older messages. These tests pin the real window keys.
"""

import app.context_engine.conversation_service as conversation_service
import app.context_engine.database as database
import app.context_engine.summary_service as summary_service
from app.context_engine.engine import _get_conversation_context


class _FakeSession:
    """Stand-in for a SQLAlchemy session used as a context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        return False


def test_conversation_context_reports_the_real_window_counts(monkeypatch):
    monkeypatch.setattr(database, "SessionLocal", _FakeSession)

    monkeypatch.setattr(
        conversation_service,
        "get_conversation_message_window",
        lambda **_kwargs: {
            "conversation_id": "conversation-1",
            "total_messages": 10,
            "recent_count": 4,
            "older_count": 6,
            "older_messages": [],
            "recent_messages": [
                {"role": "user", "content": "What did we decide?"}
            ],
        },
    )

    monkeypatch.setattr(
        summary_service,
        "get_or_update_summary",
        lambda **_kwargs: "Earlier messages were summarized.",
    )

    context = _get_conversation_context(
        conversation_id="conversation-1",
        query="What did we decide?",
    )

    assert context["summary"] == "Earlier messages were summarized."
    assert context["total_messages"] == 10
    assert context["older_count"] == 6
    assert context["recent_count"] == 4
    assert context["recent_messages"] == [
        {"role": "user", "content": "What did we decide?"}
    ]


def test_conversation_context_defaults_to_zero_without_a_window(monkeypatch):
    monkeypatch.setattr(database, "SessionLocal", _FakeSession)
    monkeypatch.setattr(
        conversation_service,
        "get_conversation_message_window",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        summary_service,
        "get_or_update_summary",
        lambda **_kwargs: "",
    )

    context = _get_conversation_context(
        conversation_id="conversation-1",
        query="Anything",
    )

    assert context == {
        "summary": "",
        "recent_messages": [],
        "total_messages": 0,
        "older_count": 0,
        "recent_count": 0,
    }
