"""Contract tests for the conversation summary service.

These run against an in-memory SQLite database and a patched summarizer,
so they need neither PostgreSQL nor an LLM.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.context_engine.database import Base

# Import the models so every table is registered on Base.metadata.
import app.context_engine.models  # noqa: F401

from app.context_engine import summary_service
from app.context_engine.conversation_service import (
    create_conversation,
    save_message,
)


@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    Base.metadata.create_all(bind=engine)

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )
    session = session_factory()

    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def seed_conversation(db, count=8):
    conversation = create_conversation(db, title="Summary Test")

    for number in range(1, count + 1):
        save_message(
            db=db,
            conversation_id=conversation.id,
            role="user" if number % 2 else "assistant",
            content=f"Test message {number}",
        )

    return conversation


def test_summary_is_returned_as_a_string(db, monkeypatch):
    conversation = seed_conversation(db)

    monkeypatch.setattr(
        summary_service,
        "generate_query_aware_summary",
        lambda older_messages, current_query: "summarized",
    )

    result = summary_service.get_or_update_summary(
        db=db,
        conversation_id=conversation.id,
        current_query="How is history stored?",
        n=2,
    )

    # The Context Engine expects a plain string, not a dictionary.
    assert isinstance(result, str)
    assert result == "summarized"


def test_repeated_query_reuses_the_stored_summary(db, monkeypatch):
    conversation = seed_conversation(db)

    calls = []

    def fake_summary(older_messages, current_query):
        calls.append(current_query)
        return "summarized"

    monkeypatch.setattr(
        summary_service,
        "generate_query_aware_summary",
        fake_summary,
    )

    first = summary_service.get_or_update_summary(
        db=db,
        conversation_id=conversation.id,
        current_query="Same question",
        n=2,
    )
    second = summary_service.get_or_update_summary(
        db=db,
        conversation_id=conversation.id,
        current_query="Same question",
        n=2,
    )

    assert first == second == "summarized"
    # The summarizer must run only once; the second call is reused.
    assert len(calls) == 1


def test_no_older_messages_returns_empty_string(db, monkeypatch):
    conversation = seed_conversation(db, count=2)

    monkeypatch.setattr(
        summary_service,
        "generate_query_aware_summary",
        lambda older_messages, current_query: "should not run",
    )

    result = summary_service.get_or_update_summary(
        db=db,
        conversation_id=conversation.id,
        current_query="Anything",
        n=2,
    )

    assert result == ""
