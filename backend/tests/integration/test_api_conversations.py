"""Tests for the conversation endpoints.

The storage dependency is replaced by an in-memory SQLite database, and
the missing-configuration test only checks that a clear error is raised;
no PostgreSQL connection is opened.
"""

from fastapi.testclient import TestClient

from app.context_engine import database
from app.context_engine.conversation_service import (
    create_conversation,
    save_message,
)
from app.main import app


def test_create_and_list_conversations(api_client):
    created = api_client.post(
        "/conversations",
        json={"title": "  First chat  "},
    )

    assert created.status_code == 201

    body = created.json()

    assert body["title"] == "First chat"
    assert body["id"]
    assert body["created_at"]

    listing = api_client.get("/conversations")

    assert listing.status_code == 200

    conversations = listing.json()["conversations"]

    assert [item["id"] for item in conversations] == [body["id"]]
    assert conversations[0]["message_count"] == 0


def test_create_conversation_without_a_title(api_client):
    response = api_client.post(
        "/conversations",
        json={"title": "   "},
    )

    assert response.status_code == 201
    assert response.json()["title"] is None


def test_conversation_messages_are_returned_in_order(
    api_client,
    sqlite_session_factory,
):
    with sqlite_session_factory() as db:
        conversation = create_conversation(db, title="Seeded")

        for number in range(1, 6):
            save_message(
                db=db,
                conversation_id=conversation.id,
                role="user",
                content=f"Message {number}",
            )

        conversation_id = conversation.id

    response = api_client.get(
        f"/conversations/{conversation_id}/messages",
        params={"limit": 3},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["conversation_id"] == conversation_id
    assert body["title"] == "Seeded"
    assert [
        message["content"] for message in body["messages"]
    ] == ["Message 3", "Message 4", "Message 5"]

    # The seeded conversations are listed newest first.
    listing = api_client.get("/conversations").json()["conversations"]

    assert listing[0]["message_count"] == 5


def test_messages_for_an_unknown_conversation_return_404(api_client):
    response = api_client.get(
        "/conversations/does-not-exist/messages"
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_conversation_pagination_is_validated(api_client):
    assert (
        api_client.get(
            "/conversations",
            params={"limit": 0},
        ).status_code
        == 422
    )

    assert (
        api_client.get(
            "/conversations",
            params={"offset": -1},
        ).status_code
        == 422
    )


def test_conversations_require_a_configured_database(monkeypatch):
    # No dependency override here: the real dependency must report the
    # missing configuration clearly instead of crashing the request.
    monkeypatch.setattr(database, "_engine", None)
    monkeypatch.setattr(database, "_session_factory", None)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with TestClient(app) as client:
        response = client.get("/conversations")

    assert response.status_code == 503
    assert "DATABASE_URL" in response.json()["detail"]


def test_health_still_works_without_a_database(monkeypatch):
    monkeypatch.setattr(database, "_engine", None)
    monkeypatch.setattr(database, "_session_factory", None)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
