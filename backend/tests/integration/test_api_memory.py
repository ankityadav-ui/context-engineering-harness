"""Tests for the long-term memory endpoints.

The storage dependency is replaced by an in-memory SQLite database, so no
PostgreSQL connection is opened.
"""


def test_create_list_and_delete_memory(api_client):
    created = api_client.post(
        "/memory",
        json={
            "user_id": "user-1",
            "memory_type": "preference",
            "content": "Prefers concise answers.",
            "importance": 4,
        },
    )

    assert created.status_code == 201

    memory = created.json()

    assert memory["user_id"] == "user-1"
    assert memory["memory_type"] == "preference"
    assert memory["importance"] == 4
    assert memory["conversation_id"] is None
    assert memory["created_at"]

    listing = api_client.get(
        "/memory",
        params={"user_id": "user-1"},
    )

    assert listing.status_code == 200
    assert [
        item["id"] for item in listing.json()["memories"]
    ] == [memory["id"]]

    deleted = api_client.delete(
        f"/memory/{memory['id']}",
        params={"user_id": "user-1"},
    )

    assert deleted.status_code == 200
    deleted_body = deleted.json()
    assert deleted_body["deleted"] is True
    assert deleted_body["memory_id"] == memory["id"]
    assert "dry_run" in deleted_body
    assert deleted_body["dry_run"] is False
    assert "other_memories_preserved" in deleted_body
    assert deleted_body["other_memories_preserved"] is True
    assert "remaining_memory_count" in deleted_body
    assert deleted_body["remaining_memory_count"] == 0

    assert (
        api_client.get(
            "/memory",
            params={"user_id": "user-1"},
        ).json()["memories"]
        == []
    )


def test_memories_are_scoped_to_one_user(api_client):
    api_client.post(
        "/memory",
        json={
            "user_id": "user-a",
            "memory_type": "fact",
            "content": "Belongs to A.",
        },
    )
    api_client.post(
        "/memory",
        json={
            "user_id": "user-b",
            "memory_type": "fact",
            "content": "Belongs to B.",
        },
    )

    listing = api_client.get(
        "/memory",
        params={"user_id": "user-a"},
    ).json()

    assert [item["content"] for item in listing["memories"]] == [
        "Belongs to A."
    ]


def test_create_memory_rejects_an_unsupported_type(api_client):
    response = api_client.post(
        "/memory",
        json={
            "user_id": "user-1",
            "memory_type": "secrets",
            "content": "Anything.",
        },
    )

    assert response.status_code == 400
    assert "Unsupported memory type" in response.json()["detail"]


def test_create_memory_rejects_an_unknown_conversation(api_client):
    response = api_client.post(
        "/memory",
        json={
            "user_id": "user-1",
            "memory_type": "fact",
            "content": "Anything.",
            "conversation_id": "missing-conversation",
        },
    )

    assert response.status_code == 400
    assert "Conversation not found" in response.json()["detail"]


def test_create_memory_validates_importance(api_client):
    response = api_client.post(
        "/memory",
        json={
            "user_id": "user-1",
            "memory_type": "fact",
            "content": "Anything.",
            "importance": 9,
        },
    )

    assert response.status_code == 422


def test_delete_unknown_memory_returns_404(api_client):
    response = api_client.delete(
        "/memory/12345",
        params={"user_id": "user-1"},
    )

    assert response.status_code == 404


def test_memory_listing_requires_a_user_id(api_client):
    assert api_client.get("/memory").status_code == 422
