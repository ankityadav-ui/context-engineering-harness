"""Tests for persistent chat conversations (POST /chat and /chat/stream).

Storage is the shared in-memory SQLite database from conftest, and the
Gemini client is replaced with an in-memory double, so no PostgreSQL
server, Pinecone index, Neo4j instance, or LLM is contacted.

Covered here:
- creating a conversation and continuing it with its id,
- user + assistant persistence (JSON and streaming paths),
- chronological ordering of stored messages,
- context assembly reading persisted history (no second copy of the
  conversation is injected; the authoritative assemble_context path
  is used),
- failed generation never persisting a misleading assistant answer,
- streaming persisting the FINAL response once (never per chunk),
- invalid conversation ids and unknown conversations,
- session-ownership isolation,
- deletion and clearing of a conversation,
- the stateless (no conversation_id) contract staying intact.
"""

import pytest
from sqlalchemy.exc import SQLAlchemyError

import app.llm as llm_module
import app.main as main_module
from app.context_engine import database
from app.context_engine.conversation_service import (
    clear_conversation,
    create_conversation,
    delete_conversation,
    save_message,
)


# ---------------------------------------------------------------- fakes


class FakeProvider:
    """Minimal in-memory Gemini chat double (text responses)."""

    def __init__(self, response="stored reply", fail=False):
        self.response = response
        self.fail = fail
        self.seen_messages = []

    def _response(self, content):
        response = type("ChatResponse", (), {})()
        response.message = type("Message", (), {})()
        response.message.content = content
        return response

    def invoke(self, messages):
        self.seen_messages.append(messages)

        if self.fail:
            raise RuntimeError("upstream exploded")

        return type("AIMessage", (), {"content": self.response})()

    def stream(self, messages):
        self.seen_messages.append(messages)
        if self.fail:
            raise RuntimeError("upstream exploded")
        yield type("AIMessageChunk", (), {"content": self.response})()

    @property
    def last_user_text(self):
        return "".join(
            message.content
            for message in self.seen_messages[-1]
            if message.type == "human"
        )


@pytest.fixture()
def fake_llm(monkeypatch):
    def install(response="stored reply", fail=False):
        provider = FakeProvider(response=response, fail=fail)
        monkeypatch.setattr(llm_module, "get_chat_model", lambda: provider)
        llm_module.reset_client()
        return provider

    return install


# The isolated_assembly fixture (in-memory SQLite for the conversation
# branch, off live Pinecone/Neo4j/summarizer) comes from conftest.py.


# ------------------------------------------------------- persistence


def test_chat_with_conversation_persists_both_turns(
    api_client, fake_llm, isolated_assembly
):
    provider = fake_llm(response="The policy allows 30 days.")

    created = api_client.post(
        "/conversations", json={"title": "Policy"}
    ).json()

    response = api_client.post(
        "/chat",
        json={
            "message": "What is the return policy?",
            "conversation_id": created["id"],
        },
    )

    assert response.status_code == 200
    body = response.json()

    # Additive keys only when a conversation is used; the stateless
    # contract is covered separately.
    assert body["response"] == "The policy allows 30 days."
    assert body["conversation_id"] == created["id"]

    messages = api_client.get(
        f"/conversations/{created['id']}/messages",
        params={"order": "chronological"},
    ).json()["messages"]

    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", "What is the return policy?"),
        ("assistant", "The policy allows 30 days."),
    ]
    assert all(m["created_at"] for m in messages)
    assert len(provider.seen_messages) == 1


def test_continuing_a_conversation_feeds_persisted_history_into_context(
    api_client, fake_llm, isolated_assembly
):
    provider = fake_llm(response="It was Acme Corp.")

    created = api_client.post(
        "/conversations", json={}
    ).json()

    for question in (
        "Who is the vendor?",
        "And the contract duration?",
    ):
        first = api_client.post(
            "/chat",
            json={
                "message": question,
                "conversation_id": created["id"],
            },
        )
        assert first.status_code == 200

    # The third turn's prompt must contain the persisted history via
    # the authoritative context assembly (RECENT CONVERSATION block),
    # not a second hand-built copy.
    third = api_client.post(
        "/chat",
        json={
            "message": "Who is the vendor again?",
            "conversation_id": created["id"],
        },
    )

    assert third.status_code == 200

    prompt = provider.last_user_text

    assert "RECENT CONVERSATION:" in prompt
    assert "Who is the vendor?" in prompt
    # The canned assistant reply of the first two turns.
    assert "It was Acme Corp." in prompt
    assert "Who is the vendor again?" in prompt

    # Chronological order of the transcript.
    messages = api_client.get(
        f"/conversations/{created['id']}/messages",
        params={"limit": 10, "order": "chronological"},
    ).json()["messages"]

    assert [m["role"] for m in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_older_messages_are_summarized_and_latest_six_stay_recent(
    api_client,
    fake_llm,
    isolated_assembly,
    sqlite_session_factory,
):
    provider = fake_llm(response="ok")

    created = api_client.post("/conversations", json={}).json()

    # Seed 8 turns directly so the window splits (6 recent + 2 older).
    with sqlite_session_factory() as db:
        for index in range(1, 9):
            save_message(
                db=db,
                conversation_id=created["id"],
                role="user" if index % 2 else "assistant",
                content=f"turn-{index}",
            )

    response = api_client.post(
        "/chat",
        json={
            "message": "Where were we?",
            "conversation_id": created["id"],
        },
    )

    assert response.status_code == 200

    prompt = provider.last_user_text

    # Older messages are summarized through the real summary service
    # (the LLM call itself is deterministic in these tests).
    assert "CONVERSATION SUMMARY:" in prompt
    assert "Older messages summarized." in prompt

    # The latest six messages (turns 3-8) stay as recent context; the
    # two older ones (turns 1-2) are only present as the summary.
    recent_block = prompt.split("RECENT CONVERSATION:")[1]
    assert "turn-1" not in recent_block
    assert "turn-2" not in recent_block
    for index in range(3, 9):
        assert f"turn-{index}" in recent_block

    # The counts-only report matches the window.
    context_report = response.json()["conversation_context"]

    assert context_report["total_messages"] == 8
    assert context_report["older_count"] == 2
    assert context_report["recent_count"] == 6
    assert context_report["summary_present"] is True


def test_context_preview_reflects_persisted_history(
    api_client, fake_llm, isolated_assembly
):
    fake_llm()
    created = api_client.post("/conversations", json={}).json()

    api_client.post(
        "/chat",
        json={
            "message": "Remember the codename Falcon.",
            "conversation_id": created["id"],
        },
    )

    preview = api_client.post(
        "/context/preview",
        json={
            "query": "What is the codename?",
            "conversation_id": created["id"],
        },
    )

    assert preview.status_code == 200

    formatted = preview.json()["formatted_context"]

    assert "RECENT CONVERSATION:" in formatted
    assert "Remember the codename Falcon." in formatted

    conversation_sources = preview.json()["sources"]["conversation"]

    assert conversation_sources["total_messages"] == 2
    assert conversation_sources["recent_count"] == 2
    assert conversation_sources["older_count"] == 0


def test_failed_generation_keeps_user_message_only(
    api_client, fake_llm, isolated_assembly
):
    fake_llm(fail=True)

    created = api_client.post("/conversations", json={}).json()

    response = api_client.post(
        "/chat",
        json={
            "message": "This will fail",
            "conversation_id": created["id"],
        },
    )

    assert response.status_code == 503

    messages = api_client.get(
        f"/conversations/{created['id']}/messages",
        params={"order": "chronological"},
    ).json()["messages"]

    # The user turn is stored; no misleading assistant answer exists.
    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", "This will fail"),
    ]


def test_assistant_persistence_failure_rolls_back_to_503(
    api_client, fake_llm, isolated_assembly, monkeypatch
):
    fake_llm(response="answer")

    created = api_client.post("/conversations", json={}).json()

    calls = {"count": 0}
    real_save_message = main_module.save_message

    def flaky_save(db, conversation_id, role, content):
        calls["count"] += 1
        if role == "assistant":
            raise SQLAlchemyError("insert failed")
        return real_save_message(
            db=db,
            conversation_id=conversation_id,
            role=role,
            content=content,
        )

    monkeypatch.setattr(main_module, "save_message", flaky_save)

    response = api_client.post(
        "/chat",
        json={
            "message": "persist me",
            "conversation_id": created["id"],
        },
    )

    assert response.status_code == 503
    assert calls["count"] == 2

    # The failed transaction did not corrupt the database: only the
    # user message exists and the session was rolled back/closed.
    monkeypatch.setattr(main_module, "save_message", real_save_message)

    messages = api_client.get(
        f"/conversations/{created['id']}/messages",
        params={"order": "chronological"},
    ).json()["messages"]

    assert [m["role"] for m in messages] == ["user"]


# ------------------------------------------------------------ streaming


def read_sse_events(response):
    events = []
    name = None
    data_lines = []

    for raw_line in response.iter_lines():
        line = raw_line.rstrip("\r")

        if line == "":
            if data_lines or name:
                import json

                events.append(
                    (name, json.loads("".join(data_lines) or "{}"))
                )
            name = None
            data_lines = []
            continue

        if line.startswith("event:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())

    return events


def test_stream_persists_the_final_response_exactly_once(
    api_client, fake_llm, isolated_assembly
):
    provider = fake_llm(response="Hello world")

    created = api_client.post("/conversations", json={}).json()

    with api_client.stream(
        "POST",
        "/chat/stream",
        json={
            "message": "Stream this",
            "conversation_id": created["id"],
        },
    ) as response:
        assert response.status_code == 200
        events = read_sse_events(response)

    done = [data for event, data in events if event == "done"]
    assert len(done) == 1
    assert done[0]["response"] == "Hello world"
    assert done[0]["conversation_id"] == created["id"]

    messages = api_client.get(
        f"/conversations/{created['id']}/messages",
        params={"order": "chronological"},
    ).json()["messages"]

    # Exactly two messages: no duplicate assistant row per SSE chunk.
    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", "Stream this"),
        ("assistant", "Hello world"),
    ]
    assert len(provider.seen_messages) == 1


def test_stream_failure_persists_only_the_user_turn(
    api_client, fake_llm, isolated_assembly
):
    fake_llm(fail=True)

    created = api_client.post("/conversations", json={}).json()

    with api_client.stream(
        "POST",
        "/chat/stream",
        json={
            "message": "Stream failure",
            "conversation_id": created["id"],
        },
    ) as response:
        events = read_sse_events(response)

    assert [event for event, _ in events if event == "error"]
    assert not [event for event, _ in events if event == "done"]

    messages = api_client.get(
        f"/conversations/{created['id']}/messages",
        params={"order": "chronological"},
    ).json()["messages"]

    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", "Stream failure"),
    ]


def test_stream_stateless_contract_is_unchanged(
    api_client, fake_llm
):
    fake_llm(response="plain")

    with api_client.stream(
        "POST", "/chat/stream", json={"message": "Hi"}
    ) as response:
        events = read_sse_events(response)

    done = [data for event, data in events if event == "done"]

    assert len(done) == 1
    assert done[0]["response"] == "plain"
    # No conversation was requested, so none is reported.
    assert "conversation_id" not in done[0]


# ------------------------------------------------ validation / isolation


def test_unknown_conversation_returns_404(api_client):
    response = api_client.post(
        "/chat",
        json={
            "message": "hello",
            "conversation_id": "11111111-2222-3333-4444-555555555555",
        },
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_malformed_conversation_id_returns_404(api_client):
    response = api_client.post(
        "/chat",
        json={"message": "hello", "conversation_id": "../../etc"},
    )

    assert response.status_code == 404


def test_session_ownership_isolates_conversations(api_client, fake_llm):
    fake_llm()
    created = api_client.post(
        "/conversations",
        json={"title": "Private", "session_id": "session-a"},
    ).json()

    assert created["session_id"] == "session-a"

    # Another session cannot continue the conversation.
    forbidden = api_client.post(
        "/chat",
        json={
            "message": "peek",
            "conversation_id": created["id"],
            "session_id": "session-b",
        },
    )

    assert forbidden.status_code == 404

    # The owner can.
    allowed = api_client.post(
        "/chat",
        json={
            "message": "mine",
            "conversation_id": created["id"],
            "session_id": "session-a",
        },
    )

    assert allowed.status_code == 200

    # Listing is scoped: session-b sees shared conversations only.
    listing_b = api_client.get(
        "/conversations", params={"session_id": "session-b"}
    ).json()["conversations"]

    assert created["id"] not in [item["id"] for item in listing_b]

    listing_a = api_client.get(
        "/conversations", params={"session_id": "session-a"}
    ).json()["conversations"]

    assert created["id"] in [item["id"] for item in listing_a]


def test_stateless_chat_works_without_storage(monkeypatch, fake_llm):
    fake_llm(response="stateless")

    monkeypatch.setattr(database, "_engine", None)
    monkeypatch.setattr(database, "_session_factory", None)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    app_dependency_overrides = dict(main_module.app.dependency_overrides)
    main_module.app.dependency_overrides.clear()

    try:
        from fastapi.testclient import TestClient

        with TestClient(main_module.app) as client:
            ok = client.post("/chat", json={"message": "Hi"})

            assert ok.status_code == 200
            assert ok.json()["response"] == "stateless"

            blocked = client.post(
                "/chat",
                json={
                    "message": "Hi",
                    "conversation_id": (
                        "11111111-2222-3333-4444-555555555555"
                    ),
                },
            )

            assert blocked.status_code == 503
            assert "DATABASE_URL" in blocked.json()["detail"]
    finally:
        main_module.app.dependency_overrides.update(
            app_dependency_overrides
        )


def test_stateless_chat_contract_keeps_original_keys(
    api_client, fake_llm
):
    fake_llm(response="plain")

    response = api_client.post("/chat", json={"message": "Hello"})

    assert response.status_code == 200
    assert set(response.json().keys()) == {
        "response",
        "system_prompt_active",
    }


# --------------------------------------------- delete / clear endpoints


def test_delete_conversation_removes_messages_and_summary(
    api_client, sqlite_session_factory, isolated_assembly
):
    created = api_client.post("/conversations", json={}).json()

    api_client.post(
        "/chat",
        json={
            "message": "delete me",
            "conversation_id": created["id"],
        },
    )

    # Store a summary row directly to prove cascade cleanup.
    with sqlite_session_factory() as db:
        from app.context_engine.models import ConversationSummary

        db.add(
            ConversationSummary(
                conversation_id=created["id"],
                summary_text="summary",
                covered_through_message_id=1,
                last_query="q",
            )
        )
        db.commit()

    deleted = api_client.delete(
        f"/conversations/{created['id']}"
    )

    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True

    follow_up = api_client.get(
        f"/conversations/{created['id']}/messages"
    )
    assert follow_up.status_code == 404

    with sqlite_session_factory() as db:
        from app.context_engine.models import (
            Conversation,
            ConversationSummary,
            Message,
        )

        assert (
            db.query(Conversation)
            .filter(Conversation.id == created["id"])
            .count()
            == 0
        )
        assert (
            db.query(Message)
            .filter(Message.conversation_id == created["id"])
            .count()
            == 0
        )
        assert (
            db.query(ConversationSummary)
            .filter(
                ConversationSummary.conversation_id == created["id"]
            )
            .count()
            == 0
        )


def test_delete_unknown_conversation_returns_404(api_client):
    response = api_client.delete(
        "/conversations/99999999-9999-9999-9999-999999999999"
    )

    assert response.status_code == 404


def test_clear_conversation_keeps_the_conversation_usable(
    api_client, fake_llm, isolated_assembly
):
    fake_llm(response="after clear")

    created = api_client.post("/conversations", json={}).json()

    api_client.post(
        "/chat",
        json={
            "message": "first",
            "conversation_id": created["id"],
        },
    )

    cleared = api_client.delete(
        f"/conversations/{created['id']}/messages"
    )

    assert cleared.status_code == 200
    assert cleared.json()["cleared"] is True
    assert cleared.json()["messages_deleted"] == 2

    # The conversation id remains valid for continued chats.
    again = api_client.post(
        "/chat",
        json={
            "message": "second",
            "conversation_id": created["id"],
        },
    )

    assert again.status_code == 200

    messages = api_client.get(
        f"/conversations/{created['id']}/messages",
        params={"order": "chronological"},
    ).json()["messages"]

    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", "second"),
        ("assistant", "after clear"),
    ]


def test_clear_is_denied_for_a_foreign_session(api_client):
    created = api_client.post(
        "/conversations", json={"session_id": "session-a"}
    ).json()

    denied = api_client.delete(
        f"/conversations/{created['id']}/messages",
        params={"session_id": "session-b"},
    )

    assert denied.status_code == 404


# --------------------------------------------------- service-level laws


def test_service_delete_and_clear_are_idempotent(
    sqlite_session_factory
):
    with sqlite_session_factory() as db:
        conversation = create_conversation(db, title="temp")
        save_message(db, conversation.id, "user", "hello")
        conversation_id = conversation.id

    with sqlite_session_factory() as db:
        first = clear_conversation(db, conversation_id)
        second = clear_conversation(db, conversation_id)

    assert first["cleared"] is True
    assert first["messages_deleted"] == 1
    assert second["cleared"] is True
    assert second["messages_deleted"] == 0

    with sqlite_session_factory() as db:
        third = delete_conversation(db, conversation_id)
        fourth = delete_conversation(db, conversation_id)

    assert third["deleted"] is True
    assert fourth["deleted"] is False
