"""Tests for conversation-specific system prompts.

Each conversation stores its own custom system prompt in PostgreSQL
(SQLite in-memory here). The LLM provider is replaced with an in-memory
double, so no paid API is ever contacted.

Covered here:
- a new conversation starts with no custom prompt (default instructions),
- saving, loading, switching, and resetting per conversation,
- refreshing (a fresh request) preserving saved prompts,
- resetting Conversation A leaving Conversation B untouched,
- built-in security instructions staying active with any custom prompt,
- /chat (and /chat/stream) using the selected conversation's prompt,
- the removed Skills endpoints and global prompt endpoint staying gone.
"""

import app.llm as llm_module
import app.main as main_module
from app.system_prompt import SECURITY_INSTRUCTIONS


# ---------------------------------------------------------------- fakes


class FakeProvider:
    """Minimal in-memory Gemini chat double that records messages."""

    def __init__(self, response="stubbed answer"):
        self.response = response
        self.seen_messages = []

    def _response(self, content):
        response = type("ChatResponse", (), {})()
        response.message = type("Message", (), {})()
        response.message.content = content
        return response

    def invoke(self, messages):
        self.seen_messages.append(messages)
        return type("AIMessage", (), {"content": self.response})()

    def stream(self, messages):
        self.seen_messages.append(messages)
        yield type("AIMessageChunk", (), {"content": self.response})()

    @property
    def last_system_text(self):
        return "".join(
            message.content
            for message in self.seen_messages[-1]
            if message.type == "system"
        )

    @property
    def last_user_text(self):
        return "".join(
            message.content
            for message in self.seen_messages[-1]
            if message.type == "human"
        )


import pytest


@pytest.fixture()
def fake_llm(monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(llm_module, "get_chat_model", lambda: provider)
    llm_module.reset_client()
    return provider


PROMPT_PATH = "/conversations/{conversation_id}/system-prompt"


def save_prompt(client, conversation_id, custom_prompt):
    return client.put(
        PROMPT_PATH.format(conversation_id=conversation_id),
        json={"custom_prompt": custom_prompt},
    )


def load_prompt(client, conversation_id):
    return client.get(
        PROMPT_PATH.format(conversation_id=conversation_id)
    )


def reset_prompt(client, conversation_id):
    return client.delete(
        PROMPT_PATH.format(conversation_id=conversation_id)
    )


def new_conversation(client, title=None):
    response = client.post(
        "/conversations",
        json={"title": title} if title else {},
    )
    assert response.status_code == 201
    return response.json()


# ------------------------------------------- save / load / switch / reset


def test_new_conversation_uses_default_instructions(api_client):
    created = new_conversation(api_client)

    loaded = load_prompt(api_client, created["id"])

    assert loaded.status_code == 200

    body = loaded.json()

    assert body["conversation_id"] == created["id"]
    assert body["custom_prompt"] == ""
    assert body["using_default"] is True
    # The built-in security instructions are reported read-only.
    assert body["security_instructions"] == SECURITY_INSTRUCTIONS


def test_saving_and_reloading_one_prompt(api_client):
    created = new_conversation(api_client)

    saved = save_prompt(
        api_client, created["id"], "Answer like a pirate."
    )

    assert saved.status_code == 200
    assert saved.json()["custom_prompt"] == "Answer like a pirate."
    assert saved.json()["using_default"] is False

    # A later request (a "browser refresh") reads the persisted value.
    loaded = load_prompt(api_client, created["id"])

    assert loaded.json()["custom_prompt"] == "Answer like a pirate."


def test_conversations_keep_different_prompts(api_client):
    conversation_a = new_conversation(api_client, "A")
    conversation_b = new_conversation(api_client, "B")

    assert (
        save_prompt(api_client, conversation_a["id"], "Prompt A.").status_code
        == 200
    )
    assert (
        save_prompt(api_client, conversation_b["id"], "Prompt B.").status_code
        == 200
    )

    # Switching conversations restores each conversation's own prompt.
    loaded_a = load_prompt(api_client, conversation_a["id"]).json()
    loaded_b = load_prompt(api_client, conversation_b["id"]).json()

    assert loaded_a["custom_prompt"] == "Prompt A."
    assert loaded_b["custom_prompt"] == "Prompt B."

    # Returning to the original conversation restores its prompt.
    loaded_a_again = load_prompt(api_client, conversation_a["id"]).json()

    assert loaded_a_again["custom_prompt"] == "Prompt A."


def test_reset_clears_only_that_conversation(api_client):
    conversation_a = new_conversation(api_client, "A")
    conversation_b = new_conversation(api_client, "B")

    save_prompt(api_client, conversation_a["id"], "Prompt A.")
    save_prompt(api_client, conversation_b["id"], "Prompt B.")

    reset = reset_prompt(api_client, conversation_a["id"])

    assert reset.status_code == 200
    assert reset.json()["using_default"] is True
    assert reset.json()["custom_prompt"] == ""

    # Conversation A returns to the default instructions...
    reloaded_a = load_prompt(api_client, conversation_a["id"]).json()
    assert reloaded_a["custom_prompt"] == ""
    assert reloaded_a["using_default"] is True

    # ...and Conversation B is untouched.
    reloaded_b = load_prompt(api_client, conversation_b["id"]).json()
    assert reloaded_b["custom_prompt"] == "Prompt B."


def test_putting_an_empty_prompt_resets_to_default(api_client):
    created = new_conversation(api_client)

    save_prompt(api_client, created["id"], "Temporary.")

    reset = save_prompt(api_client, created["id"], "   ")

    assert reset.status_code == 200
    assert reset.json()["using_default"] is True
    assert load_prompt(api_client, created["id"]).json()["custom_prompt"] == ""


def test_overlong_prompt_is_rejected(api_client):
    created = new_conversation(api_client)

    response = save_prompt(api_client, created["id"], "x" * 4001)

    assert response.status_code == 422
    # The rejected value was not stored.
    assert load_prompt(api_client, created["id"]).json()["custom_prompt"] == ""


def test_unknown_conversation_returns_404(api_client):
    unknown = "11111111-2222-3333-4444-555555555555"

    assert load_prompt(api_client, unknown).status_code == 404
    assert save_prompt(api_client, unknown, "hi").status_code == 404
    assert reset_prompt(api_client, unknown).status_code == 404


def test_malformed_conversation_id_returns_404(api_client):
    assert load_prompt(api_client, "../../etc").status_code == 404


def test_session_ownership_protects_the_prompt(api_client):
    created = api_client.post(
        "/conversations",
        json={"title": "Private", "session_id": "session-a"},
    ).json()

    saved = api_client.put(
        PROMPT_PATH.format(conversation_id=created["id"]),
        params={"session_id": "session-a"},
        json={"custom_prompt": "Owner prompt."},
    )

    assert saved.status_code == 200

    # Another session cannot read, save, or reset the prompt.
    foreign_read = api_client.get(
        PROMPT_PATH.format(conversation_id=created["id"]),
        params={"session_id": "session-b"},
    )
    foreign_save = api_client.put(
        PROMPT_PATH.format(conversation_id=created["id"]),
        params={"session_id": "session-b"},
        json={"custom_prompt": "hijack"},
    )
    foreign_reset = api_client.delete(
        PROMPT_PATH.format(conversation_id=created["id"]),
        params={"session_id": "session-b"},
    )

    assert foreign_read.status_code == 404
    assert foreign_save.status_code == 404
    assert foreign_reset.status_code == 404

    # The owner still sees the original prompt.
    owner_read = api_client.get(
        PROMPT_PATH.format(conversation_id=created["id"]),
        params={"session_id": "session-a"},
    )

    assert owner_read.status_code == 200
    assert owner_read.json()["custom_prompt"] == "Owner prompt."


def test_clearing_messages_keeps_the_prompt(api_client, isolated_assembly):
    created = new_conversation(api_client)
    save_prompt(api_client, created["id"], "Keep me.")

    cleared = api_client.delete(f"/conversations/{created['id']}/messages")

    assert cleared.status_code == 200

    reloaded = load_prompt(api_client, created["id"]).json()

    assert reloaded["custom_prompt"] == "Keep me."


def test_deleting_the_conversation_removes_its_prompt(
    api_client, sqlite_session_factory, isolated_assembly
):
    created = new_conversation(api_client)
    save_prompt(api_client, created["id"], "Doomed.")

    deleted = api_client.delete(f"/conversations/{created['id']}")

    assert deleted.status_code == 200
    assert load_prompt(api_client, created["id"]).status_code == 404

    # No orphaned prompt rows remain.
    from app.context_engine.models import Conversation

    with sqlite_session_factory() as db:
        assert (
            db.query(Conversation)
            .filter(Conversation.id == created["id"])
            .count()
            == 0
        )


# ------------------------------------------------------------ /chat wiring


def test_chat_uses_the_selected_conversations_prompt(
    api_client, fake_llm, isolated_assembly
):
    created = new_conversation(api_client, "French")

    saved = save_prompt(api_client, created["id"], "Answer in French.")
    assert saved.status_code == 200

    response = api_client.post(
        "/chat",
        json={"message": "Hello?", "conversation_id": created["id"]},
    )

    assert response.status_code == 200
    assert response.json()["system_prompt_active"] is True

    # The conversation's custom prompt and the built-in security rules
    # reached the provider as the system message; the user's message
    # stayed user content.
    assert fake_llm.last_system_text.startswith("Answer in French.")
    assert SECURITY_INSTRUCTIONS in fake_llm.last_system_text
    assert "Hello?" in fake_llm.last_user_text


def test_chat_switching_conversations_switches_prompts(
    api_client, fake_llm, isolated_assembly
):
    conversation_a = new_conversation(api_client, "A")
    conversation_b = new_conversation(api_client, "B")

    save_prompt(api_client, conversation_a["id"], "Prompt A.")
    save_prompt(api_client, conversation_b["id"], "Prompt B.")

    api_client.post(
        "/chat",
        json={"message": "to A", "conversation_id": conversation_a["id"]},
    )
    assert "Prompt A." in fake_llm.last_system_text
    assert "Prompt B." not in fake_llm.last_system_text

    api_client.post(
        "/chat",
        json={"message": "to B", "conversation_id": conversation_b["id"]},
    )
    assert "Prompt B." in fake_llm.last_system_text
    assert "Prompt A." not in fake_llm.last_system_text


def test_new_conversation_chat_uses_default_instructions(
    api_client, fake_llm, isolated_assembly
):
    created = new_conversation(api_client)

    response = api_client.post(
        "/chat",
        json={"message": "Hello?", "conversation_id": created["id"]},
    )

    assert response.status_code == 200
    assert response.json()["system_prompt_active"] is False

    # Only the built-in security instructions travel as system content.
    assert fake_llm.last_system_text == SECURITY_INSTRUCTIONS


def test_stateless_chat_uses_default_instructions(
    api_client, fake_llm, isolated_assembly
):
    # A saved custom prompt on some conversation must not leak into
    # stateless chat: there is no global active prompt.
    other = new_conversation(api_client)
    save_prompt(api_client, other["id"], "Not global.")

    response = api_client.post("/chat", json={"message": "Hello?"})

    assert response.status_code == 200
    assert response.json()["system_prompt_active"] is False
    assert "Not global." not in fake_llm.last_system_text
    assert fake_llm.last_system_text == SECURITY_INSTRUCTIONS


def test_stream_uses_the_selected_conversations_prompt(
    api_client, fake_llm, isolated_assembly
):
    created = new_conversation(api_client, "French")
    save_prompt(api_client, created["id"], "Answer in French.")

    with api_client.stream(
        "POST",
        "/chat/stream",
        json={"message": "Hello?", "conversation_id": created["id"]},
    ) as response:
        assert response.status_code == 200
        events = [
            line
            for line in response.iter_lines()
            if line.startswith("event:")
        ]

    assert any("done" in event for event in events)
    assert "Answer in French." in fake_llm.last_system_text
    assert SECURITY_INSTRUCTIONS in fake_llm.last_system_text


# --------------------------------------------- removed Skills functionality


def test_skills_endpoints_are_removed(api_client):
    # The Skills feature is gone: its endpoints no longer exist.
    assert api_client.get("/settings/skills").status_code == 404
    assert (
        api_client.post(
            "/settings/skills/select", json={"skill_id": "document_qa"}
        ).status_code
        == 404
    )


def test_global_system_prompt_endpoint_is_removed(api_client):
    # The old global prompt store is replaced by per-conversation
    # prompts; the global endpoints no longer exist.
    assert api_client.get("/settings/system-prompt").status_code == 404
    assert (
        api_client.put(
            "/settings/system-prompt",
            json={"custom_prompt": "global"},
        ).status_code
        == 404
    )


def test_no_skill_symbols_remain_in_the_api_module():
    for symbol in (
        "list_skills",
        "get_skill_text",
        "get_active_skill",
        "save_active_skill",
        "select_active_skill",
        "list_available_skills",
        "get_system_prompt",
        "save_system_prompt",
    ):
        assert not hasattr(main_module, symbol), symbol
