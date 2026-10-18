"""Tests for chat with no retrieved context (no uploaded documents).

An empty retrieval result is a legitimate outcome, not an error: the
user's question and the applicable system instructions must still reach
the configured LLM. The LLM provider is replaced with an in-memory
double, so no paid API is ever contacted.

Covered here:
- /chat calls the LLM when no context is available (stateless and with
  a conversation),
- /chat/stream handles empty context the same way,
- no misleading "No context provided" placeholder is sent as context,
- a recoverable retrieval failure does not block ordinary chat,
- relevant retrieved context is still included when it exists.
"""

import pytest

import app.context_engine.engine as engine
import app.llm as llm_module
from app.system_prompt import SECURITY_INSTRUCTIONS


class FakeProvider:
    """Minimal in-memory Ollama chat double that records prompts."""

    def __init__(self, response="llm answer"):
        self.response = response
        self.seen_messages: list[list[dict]] = []

    def _response(self, content):
        response = type("ChatResponse", (), {})()
        response.message = type("Message", (), {})()
        response.message.content = content
        return response

    def chat(self, **kwargs):
        self.seen_messages.append(kwargs.get("messages", []))

        if kwargs.get("stream"):
            return iter([self._response(self.response)])

        return self._response(self.response)

    @property
    def last_user_text(self):
        return "".join(
            message.get("content", "")
            for message in self.seen_messages[-1]
            if message.get("role") == "user"
        )

    @property
    def last_system_text(self):
        return "".join(
            message.get("content", "")
            for message in self.seen_messages[-1]
            if message.get("role") == "system"
        )


@pytest.fixture()
def fake_llm(monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(llm_module, "_get_client", lambda: provider)
    llm_module.reset_client()
    return provider


def _new_conversation(api_client):
    created = api_client.post("/conversations", json={})
    assert created.status_code == 201
    return created.json()


# ---------------------------------------------------------- empty context


def test_stateless_chat_reaches_the_llm_without_documents(
    api_client, fake_llm
):
    # No conversation, no uploaded documents: the question goes
    # straight to the model with the trusted instructions.
    response = api_client.post("/chat", json={"message": "Hello there?"})

    assert response.status_code == 200
    assert response.json()["response"] == "llm answer"
    assert len(fake_llm.seen_messages) == 1
    assert "Hello there?" in fake_llm.last_user_text
    assert SECURITY_INSTRUCTIONS in fake_llm.last_system_text


def test_chat_with_conversation_and_empty_context_calls_the_llm(
    api_client, fake_llm, isolated_assembly
):
    # isolated_assembly keeps retrieval off live services and returns
    # no results: a legitimate empty retrieval.
    created = _new_conversation(api_client)

    response = api_client.post(
        "/chat",
        json={"message": "What is 2+2?", "conversation_id": created["id"]},
    )

    assert response.status_code == 200
    assert response.json()["response"] == "llm answer"

    # The LLM was actually called with the user's question.
    prompt = fake_llm.last_user_text
    assert "What is 2+2?" in prompt

    # No misleading placeholder was passed off as retrieved context.
    assert "No context provided" not in prompt
    assert "No context" not in prompt


def test_chat_with_recoverable_retrieval_failure_still_calls_the_llm(
    api_client, fake_llm, isolated_assembly, monkeypatch
):
    # A retrieval crash (e.g. vector index unreachable) is recorded as
    # a source error; it must not block ordinary chat.
    def broken_retrieve(**_kwargs):
        raise RuntimeError("vector index unreachable")

    monkeypatch.setattr(engine, "retrieve_context", broken_retrieve)

    created = _new_conversation(api_client)

    response = api_client.post(
        "/chat",
        json={
            "message": "Anything at all?",
            "conversation_id": created["id"],
        },
    )

    assert response.status_code == 200
    assert response.json()["response"] == "llm answer"
    assert "Anything at all?" in fake_llm.last_user_text
    assert "vector index unreachable" not in fake_llm.last_user_text


def test_chat_stream_with_empty_context_completes(
    api_client, fake_llm, isolated_assembly
):
    created = _new_conversation(api_client)

    with api_client.stream(
        "POST",
        "/chat/stream",
        json={"message": "Stream this", "conversation_id": created["id"]},
    ) as response:
        assert response.status_code == 200
        events = [
            line.rstrip("\r")
            for line in response.iter_lines()
            if line.startswith("event:")
        ]

    assert any("done" in event for event in events)
    assert not any("error" in event for event in events)
    assert "Stream this" in fake_llm.last_user_text


def test_chat_stream_stateless_without_documents(api_client, fake_llm):
    with api_client.stream(
        "POST", "/chat/stream", json={"message": "Hi model"}
    ) as response:
        assert response.status_code == 200
        events = [
            line.rstrip("\r")
            for line in response.iter_lines()
            if line.startswith("event:")
        ]

    assert any("done" in event for event in events)
    assert "Hi model" in fake_llm.last_user_text


# ------------------------------------------------------- context included


def test_relevant_context_is_still_included_when_available(
    api_client, fake_llm, isolated_assembly, monkeypatch
):
    def fake_retrieve(**_kwargs):
        return [
            {
                "content_type": "text",
                "content": "The return window is 30 days.",
                "document_id": "doc-1",
                "page_number": 2,
            }
        ]

    monkeypatch.setattr(engine, "retrieve_context", fake_retrieve)

    created = _new_conversation(api_client)

    response = api_client.post(
        "/chat",
        json={
            "message": "What is the return policy?",
            "conversation_id": created["id"],
        },
    )

    assert response.status_code == 200

    prompt = fake_llm.last_user_text

    # The retrieved evidence reaches the model, clearly labelled.
    assert "RELEVANT DOCUMENT EVIDENCE:" in prompt
    assert "The return window is 30 days." in prompt
    assert "document_id=doc-1" in prompt
    # The question itself is still present.
    assert "What is the return policy?" in prompt


def test_context_preview_reports_empty_sources_without_errors(
    api_client, isolated_assembly
):
    created = _new_conversation(api_client)

    preview = api_client.post(
        "/context/preview",
        json={"query": "Anything?", "conversation_id": created["id"]},
    )

    assert preview.status_code == 200

    body = preview.json()

    assert body["sources"]["rag"] == []
    assert body["sources"]["memory"] == []
    assert body["sources"]["knowledge_graph"] == []
    # A legitimate empty retrieval is not an error.
    assert body["source_errors"] == []
    # The question is still assembled into the preview text.
    assert "Anything?" in body["formatted_context"]
