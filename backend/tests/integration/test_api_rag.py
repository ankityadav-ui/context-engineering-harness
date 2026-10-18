"""Tests for the document question-answering API contract.

External services (Pinecone retrieval, Neo4j, the LLM) are replaced
with in-memory doubles, so no paid API is ever contacted.
"""

import pytest

import app.context_engine.engine as engine
import app.llm as llm_module
import app.main as main_module
from app.system_prompt import SECURITY_INSTRUCTIONS


def _context(**overrides):
    context = {
        "query": "What did we discuss?",
        "conversation": None,
        "memory": [],
        "rag": [],
        "knowledge_graph": [],
        "source_errors": [],
    }
    context.update(overrides)
    return context


class FakeProvider:
    """Minimal in-memory Ollama chat double that records messages."""

    def __init__(self, response="The return window is 30 days."):
        self.response = response
        self.seen_messages: list[list[dict]] = []

    def _response(self, content):
        response = type("ChatResponse", (), {})()
        response.message = type("Message", (), {})()
        response.message.content = content
        return response

    def chat(self, **kwargs):
        self.seen_messages.append(kwargs.get("messages", []))
        return self._response(self.response)

    @property
    def last_system_text(self):
        return "".join(
            message.get("content", "")
            for message in self.seen_messages[-1]
            if message.get("role") == "system"
        )

    @property
    def last_user_text(self):
        return "".join(
            message.get("content", "")
            for message in self.seen_messages[-1]
            if message.get("role") == "user"
        )


@pytest.fixture()
def fake_llm(monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(llm_module, "_get_client", lambda: provider)
    llm_module.reset_client()
    return provider


# ---------------------------------------------------------- request contract


def test_document_ask_accepts_string_conversation_id(
    api_client,
    monkeypatch,
):
    # The conversation must exist: /documents/ask enforces the same
    # access rules as POST /chat instead of bypassing ownership.
    created = api_client.post("/conversations", json={}).json()
    conversation_id = created["id"]

    context = _context()
    captured = {}

    def fake_assemble_context(**kwargs):
        captured.update(kwargs)
        return context

    monkeypatch.setattr(
        main_module,
        "assemble_context",
        fake_assemble_context,
    )
    monkeypatch.setattr(
        main_module,
        "format_context",
        lambda _context: "formatted context",
    )

    # The route passes trusted instructions as the system argument, so
    # the double must accept both keyword arguments.
    def fake_generate_response(prompt, system=None):
        assert system is not None
        return "A prior discussion."

    monkeypatch.setattr(
        main_module,
        "generate_response",
        fake_generate_response,
    )

    response = api_client.post(
        "/documents/ask",
        json={
            "query": "What did we discuss?",
            "conversation_id": conversation_id,
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "A prior discussion."
    assert captured["conversation_id"] == conversation_id


def test_ask_unknown_conversation_returns_404(api_client):
    response = api_client.post(
        "/documents/ask",
        json={
            "query": "Anything?",
            "conversation_id": "11111111-2222-3333-4444-555555555555",
        },
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_ask_enforces_conversation_session_ownership(
    api_client,
    fake_llm,
    monkeypatch,
):
    created = api_client.post(
        "/conversations",
        json={"title": "Private", "session_id": "session-a"},
    ).json()

    monkeypatch.setattr(
        main_module,
        "assemble_context",
        lambda **_kwargs: _context(),
    )

    # A foreign session can never target the conversation: the request
    # is rejected before any retrieval or generation happens.
    forbidden = api_client.post(
        "/documents/ask",
        json={
            "query": "Anything?",
            "conversation_id": created["id"],
            "session_id": "session-b",
        },
    )

    assert forbidden.status_code == 404
    assert fake_llm.seen_messages == []

    # The owner may target the conversation.
    allowed = api_client.post(
        "/documents/ask",
        json={
            "query": "Anything?",
            "conversation_id": created["id"],
            "session_id": "session-a",
        },
    )

    assert allowed.status_code == 200


# ------------------------------------------------------------ response path


def test_ask_returns_retrieved_sources_and_uses_them(
    api_client,
    fake_llm,
    monkeypatch,
):
    rag_results = [
        {
            "content_type": "text",
            "content": "The return window is 30 days.",
            "document_id": "doc-1",
            "page_number": 2,
            "score": 0.91,
        }
    ]

    monkeypatch.setattr(
        main_module,
        "assemble_context",
        lambda **kwargs: _context(
            query=kwargs["query"],
            rag=rag_results,
        ),
    )

    response = api_client.post(
        "/documents/ask",
        json={"query": "What is the return policy?"},
    )

    assert response.status_code == 200

    body = response.json()

    # Source references correspond to the real retrieved records.
    assert body["sources"]["rag"] == rag_results
    assert body["source_errors"] == []

    # The retrieved context reached the LLM inside the user content,
    # clearly labelled, while instructions stayed in the system role.
    prompt = fake_llm.last_user_text

    assert "RELEVANT DOCUMENT EVIDENCE:" in prompt
    assert "The return window is 30 days." in prompt
    assert "document_id=doc-1" in prompt
    assert "What is the return policy?" in prompt
    assert SECURITY_INSTRUCTIONS in fake_llm.last_system_text
    assert "The return window is 30 days." not in fake_llm.last_system_text


def test_ask_with_no_document_or_retrieval_returns_actionable_error(
    api_client,
    monkeypatch,
):
    # /documents/ask is document-grounded: with no selected document and
    # nothing retrievable there is no evidence to answer from, so the
    # client is told what to do instead of receiving a fabricated answer.
    monkeypatch.setattr(
        main_module,
        "assemble_context",
        lambda **kwargs: _context(query=kwargs["query"]),
    )
    monkeypatch.setattr(
        main_module,
        "generate_response",
        lambda *_args, **_kwargs: pytest.fail(
            "The model must not be called without document evidence."
        ),
    )

    response = api_client.post(
        "/documents/ask",
        json={"query": "What is the return policy?"},
    )

    assert response.status_code == 400
    assert "document" in response.json()["detail"].lower()


def test_ask_llm_failure_returns_503_with_detail(
    api_client,
    monkeypatch,
):
    # An LLM failure is reported as an error, distinguishable from a
    # successful answer with no retrieved documents.
    from app.llm import LLMError

    def failing_generate_response(prompt, system=None):
        raise LLMError("The model service is unavailable.")

    monkeypatch.setattr(
        main_module,
        "assemble_context",
        lambda **_kwargs: _context(
            rag=[
                {
                    "content_type": "text",
                    "content": "The return window is 30 days.",
                    "document_id": "doc-1",
                    "page_number": 1,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        main_module,
        "generate_response",
        failing_generate_response,
    )

    response = api_client.post(
        "/documents/ask",
        json={"query": "What is the return policy?"},
    )

    assert response.status_code == 503
    assert "model service is unavailable" in response.json()["detail"]


def test_ask_uses_the_conversations_saved_system_prompt(
    api_client,
    fake_llm,
    isolated_assembly,
):
    created = api_client.post(
        "/conversations", json={"title": "French"}
    ).json()

    saved = api_client.put(
        f"/conversations/{created['id']}/system-prompt",
        json={"custom_prompt": "Answer in French."},
    )
    assert saved.status_code == 200

    response = api_client.post(
        "/documents/ask",
        json={
            "query": "What is the return policy?",
            "conversation_id": created["id"],
        },
    )

    assert response.status_code == 200

    # The selected conversation's own prompt and the built-in security
    # rules reached the provider as the system message.
    assert "Answer in French." in fake_llm.last_system_text
    assert SECURITY_INSTRUCTIONS in fake_llm.last_system_text
    assert "Answer in French." not in fake_llm.last_user_text


def test_ask_without_a_conversation_uses_default_instructions(
    api_client,
    fake_llm,
    isolated_assembly,
):
    # A saved prompt on some other conversation must not leak: there is
    # no global active prompt.
    isolated_assembly.setattr(
        engine,
        "retrieve_context",
        lambda **_kwargs: [
            {
                "content_type": "text",
                "content": "The return window is 30 days.",
                "document_id": "doc-1",
                "page_number": 1,
            }
        ],
    )

    other = api_client.post("/conversations", json={}).json()

    api_client.put(
        f"/conversations/{other['id']}/system-prompt",
        json={"custom_prompt": "Not global."},
    )

    response = api_client.post(
        "/documents/ask",
        json={"query": "What is the return policy?"},
    )

    assert response.status_code == 200
    assert fake_llm.last_system_text == SECURITY_INSTRUCTIONS


def test_ask_with_conversation_requires_configured_storage(
    monkeypatch,
    fake_llm,
):
    from fastapi.testclient import TestClient

    from app.context_engine import database
    from app.main import app

    monkeypatch.setattr(database, "_engine", None)
    monkeypatch.setattr(database, "_session_factory", None)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    overrides = dict(app.dependency_overrides)
    app.dependency_overrides.clear()

    try:
        with TestClient(app) as client:
            response = client.post(
                "/documents/ask",
                json={
                    "query": "Anything?",
                    "conversation_id": (
                        "11111111-2222-3333-4444-555555555555"
                    ),
                },
            )
    finally:
        app.dependency_overrides.update(overrides)

    assert response.status_code == 503
    assert "DATABASE_URL" in response.json()["detail"]


def test_ask_rejects_blank_query(api_client):
    response = api_client.post(
        "/documents/ask", json={"query": "   "})

    # min_length=1 lets whitespace through, so the route's own
    # emptiness check applies.
    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty."
