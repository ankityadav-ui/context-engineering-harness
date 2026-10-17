"""Tests for the document question-answering API contract."""

import app.main as main_module


def test_document_ask_accepts_string_conversation_id(
    api_client,
    monkeypatch,
):
    conversation_id = "9b19f12f-7212-4a7f-bf44-7ec0c23c9542"
    context = {
        "query": "What did we discuss?",
        "conversation": None,
        "memory": [],
        "rag": [],
        "knowledge_graph": [],
        "source_errors": [],
    }
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
