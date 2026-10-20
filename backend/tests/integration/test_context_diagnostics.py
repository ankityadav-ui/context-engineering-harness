"""Diagnostics contract tests for POST /context/preview.

External services are mocked; conversations, summaries, and memories run
against the shared in-memory SQLite database. The tests verify the
request-scoped ``diagnostics`` block describes the real assembly run:
source statuses, counts, timings, sanitized failures, and the final
assembly report.
"""

import json

import app.context_engine.engine as engine
import app.main as main_module
from app.context_engine import conversation_service


def _seed_messages(session_factory, conversation_id, count):
    with session_factory() as db:
        for number in range(1, count + 1):
            conversation_service.save_message(
                db=db,
                conversation_id=conversation_id,
                role="user" if number % 2 else "assistant",
                content=f"turn-{number}",
            )


def _rag_result(document_id="doc-1", page=2, score=0.87):
    return {
        "content_type": "text",
        "content": "The return window is 30 days.",
        "document_id": document_id,
        "page_number": page,
        "score": score,
    }


def _relationship():
    return {
        "matched_entity": "Acme",
        "source": "Acme",
        "relationship": "RELATES_TO",
        "target": "Policy",
        "direction": "outgoing",
    }


def _preview(api_client, **overrides):
    payload = {
        "query": "What is the return policy?",
        "top_k": 5,
    }
    payload.update(overrides)

    return api_client.post("/context/preview", json=payload)


def test_all_sources_report_content_counts_and_assembly(
    api_client,
    sqlite_session_factory,
    isolated_assembly,
):
    created = api_client.post("/conversations", json={}).json()
    _seed_messages(sqlite_session_factory, created["id"], 8)

    memory = api_client.post(
        "/memory",
        json={
            "user_id": "user-1",
            "memory_type": "preference",
            "content": "Prefers concise answers about returns.",
            "importance": 5,
        },
    )

    assert memory.status_code == 201

    isolated_assembly.setattr(
        engine,
        "retrieve_context",
        lambda **_kwargs: [_rag_result(), _rag_result(page=3, score=0.5)],
    )
    isolated_assembly.setattr(
        engine,
        "_get_graph_context",
        lambda _query: [_relationship()],
    )

    response = _preview(
        api_client,
        conversation_id=created["id"],
        user_id="user-1",
    )

    assert response.status_code == 200

    body = response.json()
    diagnostics = body["diagnostics"]

    assert diagnostics["status"] == "ok"
    assert isinstance(diagnostics["duration_ms"], int)
    assert diagnostics["duration_ms"] >= 0
    assert diagnostics["errors"] == []
    assert diagnostics["conversation_requested"] is True
    assert diagnostics["memory_requested"] is True

    sources = diagnostics["sources"]

    assert sources["short_term"]["status"] == "ok"
    assert sources["short_term"]["messages"] == 6
    assert sources["conversation_summary"]["status"] == "ok"
    assert sources["conversation_summary"]["older_messages"] == 2
    assert sources["conversation_summary"]["chars"] > 0
    assert sources["memory"]["status"] == "ok"
    assert sources["memory"]["items"] == 1
    assert sources["rag"]["status"] == "ok"
    # Two chunks from different pages are distinct retrieval results.
    assert sources["rag"]["chunks"] == 2
    assert sources["rag"]["documents"] == 1
    assert sources["rag"]["document_ids"] == ["doc-1"]
    assert sources["rag"]["content_types"] == ["text"]
    assert sources["knowledge_graph"]["status"] == "ok"
    assert sources["knowledge_graph"]["relationships"] == 1
    assert sources["knowledge_graph"]["entities"] == 1

    assembly = diagnostics["assembly"]

    assert assembly["empty"] is False
    assert assembly["truncated"] is False
    assert assembly["chars"] == len(body["formatted_context"])
    assert {section["name"] for section in assembly["sections"]} == {
        "conversation_summary",
        "recent_conversation",
        "memory",
        "document_evidence",
        "knowledge_graph",
    }
    assert all(section["chars"] > 0 for section in assembly["sections"])


def test_empty_not_requested_and_failed_are_distinct(
    api_client,
    isolated_assembly,
):
    isolated_assembly.setattr(engine, "retrieve_context", lambda **_k: [])

    def unavailable_graph(_query):
        raise RuntimeError("Neo4j is not configured at bolt://secret-host")

    isolated_assembly.setattr(engine, "_get_graph_context", unavailable_graph)

    response = _preview(api_client)

    assert response.status_code == 200

    body = response.json()
    diagnostics = body["diagnostics"]
    sources = diagnostics["sources"]

    # A conversation and a memory owner were not supplied: not a failure.
    assert sources["short_term"]["status"] == "not_requested"
    assert sources["conversation_summary"]["status"] == "not_requested"
    assert sources["memory"]["status"] == "not_requested"

    # An empty retrieval is reported as empty, not failed.
    assert sources["rag"]["status"] == "empty"
    assert sources["rag"]["chunks"] == 0

    # A real failure keeps its own status and a sanitized error.
    assert sources["knowledge_graph"]["status"] == "error"
    assert diagnostics["errors"] == [
        {
            "source": "knowledge_graph",
            "category": "unavailable",
            "message": "The knowledge graph could not be queried.",
        }
    ]

    serialized = json.dumps(diagnostics)

    assert "secret-host" not in serialized
    assert "Neo4j" not in serialized

    # The context is answerable from the question alone.
    assert diagnostics["assembly"]["empty"] is True
    assert diagnostics["assembly"]["sections"] == []
    assert body["formatted_context"] == (
        "CURRENT QUESTION:\nWhat is the return policy?"
    )


def test_one_source_failure_does_not_fail_the_others(
    api_client,
    isolated_assembly,
):
    def broken_retrieval(**_kwargs):
        raise RuntimeError("password=hunter2 connection refused")

    isolated_assembly.setattr(engine, "retrieve_context", broken_retrieval)
    isolated_assembly.setattr(
        engine,
        "_get_graph_context",
        lambda _query: [_relationship()],
    )
    isolated_assembly.setattr(
        engine,
        "_get_memory_context",
        lambda **_kwargs: [
            {"memory_type": "preference", "content": "Concise answers."}
        ],
    )

    response = _preview(api_client, user_id="user-1")

    assert response.status_code == 200

    body = response.json()
    diagnostics = body["diagnostics"]
    sources = diagnostics["sources"]

    assert sources["rag"]["status"] == "error"
    assert sources["memory"]["status"] == "ok"
    assert sources["knowledge_graph"]["status"] == "ok"

    rag_error = next(
        error for error in diagnostics["errors"] if error["source"] == "rag"
    )

    assert rag_error["message"] == (
        "Document retrieval failed. The vector index may be unavailable."
    )
    assert rag_error["category"] == "unavailable"
    assert "hunter2" not in json.dumps(diagnostics)
    assert "password" not in json.dumps(diagnostics).lower()

    # The legacy field shape remains stable, but its error text is sanitized.
    assert body["source_errors"][0]["source"] == "rag"
    assert "hunter2" not in body["source_errors"][0]["error"]
    assert "password" not in body["source_errors"][0]["error"].lower()


def test_summary_is_not_applicable_without_older_messages(
    api_client,
    sqlite_session_factory,
    isolated_assembly,
):
    created = api_client.post("/conversations", json={}).json()
    _seed_messages(sqlite_session_factory, created["id"], 2)

    response = _preview(api_client, conversation_id=created["id"])

    assert response.status_code == 200

    sources = response.json()["diagnostics"]["sources"]

    assert sources["short_term"]["status"] == "ok"
    assert sources["short_term"]["messages"] == 2
    # Nothing older exists, so the summary stage was not applicable.
    assert sources["conversation_summary"]["status"] == "not_requested"
    assert sources["conversation_summary"]["older_messages"] == 0


def test_preview_runs_one_retrieval_and_generates_no_answer(
    api_client,
    isolated_assembly,
):
    calls = []

    def counting_retrieval(**kwargs):
        calls.append(kwargs)
        return [_rag_result()]

    isolated_assembly.setattr(engine, "retrieve_context", counting_retrieval)
    isolated_assembly.setattr(
        engine,
        "_get_graph_context",
        lambda _query: [_relationship()],
    )

    def fail_if_asked_for_an_answer(*_args, **_kwargs):
        raise AssertionError("Diagnostics must never generate an answer.")

    isolated_assembly.setattr(
        main_module, "generate_response", fail_if_asked_for_an_answer
    )

    response = _preview(api_client)

    assert response.status_code == 200

    body = response.json()

    assert len(calls) == 1
    assert "answer" not in body
    assert "response" not in body
    assert body["diagnostics"]["sources"]["rag"]["chunks"] == 1


def test_preview_keeps_its_existing_response_fields(
    api_client,
    isolated_assembly,
):
    isolated_assembly.setattr(
        engine, "retrieve_context", lambda **_kwargs: []
    )

    response = _preview(api_client)

    assert response.status_code == 200

    body = response.json()

    for key in (
        "query",
        "sources",
        "source_errors",
        "image_references",
        "system_prompt",
        "model",
        "formatted_context",
        "diagnostics",
    ):
        assert key in body

    assert set(body["sources"]) == {
        "conversation",
        "memory",
        "rag",
        "knowledge_graph",
    }


def test_preview_rejects_empty_query_before_assembly(
    api_client,
    isolated_assembly,
):
    response = api_client.post(
        "/context/preview",
        json={"query": "   "},
    )

    # Pydantic min_length still allows spaces; the endpoint rejects them.
    assert response.status_code in (400, 422)


def test_preview_404_for_unknown_conversation_does_not_run_assembly(
    api_client,
    isolated_assembly,
):
    calls = []
    isolated_assembly.setattr(
        engine,
        "retrieve_context",
        lambda **kwargs: calls.append(kwargs) or [],
    )

    response = _preview(
        api_client,
        conversation_id="00000000-0000-0000-0000-000000000000",
    )

    assert response.status_code == 404
    assert calls == []
