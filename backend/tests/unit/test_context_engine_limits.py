"""Context-assembly bounds, deduplication, and assembly logging.

No database, Pinecone, or Neo4j connection is opened: every source helper
is replaced with an in-memory stub. Fixture text is fictional.
"""

import logging

import pytest

import app.context_engine.engine as engine


def _rag_result(content, score=0.5, page=1, content_type="text"):
    return {
        "id": f"unit-{abs(hash((content, page))) % 10000}",
        "score": score,
        "document_id": "doc-1",
        "content_type": content_type,
        "page_number": page,
        "content": content,
        "content_ref": "",
        "source": "pdf_text",
    }


@pytest.fixture()
def quiet_sources(monkeypatch):
    """Replace every context source with an in-memory stub."""

    state = {
        "rag": [],
        "graph": [],
        "memory": [],
        "retrieve_calls": [],
    }

    def fake_retrieve_context(query, top_k=5, document_id=None):
        state["retrieve_calls"].append(
            {"query": query, "top_k": top_k, "document_id": document_id}
        )
        return state["rag"]

    monkeypatch.setattr(engine, "retrieve_context", fake_retrieve_context)
    monkeypatch.setattr(engine, "_get_graph_context", lambda _q: state["graph"])
    monkeypatch.setattr(
        engine,
        "_get_memory_context",
        lambda **_kwargs: state["memory"],
    )
    monkeypatch.setattr(
        engine,
        "_get_conversation_context",
        lambda **_kwargs: {
            "summary": "",
            "recent_messages": [],
            "total_messages": 0,
            "older_count": 0,
            "recent_count": 0,
        },
    )

    return state


# --------------------------------------------------
# Deduplication
# --------------------------------------------------

def test_duplicate_rag_results_are_collapsed_keeping_the_best_score(
    quiet_sources,
):
    quiet_sources["rag"] = [
        _rag_result("The return window is 30 days.", score=0.4),
        _rag_result("The return window is 30 days.", score=0.9),
        _rag_result("Refunds are processed weekly.", score=0.6),
    ]

    context = engine.assemble_context(query="What is the return window?")

    contents = [result["content"] for result in context["rag"]]

    assert len(contents) == 2
    assert contents.count("The return window is 30 days.") == 1

    kept = next(
        result
        for result in context["rag"]
        if result["content"] == "The return window is 30 days."
    )
    assert kept["score"] == 0.9


def test_same_text_on_different_pages_is_not_collapsed(quiet_sources):
    quiet_sources["rag"] = [
        _rag_result("Repeat heading", page=1),
        _rag_result("Repeat heading", page=2),
    ]

    context = engine.assemble_context(query="heading")

    assert len(context["rag"]) == 2


def test_duplicate_graph_relationships_are_collapsed(quiet_sources):
    relationship = {
        "matched_entity": "Acme Corp.",
        "source": "Acme Corp.",
        "relationship": "PART_OF",
        "target": "Holdings",
        "direction": "outgoing",
    }

    quiet_sources["graph"] = [relationship, dict(relationship)]

    context = engine.assemble_context(query="Acme")

    assert context["knowledge_graph"] == [relationship]


# --------------------------------------------------
# Bounds
# --------------------------------------------------

def test_top_k_is_bounded_before_retrieval(quiet_sources):
    engine.assemble_context(query="anything", top_k=500)

    assert quiet_sources["retrieve_calls"][0]["top_k"] == engine.MAX_RAG_RESULTS


def test_graph_relationships_are_capped(quiet_sources):
    quiet_sources["graph"] = [
        {
            "matched_entity": f"Entity {index}",
            "source": f"Entity {index}",
            "relationship": "RELATES_TO",
            "target": "Other",
            "direction": "outgoing",
        }
        for index in range(engine.MAX_GRAPH_RELATIONSHIPS + 15)
    ]

    context = engine.assemble_context(query="Entity")

    assert len(context["knowledge_graph"]) == (
        engine.MAX_GRAPH_RELATIONSHIPS
    )


def test_format_context_truncates_a_long_unit():
    context = {
        "query": "q",
        "conversation": None,
        "memory": [],
        "rag": [_rag_result("x" * (engine.MAX_UNIT_CHARS * 2))],
        "knowledge_graph": [],
        "source_errors": [],
    }

    formatted = engine.format_context(context)

    assert "x" * engine.MAX_UNIT_CHARS in formatted
    assert "x" * (engine.MAX_UNIT_CHARS + 100) not in formatted


def test_format_context_enforces_the_total_limit():
    context = {
        "query": "q",
        "conversation": None,
        "memory": [],
        "rag": [
            _rag_result(f"unit {index} " + "y" * 3000)
            for index in range(20)
        ],
        "knowledge_graph": [],
        "source_errors": [],
    }

    formatted = engine.format_context(context, max_chars=5000)

    assert len(formatted) <= 5000 + len(engine.TRUNCATION_NOTICE) + 4
    assert engine.TRUNCATION_NOTICE in formatted


def test_small_contexts_are_not_marked_truncated():
    context = {
        "query": "q",
        "conversation": None,
        "memory": [],
        "rag": [_rag_result("short evidence")],
        "knowledge_graph": [],
        "source_errors": [],
    }

    formatted = engine.format_context(context)

    assert engine.TRUNCATION_NOTICE not in formatted


# --------------------------------------------------
# Failure isolation and logging
# --------------------------------------------------

def test_a_failing_source_does_not_hide_the_others(monkeypatch, quiet_sources):
    def failing_retrieve(**_kwargs):
        raise RuntimeError("Pinecone is not configured.")

    monkeypatch.setattr(engine, "retrieve_context", failing_retrieve)

    quiet_sources["memory"] = [
        {"memory_type": "fact", "content": "Likes concise answers."}
    ]

    context = engine.assemble_context(query="anything", user_id="user-1")

    assert context["rag"] == []
    assert context["memory"]
    assert context["source_errors"] == [
        {
            "source": "rag",
            "error": (
                "Document retrieval failed. "
                "The vector index may be unavailable."
            ),
        }
    ]


def test_blank_query_is_rejected():
    with pytest.raises(ValueError):
        engine.assemble_context(query="   ")


def test_assembly_logging_reports_counts_without_content(
    caplog,
    quiet_sources,
):
    secret_marker = "SENSITIVE-MARKER-1234"

    quiet_sources["rag"] = [_rag_result(f"evidence {secret_marker}")]
    quiet_sources["memory"] = [
        {"memory_type": "fact", "content": secret_marker}
    ]

    with caplog.at_level(logging.INFO, logger=engine.logger.name):
        engine.assemble_context(
            query=f"question {secret_marker}",
            user_id="user-1",
        )

    messages = " ".join(record.getMessage() for record in caplog.records)

    assert "Assembled context" in messages
    assert "rag=1" in messages
    assert "memory=1" in messages

    # Counts and sizes only: no query, memory, document, or entity text.
    assert secret_marker not in messages
