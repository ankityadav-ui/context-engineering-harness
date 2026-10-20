"""Unit tests for diagnostics metadata and sanitized failures."""

import app.context_engine.engine as engine


def _context(**overrides):
    context = {
        "query": "What is the policy?",
        "conversation": None,
        "memory": [],
        "rag": [],
        "knowledge_graph": [],
        "source_errors": [],
    }
    context.update(overrides)

    return context


def test_report_matches_existing_formatted_context_exactly():
    context = _context(
        conversation={
            "summary": "Older messages summarized.",
            "recent_messages": [
                {"role": "user", "content": "Remember the codename."}
            ],
        },
        memory=[
            {"memory_type": "preference", "content": "Prefers brevity."}
        ],
        knowledge_graph=[
            {
                "matched_entity": "Acme",
                "source": "Acme",
                "relationship": "RELATES_TO",
                "target": "Policy",
                "direction": "outgoing",
            }
        ],
        rag=[
            {
                "content_type": "text",
                "content": "The return window is 30 days.",
                "document_id": "doc-1",
                "page_number": 2,
            }
        ],
    )

    report = engine.build_context_report(context)

    assert report["formatted_context"] == engine.format_context(context)
    assert report["chars"] == len(report["formatted_context"])
    assert report["empty"] is False
    assert report["truncated"] is False

    sections = {
        section["name"]: section["chars"] for section in report["sections"]
    }

    assert set(sections) == {
        "conversation_summary",
        "recent_conversation",
        "memory",
        "knowledge_graph",
        "document_evidence",
    }
    assert all(chars > 0 for chars in sections.values())
    # Section sizes account for the whole text (plus joining newlines).
    assert sum(sections.values()) <= report["chars"]


def test_empty_report_contains_only_the_question():
    report = engine.build_context_report(_context())

    assert report["empty"] is True
    assert report["sections"] == []
    assert report["formatted_context"] == (
        "CURRENT QUESTION:\nWhat is the policy?"
    )


def test_truncation_is_reported_not_hidden():
    context = _context(
        rag=[
            {
                "content_type": "text",
                "content": "x" * 500,
                "document_id": "doc-1",
                "page_number": 1,
            }
        ]
    )

    report = engine.build_context_report(context, max_chars=80)

    assert report["truncated"] is True
    assert report["formatted_context"].endswith(engine.TRUNCATION_NOTICE)
    assert report["chars"] == len(report["formatted_context"])


def test_failure_messages_are_fixed_and_categorized():
    unavailable = engine._safe_failure(
        "rag", ConnectionError("host=db.internal password=hunter2")
    )

    assert unavailable == {
        "source": "rag",
        "category": "unavailable",
        "message": "Document retrieval failed. The vector index may be unavailable.",
    }
    assert "hunter2" not in str(unavailable)

    internal = engine._safe_failure(
        "memory", ValueError("unexpected row shape: secret")
    )

    assert internal["category"] == "internal"
    assert internal["message"] == "Long-term memory could not be read."
    assert "secret" not in str(internal)


def test_unknown_source_gets_a_generic_message():
    failure = engine._safe_failure("other", RuntimeError("boom"))

    assert failure["message"] == "This context source failed."
    assert failure["category"] == "internal"
