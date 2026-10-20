"""Pure unit tests for context formatting and assembly.

These do not call PostgreSQL, Pinecone, or Neo4j.
"""

from pathlib import Path

from app.context_engine import engine
from app.context_engine.engine import format_context
from app.rag import document_service


def build_sample_context() -> dict:
    """Build the combined result that the Context Engine would return."""

    return {
        "query": "What does the document say about the heart?",
        "conversation": {
            "summary": "The user previously discussed heart anatomy.",
            "recent_messages": [
                {
                    "role": "user",
                    "content": "Explain the heart diagram.",
                },
                {
                    "role": "assistant",
                    "content": "The heart pumps blood.",
                },
            ],
        },
        "memory": [
            {
                "memory_type": "preference",
                "content": "The user prefers simple explanations.",
            }
        ],
        "rag": [
            {
                "content_type": "text",
                "content": "The heart circulates blood.",
                "document_id": "test-doc-001",
                "page_number": 1,
            },
            {
                "content_type": "table",
                "content": (
                    "Chamber | Function: Left ventricle | Pumps blood"
                ),
                "document_id": "test-doc-001",
                "page_number": 2,
            },
        ],
        "knowledge_graph": [
            {
                "source": "Heart",
                "relationship": "PART_OF",
                "target": "Circulatory System",
            }
        ],
        "source_errors": [],
    }


def test_format_context_includes_all_populated_sections():
    formatted = format_context(build_sample_context())

    expected_sections = [
        "CURRENT QUESTION:",
        "CONVERSATION SUMMARY:",
        "RECENT CONVERSATION:",
        "RELEVANT LONG-TERM MEMORIES:",
        "RELEVANT DOCUMENT EVIDENCE:",
        "KNOWLEDGE GRAPH RELATIONSHIPS:",
    ]

    for section in expected_sections:
        assert section in formatted, (
            f"Missing expected section: {section}"
        )


def test_format_context_includes_source_details():
    formatted = format_context(build_sample_context())

    # Document metadata should be preserved for traceability.
    assert "document_id=test-doc-001" in formatted
    assert "page=1" in formatted

    # The graph relationship should be rendered as source -> relation -> target.
    assert "Heart --PART_OF--> Circulatory System" in formatted


def test_format_context_omits_empty_sources():
    minimal_context = {
        "query": "Only a question.",
        "conversation": None,
        "memory": [],
        "rag": [],
        "knowledge_graph": [],
        "source_errors": [],
    }

    formatted = format_context(minimal_context)

    assert "CURRENT QUESTION:" in formatted
    assert "CONVERSATION SUMMARY:" not in formatted
    assert "RELEVANT DOCUMENT EVIDENCE:" not in formatted
    assert "KNOWLEDGE GRAPH RELATIONSHIPS:" not in formatted


def test_format_context_returns_a_string_when_empty():
    assert format_context({}) == ""


def test_selected_document_graph_context_is_scoped_and_query_relevant(
    monkeypatch,
):
    import app.context_engine.graph_service as graph_service

    relationship_calls = []

    monkeypatch.setattr(
        graph_service,
        "find_entities",
        lambda *_args, **_kwargs: pytest.fail(
            "document-scoped graph lookup must not search globally"
        ),
    )
    monkeypatch.setattr(
        graph_service,
        "find_document_entities",
        lambda document_id: [
            {
                "entity_id": "person:ada",
                "name": "Ada Lovelace",
                "entity_type": "PERSON",
            },
            {
                "entity_id": "organization:acme",
                "name": "Acme Corp",
                "entity_type": "ORGANIZATION",
            },
        ]
        if document_id == "doc-1"
        else [],
    )

    def get_relationships(entity_id, document_id=None):
        relationship_calls.append((entity_id, document_id))
        return [
            {
                "source": "Ada Lovelace",
                "relationship": "RELATES_TO",
                "target": "Acme Corp",
            }
        ]

    monkeypatch.setattr(
        graph_service,
        "get_entity_relationships",
        get_relationships,
    )

    context = engine._get_graph_context(
        "What does Ada Lovelace say?",
        document_id="doc-1",
    )

    assert context[0]["matched_entity"] == "Ada Lovelace"
    assert relationship_calls == [("person:ada", "doc-1")]


# --------------------------------------------------
# Document-scoped summary assembly
# --------------------------------------------------


def _chunk() -> dict:
    return {
        "content_type": "text",
        "content": "A retrieved chunk with [PERSON_1] and [ORG_1].",
        "document_id": "doc-1",
        "page_number": 1,
    }


def _patch_sources(monkeypatch, masked_text):
    monkeypatch.setattr(
        engine, "retrieve_context", lambda **_kwargs: [_chunk()]
    )
    monkeypatch.setattr(engine, "_get_graph_context", lambda _query: [])
    monkeypatch.setattr(
        document_service,
        "read_masked_text_artifact",
        lambda _document_id: masked_text,
    )
    monkeypatch.setattr(
        document_service,
        "read_masked_content_units",
        lambda _document_id: None,
    )
    # No raw PDF is stored for a masked upload.
    monkeypatch.setattr(
        document_service,
        "resolve_stored_document_path",
        lambda _document_id: Path("missing.pdf"),
    )
    from app.rag import pinecone_service

    monkeypatch.setattr(
        pinecone_service,
        "get_document_searchable_units",
        lambda document_id: {
            "document_id": document_id,
            "document_found": False,
            "units": [],
            "page_count": None,
            "pages_without_text": [],
            "units_truncated": False,
        },
    )


def test_legacy_masked_artifact_without_pages_is_not_summarized(
    monkeypatch,
):
    _patch_sources(monkeypatch, "Whole masked document text.")

    context = engine.assemble_context("summarise this", document_id="doc-1")

    assert context["rag"] == []
    assert context["document_summary_info"][
        "source_provenance_unavailable"
    ] is True


def test_a_normal_question_keeps_its_retrieved_chunks(monkeypatch):
    _patch_sources(monkeypatch, "Whole masked document text.")

    context = engine.assemble_context(
        "What does the document say about marks?",
        document_id="doc-1",
    )

    assert len(context["rag"]) == 1
    assert context["rag"][0]["content"] == _chunk()["content"]


def test_summary_without_stored_document_returns_no_semantic_chunks(
    monkeypatch,
):
    _patch_sources(monkeypatch, None)

    context = engine.assemble_context("summarise this", document_id="doc-1")

    assert context["rag"] == []


def test_is_summary_request_recognises_common_spellings():
    for query in (
        "summarise this",
        "Summarize the document",
        "give me a summary",
        "tl;dr",
    ):
        assert engine.is_summary_request(query) is True

    for query in ("what is the deadline?", "who wrote this?"):
        assert engine.is_summary_request(query) is False
