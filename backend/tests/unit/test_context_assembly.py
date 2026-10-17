"""Pure unit tests for context formatting.

These do not call PostgreSQL, Pinecone, or Neo4j.
"""

from app.context_engine.engine import format_context


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
