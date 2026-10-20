from app.context_engine import engine


def test_document_summary_context_is_scoped_and_excludes_other_sources(
    monkeypatch,
):
    def unexpected_source(**_kwargs):
        raise AssertionError("document-only summary read unrelated context")

    monkeypatch.setattr(engine, "retrieve_context", unexpected_source)
    monkeypatch.setattr(engine, "_get_conversation_context", unexpected_source)
    monkeypatch.setattr(engine, "_get_memory_context", unexpected_source)
    monkeypatch.setattr(engine, "_get_graph_context", unexpected_source)

    def indexed_units(document_id):
        return {
            "document_id": document_id,
            "document_found": True,
            "page_count": 3,
            "pages_without_text": [2],
            "units_truncated": False,
            "units": [
                {
                    "unit_id": "chunk-page-1",
                    "document_id": document_id,
                    "content_type": "text",
                    "page_number": 1,
                    "content": "Document-specific first page text.",
                    "source": "pdf_text",
                },
                {
                    "unit_id": "duplicate-chunk",
                    "document_id": document_id,
                    "content_type": "text",
                    "page_number": 1,
                    "content": "Document-specific first page text.",
                    "source": "pdf_text",
                },
                {
                    "unit_id": "wrong-document",
                    "document_id": "another-document",
                    "content_type": "text",
                    "page_number": 9,
                    "content": "Unrelated content must never enter.",
                    "source": "pdf_text",
                },
                {
                    "unit_id": "chunk-page-3",
                    "document_id": document_id,
                    "content_type": "table",
                    "page_number": 3,
                    "content": "Revenue | 20",
                    "source": "pdf_table",
                },
            ],
        }

    from app.rag import pinecone_service

    monkeypatch.setattr(
        pinecone_service,
        "get_document_searchable_units",
        indexed_units,
    )

    context = engine.assemble_context(
        query="summarize this",
        document_id="selected-document",
        conversation_id="conversation-1",
        user_id="user-1",
    )

    assert [unit["unit_id"] for unit in context["rag"]] == [
        "chunk-page-1",
        "chunk-page-3",
    ]
    assert [unit["page_number"] for unit in context["rag"]] == [1, 3]
    assert context["memory"] == []
    assert context["knowledge_graph"] == []
    assert context["conversation"] is None
    assert context["document_summary_info"]["pages_without_text"] == [2]
