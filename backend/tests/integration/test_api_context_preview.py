"""Tests for the read-only context preview endpoint."""

import app.main as main_module


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


def test_context_preview_reports_sources_without_generating_an_answer(
    api_client,
    monkeypatch,
):
    captured = {}
    context = _context(
        memory=[{"memory_type": "fact", "content": "Likes concise answers."}],
        rag=[
            {
                "content_type": "text",
                "content": "The return window is 30 days.",
                "document_id": "doc-1",
                "page_number": 2,
            }
        ],
        knowledge_graph=[
            {
                "source": "Acme",
                "relationship": "RELATES_TO",
                "target": "Policy",
            }
        ],
    )

    def fake_assemble_context(**kwargs):
        captured.update(kwargs)
        return context

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("The preview must not generate an answer.")

    monkeypatch.setattr(
        main_module,
        "assemble_context",
        fake_assemble_context,
    )
    monkeypatch.setattr(main_module, "generate_response", fail_if_called)
    monkeypatch.setattr(
        main_module,
        "generate_multimodal_response",
        fail_if_called,
    )

    response = api_client.post(
        "/context/preview",
        json={
            "query": "  What is the policy?  ",
            "top_k": 3,
            "conversation_id": "9b19f12f-7212-4a7f-bf44-7ec0c23c9542",
            "user_id": "user-1",
        },
    )

    assert response.status_code == 200
    body = response.json()

    # The query is trimmed, the limits reach assemble_context unchanged,
    # and every source is reported as structured data.
    assert captured == {
        "query": "What is the policy?",
        "top_k": 3,
        "document_id": None,
        "conversation_id": "9b19f12f-7212-4a7f-bf44-7ec0c23c9542",
        "user_id": "user-1",
    }
    assert body["query"] == "What is the policy?"
    assert body["sources"]["memory"] == context["memory"]
    assert body["sources"]["rag"] == context["rag"]
    assert body["sources"]["knowledge_graph"] == context["knowledge_graph"]
    assert body["sources"]["conversation"] is None
    assert body["source_errors"] == []
    assert body["formatted_context"] == main_module.format_context(context)


def test_context_preview_counts_only_resolvable_image_units(
    api_client,
    monkeypatch,
):
    context = _context(
        rag=[
            {"content_type": "image", "image_id": "known.png"},
            {"content_type": "image", "image_id": "missing.png"},
            {"content_type": "text", "content": "not an image"},
        ]
    )

    monkeypatch.setattr(
        main_module,
        "assemble_context",
        lambda **_kwargs: context,
    )
    monkeypatch.setattr(
        main_module,
        "resolve_image_path",
        lambda result: (
            "C:/server/storage/images/known.png"
            if result.get("image_id") == "known.png"
            else None
        ),
    )

    response = api_client.post(
        "/context/preview",
        json={"query": "Show the chart"},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["image_references"] == 1
    # Unit metadata is reported, but the resolved server-side file path
    # must never be exposed to the client.
    assert body["sources"]["rag"][0]["image_id"] == "known.png"
    assert "C:/server" not in str(body)
    assert "storage" not in body["formatted_context"]


def test_context_preview_surfaces_source_errors(
    api_client,
    monkeypatch,
):
    context = _context(
        source_errors=[
            {"source": "knowledge_graph", "error": "Neo4j is not configured."}
        ]
    )

    monkeypatch.setattr(
        main_module,
        "assemble_context",
        lambda **_kwargs: context,
    )

    response = api_client.post(
        "/context/preview",
        json={"query": "Anything"},
    )

    assert response.status_code == 200
    assert response.json()["source_errors"] == context["source_errors"]


def test_context_preview_rejects_blank_query(api_client):
    response = api_client.post("/context/preview", json={"query": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "Query cannot be empty."


def test_context_preview_rejects_missing_query(api_client):
    response = api_client.post("/context/preview", json={})

    assert response.status_code == 422


def test_context_preview_rejects_top_k_above_the_limit(api_client):
    response = api_client.post(
        "/context/preview",
        json={"query": "Anything", "top_k": 21},
    )

    assert response.status_code == 422


def test_context_preview_reports_assembly_failure(api_client, monkeypatch):
    def failing_assemble_context(**_kwargs):
        raise RuntimeError("unexpected assembly failure")

    monkeypatch.setattr(
        main_module,
        "assemble_context",
        failing_assemble_context,
    )

    response = api_client.post(
        "/context/preview",
        json={"query": "Anything"},
    )

    assert response.status_code == 500
    assert "Context assembly failed" in response.json()["detail"]
