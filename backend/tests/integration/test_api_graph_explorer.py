"""Graph Explorer endpoint tests.

The graph service calls are replaced with recorders, so these run without
Neo4j and pin the request/response contract, the parameter passing, and
the 400/503 error mapping.
"""

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.context_engine import graph_service
from app.main import app


@pytest.fixture()
def graph_configured(monkeypatch):
    monkeypatch.setattr(graph_service, "is_configured", lambda: True)


def test_status_reports_reachability_and_counts(
    graph_configured, monkeypatch
):
    verified = []
    monkeypatch.setattr(
        main_module,
        "verify_graph_connection",
        lambda: verified.append(True),
    )
    monkeypatch.setattr(
        main_module,
        "graph_counts",
        lambda: {
            "entities": 4,
            "documents": 2,
            "relationships": 6,
            "entity_types": [{"entity_type": "PERSON", "entities": 3}],
        },
    )

    with TestClient(app) as client:
        response = client.get("/knowledge-graph/status")

    assert response.status_code == 200
    assert response.json() == {
        "available": True,
        "counts": {
            "entities": 4,
            "documents": 2,
            "relationships": 6,
            "entity_types": [{"entity_type": "PERSON", "entities": 3}],
        },
    }
    assert verified == [True]


def test_status_reports_missing_configuration(monkeypatch):
    monkeypatch.setattr(graph_service, "is_configured", lambda: False)

    with TestClient(app) as client:
        response = client.get("/knowledge-graph/status")

    assert response.status_code == 503
    assert "NEO4J_URI" in response.json()["detail"]


def test_status_reports_unreachable_graph_without_leaking_details(
    graph_configured, monkeypatch
):
    def explode():
        raise OSError(
            "connection refused to bolt://user:secret@internal-host:7687"
        )

    monkeypatch.setattr(main_module, "verify_graph_connection", explode)

    with TestClient(app) as client:
        response = client.get("/knowledge-graph/status")

    assert response.status_code == 503

    detail = response.json()["detail"]

    # The caller learns the graph is down, not the host or credentials.
    assert detail == "Knowledge graph is unreachable."
    assert "secret" not in detail
    assert "internal-host" not in detail


def test_status_maps_missing_driver_to_503(graph_configured, monkeypatch):
    def explode():
        raise RuntimeError(
            "Neo4j is not configured. Set NEO4J_URI and NEO4J_PASSWORD."
        )

    monkeypatch.setattr(main_module, "verify_graph_connection", explode)

    with TestClient(app) as client:
        response = client.get("/knowledge-graph/status")

    assert response.status_code == 503
    assert "NEO4J_URI" in response.json()["detail"]


def test_entities_are_listed_with_paging_and_type_filter(
    graph_configured, monkeypatch
):
    calls = []

    def fake_list(limit, offset=0, entity_type=None):
        calls.append(
            {
                "limit": limit,
                "offset": offset,
                "entity_type": entity_type,
            }
        )
        return [
            {
                "entity_id": "person:ada",
                "name": "Ada",
                "entity_type": "PERSON",
            }
        ]

    monkeypatch.setattr(main_module, "list_entities", fake_list)

    with TestClient(app) as client:
        response = client.get(
            "/knowledge-graph/entities",
            params={"limit": 10, "offset": 20, "entity_type": " person "},
        )

    assert response.status_code == 200
    assert response.json() == {
        "entities": [
            {
                "entity_id": "person:ada",
                "name": "Ada",
                "entity_type": "PERSON",
            }
        ],
        "search": None,
    }
    assert calls == [
        {"limit": 10, "offset": 20, "entity_type": "person"}
    ]


def test_entities_search_passes_limit_and_type(graph_configured, monkeypatch):
    calls = []

    def fake_find(search_text, limit=20, entity_type=None):
        calls.append(
            {
                "search_text": search_text,
                "limit": limit,
                "entity_type": entity_type,
            }
        )
        return []

    monkeypatch.setattr(main_module, "find_entities", fake_find)

    with TestClient(app) as client:
        response = client.get(
            "/knowledge-graph/entities",
            params={"search": " ada ", "limit": 5, "entity_type": "ORG"},
        )

    assert response.status_code == 200
    assert response.json() == {"entities": [], "search": "ada"}
    assert calls == [
        {"search_text": "ada", "limit": 5, "entity_type": "ORG"}
    ]


def test_entities_reject_out_of_range_paging(graph_configured):
    with TestClient(app) as client:
        too_many = client.get(
            "/knowledge-graph/entities", params={"limit": 500}
        )
        negative = client.get(
            "/knowledge-graph/entities", params={"offset": -5}
        )

    assert too_many.status_code == 422
    assert negative.status_code == 422


def test_entities_service_value_error_is_a_client_error(
    graph_configured, monkeypatch
):
    def explode(limit, offset=0, entity_type=None):
        raise ValueError("limit must be between 1 and 100.")

    monkeypatch.setattr(main_module, "list_entities", explode)

    with TestClient(app) as client:
        response = client.get("/knowledge-graph/entities")

    assert response.status_code == 400
    assert "limit" in response.json()["detail"]


def test_entities_report_missing_configuration(monkeypatch):
    def no_driver(limit, offset=0, entity_type=None):
        raise RuntimeError(
            "Neo4j is not configured. Set NEO4J_URI and "
            "NEO4J_PASSWORD in the root .env file."
        )

    monkeypatch.setattr(main_module, "list_entities", no_driver)

    with TestClient(app) as client:
        response = client.get("/knowledge-graph/entities")

    assert response.status_code == 503
    assert "NEO4J_URI" in response.json()["detail"]


def test_neighborhood_returns_entity_and_edges(graph_configured, monkeypatch):
    monkeypatch.setattr(
        main_module,
        "get_entity_neighborhood",
        lambda entity_id, limit=25: {
            "entity": {
                "entity_id": entity_id,
                "name": "Ada",
                "entity_type": "PERSON",
            },
            "edges": [
                {
                    "source_id": entity_id,
                    "source": "Ada",
                    "relationship": "RELATES_TO",
                    "target_id": "org:acme",
                    "target": "Acme",
                    "direction": "outgoing",
                }
            ],
        },
    )

    with TestClient(app) as client:
        response = client.get(
            "/knowledge-graph/entities/person:ada/neighborhood"
        )

    assert response.status_code == 200

    body = response.json()

    assert body["entity_id"] == "person:ada"
    assert body["entity"]["name"] == "Ada"
    assert body["edges"][0]["target_id"] == "org:acme"


def test_neighborhood_of_an_unknown_entity_is_empty(
    graph_configured, monkeypatch
):
    monkeypatch.setattr(
        main_module,
        "get_entity_neighborhood",
        lambda entity_id, limit=25: {"entity": None, "edges": []},
    )

    with TestClient(app) as client:
        response = client.get(
            "/knowledge-graph/entities/person:missing/neighborhood"
        )

    assert response.status_code == 200
    assert response.json() == {
        "entity_id": "person:missing",
        "entity": None,
        "edges": [],
    }


def test_neighborhood_rejects_a_blank_entity_id(graph_configured):
    with TestClient(app) as client:
        response = client.get(
            "/knowledge-graph/entities/%20/neighborhood"
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "entity_id cannot be empty."


def test_neighborhood_maps_a_value_error_to_400(
    graph_configured, monkeypatch
):
    def explode(entity_id, limit=25):
        raise ValueError("limit must be between 1 and 100.")

    monkeypatch.setattr(main_module, "get_entity_neighborhood", explode)

    with TestClient(app) as client:
        response = client.get(
            "/knowledge-graph/entities/person:ada/neighborhood"
        )

    assert response.status_code == 400
    assert "limit" in response.json()["detail"]


def test_neighborhood_maps_an_unreachable_graph_to_503(
    graph_configured, monkeypatch
):
    def unreachable(entity_id, limit=25):
        raise RuntimeError(
            "Neo4j is not configured. Set NEO4J_URI and "
            "NEO4J_PASSWORD in the root .env file."
        )

    monkeypatch.setattr(main_module, "get_entity_neighborhood", unreachable)

    with TestClient(app) as client:
        response = client.get(
            "/knowledge-graph/entities/person:ada/neighborhood"
        )

    assert response.status_code == 503
    assert "NEO4J_URI" in response.json()["detail"]


def test_neighborhood_limit_is_validated(graph_configured):
    with TestClient(app) as client:
        response = client.get(
            "/knowledge-graph/entities/person:ada/neighborhood",
            params={"limit": 0},
        )

    assert response.status_code == 422


def test_entity_documents_are_listed(graph_configured, monkeypatch):
    calls = []

    def fake_documents(entity_id, limit=50):
        calls.append({"entity_id": entity_id, "limit": limit})
        return [
            {
                "document_id": "doc-1",
                "title": "Resume",
                "source": "document-upload",
            }
        ]

    monkeypatch.setattr(main_module, "find_entity_documents", fake_documents)

    with TestClient(app) as client:
        response = client.get(
            "/knowledge-graph/entities/person:ada/documents",
            params={"limit": 5},
        )

    assert response.status_code == 200
    assert response.json() == {
        "entity_id": "person:ada",
        "documents": [
            {
                "document_id": "doc-1",
                "title": "Resume",
                "source": "document-upload",
            }
        ],
    }
    assert calls == [{"entity_id": "person:ada", "limit": 5}]


def test_entity_documents_reject_a_blank_id(graph_configured):
    with TestClient(app) as client:
        response = client.get("/knowledge-graph/entities/%20/documents")

    assert response.status_code == 400
    assert response.json()["detail"] == "entity_id cannot be empty."


def test_entity_documents_report_missing_configuration(monkeypatch):
    def no_driver(entity_id, limit=50):
        raise RuntimeError(
            "Neo4j is not configured. Set NEO4J_URI and "
            "NEO4J_PASSWORD in the root .env file."
        )

    monkeypatch.setattr(main_module, "find_entity_documents", no_driver)

    with TestClient(app) as client:
        response = client.get(
            "/knowledge-graph/entities/person:ada/documents"
        )

    assert response.status_code == 503
    assert "NEO4J_URI" in response.json()["detail"]
