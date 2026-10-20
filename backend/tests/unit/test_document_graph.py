"""Unit tests for the document-scoped knowledge graph query.

Neo4j is replaced by a recorder: no graph is contacted and nothing is
written.
"""

import pytest

from app.context_engine import graph_service


class RecordingSession:
    def __init__(self):
        self.runs = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def run(self, query, **parameters):
        self.runs.append((query, parameters))

        if "MENTIONS]->(entity:Entity)" in query:
            return iter(
                [
                    {
                        "entity_id": "person:ada",
                        "name": "Ada",
                        "entity_type": "PERSON",
                    }
                ]
            )

        return iter(
            [
                {
                    "source_id": "person:ada",
                    "source": "Ada",
                    "relationship": "RELATES_TO",
                    "target_id": "org:acme",
                    "target": "Acme",
                }
            ]
        )


class RecordingDriver:
    def __init__(self):
        self.session_log = RecordingSession()

    def session(self, database=None):
        return self.session_log


def test_document_graph_returns_entities_and_internal_edges(monkeypatch):
    driver = RecordingDriver()
    monkeypatch.setattr(graph_service, "get_driver", lambda: driver)

    graph = graph_service.find_document_graph("doc-1", limit=10)

    assert graph["document_id"] == "doc-1"
    assert graph["entities"] == [
        {
            "entity_id": "person:ada",
            "name": "Ada",
            "entity_type": "PERSON",
        }
    ]
    assert graph["edges"] == [
        {
            "source_id": "person:ada",
            "source": "Ada",
            "relationship": "RELATES_TO",
            "target_id": "org:acme",
            "target": "Acme",
        }
    ]

    edge_query, parameters = driver.session_log.runs[1]
    assert parameters == {"document_id": "doc-1", "limit": 10}
    # Both ends must be mentioned by the same document.
    assert "MATCH (document)-[:MENTIONS]->(target)" in edge_query
    assert (
        "$document_id IN coalesce(relationship.document_ids, [])"
        in edge_query
    )


def test_document_graph_queries_never_write(monkeypatch):
    driver = RecordingDriver()
    monkeypatch.setattr(graph_service, "get_driver", lambda: driver)

    graph_service.find_document_graph("doc-1")

    for query, _parameters in driver.session_log.runs:
        upper = query.upper()
        assert upper.strip().startswith("MATCH")
        for keyword in ("MERGE", "CREATE", "SET ", "DELETE", "REMOVE"):
            assert keyword not in upper


def test_document_graph_rejects_an_empty_document_id():
    with pytest.raises(ValueError):
        graph_service.find_document_graph("   ")


def test_document_graph_bounds_the_edge_limit():
    with pytest.raises(ValueError):
        graph_service.find_document_graph("doc-1", limit=0)

    with pytest.raises(ValueError):
        graph_service.find_document_graph("doc-1", limit=501)
