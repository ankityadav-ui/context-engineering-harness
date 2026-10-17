"""Tests for the knowledge-graph endpoints.

Neo4j is replaced by a fake driver, so no graph connection is opened.
The missing-configuration test uses the service's own guard.
"""

import pytest
from fastapi.testclient import TestClient

from app.context_engine import graph_service
from app.main import app


class FakeResult:
    def __init__(self, records):
        self.records = records

    def __iter__(self):
        return iter(self.records)

    def single(self):
        if not self.records:
            return None

        return self.records[0]


class FakeSession:
    def __init__(self, records):
        self.records = records

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def run(self, query, **parameters):
        return FakeResult(self.records)


class FakeDriver:
    def __init__(self, records):
        self.records = records

    def session(self, database=None):
        return FakeSession(self.records)


ENTITY_RECORDS = [
    {
        "entity_id": "entity-1",
        "name": "Heart",
        "entity_type": "ORGAN",
    },
    {
        "entity_id": "entity-2",
        "name": "Lung",
        "entity_type": "ORGAN",
    },
]


@pytest.fixture()
def fake_driver(monkeypatch):
    def use_records(records):
        monkeypatch.setattr(
            graph_service,
            "get_driver",
            lambda: FakeDriver(records),
        )

    return use_records


def test_entities_are_listed(fake_driver):
    fake_driver(ENTITY_RECORDS)

    with TestClient(app) as client:
        response = client.get("/knowledge-graph/entities")

    assert response.status_code == 200
    assert response.json() == {
        "entities": ENTITY_RECORDS,
        "search": None,
    }


def test_entities_can_be_searched_by_name(fake_driver):
    fake_driver(ENTITY_RECORDS[:1])

    with TestClient(app) as client:
        response = client.get(
            "/knowledge-graph/entities",
            params={"search": " heart "},
        )

    assert response.status_code == 200

    body = response.json()

    assert body["search"] == "heart"
    assert body["entities"] == ENTITY_RECORDS[:1]


def test_entity_relationships_are_returned(fake_driver):
    fake_driver(
        [
            {
                "outgoing_relationships": [
                    {
                        "source": "Heart",
                        "relationship": "PART_OF",
                        "target": "Body",
                        "direction": "outgoing",
                    }
                ],
                "incoming_relationships": [None],
            }
        ]
    )

    with TestClient(app) as client:
        response = client.get(
            "/knowledge-graph/entities/entity-1/relationships"
        )

    assert response.status_code == 200

    # Empty optional matches are dropped from the response.
    assert response.json() == {
        "entity_id": "entity-1",
        "relationships": [
            {
                "source": "Heart",
                "relationship": "PART_OF",
                "target": "Body",
                "direction": "outgoing",
            }
        ],
    }


def test_unknown_entity_returns_an_empty_relationship_list(fake_driver):
    fake_driver([])

    with TestClient(app) as client:
        response = client.get(
            "/knowledge-graph/entities/entity-missing/relationships"
        )

    assert response.status_code == 200
    assert response.json()["relationships"] == []


def test_knowledge_graph_reports_missing_configuration(monkeypatch):
    monkeypatch.setattr(graph_service, "driver", None)

    with TestClient(app) as client:
        entities = client.get("/knowledge-graph/entities")
        relationships = client.get(
            "/knowledge-graph/entities/entity-1/relationships"
        )

    assert entities.status_code == 503
    assert "NEO4J_URI" in entities.json()["detail"]

    assert relationships.status_code == 503
    assert "NEO4J_URI" in relationships.json()["detail"]


def test_entity_limit_is_validated():
    with TestClient(app) as client:
        response = client.get(
            "/knowledge-graph/entities",
            params={"limit": 500},
        )

    assert response.status_code == 422
