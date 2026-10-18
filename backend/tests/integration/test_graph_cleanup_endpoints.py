"""Tests for the knowledge-graph cleanup endpoints.

Neo4j is replaced by a fake driver: the GET report must issue only
read-only queries, and the POST endpoint must not write anything
unless confirm=true is sent explicitly.
"""

import pytest
from fastapi.testclient import TestClient

from app.context_engine import graph_cleanup, graph_service
from app.main import app


WRITE_KEYWORDS = (
    "MERGE",
    "CREATE",
    "DELETE",
    "DETACH",
    "SET ",
    "REMOVE ",
    "DROP",
    "FOREACH",
)


class FakeResult:
    def __init__(self, records=()):
        self._records = list(records)

    def __iter__(self):
        return iter(self._records)

    def single(self):
        return self._records[0] if self._records else None


class FakeSession:
    def __init__(self, reads):
        self.reads = reads
        self.all_queries: list[str] = []
        # Simulates DETACH DELETE so the post-execution re-plan sees
        # the graph the way it would look after the real cleanup.
        self.deleted_entity_ids: set[str] = set()

    def _visible(self, records):
        if not self.deleted_entity_ids:
            return records

        visible = []

        for record in records:
            if record.get("entity_id") in self.deleted_entity_ids:
                continue

            start = record.get("start")
            end = record.get("end")

            if start in self.deleted_keys or end in self.deleted_keys:
                continue

            visible.append(record)

        return visible

    @property
    def deleted_keys(self) -> set[str]:
        keys = set()

        for record in self.reads.get(
            "WHERE n:Entity OR n:Document OR n:Case", []
        ):
            if record.get("entity_id") in self.deleted_entity_ids:
                keys.add(record["key"])

        return keys

    def run(self, query, **parameters):
        stripped = " ".join(query.split())
        self.all_queries.append(stripped)

        if (
            "DETACH DELETE dup" in stripped
            and parameters.get("dup")
        ):
            self.deleted_entity_ids.add(parameters["dup"])

        for fragment, records in self.reads.items():
            if fragment in stripped:
                return FakeResult(self._visible(records))

        return FakeResult()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeDriver:
    def __init__(self, session):
        self._session = session

    def session(self, database=None):
        return self._session


@pytest.fixture()
def fake_graph(monkeypatch):
    """A graph with one duplicate pair and one placeholder entity."""

    session = FakeSession(
        reads={
            # fetch_snapshot: nodes
            "WHERE n:Entity OR n:Document OR n:Case": [
                {
                    "key": "d1",
                    "labels": ["Document"],
                    "entity_id": None,
                    "name": None,
                    "entity_type": None,
                    "document_id": "doc-1",
                    "source": "document-upload",
                },
                {
                    "key": "e1",
                    "labels": ["Entity"],
                    "entity_id": "org:acme corp",
                    "name": "Acme Corp.",
                    "entity_type": "ORG",
                    "document_id": None,
                    "source": None,
                },
                {
                    "key": "e2",
                    "labels": ["Entity"],
                    "entity_id": "org:acme_corp",
                    "name": "Acme Corp",
                    "entity_type": "ORG",
                    "document_id": None,
                    "source": None,
                },
            ],
            # fetch_snapshot: relationships
            "WHERE a:Entity OR b:Entity": [
                {"start": "d1", "end": "e1", "type": "MENTIONS"},
                {"start": "d1", "end": "e2", "type": "MENTIONS"},
            ],
            # label counts
            "UNWIND labels(n) AS label": [
                {"label": "Document", "count": 1},
                {"label": "Entity", "count": 2},
                {"label": "Movie", "count": 40},
            ],
            # relationship counts
            "MATCH ()-[r]->()": [
                {"type": "MENTIONS", "count": 2},
                {"type": "ACTED_IN", "count": 37},
            ],
        }
    )

    driver = FakeDriver(session)

    monkeypatch.setattr(graph_cleanup, "get_driver", lambda: driver)
    monkeypatch.setattr(graph_service, "driver", driver)

    yield session


def test_cleanup_report_is_read_only(fake_graph):
    with TestClient(app) as client:
        response = client.get("/knowledge-graph/cleanup-report")

    assert response.status_code == 200

    body = response.json()

    assert body["destructive_changes_performed"] is False
    assert "dry-run" in body["mode"]
    assert body["counts"]["labels"]["Movie"] == 40
    assert body["scope"]["owned_entities"] == 2
    assert body["planned_actions"]["merges"] == 1

    # Every query issued for the report is read-only.
    for query in fake_graph.all_queries:
        for keyword in WRITE_KEYWORDS:
            assert keyword not in query.upper().replace("count(", "")


def test_cleanup_post_defaults_to_dry_run(fake_graph):
    before = len(fake_graph.all_queries)

    with TestClient(app) as client:
        response = client.post("/knowledge-graph/cleanup", json={})

    assert response.status_code == 200

    body = response.json()

    assert body["dry_run"] is True
    assert body["executed"] is False
    assert body["planned"]["merges"] == 1

    # No write query was issued after the dry-run request started.
    new_queries = fake_graph.all_queries[before:]

    for query in new_queries:
        for keyword in WRITE_KEYWORDS:
            assert keyword not in query.upper().replace("count(", "")


def test_cleanup_with_confirm_executes_and_reports_idempotency(
    fake_graph,
):
    with TestClient(app) as client:
        response = client.post(
            "/knowledge-graph/cleanup",
            json={"confirm": True},
        )

    assert response.status_code == 200

    body = response.json()

    assert body["dry_run"] is False
    assert body["executed"] is True
    assert body["applied"]["errors"] == []
    assert len(body["applied"]["merges"]) == 1

    # After the merge, the fresh plan reports no remaining actions,
    # proving the cleanup is idempotent.
    assert body["remaining_after_cleanup"]["merges"] == 0
    assert body["remaining_after_cleanup"]["removals"] == 0


def test_cleanup_reports_missing_configuration(monkeypatch):
    monkeypatch.setattr(graph_service, "driver", None)

    with TestClient(app) as client:
        report = client.get("/knowledge-graph/cleanup-report")
        execute = client.post(
            "/knowledge-graph/cleanup", json={"confirm": True}
        )

    assert report.status_code == 503
    assert "NEO4J_URI" in report.json()["detail"]
    assert execute.status_code == 503
    assert "NEO4J_URI" in execute.json()["detail"]
