"""Tests for the knowledge-graph endpoints.

Neo4j is replaced by a fake driver, so no graph connection is opened.
The missing-configuration test uses the service's own guard.
"""

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.context_engine import graph_service
from app.context_engine import llm_graph_transformer
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


# --------------------------------------------------
# Ingestion wiring: the upload route really reaches the LLM transformer
# --------------------------------------------------


class GraphDocument:
    def __init__(self, nodes, relationships):
        self.nodes = nodes
        self.relationships = relationships


class PromptRecordingTransformer:
    def __init__(self):
        self.calls = []

    def convert_to_graph_documents(self, documents):
        self.calls.append(list(documents))

        return [
            GraphDocument(
                nodes=[{"id": "Ada Lovelace", "type": "Person"}],
                relationships=[],
            )
            for _ in documents
        ]


class WriteRecordingSession:
    def __init__(self):
        self.runs = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def run(self, query, **parameters):
        self.runs.append((query, parameters))

        class _Result:
            def __iter__(self):
                return iter(())

            def single(self):
                if "MERGE (document:Document" in query:
                    return dict(parameters)
                if "MERGE (entity:Entity" in query:
                    return dict(parameters)

                return None

        return _Result()

    def execute_write(self, callback):
        return callback(self)


class WriteRecordingDriver:
    def __init__(self):
        self.session_log = WriteRecordingSession()

    def session(self, database=None):
        return self.session_log


def test_upload_ingests_through_the_llm_transformer(
    tmp_path, monkeypatch
):
    """The production upload path must call the LangChain transformer."""

    units = [
        {
            "id": "source-unit-2",
            "content_type": "text",
            "page_number": 2,
            "content": "Ada Lovelace worked at Acme Corp.",
        }
    ]

    monkeypatch.setattr(main_module, "DOCUMENT_DIR", tmp_path)
    monkeypatch.setattr(
        main_module,
        "process_document",
        lambda **_kwargs: (
            units,
            0,
            [],
            [],
            {"applied": False, "safety_passed": True, "masked_entities": 0},
        ),
    )
    monkeypatch.setattr(graph_service, "is_configured", lambda: True)

    driver = WriteRecordingDriver()
    monkeypatch.setattr(graph_service, "get_driver", lambda: driver)

    transformer = PromptRecordingTransformer()
    monkeypatch.setattr(
        llm_graph_transformer,
        "build_transformer",
        lambda **_kwargs: transformer,
    )

    with TestClient(app) as client:
        response = client.post(
            "/documents/upload",
            files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )

        assert response.status_code == 202, response.text
        job = client.get(
            f"/documents/jobs/{response.json()['job_id']}"
        ).json()
        assert job["status"] == "completed", job

    body = {
        "document_id": job["document_id"],
        **job["result"],
    }

    assert body["knowledge_graph"]["ingested"] is True
    assert body["knowledge_graph"]["error"] is None
    assert body["knowledge_graph"]["entities"] == 1

    # The model really ran, on the page text with its provenance intact.
    assert len(transformer.calls) == 1
    sent = transformer.calls[0][0]
    assert sent.page_content == "Ada Lovelace worked at Acme Corp."
    assert sent.metadata["page_number"] == 2
    assert sent.metadata["document_id"] == body["document_id"]
    assert sent.metadata["source_unit_id"] == "source-unit-2"

    # And the validated node reached Neo4j with page provenance.
    mention_runs = [
        parameters
        for query, parameters in driver.session_log.runs
        if "[mention:MENTIONS]" in query and "entity_id" in parameters
    ]

    assert len(mention_runs) == 1
    assert mention_runs[0]["entity_id"] == "person:ada_lovelace"
    assert mention_runs[0]["page_numbers"] == [2]
    assert mention_runs[0]["document_id"] == body["document_id"]
    assert mention_runs[0]["source_refs"] == [
        f"{body['document_id']}#page=2#unit=source-unit-2"
    ]


def test_document_graph_endpoint_returns_entities_and_edges(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "find_document_graph",
        lambda document_id: {
            "document_id": document_id,
            "entities": ENTITY_RECORDS,
            "edges": [
                {
                    "source_id": "entity-1",
                    "source": "Heart",
                    "relationship": "RELATES_TO",
                    "target_id": "entity-2",
                    "target": "Lung",
                }
            ],
        },
    )

    with TestClient(app) as client:
        response = client.get("/knowledge-graph/documents/doc-1/graph")

    assert response.status_code == 200

    body = response.json()

    assert body["document_id"] == "doc-1"
    assert body["entities"] == ENTITY_RECORDS
    assert body["edges"][0]["relationship"] == "RELATES_TO"


def test_entity_limit_is_validated():
    with TestClient(app) as client:
        response = client.get(
            "/knowledge-graph/entities",
            params={"limit": 500},
        )

    assert response.status_code == 422
