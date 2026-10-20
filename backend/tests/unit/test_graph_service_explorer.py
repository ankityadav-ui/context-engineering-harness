"""Read-only Graph Explorer service tests.

Neo4j is replaced by a recording driver, so no connection is opened and
every query is inspected for its parameters. These pin the paging,
type-filter, document-link, and neighborhood behaviour the explorer
depends on.
"""

import pytest

from app.context_engine import graph_service


class FakeResult:
    def __init__(self, records):
        self.records = records

    def __iter__(self):
        return iter(self.records)

    def single(self):
        if not self.records:
            return None

        return self.records[0]


class RecordingSession:
    """Return scripted records and remember every query and parameter."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def run(self, query, **parameters):
        self.calls.append({"query": query, "parameters": parameters})

        for marker, records in self.responses:
            if marker in query:
                return FakeResult(records)

        return FakeResult([])


class RecordingDriver:
    def __init__(self, responses):
        self.session_instance = RecordingSession(responses)

    def session(self, database=None):
        return self.session_instance


@pytest.fixture()
def driver(monkeypatch):
    def install(responses):
        recording = RecordingDriver(responses)

        monkeypatch.setattr(
            graph_service, "get_driver", lambda: recording
        )

        return recording

    return install


def test_graph_counts_reports_totals_and_types(driver):
    recording = driver(
        [
            ("MATCH (entity:Entity) RETURN count(entity)", [{"count": 7}]),
            (
                "MATCH (document:Document) RETURN count(document)",
                [{"count": 3}],
            ),
            ("MATCH ()-[r]->() RETURN count(r)", [{"count": 9}]),
            (
                "RETURN entity.entity_type AS entity_type",
                [
                    {"entity_type": "PERSON", "entities": 4},
                    {"entity_type": "ORG", "entities": 3},
                ],
            ),
        ]
    )

    counts = graph_service.graph_counts()

    assert counts == {
        "entities": 7,
        "documents": 3,
        "relationships": 9,
        "entity_types": [
            {"entity_type": "PERSON", "entities": 4},
            {"entity_type": "ORG", "entities": 3},
        ],
    }
    # Read-only: only MATCH/RETURN queries were issued.
    assert all(
        call["query"].lstrip().upper().startswith(("MATCH", "WITH"))
        for call in recording.session_instance.calls
    )


def test_list_entities_pages_and_filters_by_type(driver):
    recording = driver(
        [("MATCH (entity:Entity)", [{"entity_id": "person:ada"}])]
    )

    entities = graph_service.list_entities(
        limit=10, offset=20, entity_type="person"
    )

    assert entities == [{"entity_id": "person:ada"}]

    call = recording.session_instance.calls[0]

    # Type and paging values travel as parameters, never interpolated.
    assert call["parameters"] == {
        "limit": 10,
        "offset": 20,
        "entity_type": "PERSON",
    }
    assert "$entity_type" in call["query"]
    assert "SKIP $offset" in call["query"]


def test_list_entities_rejects_out_of_range_paging():
    with pytest.raises(ValueError):
        graph_service.list_entities(limit=0)

    with pytest.raises(ValueError):
        graph_service.list_entities(limit=101)

    with pytest.raises(ValueError):
        graph_service.list_entities(offset=-1)


def test_find_entities_requires_search_text():
    with pytest.raises(ValueError):
        graph_service.find_entities("   ")


def test_find_entities_applies_type_filter(driver):
    recording = driver([("CONTAINS toLower", [])])

    graph_service.find_entities("ada", limit=5, entity_type="ORG")

    parameters = recording.session_instance.calls[0]["parameters"]

    assert parameters["search_text"] == "ada"
    assert parameters["limit"] == 5
    assert parameters["entity_type"] == "ORG"


def test_find_entity_documents_returns_graph_links(driver):
    recording = driver(
        [
            (
                "-[:MENTIONS]->",
                [
                    {
                        "document_id": "doc-1",
                        "title": "Resume",
                        "source": "document-upload",
                    }
                ],
            )
        ]
    )

    documents = graph_service.find_entity_documents("person:ada", limit=5)

    assert documents[0]["document_id"] == "doc-1"

    parameters = recording.session_instance.calls[0]["parameters"]

    assert parameters == {"entity_id": "person:ada", "limit": 5}


def test_find_entity_documents_requires_an_entity_id():
    with pytest.raises(ValueError):
        graph_service.find_entity_documents("   ")


def test_neighborhood_keeps_ids_and_drops_empty_optionals(driver):
    driver(
        [
            (
                "MATCH (center:Entity",
                [
                    {
                        "center_id": "person:ada",
                        "center_name": "Ada",
                        "center_type": "PERSON",
                        "outgoing_edges": [
                            {
                                "entity_id": "org:acme",
                                "name": "Acme",
                                "entity_type": "ORG",
                                "relationship": "RELATES_TO",
                            },
                            {
                                "entity_id": None,
                                "name": None,
                                "entity_type": None,
                                "relationship": None,
                            },
                        ],
                        "incoming_edges": [
                            {
                                "entity_id": "gpe:london",
                                "name": "London",
                                "entity_type": "GPE",
                                "relationship": "PART_OF",
                            }
                        ],
                    }
                ],
            )
        ]
    )

    result = graph_service.get_entity_neighborhood("person:ada", limit=10)

    assert result["entity"] == {
        "entity_id": "person:ada",
        "name": "Ada",
        "entity_type": "PERSON",
    }

    assert result["edges"] == [
        {
            "source_id": "person:ada",
            "source": "Ada",
            "relationship": "RELATES_TO",
            "target_id": "org:acme",
            "target": "Acme",
            "direction": "outgoing",
        },
        {
            "source_id": "gpe:london",
            "source": "London",
            "relationship": "PART_OF",
            "target_id": "person:ada",
            "target": "Ada",
            "direction": "incoming",
        },
    ]


def test_neighborhood_deduplicates_edges(driver):
    edge = {
        "entity_id": "org:acme",
        "name": "Acme",
        "entity_type": "ORG",
        "relationship": "MENTIONS",
    }

    driver(
        [
            (
                "MATCH (center:Entity",
                [
                    {
                        "center_id": "person:ada",
                        "center_name": "Ada",
                        "center_type": "PERSON",
                        "outgoing_edges": [edge, dict(edge)],
                        "incoming_edges": [],
                    }
                ],
            )
        ]
    )

    result = graph_service.get_entity_neighborhood("person:ada")

    assert len(result["edges"]) == 1


def test_neighborhood_of_an_unknown_entity_is_empty(driver):
    driver([("MATCH (center:Entity", [])])

    assert graph_service.get_entity_neighborhood("person:missing") == {
        "entity": None,
        "edges": [],
    }


def test_neighborhood_validates_inputs():
    with pytest.raises(ValueError):
        graph_service.get_entity_neighborhood("  ")

    with pytest.raises(ValueError):
        graph_service.get_entity_neighborhood("person:ada", limit=0)
