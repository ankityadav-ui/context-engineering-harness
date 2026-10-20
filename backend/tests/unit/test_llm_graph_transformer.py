"""Focused tests for the LLMGraphTransformer ingestion path.

Every test mocks the model and Neo4j: no Gemini request is made and no live
graph is touched.
"""

import pytest

import app.llm as llm
import app.context_engine.graph_service as graph_service
import app.context_engine.llm_graph_transformer as transformer_module
from app.context_engine.llm_graph_transformer import (
    ALLOWED_ENTITY_TYPES,
    ALLOWED_RELATIONSHIP_TYPES,
    ExtractedNode,
    GraphExtractionError,
    audit_suspicious_nodes,
    build_entity_id,
    canonical_entity_type,
    extract_graph,
    is_technical_term,
    validate_and_clean_graph,
)


# --------------------------------------------------
# Test doubles
# --------------------------------------------------


class FakeNode:
    def __init__(self, node_id, node_type):
        self.id = node_id
        self.type = node_type
        self.properties = {}


class FakeRelationship:
    def __init__(self, source, target, rel_type, evidence=""):
        self.source = FakeNode(source, "Node")
        self.target = FakeNode(target, "Node")
        self.type = rel_type
        self.properties = {"evidence": evidence} if evidence else {}


class FakeGraphDocument:
    def __init__(self, nodes=None, relationships=None):
        self.nodes = nodes
        self.relationships = relationships


class FakeTransformer:
    """Returns one prepared graph document per input document."""

    def __init__(self, graph_documents=None, error=None):
        self._graph_documents = graph_documents or []
        self._error = error
        self.calls = []

    def convert_to_graph_documents(self, documents):
        self.calls.append(list(documents))

        if self._error is not None:
            raise self._error

        return self._graph_documents


class RecordingSession:
    def __init__(self, fail_on=None):
        self.runs = []
        self._fail_on = fail_on

    def run(self, query, **parameters):
        if self._fail_on and self._fail_on in query:
            raise OSError("Neo4j is unreachable")

        self.runs.append((query, parameters))

        payload = (
            dict(parameters)
            if "MERGE (document:Document" in query
            or "MERGE (entity:Entity" in query
            else None
        )

        class _Result:
            def __iter__(self):
                return iter(())

            def single(self):
                return payload

        return _Result()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute_write(self, callback):
        previous_runs = list(self.runs)
        try:
            return callback(self)
        except Exception:
            self.runs[:] = previous_runs
            raise


class RecordingDriver:
    def __init__(self, fail_on=None):
        self.session_log = RecordingSession(fail_on=fail_on)

    def session(self, database=None):
        return self.session_log


# --------------------------------------------------
# Schema mapping
# --------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Person", "PERSON"),
        ("person", "PERSON"),
        ("Organization", "ORGANIZATION"),
        ("Company", "ORGANIZATION"),
        ("Technology", "TECHNOLOGY"),
        ("Framework", "TECHNOLOGY"),
        ("Product", "PRODUCT"),
        ("Location", "LOCATION"),
        ("GPE", "LOCATION"),
    ],
)
def test_llm_labels_map_onto_stored_schema(raw, expected):
    assert canonical_entity_type(raw) == expected


@pytest.mark.parametrize("raw", ["Chemical", "Disease", "Quantity", "", None])
def test_unknown_llm_types_are_rejected(raw):
    assert canonical_entity_type(raw) is None


def test_entity_id_is_deterministic_and_slugged():
    assert build_entity_id("PERSON", "  Ada   Lovelace ") == "person:ada_lovelace"
    assert build_entity_id("PERSON", "ada lovelace") == "person:ada_lovelace"


@pytest.mark.parametrize(
    "term",
    [
        "AI",
        "ML",
        "LLM",
        "RAG",
        "SQL",
        "FastAPI",
        "Neo4j",
        "PostgreSQL",
        "Pinecone",
        "Python",
        "Flutter",
        "Docker",
        "LangChain",
    ],
)
def test_named_technical_terms_are_flagged(term):
    assert is_technical_term(term)
    assert is_technical_term(f"  {term}  ")


# --------------------------------------------------
# Technical-term misclassification
# --------------------------------------------------


def test_technical_terms_are_not_stored_as_people_or_places():
    nodes = [
        FakeNode("AI", "Person"),
        FakeNode("RAG", "Location"),
        FakeNode("FastAPI", "GPE"),
        FakeNode("PostgreSQL", "Person"),
        FakeNode("Flutter", "LOC"),
    ]

    cleaned, relationships, dropped = validate_and_clean_graph(nodes, [])

    assert cleaned == []
    assert relationships == []
    reasons = {entry["reason"] for entry in dropped}
    assert "technical_term_as_person" in reasons
    assert "technical_term_as_location" in reasons


def test_technical_terms_kept_when_typed_as_technology_or_product():
    nodes = [
        FakeNode("AI", "Technology"),
        FakeNode("RAG", "Technology"),
        FakeNode("FastAPI", "Product"),
        FakeNode("PostgreSQL", "Technology"),
        FakeNode("Flutter", "Technology"),
    ]

    cleaned, _, dropped = validate_and_clean_graph(nodes, [])

    assert {node.name for node in cleaned} == {
        "AI",
        "RAG",
        "FastAPI",
        "PostgreSQL",
        "Flutter",
    }
    assert [node.entity_type for node in cleaned] == [
        "TECHNOLOGY",
        "TECHNOLOGY",
        "PRODUCT",
        "TECHNOLOGY",
        "TECHNOLOGY",
    ]
    assert dropped == []


def test_concept_phrases_are_not_stored_as_people_or_organizations():
    cleaned, _, dropped = validate_and_clean_graph(
        [
            FakeNode("Machine Learning", "Person"),
            FakeNode("Data Engineering", "Organization"),
        ],
        [],
    )

    assert cleaned == []
    assert [entry["reason"] for entry in dropped] == [
        "technical_term_as_person",
        "implausible_type:organization",
    ]


def test_end_to_end_extraction_drops_mislabelled_technology():
    transformer = FakeTransformer(
        [
            FakeGraphDocument(
                nodes=[
                    FakeNode("Ada Lovelace", "Person"),
                    FakeNode("Python", "Person"),
                    FakeNode("Neo4j", "Location"),
                ],
                relationships=[
                    FakeRelationship(
                        "Ada Lovelace",
                        "Python",
                        "RELATES_TO",
                        "Ada Lovelace used Python",
                    ),
                ],
            )
        ]
    )

    result = extract_graph(
        "Ada Lovelace used Python with Neo4j.",
        document_id="doc-terms",
        transformer=transformer,
    )

    assert [node.name for node in result.nodes] == ["Ada Lovelace"]
    assert result.relationships == []
    assert any(
        entry["reason"] == "technical_term_as_person" for entry in result.dropped
    )


def test_product_with_hyphenated_model_number_and_supported_relation_survive():
    source = (
        "Asterion Instruments developed the Luma-4 Sensor for calibration."
    )
    transformer = FakeTransformer(
        [
            FakeGraphDocument(
                nodes=[
                    FakeNode("Asterion Instruments", "Organization"),
                    FakeNode("Luma-4 Sensor", "Product"),
                ],
                relationships=[
                    FakeRelationship(
                        "Asterion Instruments",
                        "Luma-4 Sensor",
                        "RELATES_TO",
                        source,
                    ),
                ],
            )
        ]
    )

    result = extract_graph(
        source,
        document_id="doc-model-number",
        transformer=transformer,
    )

    assert {node.name for node in result.nodes} == {
        "Asterion Instruments",
        "Luma-4 Sensor",
    }
    assert len(result.relationships) == 1
    assert result.relationships[0].source_id == build_entity_id(
        "ORGANIZATION", "Asterion Instruments"
    )
    assert result.relationships[0].target_id == build_entity_id(
        "PRODUCT", "Luma-4 Sensor"
    )


# --------------------------------------------------
# Ambiguous / unknown entities
# --------------------------------------------------


def test_ambiguous_name_is_dropped_instead_of_assigned_a_guessed_type():
    nodes = [
        FakeNode("Mercury", "Person"),
        FakeNode("Mercury", "Technology"),
    ]

    cleaned, _, dropped = validate_and_clean_graph(nodes, [])

    # A conflicting label is not resolved by an arbitrary type priority.
    assert cleaned == []
    assert any(
        entry["reason"].startswith("ambiguous_type:") for entry in dropped
    )


def test_generic_anonymized_alias_is_not_stored_as_a_real_entity():
    nodes, relationships, dropped = validate_and_clean_graph(
        [FakeNode("Organization B", "Organization")],
        [],
    )

    assert nodes == []
    assert relationships == []
    assert dropped == [
        {
            "name": "Organization B",
            "reason": "malformed_or_anonymized_name",
        }
    ]


def test_relationship_requires_verbatim_evidence_with_endpoints_and_cue():
    nodes, relationships, dropped = validate_and_clean_graph(
        [
            FakeNode("Ada Lovelace", "Person"),
            FakeNode("Acme Corp", "Organization"),
        ],
        [
            FakeRelationship(
                "Ada Lovelace",
                "Acme Corp",
                "RELATES_TO",
                "Ada Lovelace and Acme Corp.",
            )
        ],
        source_text="Ada Lovelace and Acme Corp. appear in the same paragraph.",
    )

    assert nodes
    assert relationships == []
    assert dropped == [
        {
            "name": "Ada Lovelace->Acme Corp",
            "reason": "unsupported_relationship_evidence",
        }
    ]


def test_relationship_quote_must_be_an_exact_source_excerpt():
    nodes, relationships, dropped = validate_and_clean_graph(
        [
            FakeNode("Ada Lovelace", "Person"),
            FakeNode("Acme Corp", "Organization"),
        ],
        [
            FakeRelationship(
                "Ada Lovelace",
                "Acme Corp",
                "RELATES_TO",
                "Ada Lovelace works with Acme Corp.",
            )
        ],
        source_text="Ada Lovelace works with Acme Corp in the source.",
    )

    assert nodes
    assert relationships == []
    assert dropped[0]["reason"] == "unsupported_relationship_evidence"


def test_unsupported_entity_type_is_dropped_with_reason():
    cleaned, _, dropped = validate_and_clean_graph(
        [FakeNode("Aspirin", "Chemical")], []
    )

    assert cleaned == []
    assert dropped == [
        {"name": "Aspirin", "reason": "unsupported_type:Chemical"}
    ]


def test_empty_name_is_dropped():
    cleaned, _, dropped = validate_and_clean_graph([FakeNode("", "Person")], [])

    assert cleaned == []
    assert dropped == [{"name": "", "reason": "empty_name"}]


def test_nodes_without_source_evidence_are_rejected():
    cleaned, _, dropped = validate_and_clean_graph(
        [
            FakeNode("Ada Lovelace", "Person"),
            FakeNode("Python", "Technology"),
            FakeNode("Grace Hopper", "Person"),
        ],
        [],
        source_text="Ada Lovelace used Python.",
    )

    assert {node.name for node in cleaned} == {"Ada Lovelace", "Python"}
    assert dropped == [
        {
            "name": "Grace Hopper",
            "reason": "unsupported_source_evidence",
        }
    ]


def test_relationship_between_unknown_nodes_is_dropped():
    cleaned, relationships, dropped = validate_and_clean_graph(
        [FakeNode("Ada Lovelace", "Person")],
        [FakeRelationship("Ada Lovelace", "Ghost", "RELATES_TO")],
    )

    assert len(cleaned) == 1
    assert relationships == []
    assert dropped == [
        {
            "name": "Ada Lovelace->Ghost",
            "reason": "relationship_missing_node",
        }
    ]


# --------------------------------------------------
# Duplicates
# --------------------------------------------------


def test_duplicate_nodes_and_relationships_collapse():
    nodes = [
        FakeNode("Acme Corp", "Organization"),
        FakeNode("acme corp", "Company"),
        FakeNode("Ada", "Person"),
    ]
    relationships = [
        FakeRelationship("Acme Corp", "Ada", "RELATES_TO"),
        FakeRelationship("acme corp", "Ada", "relates_to"),
    ]

    cleaned, cleaned_relationships, dropped = validate_and_clean_graph(
        nodes, relationships
    )

    assert [node.entity_id for node in cleaned] == [
        "organization:acme_corp",
        "person:ada",
    ]
    assert len(cleaned_relationships) == 1
    assert cleaned_relationships[0].relationship_type == "RELATES_TO"
    reasons = [entry["reason"] for entry in dropped]
    assert "duplicate_node" in reasons
    assert "duplicate_relationship" in reasons


def test_self_loop_is_dropped():
    cleaned, relationships, dropped = validate_and_clean_graph(
        [FakeNode("Ada", "Person")],
        [FakeRelationship("Ada", "Ada", "RELATES_TO")],
    )

    assert relationships == []
    assert dropped == [{"name": "Ada->Ada", "reason": "self_loop"}]


def test_disallowed_relationship_type_is_dropped():
    cleaned, relationships, dropped = validate_and_clean_graph(
        [FakeNode("Ada", "Person"), FakeNode("Acme", "Organization")],
        [FakeRelationship("Ada", "Acme", "MARRIED_TO")],
    )

    assert relationships == []
    assert dropped == [
        {
            "name": "Ada->Acme",
            "reason": "unsupported_relationship:MARRIED_TO",
        }
    ]


# --------------------------------------------------
# Malformed / empty model output
# --------------------------------------------------


def test_empty_model_output_yields_empty_graph_without_crashing():
    transformer = FakeTransformer([FakeGraphDocument(nodes=None, relationships=None)])

    result = extract_graph(
        "Some content.", document_id="doc-empty", transformer=transformer
    )

    assert result.nodes == []
    assert result.relationships == []


def test_malformed_output_ignores_non_node_entries():
    transformer = FakeTransformer(
        [FakeGraphDocument(nodes=[object(), None], relationships=[object()])]
    )

    result = extract_graph(
        "Some content.", document_id="doc-malformed", transformer=transformer
    )

    assert result.nodes == []
    assert result.relationships == []
    assert result.dropped


def test_empty_document_content_is_rejected():
    with pytest.raises(GraphExtractionError):
        extract_graph("   ", document_id="doc-blank", transformer=FakeTransformer())


# --------------------------------------------------
# Structured-output failure / timeout
# --------------------------------------------------


def test_structured_output_unsupported_is_reported():
    transformer = FakeTransformer(error=NotImplementedError("no tools"))

    with pytest.raises(GraphExtractionError) as error:
        extract_graph(
            "Some content.", document_id="doc-so", transformer=transformer
        )

    assert "structured output" in str(error.value)


def test_model_timeout_is_reported():
    transformer = FakeTransformer(error=TimeoutError("slow"))

    with pytest.raises(GraphExtractionError) as error:
        extract_graph(
            "Some content.", document_id="doc-timeout", transformer=transformer
        )

    assert "timed out" in str(error.value)


def test_model_transport_error_is_reported_without_leaking_details():
    transformer = FakeTransformer(error=ConnectionError("bolt://user:pw@host"))

    with pytest.raises(GraphExtractionError) as error:
        extract_graph(
            "Some content.", document_id="doc-conn", transformer=transformer
        )

    message = str(error.value)
    assert "could not reach the Gemini service" in message
    assert "host" not in message


# --------------------------------------------------
# Ingestion wiring + Neo4j write failure
# --------------------------------------------------


@pytest.fixture()
def fake_driver(monkeypatch):
    driver = RecordingDriver()
    monkeypatch.setattr(graph_service, "get_driver", lambda: driver)
    return driver


def _document_node_run(session):
    return [
        parameters
        for query, parameters in session.runs
        if "MERGE (document:Document" in query
    ]


def test_ingestion_uses_llm_by_default(fake_driver):
    transformer = FakeTransformer(
        [
            FakeGraphDocument(
                nodes=[FakeNode("Ada Lovelace", "Person")],
                relationships=[],
            )
        ]
    )

    result = graph_service.ingest_document_graph(
        document_id="doc-llm",
        title="Ada",
        content="Ada Lovelace wrote notes.",
        transformer=transformer,
    )

    assert result["extraction_backend"] == "llm"
    assert result["entities_ingested"] == 1

    entity_runs = [
        parameters
        for query, parameters in fake_driver.session_log.runs
        if "MERGE (entity:Entity" in query
    ]
    assert entity_runs[0]["entity_type"] == "PERSON"
    assert entity_runs[0]["entity_id"] == "person:ada_lovelace"


def test_validation_runs_before_any_neo4j_write(fake_driver):
    transformer = FakeTransformer(
        [
            FakeGraphDocument(
                nodes=[
                    FakeNode("Ada Lovelace", "Person"),
                    FakeNode("Python", "Person"),
                    FakeNode("Aspirin", "Chemical"),
                ],
                relationships=[],
            )
        ]
    )

    graph_service.ingest_document_graph(
        document_id="doc-validate",
        title="Validation",
        content="Ada Lovelace used Python.",
        transformer=transformer,
    )

    written_ids = [
        parameters["entity_id"]
        for query, parameters in fake_driver.session_log.runs
        if "MERGE (entity:Entity" in query
    ]

    assert written_ids == ["person:ada_lovelace"]


def test_neo4j_write_failure_propagates(fake_driver, monkeypatch):
    failing = RecordingDriver(fail_on="MERGE (entity:Entity")
    monkeypatch.setattr(graph_service, "get_driver", lambda: failing)

    transformer = FakeTransformer(
        [FakeGraphDocument(nodes=[FakeNode("Ada", "Person")], relationships=[])]
    )

    with pytest.raises(OSError):
        graph_service.ingest_document_graph(
            document_id="doc-write-fail",
            title="Write failure",
            content="Ada.",
            transformer=transformer,
        )

    assert failing.session_log.runs == []


def test_document_write_failure_stops_before_extraction(monkeypatch):
    failing = RecordingDriver(fail_on="MERGE (document:Document")
    monkeypatch.setattr(graph_service, "get_driver", lambda: failing)

    transformer = FakeTransformer(
        [FakeGraphDocument(nodes=[FakeNode("Ada", "Person")], relationships=[])]
    )

    with pytest.raises(OSError):
        graph_service.ingest_document_graph(
            document_id="doc-doc-fail",
            title="Document write failure",
            content="Ada.",
            transformer=transformer,
        )

    # Extraction precedes writes; a failed document write leaves no partial
    # graph writes in the mocked transaction.
    assert len(transformer.calls) == 1
    assert failing.session_log.runs == []


def test_relationship_write_failure_is_not_swallowed(monkeypatch):
    failing = RecordingDriver(fail_on="RELATES_TO")
    monkeypatch.setattr(graph_service, "get_driver", lambda: failing)

    transformer = FakeTransformer(
        [
            FakeGraphDocument(
                nodes=[
                    FakeNode("Ada", "Person"),
                    FakeNode("Acme Corp", "Organization"),
                ],
                relationships=[
                    FakeRelationship(
                        "Ada",
                        "Acme Corp",
                        "RELATES_TO",
                        "Ada works with Acme Corp.",
                    ),
                ],
            )
        ]
    )

    with pytest.raises(OSError):
        graph_service.ingest_document_graph(
            document_id="doc-rel-fail",
            title="Rel failure",
            content="Ada works with Acme Corp.",
            transformer=transformer,
        )

    assert failing.session_log.runs == []


def test_reingestion_uses_stable_document_id_and_refreshes_owned_edges(
    fake_driver,
):
    first = FakeTransformer(
        [
            FakeGraphDocument(
                nodes=[
                    FakeNode("Ada Lovelace", "Person"),
                    FakeNode("Acme Corp", "Organization"),
                ],
                relationships=[
                    FakeRelationship(
                        "Ada Lovelace",
                        "Acme Corp",
                        "RELATES_TO",
                        "Ada Lovelace works with Acme Corp.",
                    )
                ],
            )
        ]
    )
    second = FakeTransformer(
        [
            FakeGraphDocument(
                nodes=[FakeNode("Ada Lovelace", "Person")],
                relationships=[],
            )
        ]
    )

    graph_service.ingest_document_graph(
        "doc-reprocess",
        "First extraction",
        "Ada Lovelace works with Acme Corp.",
        transformer=first,
    )
    graph_service.ingest_document_graph(
        "doc-reprocess",
        "Updated extraction",
        "Ada Lovelace wrote notes.",
        transformer=second,
    )

    document_writes = _document_node_run(fake_driver.session_log)
    assert [write["document_id"] for write in document_writes] == [
        "doc-reprocess",
        "doc-reprocess",
    ]
    assert any(
        "DELETE mention" in query
        for query, _parameters in fake_driver.session_log.runs
    )
    assert any(
        "remaining_document_ids" in query
        for query, _parameters in fake_driver.session_log.runs
    )
    assert any(
        "DELETE entity" in query
        for query, _parameters in fake_driver.session_log.runs
    )

# --------------------------------------------------
# Provenance: document id, page numbers
# --------------------------------------------------


def test_page_numbers_are_preserved_per_page(fake_driver):
    transformer = FakeTransformer(
        [
            FakeGraphDocument(
                nodes=[FakeNode("Ada Lovelace", "Person")],
                relationships=[],
            ),
            FakeGraphDocument(
                nodes=[
                    FakeNode("Ada Lovelace", "Person"),
                    FakeNode("Acme Corp", "Organization"),
                ],
                relationships=[
                    FakeRelationship(
                        "Ada Lovelace",
                        "Acme Corp",
                        "PART_OF",
                        "Ada Lovelace is a member of Acme Corp",
                    ),
                ],
            ),
        ]
    )

    result = graph_service.ingest_document_graph(
        document_id="doc-pages",
        title="Paged document",
        content=(
            "Ada Lovelace is on page one. "
            "Ada Lovelace is a member of Acme Corp on page two."
        ),
        pages=[
            {"page_number": 1, "content": "Ada Lovelace is on page one."},
            {
                "page_number": 2,
                "content": (
                    "Ada Lovelace is a member of Acme Corp on page two."
                ),
            },
        ],
        transformer=transformer,
    )

    mention_runs = [
        parameters
        for query, parameters in fake_driver.session_log.runs
        if "[mention:MENTIONS]" in query and "entity_id" in parameters
    ]

    by_id = {run["entity_id"]: run for run in mention_runs}

    assert by_id["person:ada_lovelace"]["page_numbers"] == [1, 2]
    assert by_id["organization:acme_corp"]["page_numbers"] == [2]
    assert by_id["person:ada_lovelace"]["document_id"] == "doc-pages"
    assert by_id["person:ada_lovelace"]["source_refs"] == [
        "doc-pages#page=1",
        "doc-pages#page=2",
    ]

    relationship_runs = [
        parameters
        for query, parameters in fake_driver.session_log.runs
        if "PART_OF]->" in query and "source_id" in parameters
    ]
    assert relationship_runs[0]["page_numbers"] == [2]
    assert relationship_runs[0]["document_id"] == "doc-pages"
    assert relationship_runs[0]["source_refs"] == ["doc-pages#page=2"]

    assert result["document_id"] == "doc-pages"
    assert _document_node_run(fake_driver.session_log)[0]["document_id"] == "doc-pages"


def test_source_unit_ids_are_preserved_with_page_provenance(fake_driver):
    transformer = FakeTransformer(
        [
            FakeGraphDocument(
                nodes=[FakeNode("Ada Lovelace", "Person")],
                relationships=[],
            )
        ]
    )

    graph_service.ingest_document_graph(
        document_id="doc-units",
        title="Source units",
        content="Ada Lovelace wrote notes.",
        pages=[
            {
                "page_number": 3,
                "source_unit_id": "unit-page-3-text",
                "content": "Ada Lovelace wrote notes.",
            }
        ],
        transformer=transformer,
    )

    assert transformer.calls[0][0].metadata == {
        "document_id": "doc-units",
        "page_number": 3,
        "source_unit_id": "unit-page-3-text",
    }
    mention = next(
        parameters
        for query, parameters in fake_driver.session_log.runs
        if "[mention:MENTIONS]" in query and "entity_id" in parameters
    )
    assert mention["source_refs"] == [
        "doc-units#page=3#unit=unit-page-3-text"
    ]


def test_same_name_with_conflicting_types_across_pages_is_dropped():
    transformer = FakeTransformer(
        [
            FakeGraphDocument(
                nodes=[FakeNode("Mercury", "Technology")],
                relationships=[],
            ),
            FakeGraphDocument(
                nodes=[FakeNode("Mercury", "Person")],
                relationships=[],
            ),
        ]
    )

    result = extract_graph(
        "Mercury is discussed in both passages.",
        document_id="doc-ambiguous",
        transformer=transformer,
        pages=[
            {"page_number": 1, "content": "Mercury powers retrieval."},
            {"page_number": 2, "content": "Mercury joined the project."},
        ],
    )

    assert result.nodes == []
    assert any(
        entry["reason"] == "ambiguous_type_across_units"
        for entry in result.dropped
    )


def test_cross_page_duplicate_relationship_unions_page_provenance(fake_driver):
    transformer = FakeTransformer(
        [
            FakeGraphDocument(
                nodes=[
                    FakeNode("Ada Lovelace", "Person"),
                    FakeNode("Acme Corp", "Organization"),
                ],
                relationships=[
                    FakeRelationship(
                        "Ada Lovelace",
                        "Acme Corp",
                        "PART_OF",
                        "Ada Lovelace is a member of Acme Corp",
                    ),
                ],
            ),
            FakeGraphDocument(
                nodes=[
                    FakeNode("Ada Lovelace", "Person"),
                    FakeNode("Acme Corp", "Organization"),
                ],
                relationships=[
                    FakeRelationship(
                        "Ada Lovelace",
                        "Acme Corp",
                        "PART_OF",
                        "Ada Lovelace is a member of Acme Corp",
                    ),
                ],
            ),
        ]
    )

    result = graph_service.ingest_document_graph(
        document_id="doc-cross-page",
        title="Two pages",
        content="Ada Lovelace is a member of Acme Corp on both pages.",
        pages=[
            {
                "page_number": 1,
                "content": "Ada Lovelace is a member of Acme Corp.",
            },
            {
                "page_number": 2,
                "content": "Ada Lovelace is a member of Acme Corp.",
            },
        ],
        transformer=transformer,
    )

    # The edge is written once, with both pages as provenance.
    assert result["entity_relations_ingested"] == 1

    relationship_runs = [
        parameters
        for query, parameters in fake_driver.session_log.runs
        if "PART_OF]->" in query and "source_id" in parameters
    ]

    assert len(relationship_runs) == 1
    assert relationship_runs[0]["page_numbers"] == [1, 2]
    assert relationship_runs[0]["source_refs"] == [
        "doc-cross-page#page=1",
        "doc-cross-page#page=2",
    ]
    relationship_query = next(
        query
        for query, parameters in fake_driver.session_log.runs
        if "PART_OF]->" in query and "source_id" in parameters
    )
    assert "unique_values = []," in relationship_query
    assert (
        "value IN coalesce(relationship.document_ids, []) "
        "+ [$document_id] |"
    ) in relationship_query
    assert "values = coalesce(" not in relationship_query


def test_transformer_returning_a_wrong_document_count_is_rejected():
    # Two page documents in, one graph document back: the model dropped a
    # page, so provenance would be lost. Report instead of guessing.
    transformer = FakeTransformer(
        [FakeGraphDocument(nodes=[FakeNode("Ada", "Person")], relationships=[])]
    )

    with pytest.raises(GraphExtractionError) as error:
        extract_graph(
            "page one\npage two",
            document_id="doc-count",
            transformer=transformer,
            pages=[
                {"page_number": 1, "content": "page one"},
                {"page_number": 2, "content": "page two"},
            ],
        )

    assert "different number of graph documents" in str(error.value)


def test_document_metadata_reaches_the_transformer(fake_driver):
    transformer = FakeTransformer(
        [FakeGraphDocument(nodes=[], relationships=[])]
    )

    graph_service.ingest_document_graph(
        document_id="doc-meta",
        title="Meta",
        content="page one",
        pages=[{"page_number": 7, "content": "page one"}],
        transformer=transformer,
    )

    sent_document = transformer.calls[0][0]

    assert sent_document.metadata["document_id"] == "doc-meta"
    assert sent_document.metadata["page_number"] == 7


# --------------------------------------------------
# Read-only audit
# --------------------------------------------------


class AuditSession:
    def __init__(self, type_rows):
        self._type_rows = type_rows
        self.queries = []

    def run(self, query, **parameters):
        self.queries.append(query)

        rows = self._type_rows if "count(entity)" in query else [
            {"entity_id": "person:ada", "name": "Ada"}
        ]

        return iter(rows)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class AuditDriver:
    def __init__(self, type_rows):
        self.session_log = AuditSession(type_rows)

    def session(self, database=None):
        return self.session_log


class AuditGraphService:
    NEO4J_DATABASE = "neo4j"

    def __init__(self, type_rows):
        self.driver = AuditDriver(type_rows)

    def get_driver(self):
        return self.driver


def test_audit_reports_suspicious_types_without_writing():
    service = AuditGraphService(
        [
            {"entity_type": "PERSON", "nodes": 4},
            {"entity_type": "Technology", "nodes": 2},
            {"entity_type": "Chemical", "nodes": 1},
        ]
    )

    findings = audit_suspicious_nodes(service)

    assert [finding["entity_type"] for finding in findings] == ["Technology", "Chemical"]

    technology = findings[0]
    assert technology["proposed_correction"] == "TECHNOLOGY"
    assert technology["is_allowed"] is False
    assert technology["evidence"]["sample_nodes"][0]["name"] == "Ada"
    assert technology["confidence"] > 0.5

    # Only read statements were issued: no writes of any kind.
    forbidden = ("MERGE", "CREATE", "SET ", "DELETE", "REMOVE")
    for query in service.driver.session_log.queries:
        upper = query.upper()
        assert upper.strip().startswith("MATCH")
        assert "RETURN" in upper
        for keyword in forbidden:
            assert keyword not in upper


def test_audit_reports_an_unknown_type_without_a_correction():
    service = AuditGraphService([{"entity_type": "Chemical", "nodes": 3}])

    findings = audit_suspicious_nodes(service)

    assert findings == [
        {
            "entity_type": "Chemical",
            "node_count": 3,
            "is_allowed": False,
            "proposed_correction": None,
            "evidence": {
                "sample_nodes": [{"entity_id": "person:ada", "name": "Ada"}]
            },
            "confidence": 0.4,
        }
    ]

    for query in service.driver.session_log.queries:
        assert query.upper().strip().startswith("MATCH")


def test_audit_reports_nothing_for_allowed_types_only():
    service = AuditGraphService(
        [
            {"entity_type": type_name, "nodes": 1}
            for type_name in ALLOWED_ENTITY_TYPES
        ]
    )

    assert audit_suspicious_nodes(service) == []


# --------------------------------------------------
# Backend selection
# --------------------------------------------------


def test_default_backend_is_the_llm_transformer(monkeypatch):
    monkeypatch.delenv("GRAPH_EXTRACTION_BACKEND", raising=False)

    assert transformer_module.llm_extraction_enabled() is True
    assert graph_service._resolve_extraction_backend(None) == "llm"


def test_spacy_backend_must_be_requested_explicitly(monkeypatch):
    monkeypatch.setenv("GRAPH_EXTRACTION_BACKEND", "spacy")

    assert transformer_module.llm_extraction_enabled() is False
    assert graph_service._resolve_extraction_backend(None) == "spacy"
    # An explicit argument still wins.
    assert graph_service._resolve_extraction_backend("llm") == "llm"


def test_graph_extraction_uses_gemini_without_provider_switching(monkeypatch):
    monkeypatch.setenv("GRAPH_LLM_PROVIDER", "ollama")
    providers = []
    transformer = FakeTransformer(
        [
            FakeGraphDocument(
                nodes=[FakeNode("Ada Lovelace", "Person")],
                relationships=[],
            )
        ]
    )

    def build_transformer(*, provider):
        providers.append(provider)
        return transformer

    monkeypatch.setattr(
        transformer_module,
        "build_transformer",
        build_transformer,
    )

    result = extract_graph(
        "Ada Lovelace wrote notes.",
        document_id="doc-gemini-only",
    )

    assert result.provider == "gemini"
    assert [node.name for node in result.nodes] == ["Ada Lovelace"]
    assert providers == ["gemini"]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ConnectionError("host unavailable"), "could not reach"),
        (
            type("AuthError", (RuntimeError,), {"status_code": 401})(
                "secret-token"
            ),
            "authentication",
        ),
        (
            type("QuotaError", (RuntimeError,), {"status_code": 429})(
                "secret-token"
            ),
            "quota or rate limit",
        ),
        (NotImplementedError("structured output unavailable"), "structured output"),
    ],
)
def test_graph_extraction_surfaces_safe_gemini_errors(error, expected):
    with pytest.raises(GraphExtractionError, match=expected) as caught:
        extract_graph(
            "Ada Lovelace wrote notes.",
            document_id="doc-gemini-failure",
            transformer=FakeTransformer(error=error),
        )

    assert "secret-token" not in str(caught.value)


def test_gemini_graph_builder_rejects_obsolete_provider():
    with pytest.raises(llm.LLMError, match="Gemini only"):
        transformer_module.build_transformer(llm=object(), provider="ollama")


def test_configured_gemini_model_supports_structured_graph_output():
    transformer = transformer_module.build_transformer()

    assert transformer.allowed_nodes == list(ALLOWED_ENTITY_TYPES)
    assert transformer.allowed_relationships == list(
        ALLOWED_RELATIONSHIP_TYPES
    )


def test_transformer_construction_failure_is_reported_safely(monkeypatch):
    def fail_to_build(**_kwargs):
        raise NotImplementedError("upstream details")

    monkeypatch.setattr(
        transformer_module,
        "build_transformer",
        fail_to_build,
    )

    with pytest.raises(GraphExtractionError, match="structured output"):
        extract_graph("Some content.", document_id="doc-build-failure")


def test_graph_dependency_failure_is_reported_without_details():
    error = transformer_module._translate_graph_error(
        ImportError("private import path")
    )

    assert "structured-output dependencies" in str(error)
    assert "private import path" not in str(error)


def test_unknown_extractor_is_rejected():
    with pytest.raises(ValueError):
        graph_service._resolve_extraction_backend("magic")


# --------------------------------------------------
# Schema constants
# --------------------------------------------------


def test_schema_constants_cover_expected_types():
    assert set(ALLOWED_ENTITY_TYPES) == {
        "PERSON",
        "ORGANIZATION",
        "TECHNOLOGY",
        "PRODUCT",
        "LOCATION",
        "DOCUMENT",
    }
    assert set(ALLOWED_RELATIONSHIP_TYPES) == {
        "RELATES_TO",
        "CONCERNS",
        "PRESCRIBED",
        "MENTIONS",
        "PART_OF",
    }


def test_extraction_result_is_json_safe():
    result = extract_graph(
        "Ada Lovelace wrote notes.",
        document_id="doc-json",
        transformer=FakeTransformer(
            [
                FakeGraphDocument(
                    nodes=[FakeNode("Ada Lovelace", "Person")],
                    relationships=[],
                )
            ]
        ),
    )

    payload = result.as_dict()
    assert isinstance(payload["nodes"], list)
    assert isinstance(payload["relationships"], list)
    assert isinstance(payload["dropped"], list)
