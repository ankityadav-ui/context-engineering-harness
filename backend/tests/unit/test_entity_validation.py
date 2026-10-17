"""Regression tests for graph-specific entity validation.

Known spaCy failures on PDF/OCR text (reported during testing):
- "Nachine Learning" classified as PERSON,
- "Nagpur" classified as both GPE and PERSON,
- "Maharashtra" classified as ORG,
- "Topology" classified as ORG,
- standalone placeholders such as "GPE" or "ORG".

These tests pin the fixes and, just as importantly, pin that legitimate
people, organizations, and places survive. The shared masking NER service
is not modified by this validation layer.

No Neo4j connection is opened: the driver is replaced by a recorder.
"""

import pytest

import app.context_engine.graph_service as graph_service
import app.masking.ner_service as ner_service
from app.context_engine.entity_validation import (
    entity_name_key,
    is_plausible_entity,
    is_usable_entity_name,
    looks_like_person_name,
    resolve_entity_types,
)


# --------------------------------------------------
# Name quality (independent of entity type)
# --------------------------------------------------

@pytest.mark.parametrize(
    "name",
    [
        "PERSON_4",
        "ORG_12",
        "GPE",
        "ORG",
        "[ORG_3]",
        "GPE_4]<e",
        "0nd",
        "12",
        "",
        " ",
        "A",
    ],
)
def test_malformed_names_are_rejected(name):
    assert is_usable_entity_name(name) is False


@pytest.mark.parametrize(
    "name",
    [
        "Priya Sharma",
        "Dr. Anil Kumar",
        "Acme Corp.",
        "Supreme Court of India",
        "Nagpur",
        "Li",
        "O'Brien & Sons",
    ],
)
def test_plausible_names_are_accepted(name):
    assert is_usable_entity_name(name) is True


# --------------------------------------------------
# Type plausibility (separate from name quality)
# --------------------------------------------------

def test_concept_phrase_is_not_a_person():
    # The name itself is well formed; only the type is wrong.
    assert is_usable_entity_name("Nachine Learning") is True
    assert is_plausible_entity("Nachine Learning", "PERSON") is False
    assert looks_like_person_name("Nachine Learning") is False


def test_concept_words_are_not_organizations():
    assert is_plausible_entity("Topology", "ORG") is False
    assert is_plausible_entity("Machine Learning", "ORG") is False


def test_synthetic_marker_is_not_an_organization():
    assert is_plausible_entity(
        "GreenField Analytics Fictional", "ORG"
    ) is False


def test_legitimate_entities_keep_their_type():
    assert is_plausible_entity("Priya Sharma", "PERSON") is True
    assert is_plausible_entity("Dr. Anil Kumar", "PERSON") is True
    assert is_plausible_entity("Li", "PERSON") is True
    assert is_plausible_entity("Microsoft", "ORG") is True
    assert is_plausible_entity("Acme Corp.", "ORG") is True
    assert is_plausible_entity("Supreme Court of India", "ORG") is True
    assert is_plausible_entity("Nagpur", "GPE") is True
    assert is_plausible_entity("Paris", "LOC") is True


def test_organizational_designator_does_not_make_a_person():
    assert is_plausible_entity("Acme Corp.", "PERSON") is False


def test_law_requires_a_legal_term():
    assert is_plausible_entity("Clean Air Act", "LAW") is True
    assert is_plausible_entity("Somewhere", "LAW") is False


def test_unknown_entity_type_is_rejected():
    assert is_plausible_entity("Priya Sharma", "MONEY") is False


# --------------------------------------------------
# Document-level type conflict resolution
# --------------------------------------------------

def test_place_wins_over_person_for_the_same_name():
    resolved = resolve_entity_types([
        ("Nagpur", "PERSON"),
        ("Nagpur", "GPE"),
    ])

    entry = resolved[entity_name_key("Nagpur")]

    assert entry["entity_type"] == "GPE"
    assert entry["observed_types"] == {"PERSON", "GPE"}


def test_place_wins_over_organization_for_the_same_name():
    resolved = resolve_entity_types([
        ("Maharashtra", "ORG"),
        ("Maharashtra", "GPE"),
    ])

    assert resolved["maharashtra"]["entity_type"] == "GPE"


def test_consistent_type_is_unchanged():
    resolved = resolve_entity_types([("Microsoft", "ORG")])

    assert resolved["microsoft"]["entity_type"] == "ORG"


def test_first_spelling_is_preserved():
    resolved = resolve_entity_types([
        ("Acme Corp.", "ORG"),
        ("acme corp", "ORG"),
    ])

    assert resolved["acme corp"]["name"] == "Acme Corp."


# --------------------------------------------------
# Graph ingestion with the reported failures
# --------------------------------------------------

class _RecordingResult:
    def __init__(self, records=()):
        self._records = list(records)

    def single(self):
        return self._records[0] if self._records else None

    def __iter__(self):
        return iter(self._records)


class _RecordingSession:
    def __init__(self):
        self.runs: list[tuple[str, dict]] = []

    def run(self, query, **parameters):
        self.runs.append((query, parameters))

        # Mirror the real driver: the document MERGE returns one record.
        if "MERGE (document:Document" in query:
            return _RecordingResult([
                {
                    "document_id": parameters.get("document_id"),
                    "title": parameters.get("title"),
                    "source": parameters.get("source"),
                }
            ])

        return _RecordingResult()

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        return False


class _RecordingDriver:
    def __init__(self):
        self.session_log = _RecordingSession()

    def session(self, database=None):
        return self.session_log


def _install_ner(monkeypatch, entities_per_sentence):
    """Install a deterministic NER stand-in (no spaCy model needed)."""

    class _Entity:
        def __init__(self, text, label):
            self.text = text
            self.label_ = label

    class _Sentence:
        def __init__(self, entities):
            self.ents = entities

    class _Doc:
        def __init__(self, sentences):
            self.sents = sentences

    class _NLP:
        def __call__(self, _text):
            return _Doc([
                _Sentence([_Entity(text, label) for text, label in entities])
                for entities in entities_per_sentence
            ])

    monkeypatch.setattr(ner_service, "get_nlp", lambda: _NLP())


def _ingested_entities(driver):
    return [
        parameters
        for query, parameters in driver.session_log.runs
        if "MERGE (entity:Entity" in query
    ]


def test_ingestion_rejects_noisy_entities_and_keeps_valid_ones(monkeypatch):
    driver = _RecordingDriver()
    monkeypatch.setattr(graph_service, "get_driver", lambda: driver)

    _install_ner(
        monkeypatch,
        [
            # OCR noise and concept misclassification.
            [("Nachine Learning", "PERSON"), ("Topology", "ORG")],
            # Contradictory labels for the same place.
            [("Nagpur", "GPE"), ("Nagpur", "PERSON")],
            # A state labelled both ORG and GPE.
            [("Maharashtra", "ORG"), ("Maharashtra", "GPE")],
            # Standalone placeholders leaked from masking.
            [("GPE", "GPE"), ("ORG", "ORG"), ("PERSON_2", "PERSON")],
            # Legitimate entities that must survive.
            [
                ("Dr. Anil Kumar", "PERSON"),
                ("Acme Corp.", "ORG"),
                ("Supreme Court of India", "ORG"),
            ],
        ],
    )

    result = graph_service.ingest_document_graph(
        document_id="doc-regression",
        title="Regression sample",
        content="Synthetic regression content.",
    )

    entities = {
        parameters["name"]: parameters
        for parameters in _ingested_entities(driver)
    }

    assert "Nachine Learning" not in entities
    assert "Topology" not in entities
    assert "GPE" not in entities
    assert "ORG" not in entities
    assert "PERSON_2" not in entities

    # Nagpur is stored once, as a place.
    nagpur = [name for name in entities if name.casefold() == "nagpur"]
    assert nagpur == ["Nagpur"]
    assert entities["Nagpur"]["entity_type"] == "GPE"

    # Maharashtra is stored once, as a place.
    maharashtra = [
        name for name in entities if name.casefold() == "maharashtra"
    ]
    assert maharashtra == ["Maharashtra"]
    assert entities["Maharashtra"]["entity_type"] == "GPE"

    # Legitimate entities survive with their types.
    assert entities["Dr. Anil Kumar"]["entity_type"] == "PERSON"
    assert entities["Acme Corp."]["entity_type"] == "ORG"
    assert entities["Supreme Court of India"]["entity_type"] == "ORG"

    assert result["entities_ingested"] == len(entities)


def test_ingestion_is_idempotent_and_uses_parameterized_cypher(monkeypatch):
    driver = _RecordingDriver()
    monkeypatch.setattr(graph_service, "get_driver", lambda: driver)

    _install_ner(monkeypatch, [[("Acme Corp.", "ORG")]])

    graph_service.ingest_document_graph(
        document_id="doc-idempotent",
        title="Idempotent",
        content="Acme Corp. content.",
    )
    first_run = len(driver.session_log.runs)

    graph_service.ingest_document_graph(
        document_id="doc-idempotent",
        title="Idempotent",
        content="Acme Corp. content.",
    )

    # The second ingestion issues the same MERGE statements: no inserts.
    assert all(
        "MERGE" in query
        for query, _parameters in driver.session_log.runs
    )

    names = [
        parameters["name"]
        for parameters in _ingested_entities(driver)
    ]
    assert names == ["Acme Corp.", "Acme Corp."]
    assert len(driver.session_log.runs) == first_run * 2
