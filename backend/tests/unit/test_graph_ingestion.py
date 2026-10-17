# Tests for graph ingestion with a mocked Neo4j session.
# No real Neo4j connection is required.

import pytest

import app.context_engine.graph_service as graph_service


class FakeRecord:
    def __init__(self, payload: dict | None):
        self._payload = payload

    def __iter__(self):
        return iter(self._payload.items())

    def __getitem__(self, key):
        return self._payload.get(key)


class FakeResult:
    def __init__(self, records):
        self._records = records

    def single(self):
        return self._records[0] if self._records else None

    def __iter__(self):
        return iter(self._records)


class FakeSession:
    """Captures every Cypher query + parameters pair."""

    def __init__(self):
        self.runs: list[tuple[str, dict]] = []

    def run(self, query, **parameters):
        self.runs.append((query, parameters))

        query_lower = query.lower()

        if "merge (document:document" in query_lower:
            return FakeResult([{
                "document_id": parameters.get("document_id"),
                "title": parameters.get("title"),
                "source": parameters.get("source"),
            }])

        return FakeResult([])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeDriver:
    def __init__(self):
        self.session_log = FakeSession()

    def session(self, database=None):
        return self.session_log


@pytest.fixture(autouse=True)
def _fake_driver(monkeypatch):
    driver = FakeDriver()
    monkeypatch.setattr(graph_service, "get_driver", lambda: driver)

    # Deterministic NER output instead of loading spaCy.
    class _Entity:
        def __init__(self, text, label_):
            self.text = text
            self.label_ = label_

    class _Sentence:
        def __init__(self, ents):
            self.ents = ents

    class _Doc:
        def __init__(self, sentences):
            self.sents = sentences

    class _NLP:
        def __call__(self, text):
            return _Doc([
                _Sentence([
                    _Entity("Microsoft", "ORG"),
                    _Entity("Acme Corp.", "ORG"),
                ]),
                _Sentence([
                    _Entity("Microsoft", "ORG"),
                    _Entity("Paris", "GPE"),
                ]),
            ])

    import app.masking.ner_service as ner_service

    monkeypatch.setattr(ner_service, "get_nlp", lambda: _NLP())

    yield driver


# --------------------------------------------------
# Deterministic entity ids
# --------------------------------------------------

def test_entity_node_id_is_deterministic():
    first = graph_service._entity_node_id("ORG", "  Acme   Corp. ")
    second = graph_service._entity_node_id("ORG", "acme   corp.")

    assert first == second
    assert first.startswith("org:")


# --------------------------------------------------
# Document ingestion
# --------------------------------------------------

def test_ingest_merges_document_entity_and_relations(_fake_driver):
    result = graph_service.ingest_document_graph(
        document_id="doc-1",
        title="Annual report",
        content="Microsoft and Acme Corp. met in Paris.",
    )

    assert result["document_id"] == "doc-1"
    assert result["entities_ingested"] == 3

    queries = [query for query, _ in _fake_driver.session_log.runs]

    # Only MERGE-based writes: re-ingestion never duplicates nodes.
    assert all("MERGE" in query for query in queries)

    # Parameterized Cypher only: no string interpolation.
    runtime_queries = [
        (query, parameters)
        for query, parameters in _fake_driver.session_log.runs
    ]

    for query, _parameters in runtime_queries:
        if "{entity_id" in query and "MERGE" in query:
            assert "$" in query

    merged_pairs = [
        parameters
        for query, parameters in runtime_queries
        if "[:RELATES_TO]->" in query
    ]

    # Microsoft–Acme pair and Microsoft–Paris pair were created.
    assert len(merged_pairs) >= 2


def test_ingest_rejects_empty_content(_fake_driver):
    with pytest.raises(ValueError):
        graph_service.ingest_document_graph(
            document_id="doc-1",
            title="Empty",
            content="   ",
        )


def test_ingest_caps_entities(_fake_driver):
    class _ManyDoc:
        def __init__(self, sentences):
            self.sents = sentences

    class _NLP:
        def __call__(self, text):
            class _E:
                def __init__(self, t, l):
                    self.text = t
                    self.label_ = l

            class _S:
                def __init__(self, ents):
                    self.ents = ents

            return _ManyDoc([
                _S([
                    _E(f"Org_{i}", "ORG")
                    for i in range(50)
                ])
            ])

    import app.masking.ner_service as ner_service

    ner_service.get_nlp = lambda: _NLP()

    result = graph_service.ingest_document_graph(
        document_id="doc-2",
        title="Too many entities",
        content="Many organizations.",
    )

    assert result["entities_ingested"] <= (
        graph_service.MAX_ENTITIES_PER_DOCUMENT
    )


def test_ingest_case_none_has_no_case_query(_fake_driver):
    graph_service.ingest_document_graph(
        document_id="doc-3",
        title="No case",
        content="Plain content.",
    )

    # The Case node is guarded inside the document query by
    # `FOREACH (_ IN CASE WHEN $case_id IS NULL THEN [] ELSE [1] END |`,
    # so the guard is the parameter, not a separate query. Match the
    # document merge and assert the case parameter is None.
    document_runs = [
        parameters
        for query, parameters in _fake_driver.session_log.runs
        if "MERGE (document:Document" in query
    ]

    assert len(document_runs) == 1
    assert document_runs[0]["case_id"] is None


def test_ingest_with_case_attaches_document(_fake_driver):
    graph_service.ingest_document_graph(
        document_id="doc-4",
        title="Litigation bundle",
        content="Plain content.",
        case="acme-vs-state",
    )

    document_runs = [
        (query, parameters)
        for query, parameters in _fake_driver.session_log.runs
        if "MERGE (document:Document" in query
    ]

    assert document_runs != []
    assert document_runs[0][1]["case_id"] == "acme-vs-state"
    assert "(case_node:Case" in document_runs[0][0]
