"""Tests for the Neo4j legacy cleanup (report + plan + guarded writes).

Everything here runs against pure snapshots and a fake driver, so no
Neo4j connection is opened. The live behavior is covered by the
opt-in integration tests (marked `integration`).
"""

from app.context_engine import graph_cleanup


# ------------------------------------------------------------- helpers


def _node(key, labels, **fields):
    node = {
        "key": key,
        "labels": set(labels),
        "entity_id": None,
        "name": None,
        "entity_type": None,
        "document_id": None,
        "source": None,
    }
    node.update(fields)
    return node


def _rel(start, end, rel_type):
    return {"start": start, "end": end, "type": rel_type}


def _snapshot(nodes, relationships=None):
    if relationships is None:
        # Accept one mixed list of nodes and relationships; the
        # relationship dicts are exactly the ones carrying "type".
        relationships = [
            item for item in nodes if "type" in item
        ]
        nodes = [item for item in nodes if "type" not in item]

    label_counts: dict[str, int] = {}
    relationship_counts: dict[str, int] = {}

    for node in nodes:
        for label in node["labels"]:
            label_counts[label] = label_counts.get(label, 0) + 1

    for relationship in relationships:
        relationship_counts[relationship["type"]] = (
            relationship_counts.get(relationship["type"], 0) + 1
        )

    return {
        "nodes": nodes,
        "relationships": relationships,
        "label_counts": label_counts,
        "relationship_counts": relationship_counts,
    }


def _app_document(key="d1", document_id="doc-1"):
    return _node(
        key,
        ["Document"],
        document_id=document_id,
        source="document-upload",
    )


def _entity(key, entity_id, name, entity_type):
    return _node(
        key,
        ["Entity"],
        entity_id=entity_id,
        name=name,
        entity_type=entity_type,
    )


# ------------------------------------------------------- report is pure


def test_report_never_touches_foreign_nodes():
    snapshot = _snapshot(
        [
            # Unrelated movie graph.
            _node("m1", ["Movie"], name="Heat"),
            _node("p1", ["Person"], name="Neil"),
            _rel("p1", "m1", "ACTED_IN"),
            # Application-owned pair with one duplicate spelling.
            _app_document(),
            _entity(
                "e1", "org:acme corp", "Acme Corp.", "ORG"
            ),
            _entity("e2", "org:acme_corp", "Acme Corp", "ORG"),
            _rel("d1", "e1", "MENTIONS"),
            _rel("d1", "e2", "MENTIONS"),
        ]
    )

    plan = graph_cleanup.plan_cleanup(snapshot)
    report = graph_cleanup.build_report(snapshot, plan)

    # Foreign nodes are counted but never part of any action.
    assert report["counts"]["labels"]["Movie"] == 1
    assert report["counts"]["labels"]["Person"] == 1
    assert report["scope"]["foreign_nodes_untouched"] == 2

    planned_ids = set()

    for merge in plan["merges"]:
        planned_ids.add(merge["canonical"]["entity_id"])
        planned_ids.update(
            item["entity_id"] for item in merge["duplicates"]
        )

    planned_ids.update(item["entity_id"] for item in plan["removals"])

    assert "m1" not in planned_ids
    assert not any(
        entity_id.startswith(("movie", "person"))
        for entity_id in planned_ids
    )
    assert report["destructive_changes_performed"] is False


def test_duplicate_spellings_are_planned_for_merge_with_provenance():
    snapshot = _snapshot(
        [
            _app_document("d1", "doc-1"),
            _app_document("d2", "doc-2"),
            _entity("e1", "org:acme corp", "Acme Corp.", "ORG"),
            _entity("e2", "org:acme_corp", "Acme Corp", "ORG"),
            _rel("d1", "e1", "MENTIONS"),
            _rel("d2", "e2", "MENTIONS"),
            _entity("e3", "person:jane doe", "Jane Doe", "PERSON"),
        ]
    )

    plan = graph_cleanup.plan_cleanup(snapshot)

    assert len(plan["merges"]) == 1

    merge = plan["merges"][0]

    assert merge["comparison_key"] == "acme corp"
    assert merge["entity_type"] == "ORG"
    assert len(merge["duplicates"]) == 1

    # Deterministic canonical choice and preserved provenance.
    assert merge["canonical"]["entity_id"] in {
        "org:acme corp",
        "org:acme_corp",
    }
    assert merge["canonical"]["document_ids"] in (
        ["doc-1"],
        ["doc-2"],
    )
    assert merge["duplicates"][0]["document_ids"] in (
        ["doc-1"],
        ["doc-2"],
    )


def test_conflicting_labels_for_the_same_name_are_flagged_not_merged():
    snapshot = _snapshot(
        [
            _app_document(),
            _entity("e1", "gpe:nagpur", "Nagpur", "GPE"),
            _entity("e2", "person:nagpur", "Nagpur", "PERSON"),
            _rel("d1", "e1", "MENTIONS"),
            _rel("d1", "e2", "MENTIONS"),
        ]
    )

    plan = graph_cleanup.plan_cleanup(snapshot)

    assert plan["merges"] == []
    assert len(plan["conflicting_labels"]) == 1

    flagged = plan["conflicting_labels"][0]

    assert flagged["comparison_key"] == "nagpur"
    assert flagged["entity_types"] == ["GPE", "PERSON"]
    assert len(flagged["entities"]) == 2


def test_placeholder_and_malformed_entities_are_planned_for_removal():
    snapshot = _snapshot(
        [
            _app_document(),
            _entity("e1", "person:pdf", "PDF", "PERSON"),
            _entity("e2", "person:person_4", "PERSON_4", "PERSON"),
            _entity("e3", "gpe:0nd", "0nd", "GPE"),
            _entity("e4", "person:jane doe", "Jane Doe", "PERSON"),
            _rel("d1", "e1", "MENTIONS"),
            _rel("d1", "e4", "MENTIONS"),
            _rel("e1", "e4", "RELATES_TO"),
        ]
    )

    plan = graph_cleanup.plan_cleanup(snapshot)

    removal_ids = {item["entity_id"] for item in plan["removals"]}

    assert removal_ids == {
        "person:pdf",
        "person:person_4",
        "gpe:0nd",
    }

    # The relationship into the valid entity is reported with its
    # endpoints so reviewers can inspect what disappears with the node.
    assert {
        "type": "MENTIONS",
        "start": "doc-1",
        "end": "person:pdf",
    } in plan["relationships_to_invalid"]

    assert {
        "type": "RELATES_TO",
        "start": "person:pdf",
        "end": "person:jane doe",
    } in plan["relationships_to_invalid"]

    # A valid uncommon name survives.
    assert "person:jane doe" not in removal_ids


def test_near_duplicates_with_similar_names_are_flagged_for_review():
    snapshot = _snapshot(
        [
            _app_document(),
            _entity(
                "e1",
                "org:acme corporation",
                "Acme Corporation",
                "ORG",
            ),
            _entity(
                "e2",
                "org:acme corporaton",
                "Acme Corporaton",
                "ORG",
            ),
            _rel("d1", "e1", "MENTIONS"),
            _rel("d1", "e2", "MENTIONS"),
        ]
    )

    plan = graph_cleanup.plan_cleanup(snapshot)

    # Similar-but-not-identical names are never merged automatically.
    assert plan["merges"] == []
    assert len(plan["ambiguous"]) == 1
    assert plan["ambiguous"][0]["similarity"] >= 0.9


def test_ambiguous_ownership_is_flagged_and_never_planned():
    snapshot = _snapshot(
        [
            _app_document(),
            # Deterministic-looking id, but wired into an unrelated
            # graph: ownership is ambiguous.
            _entity("e1", "person:neo guy", "Neo Guy", "PERSON"),
            _node("m1", ["Movie"], name="Matrix"),
            _rel("e1", "m1", "ACTED_IN"),
        ]
    )

    plan = graph_cleanup.plan_cleanup(snapshot)

    assert plan["merges"] == []
    assert plan["removals"] == []
    assert len(plan["needs_review_ownership"]) == 1


def test_document_without_ingest_source_is_foreign():
    snapshot = _snapshot(
        [
            _node(
                "d9",
                ["Document"],
                document_id="some-file.pdf",
                source=None,
            ),
        ]
    )

    plan = graph_cleanup.plan_cleanup(snapshot)

    assert plan["_owned"]["app_document_keys"] == set()


# --------------------------------------------------------- idempotency


def _apply_plan_to_snapshot(snapshot, plan):
    """Tiny simulator: apply planned actions to an in-memory copy."""

    nodes = [dict(node) for node in snapshot["nodes"]]
    relationships = [
        dict(relationship)
        for relationship in snapshot["relationships"]
    ]

    remove_ids = {
        item["entity_id"] for item in plan["removals"]
    }
    node_by_entity_id = {
        node["entity_id"]: node
        for node in nodes
        if node.get("entity_id")
    }

    for merge in plan["merges"]:
        canonical = node_by_entity_id[
            merge["canonical"]["entity_id"]
        ]

        for duplicate in merge["duplicates"]:
            duplicate_node = node_by_entity_id[
                duplicate["entity_id"]
            ]
            remove_ids.add(duplicate["entity_id"])

            for relationship in relationships:
                if relationship["start"] == duplicate_node["key"]:
                    relationship["start"] = canonical["key"]
                if relationship["end"] == duplicate_node["key"]:
                    relationship["end"] = canonical["key"]

    nodes = [
        node
        for node in nodes
        if node.get("entity_id") not in remove_ids
    ]

    valid_keys = {node["key"] for node in nodes}

    relationships = [
        relationship
        for relationship in relationships
        if relationship["start"] in valid_keys
        and relationship["end"] in valid_keys
    ]

    return _snapshot(nodes, relationships)


def test_cleanup_plan_is_idempotent():
    snapshot = _snapshot(
        [
            _app_document(),
            _entity("e1", "org:acme corp", "Acme Corp.", "ORG"),
            _entity("e2", "org:acme_corp", "Acme Corp", "ORG"),
            _entity("e3", "person:pdf", "PDF", "PERSON"),
            _entity(
                "e4",
                "org:acme corporation",
                "Acme Corporation",
                "ORG",
            ),
            _rel("d1", "e1", "MENTIONS"),
            _rel("d1", "e2", "MENTIONS"),
            _rel("d1", "e3", "MENTIONS"),
            _rel("d1", "e4", "MENTIONS"),
        ]
    )

    first_plan = graph_cleanup.plan_cleanup(snapshot)

    assert len(first_plan["merges"]) == 1
    assert len(first_plan["removals"]) == 1

    cleaned = _apply_plan_to_snapshot(snapshot, first_plan)

    second_plan = graph_cleanup.plan_cleanup(cleaned)

    # No further changes: merges and removals are exhausted. Review
    # flags (the near-duplicate) may persist but never act.
    assert second_plan["merges"] == []
    assert second_plan["removals"] == []

    # Provenance survived the merge: the canonical node is still
    # mentioned by the document.
    mentioned = {
        relationship["end"]
        for relationship in cleaned["relationships"]
        if relationship["type"] == "MENTIONS"
    }

    assert mentioned == {"e1", "e4"}


# ------------------------------------------------------------- execution


class FakeResult:
    def __init__(self, records=()):
        self._records = list(records)

    def __iter__(self):
        return iter(self._records)

    def single(self):
        return self._records[0] if self._records else None


class FakeSession:
    def __init__(self, reads=None):
        self.writes: list[tuple[str, dict]] = []
        self.reads = reads or {}

    def run(self, query, **parameters):
        stripped = " ".join(query.split())

        for fragment, records in self.reads.items():
            if fragment in stripped:
                return FakeResult(records)

        self.writes.append((stripped, parameters))
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


def _plan_for_execution():
    snapshot = _snapshot(
        [
            _app_document(),
            _entity("e1", "org:acme corp", "Acme Corp.", "ORG"),
            _entity("e2", "org:acme_corp", "Acme Corp", "ORG"),
            _entity("e3", "person:pdf", "PDF", "PERSON"),
            _rel("d1", "e1", "MENTIONS"),
            _rel("d1", "e2", "MENTIONS"),
            _rel("d1", "e3", "MENTIONS"),
        ]
    )

    return graph_cleanup.plan_cleanup(snapshot)


def test_execution_only_touches_planned_entities_and_guards_ids(
    monkeypatch,
):
    plan = _plan_for_execution()

    duplicate_id = plan["merges"][0]["duplicates"][0]["entity_id"]
    canonical_id = plan["merges"][0]["canonical"]["entity_id"]

    session = FakeSession(
        reads={
            # The duplicate's current relationships, both directions.
            "MATCH (dup:Entity {entity_id: $dup})-[r]->(other)": [
                {"rel_type": "MENTIONS", "other_key": "d1"}
            ],
            "MATCH (other)-[r]->(dup:Entity {entity_id: $dup})": [],
        }
    )

    monkeypatch.setattr(
        graph_cleanup,
        "get_driver",
        lambda: FakeDriver(session),
    )

    applied = graph_cleanup.execute_cleanup(plan)

    assert applied["errors"] == []
    assert applied["merges"] == [
        {
            "canonical_entity_id": canonical_id,
            "merged_entity_ids": [duplicate_id],
        }
    ]
    assert applied["removals"] == [
        {"entity_id": "person:pdf"}
    ]

    write_queries = [query for query, _ in session.writes]

    # Relationship rewiring uses MERGE onto the canonical node...
    assert any(
        "MERGE (canonical)-[:MENTIONS]->(other)" in query
        for query in write_queries
    )

    # ...the duplicate is removed with DETACH DELETE...
    assert any(
        "DETACH DELETE dup" in query and duplicate_id == params["dup"]
        for query, params in session.writes
        if "detach delete dup" in query.lower()
    )

    # ...and every destructive query is scoped by entity_id with the
    # deterministic id-pattern guard inside the query itself.
    for query, params in session.writes:
        if "DETACH DELETE" in query:
            assert "entity_id" in params or "dup" in params

    removal_writes = [
        (query, params)
        for query, params in session.writes
        if params.get("entity_id") == "person:pdf"
    ]

    assert removal_writes
    assert "id_pattern" in removal_writes[0][1]
    assert removal_writes[0][1]["id_pattern"] == (
        graph_cleanup.ENTITY_ID_PATTERN.pattern
    )


def test_execution_skips_unexpected_relationship_types(monkeypatch):
    plan = _plan_for_execution()

    session = FakeSession(
        reads={
            "MATCH (dup:Entity {entity_id: $dup})-[r]->(other)": [
                {"rel_type": "bad type!", "other_key": "d1"}
            ],
            "MATCH (other)-[r]->(dup:Entity {entity_id: $dup})": [],
        }
    )

    monkeypatch.setattr(
        graph_cleanup,
        "get_driver",
        lambda: FakeDriver(session),
    )

    applied = graph_cleanup.execute_cleanup(plan)

    # The invalid type was skipped, the merge itself still completed,
    # and nothing interpolated an unvalidated identifier.
    assert applied["errors"] == []
    assert not any(
        "bad type!" in query for query, _ in session.writes
    )


def test_run_cleanup_defaults_to_dry_run(monkeypatch):
    plan = _plan_for_execution()

    session = FakeSession()

    monkeypatch.setattr(
        graph_cleanup,
        "fetch_snapshot",
        lambda: _snapshot([], []),
    )
    monkeypatch.setattr(
        graph_cleanup,
        "plan_cleanup",
        lambda _snapshot: plan,
    )
    monkeypatch.setattr(
        graph_cleanup,
        "get_driver",
        lambda: FakeDriver(session),
    )

    result = graph_cleanup.run_cleanup()

    assert result["dry_run"] is True
    assert result["executed"] is False
    assert result["planned"]["removals"] == 1
    assert result["planned"]["merges"] == 1
    # A dry run issues no writes whatsoever.
    assert session.writes == []
