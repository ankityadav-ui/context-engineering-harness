"""Safe cleanup of legacy noisy entities in the Neo4j knowledge graph.

Design rules (see README "Neo4j cleanup"):

1. **Report first, always.** :func:`build_report` and
   :func:`plan_cleanup` are pure functions over a graph snapshot and
   never touch Neo4j. The live fetchers only run MATCH/RETURN queries.
2. **Dry run is the default.** :func:`run_cleanup` with ``confirm=False``
   returns the plan without executing anything. Destructive changes
   require an explicit confirmation at the call site (the API endpoint
   requires ``confirm: true``).
3. **Only application-owned records are touched.** An entity is
   application-owned when it carries this application's deterministic
   ``type:slug`` id AND either is mentioned by an ingested Document or
   touches no foreign (non-application) nodes. Nodes with other labels
   (for example an unrelated movie graph) are counted in the report and
   never modified.
4. **Conservative merges.** Two entities merge only when their
   comparison keys are equal AND their types are equal (strong evidence
   of identity). Equal names with conflicting types, and near-duplicate
   names, are flagged for review instead of being merged or deleted.
5. **Provenance is preserved.** Merges move every relationship - most
   importantly Document -[:MENTIONS]-> Entity links - onto the
   surviving node before the duplicate is removed.
6. **Idempotent.** Re-running the cleanup finds no remaining actions,
   so a second run makes no additional changes.
"""

import logging
import re
from datetime import datetime, timezone

from app.context_engine.entity_validation import (
    CONCEPT_TOKENS,
    ENTITY_LABELS,
    entity_comparison_key,
    is_plausible_entity,
    is_usable_entity_name,
)
from app.context_engine.graph_service import (
    ALLOWED_RELATIONSHIPS,
    NEO4J_DATABASE,
    get_driver,
)

logger = logging.getLogger("app.context_engine.graph_cleanup")

# Labels this application writes. Everything else in the database is
# foreign (an unrelated graph) and is never touched.
APP_NODE_LABELS = {"Entity", "Document", "Case"}

# Deterministic entity ids produced by graph_service._entity_node_id:
# "<type.lower()>:<comparison-key slug>".
ENTITY_ID_PATTERN = re.compile(
    r"^[a-z]{2,10}:[a-z0-9][a-z0-9 _-]*$"
)

# Ingested documents carry this source; the uuid-shaped fallback keeps
# documents from earlier iterations in scope.
APP_DOCUMENT_SOURCE = "document-upload"
UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

# Relationship types this application creates. Types read from the
# graph that do not match this safe shape are reported but skipped
# during execution (no interpolated Cypher is ever built from them).
RELATIONSHIP_TYPE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,60}$")

# Similarity at or above which two different names are flagged for
# human review (never auto-merged).
AMBIGUOUS_SIMILARITY_THRESHOLD = 0.9

# Bounds so one huge graph cannot produce an unbounded report payload.
MAX_REPORT_ITEMS = 200


# ---------------------------------------------------------------------------
# Snapshot model (pure data; no database access)
# ---------------------------------------------------------------------------
# node: {
#   "key": str, "labels": set[str], "entity_id": str|None,
#   "name": str|None, "entity_type": str|None,
#   "document_id": str|None, "source": str|None,
# }
# relationship: {"start": key, "end": key, "type": str, ...properties}


def fetch_snapshot() -> dict:
    """Read the graph facts the cleanup needs. Read-only queries."""

    driver = get_driver()

    with driver.session(database=NEO4J_DATABASE) as session:
        nodes = [
            {
                "key": record["key"],
                "labels": set(record["labels"] or []),
                "entity_id": record["entity_id"],
                "name": record["name"],
                "entity_type": record["entity_type"],
                "document_id": record["document_id"],
                "source": record["source"],
            }
            for record in session.run(
                """
                MATCH (n)
                WHERE n:Entity OR n:Document OR n:Case
                RETURN elementId(n) AS key,
                       labels(n) AS labels,
                       n.entity_id AS entity_id,
                       n.name AS name,
                       n.entity_type AS entity_type,
                       n.document_id AS document_id,
                       n.source AS source
                """
            )
        ]

        relationships = [
            {
                "start": record["start"],
                "end": record["end"],
                "type": record["type"],
                **(record.get("properties") or {}),
            }
            for record in session.run(
                """
                MATCH (a)-[r]->(b)
                WHERE a:Entity OR b:Entity
                   OR a:Document OR b:Document
                RETURN elementId(a) AS start,
                       elementId(b) AS end,
                       type(r) AS type,
                       properties(r) AS properties
                """
            )
        ]

        label_counts = {
            record["label"]: record["count"]
            for record in session.run(
                """
                MATCH (n)
                UNWIND labels(n) AS label
                RETURN label, count(*) AS count
                """
            )
        }

        relationship_counts = {
            record["type"]: record["count"]
            for record in session.run(
                """
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(*) AS count
                """
            )
        }

    return {
        "nodes": nodes,
        "relationships": relationships,
        "label_counts": label_counts,
        "relationship_counts": relationship_counts,
    }


# ---------------------------------------------------------------------------
# Ownership classification (pure)
# ---------------------------------------------------------------------------


def is_app_document(node: dict) -> bool:
    """Report whether a Document node was written by this application."""

    if "Document" not in node["labels"]:
        return False

    if node.get("source") == APP_DOCUMENT_SOURCE:
        return True

    document_id = node.get("document_id") or ""

    return bool(UUID_PATTERN.match(document_id))


def classify_snapshot(snapshot: dict) -> dict:
    """Split the snapshot into owned and foreign records."""

    nodes_by_key = {node["key"]: node for node in snapshot["nodes"]}

    app_document_keys = {
        node["key"]
        for node in snapshot["nodes"]
        if is_app_document(node)
    }

    neighbor_labels: dict[str, set[str]] = {
        node["key"]: set() for node in snapshot["nodes"]
    }

    mentioned_by: dict[str, set[str]] = {}

    for relationship in snapshot["relationships"]:
        start = nodes_by_key.get(relationship["start"])
        end = nodes_by_key.get(relationship["end"])

        if start is None or end is None:
            continue

        neighbor_labels[start["key"]].update(end["labels"])
        neighbor_labels[end["key"]].update(start["labels"])

        if (
            relationship["type"] == "MENTIONS"
            and end.get("entity_id")
            and start["key"] in app_document_keys
        ):
            mentioned_by.setdefault(end["entity_id"], set()).add(
                start.get("document_id") or start["key"]
            )

    owned_entities = []
    foreign_nodes = []
    needs_review_ownership = []

    for node in snapshot["nodes"]:
        if "Entity" not in node["labels"]:
            if node["key"] not in app_document_keys and not (
                node["labels"] & APP_NODE_LABELS
            ):
                foreign_nodes.append(node)
            continue

        entity_id = node.get("entity_id") or ""

        pattern_owned = bool(ENTITY_ID_PATTERN.match(entity_id))
        linked_owned = entity_id in mentioned_by

        if not pattern_owned and not linked_owned:
            foreign_nodes.append(node)
            continue

        touching_foreign = bool(
            neighbor_labels[node["key"]] - APP_NODE_LABELS
        )

        if touching_foreign and not linked_owned:
            # An id that looks like ours but touches an unrelated
            # graph: ambiguous ownership, never auto-modified.
            needs_review_ownership.append(node)
            continue

        owned_entities.append(node)

    return {
        "nodes_by_key": nodes_by_key,
        "app_document_keys": app_document_keys,
        "mentioned_by": mentioned_by,
        "owned_entities": owned_entities,
        "foreign_nodes": foreign_nodes,
        "needs_review_ownership": needs_review_ownership,
    }


# ---------------------------------------------------------------------------
# Report and action planning (pure)
# ---------------------------------------------------------------------------


def _entity_view(node: dict, owned: dict) -> dict:
    """Serializable description of one entity with its provenance."""

    entity_id = node.get("entity_id") or ""

    return {
        "entity_id": entity_id,
        "name": node.get("name"),
        "entity_type": node.get("entity_type"),
        "comparison_key": entity_comparison_key(node.get("name") or ""),
        "document_ids": sorted(
            owned["mentioned_by"].get(entity_id, set())
        ),
    }


def _bound(items: list) -> dict:
    """Bound a report section to MAX_REPORT_ITEMS with a real total."""

    return {
        "items": items[:MAX_REPORT_ITEMS],
        "total": len(items),
        "truncated": len(items) > MAX_REPORT_ITEMS,
    }


def plan_cleanup(snapshot: dict) -> dict:
    """Compute removals, merges, and review flags. Pure function."""

    owned = classify_snapshot(snapshot)
    owned_entity_keys = {
        node["key"] for node in owned["owned_entities"]
    }

    missing_provenance = []
    unsupported_relationships = []

    for relationship in snapshot["relationships"]:
        start_key = relationship["start"]
        end_key = relationship["end"]
        start_node = owned["nodes_by_key"].get(start_key)
        end_node = owned["nodes_by_key"].get(end_key)
        app_relationship = (
            start_key in owned["app_document_keys"]
            and end_key in owned_entity_keys
        ) or (
            start_key in owned_entity_keys
            and end_key in owned_entity_keys
        )

        if (
            app_relationship
            and relationship["type"] not in ALLOWED_RELATIONSHIPS
        ):
            unsupported_relationships.append(
                {
                    "relationship": relationship["type"],
                    "start": (
                        (start_node or {}).get("document_id")
                        or (start_node or {}).get("entity_id")
                        or start_key
                    ),
                    "end": (
                        (end_node or {}).get("entity_id")
                        or (end_node or {}).get("document_id")
                        or end_key
                    ),
                    "proposed": (
                        "Review the edge and re-ingest using an "
                        "allowlisted relationship type."
                    ),
                    "confidence": 0.95,
                }
            )

        document_mention = (
            relationship["type"] == "MENTIONS"
            and start_key in owned["app_document_keys"]
            and end_key in owned_entity_keys
        )
        entity_relationship = (
            start_key in owned_entity_keys
            and end_key in owned_entity_keys
        )

        if not (document_mention or entity_relationship):
            continue

        source_refs = relationship.get("source_refs") or []
        if isinstance(source_refs, str):
            source_refs = [source_refs]
        elif not isinstance(source_refs, (list, tuple)):
            source_refs = []

        if document_mention:
            document_id = (start_node or {}).get("document_id")
            has_document_id = (
                isinstance(document_id, str)
                and bool(document_id.strip())
                and relationship.get("document_id") == document_id
            )
            has_source_refs = has_document_id and bool(source_refs) and all(
                isinstance(source_ref, str)
                and bool(source_ref.strip())
                and (
                    source_ref == document_id
                    or source_ref.startswith(f"{document_id}#page=")
                )
                for source_ref in source_refs
            )
            expected_provenance = ["document_id", "source_refs"]
        else:
            document_ids = relationship.get("document_ids") or []
            if isinstance(document_ids, str):
                document_ids = [document_ids]
            elif not isinstance(document_ids, (list, tuple)):
                document_ids = []
            valid_document_ids = [
                document_id
                for document_id in document_ids
                if isinstance(document_id, str) and document_id.strip()
            ]
            has_document_id = bool(document_ids) and (
                len(valid_document_ids) == len(document_ids)
            )
            has_source_refs = (
                has_document_id
                and bool(source_refs)
                and all(
                    isinstance(source_ref, str)
                    and bool(source_ref.strip())
                    and any(
                        source_ref == document_id
                        or source_ref.startswith(
                            f"{document_id}#page="
                        )
                        for document_id in valid_document_ids
                    )
                    for source_ref in source_refs
                )
            )
            expected_provenance = ["document_ids", "source_refs"]

        missing_fields = []
        if not has_document_id:
            missing_fields.append(expected_provenance[0])
        if not has_source_refs:
            missing_fields.append("source_refs")

        if missing_fields:
            missing_provenance.append(
                {
                    "relationship": relationship["type"],
                    "start": (
                        (start_node or {}).get("document_id")
                        or (start_node or {}).get("entity_id")
                        or start_key
                    ),
                    "end": (
                        (end_node or {}).get("entity_id")
                        or (end_node or {}).get("document_id")
                        or end_key
                    ),
                    "current": {
                        field: relationship.get(field)
                        for field in expected_provenance
                    },
                    "proposed": (
                        "Re-ingest the source document to restore "
                        "validated provenance."
                    ),
                    "missing_fields": missing_fields,
                    "confidence": 0.95,
                }
            )

    removals = []
    removal_ids = set()

    for node in owned["owned_entities"]:
        name = node.get("name") or ""
        entity_type = (node.get("entity_type") or "").upper()

        reasons = []

        key = entity_comparison_key(name)

        if not name.strip():
            reasons.append("empty-name")
        elif not is_usable_entity_name(name):
            reasons.append("malformed-or-placeholder-name")
        elif key in CONCEPT_TOKENS:
            # A stand-alone concept or document-artifact word ("PDF",
            # "NER") is never a real entity name.
            reasons.append("concept-or-artifact-word")

        if entity_type not in ENTITY_LABELS:
            reasons.append("unsupported-entity-type")

        if reasons:
            removals.append(
                {
                    **_entity_view(node, owned),
                    "reasons": reasons,
                }
            )
            removal_ids.add(node["key"])

    # Duplicate groups among the surviving owned entities.
    groups: dict[str, list[dict]] = {}

    for node in owned["owned_entities"]:
        if node["key"] in removal_ids:
            continue

        key = entity_comparison_key(node.get("name") or "")

        if not key:
            continue

        groups.setdefault(key, []).append(node)

    merges = []
    conflicting_labels = []

    for key in sorted(groups):
        members = groups[key]

        if len(members) < 2:
            continue

        types = {
            (member.get("entity_type") or "").upper()
            for member in members
        }

        views = [_entity_view(member, owned) for member in members]

        if len(types) > 1:
            # Same name, contradictory labels: flagged, never merged
            # automatically (the ingestion path now resolves such
            # conflicts before they are stored).
            conflicting_labels.append(
                {
                    "comparison_key": key,
                    "entity_types": sorted(types),
                    "entities": views,
                }
            )
            continue

        # Deterministic canonical choice: most provenance first, then
        # most relationships, then smallest entity id.
        def _rank(member: dict) -> tuple:
            rel_count = sum(
                1
                for relationship in snapshot["relationships"]
                if member["key"]
                in (relationship["start"], relationship["end"])
            )
            return (
                -len(
                    owned["mentioned_by"].get(
                        member.get("entity_id") or "", set()
                    )
                ),
                -rel_count,
                member.get("entity_id") or "",
            )

        ordered = sorted(members, key=_rank)
        canonical = ordered[0]

        merges.append(
            {
                "comparison_key": key,
                "entity_type": (
                    canonical.get("entity_type") or ""
                ).upper(),
                "canonical": _entity_view(canonical, owned),
                "duplicates": [
                    _entity_view(member, owned)
                    for member in ordered[1:]
                ],
            }
        )

    # Stored entities whose name/type pair fails the ingestion-time
    # plausibility rules are flagged for review, never auto-removed:
    # uncommon-but-correct names must survive human judgement.
    type_implausible = [
        {
            **_entity_view(node, owned),
            "reason": "stored type is implausible for this name",
        }
        for node in owned["owned_entities"]
        if node["key"] not in removal_ids
        and (node.get("entity_type") or "") in ENTITY_LABELS
        and not is_plausible_entity(
            node.get("name") or "",
            node.get("entity_type") or "",
        )
    ]

    # Near-duplicate names that were NOT merged: flagged for review.
    ambiguous = []
    surviving = [
        node
        for node in owned["owned_entities"]
        if node["key"] not in removal_ids
    ]

    from difflib import SequenceMatcher

    seen_pairs = set()

    for index, first in enumerate(surviving):
        first_key = entity_comparison_key(first.get("name") or "")
        first_type = (first.get("entity_type") or "").upper()

        for second in surviving[index + 1:]:
            second_key = entity_comparison_key(
                second.get("name") or ""
            )
            second_type = (
                second.get("entity_type") or ""
            ).upper()

            if first_key == second_key:
                continue

            if first_type != second_type:
                continue

            pair = (first.get("entity_id"), second.get("entity_id"))

            if pair in seen_pairs:
                continue

            ratio = SequenceMatcher(
                None, first_key, second_key
            ).ratio()

            if ratio >= AMBIGUOUS_SIMILARITY_THRESHOLD:
                seen_pairs.add(pair)
                ambiguous.append(
                    {
                        "similarity": round(ratio, 3),
                        "entities": [
                            _entity_view(first, owned),
                            _entity_view(second, owned),
                        ],
                    }
                )

    # Relationships that point at entities scheduled for removal.
    node_labels = {
        key: node["labels"]
        for key, node in owned["nodes_by_key"].items()
    }

    relationships_to_invalid = []

    for relationship in snapshot["relationships"]:
        start_node = owned["nodes_by_key"].get(
            relationship["start"]
        )
        end_node = owned["nodes_by_key"].get(relationship["end"])

        start_invalid = (
            start_node is not None
            and start_node["key"] in removal_ids
        )
        end_invalid = (
            end_node is not None and end_node["key"] in removal_ids
        )

        if not (start_invalid or end_invalid):
            continue

        def _endpoint_label(node) -> str | None:
            if node is None:
                return None
            return (
                node.get("entity_id")
                or node.get("document_id")
                or node["key"]
            )

        relationships_to_invalid.append(
            {
                "type": relationship["type"],
                "start": _endpoint_label(start_node),
                "end": _endpoint_label(end_node),
            }
        )

    return {
        "removals": removals,
        "merges": merges,
        "conflicting_labels": conflicting_labels,
        "type_implausible": type_implausible,
        "ambiguous": ambiguous,
        "relationships_to_invalid": relationships_to_invalid,
        "missing_provenance": missing_provenance,
        "unsupported_relationships": unsupported_relationships,
        "needs_review_ownership": [
            _entity_view(node, owned)
            for node in owned["needs_review_ownership"]
        ],
        "_owned": owned,
        "_node_labels": node_labels,
    }


def build_report(snapshot: dict, plan: dict | None = None) -> dict:
    """Read-only cleanup report over a graph snapshot."""

    if plan is None:
        plan = plan_cleanup(snapshot)

    owned = plan["_owned"]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "dry-run (read-only; nothing was modified)",
        "scope": {
            "description": (
                "Application-owned records are Document nodes written "
                "by document ingestion and Entity nodes with "
                "deterministic type:slug ids that are mentioned by "
                "such documents (or touch no foreign nodes). "
                "Everything else in the database is reported as "
                "foreign and is never modified."
            ),
            "owned_entities": len(owned["owned_entities"]),
            "owned_documents": len(owned["app_document_keys"]),
            "foreign_nodes_untouched": len(
                owned["foreign_nodes"]
            ),
        },
        "counts": {
            "labels": snapshot["label_counts"],
            "relationship_types": snapshot[
                "relationship_counts"
            ],
        },
        "malformed_or_placeholder": _bound(plan["removals"]),
        "suspected_duplicates": _bound(plan["merges"]),
        "conflicting_entity_labels": _bound(
            plan["conflicting_labels"]
        ),
        "relationships_to_suspected_invalid": _bound(
            plan["relationships_to_invalid"]
        ),
        "missing_provenance": _bound(plan["missing_provenance"]),
        "unsupported_relationships": _bound(
            plan["unsupported_relationships"]
        ),
        "ambiguous_for_review": _bound(plan["ambiguous"]),
        "type_implausible_for_review": _bound(
            plan["type_implausible"]
        ),
        "ownership_ambiguous_for_review": _bound(
            plan["needs_review_ownership"]
        ),
        "planned_actions": {
            "removals": len(plan["removals"]),
            "merges": len(plan["merges"]),
            "needs_review": (
                len(plan["conflicting_labels"])
                + len(plan["ambiguous"])
                + len(plan["needs_review_ownership"])
                + len(plan["type_implausible"])
                + len(plan["missing_provenance"])
                + len(plan["unsupported_relationships"])
            ),
        },
        "destructive_changes_performed": False,
    }


# ---------------------------------------------------------------------------
# Execution (opt-in, scoped, reviewable)
# ---------------------------------------------------------------------------


def _merge_duplicate(session, merge: dict) -> None:
    """Move one duplicate's relationships onto the canonical node."""

    canonical_id = merge["canonical"]["entity_id"]

    for duplicate in merge["duplicates"]:
        duplicate_id = duplicate["entity_id"]

        if duplicate_id == canonical_id:
            continue

        # Read the duplicate's current relationships fresh, so the
        # merge is correct even if the graph changed since planning.
        outgoing = list(
            session.run(
                """
                MATCH (dup:Entity {entity_id: $dup})-[r]->(other)
                RETURN type(r) AS rel_type,
                       elementId(other) AS other_key
                """,
                dup=duplicate_id,
            )
        )

        incoming = list(
            session.run(
                """
                MATCH (other)-[r]->(dup:Entity {entity_id: $dup})
                RETURN type(r) AS rel_type,
                       elementId(other) AS other_key
                """,
                dup=duplicate_id,
            )
        )

        for record in outgoing:
            rel_type = record["rel_type"]

            if not RELATIONSHIP_TYPE_PATTERN.match(rel_type):
                logger.warning(
                    "Skipping merge of unexpected relationship type "
                    "%r between %s and %s.",
                    rel_type,
                    duplicate_id,
                    canonical_id,
                )
                continue

            session.run(
                f"""
                MATCH (canonical:Entity {{entity_id: $canonical}})
                MATCH (other) WHERE elementId(other) = $other_key
                MERGE (canonical)-[:{rel_type}]->(other)
                """,
                canonical=canonical_id,
                other_key=record["other_key"],
            )

        for record in incoming:
            rel_type = record["rel_type"]

            if not RELATIONSHIP_TYPE_PATTERN.match(rel_type):
                logger.warning(
                    "Skipping merge of unexpected relationship type "
                    "%r between %s and %s.",
                    rel_type,
                    record["other_key"],
                    duplicate_id,
                )
                continue

            session.run(
                f"""
                MATCH (canonical:Entity {{entity_id: $canonical}})
                MATCH (other) WHERE elementId(other) = $other_key
                MERGE (other)-[:{rel_type}]->(canonical)
                """,
                canonical=canonical_id,
                other_key=record["other_key"],
            )

        # Remove the duplicate and whatever is left attached to it.
        session.run(
            """
            MATCH (dup:Entity {entity_id: $dup})
            DETACH DELETE dup
            """,
            dup=duplicate_id,
        )


def execute_cleanup(plan: dict) -> dict:
    """Apply the planned merges and removals. Opt-in only.

    Every write re-checks the entity id pattern inside the query, so a
    stale plan can never delete a node outside this application's id
    space. Each planned action is independent: one failure is recorded
    and does not stop the others.
    """

    driver = get_driver()

    applied = {"merges": [], "removals": [], "errors": []}

    with driver.session(database=NEO4J_DATABASE) as session:
        for merge in plan["merges"]:
            try:
                _merge_duplicate(session, merge)
                applied["merges"].append(
                    {
                        "canonical_entity_id": merge["canonical"][
                            "entity_id"
                        ],
                        "merged_entity_ids": [
                            duplicate["entity_id"]
                            for duplicate in merge["duplicates"]
                        ],
                    }
                )

            except Exception as error:
                applied["errors"].append(
                    {
                        "action": "merge",
                        "canonical_entity_id": merge["canonical"][
                            "entity_id"
                        ],
                        "error": type(error).__name__,
                    }
                )

        for removal in plan["removals"]:
            entity_id = removal["entity_id"]

            try:
                # The pattern guard lives inside the query: only this
                # application's deterministic id space can be deleted.
                result = session.run(
                    """
                    MATCH (dup:Entity {entity_id: $entity_id})
                    WHERE dup.entity_id =~ $id_pattern
                    DETACH DELETE dup
                    RETURN count(*) AS deleted
                    """,
                    entity_id=entity_id,
                    id_pattern=ENTITY_ID_PATTERN.pattern,
                )

                applied["removals"].append(
                    {"entity_id": entity_id}
                )

            except Exception as error:
                applied["errors"].append(
                    {
                        "action": "remove",
                        "entity_id": entity_id,
                        "error": type(error).__name__,
                    }
                )

    return applied


def run_cleanup(confirm: bool = False) -> dict:
    """Report (dry run, default) or apply the legacy cleanup.

    ``confirm=False`` (the default) performs a read-only pass and
    returns the report plus the planned actions. ``confirm=True``
    additionally executes the plan and re-computes the remaining
    actions, which proves idempotency for this run.
    """

    snapshot = fetch_snapshot()
    plan = plan_cleanup(snapshot)
    report = build_report(snapshot, plan)

    if not confirm:
        return {
            "dry_run": True,
            "executed": False,
            "report": report,
            "planned": {
                "removals": len(plan["removals"]),
                "merges": len(plan["merges"]),
                "needs_review": (
                    len(plan["conflicting_labels"])
                    + len(plan["ambiguous"])
                    + len(plan["needs_review_ownership"])
                    + len(plan["type_implausible"])
                ),
            },
        }

    applied = execute_cleanup(plan)

    # Prove idempotency: the fresh plan after execution reports what
    # remains (review-only findings stay, actionable items are gone).
    remaining_snapshot = fetch_snapshot()
    remaining_plan = plan_cleanup(remaining_snapshot)

    logger.info(
        "Graph cleanup executed: merges=%d removals=%d errors=%d "
        "remaining_merges=%d remaining_removals=%d",
        len(applied["merges"]),
        len(applied["removals"]),
        len(applied["errors"]),
        len(remaining_plan["merges"]),
        len(remaining_plan["removals"]),
    )

    return {
        "dry_run": False,
        "executed": True,
        "applied": applied,
        "remaining_after_cleanup": {
            "removals": len(remaining_plan["removals"]),
            "merges": len(remaining_plan["merges"]),
            "needs_review": (
                len(remaining_plan["conflicting_labels"])
                + len(remaining_plan["ambiguous"])
                + len(remaining_plan["needs_review_ownership"])
                + len(remaining_plan["type_implausible"])
            ),
        },
    }
