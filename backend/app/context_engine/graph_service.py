
# Import libraries for environment variables, paths, and Neo4j.
import os
import time
from pathlib import Path
from dotenv import load_dotenv
import certifi
from neo4j import GraphDatabase, TrustCustomCAs
from collections.abc import Callable

# Graph-specific validation lives outside the masking NER service so
# improving graph precision can never reduce masking coverage.
from app.context_engine.entity_validation import (
    ENTITY_LABELS as _ENTITY_LABELS,
    entity_comparison_key,
    entity_name_key,
    is_plausible_entity,
    is_usable_entity_name,
    normalize_entity_name,
    resolve_entity_types,
)


# Locate the project's root directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Load environment variables from the root .env file.
load_dotenv(PROJECT_ROOT / ".env")


# Read Neo4j connection settings.
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


# Define the relationship types the application is allowed to create.
ALLOWED_RELATIONSHIPS = {
    "RELATES_TO",
    "CONCERNS",
    "PRESCRIBED",
    "MENTIONS",
    "PART_OF",
}


# Create the Neo4j driver only when credentials are configured.
# This prevents importing the module from immediately failing
# when Neo4j has not been configured yet.
driver = None

if NEO4J_URI and NEO4J_PASSWORD:
    # Use the standard routing URI so custom TLS settings are accepted.
    # Preserve the original URI in the environment configuration.
    driver_uri = NEO4J_URI.replace(
        "neo4j+s://",
        "neo4j://",
        1,
    )

    driver = GraphDatabase.driver(
        driver_uri,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),

        # Keep TLS encryption enabled and verify certificates
        # against the trusted CA bundle.
        encrypted=True,
        trusted_certificates=TrustCustomCAs(certifi.where()),
    )

def is_configured() -> bool:
    """Report whether a Neo4j driver was created from the environment."""

    return driver is not None


def get_driver():
    """Return the configured Neo4j driver or raise a clear error."""

    if driver is None:
        raise RuntimeError(
            "Neo4j is not configured. Set NEO4J_URI and "
            "NEO4J_PASSWORD in the root .env file."
        )

    return driver


def verify_graph_connection():
    """Check whether the application can connect to Neo4j."""

    get_driver().verify_connectivity()
    return True


def create_entity(entity_id: str, name: str, entity_type: str):
    """Create an entity or update it if its ID already exists."""

    # Validate the required entity fields.
    if not entity_id.strip():
        raise ValueError("Entity ID cannot be empty.")

    if not name.strip():
        raise ValueError("Entity name cannot be empty.")

    if not entity_type.strip():
        raise ValueError("Entity type cannot be empty.")

    # Use MERGE to avoid creating duplicate entities with the same ID.
    query = """
    MERGE (entity:Entity {entity_id: $entity_id})
    SET entity.name = $name,
        entity.entity_type = $entity_type
    RETURN
        entity.entity_id AS entity_id,
        entity.name AS name,
        entity.entity_type AS entity_type
    """

    # Execute the query using parameters rather than string interpolation.
    with get_driver().session(database=NEO4J_DATABASE) as session:
        record = session.run(
            query,
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
        ).single()

        return dict(record)


def create_relationship(
    source_id: str,
    target_id: str,
    relationship_type: str,
):
    """Create a directed relationship between two existing entities."""

    # Validate the relationship type against the allowlist.
    if relationship_type not in ALLOWED_RELATIONSHIPS:
        raise ValueError(
            f"Unsupported relationship type: {relationship_type}"
        )

    # Relationship types cannot be passed as ordinary Cypher parameters,
    # so select the query fragment from this trusted allowlist.
    relationship_queries = {
        "RELATES_TO": "MERGE (source)-[:RELATES_TO]->(target)",
        "CONCERNS": "MERGE (source)-[:CONCERNS]->(target)",
        "PRESCRIBED": "MERGE (source)-[:PRESCRIBED]->(target)",
        "MENTIONS": "MERGE (source)-[:MENTIONS]->(target)",
        "PART_OF": "MERGE (source)-[:PART_OF]->(target)",
    }

    query = f"""
    MATCH (source:Entity {{entity_id: $source_id}})
    MATCH (target:Entity {{entity_id: $target_id}})
    {relationship_queries[relationship_type]}
    RETURN
        source.name AS source,
        $relationship_type AS relationship,
        target.name AS target
    """

    # Match both entities before creating the relationship.
    with get_driver().session(database=NEO4J_DATABASE) as session:
        record = session.run(
            query,
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
        ).single()

        # MATCH returns no record if either entity does not exist.
        if record is None:
            raise ValueError(
                "Source or target entity was not found. "
                "Create both entities before linking them."
            )

        return dict(record)


def list_entities(
    limit: int = 25,
    offset: int = 0,
    entity_type: str | None = None,
):
    """Return entities from the graph, ordered by name."""

    # Keep a single request bounded.
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100.")

    if offset < 0:
        raise ValueError("offset cannot be negative.")

    entity_filter = (
        "WHERE entity.entity_type = $entity_type "
        if entity_type
        else ""
    )

    query = f"""
    MATCH (entity:Entity)
    {entity_filter}
    RETURN
        entity.entity_id AS entity_id,
        entity.name AS name,
        entity.entity_type AS entity_type
    ORDER BY entity.name
    SKIP $offset
    LIMIT $limit
    """

    # Return matching entities as ordinary Python dictionaries.
    with get_driver().session(database=NEO4J_DATABASE) as session:
        params = {"limit": limit, "offset": offset}
        if entity_type:
            params["entity_type"] = entity_type.strip().upper()
        records = session.run(query, **params)

        return [dict(record) for record in records]


def find_entities(
    search_text: str,
    limit: int = 20,
    entity_type: str | None = None,
    offset: int = 0,
):
    """Find entities whose names contain the supplied text."""

    # Reject empty searches.
    if not search_text or not search_text.strip():
        raise ValueError("Search text cannot be empty.")

    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100.")

    if offset < 0:
        raise ValueError("offset cannot be negative.")

    type_filter = (
        "AND entity.entity_type = $entity_type "
        if entity_type
        else ""
    )
    offset_clause = "SKIP $offset" if offset else ""

    query = f"""
    MATCH (entity:Entity)
    WHERE toLower(entity.name) CONTAINS toLower($search_text)
    {type_filter}
    RETURN
        entity.entity_id AS entity_id,
        entity.name AS name,
        entity.entity_type AS entity_type
    ORDER BY entity.name
    {offset_clause}
    LIMIT $limit
    """

    # Return matching entities as ordinary Python dictionaries.
    with get_driver().session(database=NEO4J_DATABASE) as session:
        parameters = {
            "search_text": search_text.strip(),
            "limit": limit,
        }
        if entity_type:
            parameters["entity_type"] = entity_type.strip().upper()
        if offset:
            parameters["offset"] = offset
        records = session.run(
            query,
            **parameters,
        )

        return [dict(record) for record in records]


def graph_counts() -> dict:
    """Return graph totals and entity-type counts using read-only queries."""

    queries = {
        "entities": "MATCH (entity:Entity) RETURN count(entity) AS count",
        "documents": (
            "MATCH (document:Document) RETURN count(document) AS count"
        ),
        "relationships": "MATCH ()-[r]->() RETURN count(r) AS count",
    }

    with get_driver().session(database=NEO4J_DATABASE) as session:
        counts = {}
        for name, query in queries.items():
            record = session.run(query).single()
            counts[name] = int(record["count"]) if record else 0

        type_query = """
        MATCH (entity:Entity)
        RETURN entity.entity_type AS entity_type, count(entity) AS entities
        ORDER BY entity_type
        """
        type_records = session.run(type_query)

        counts["entity_types"] = [
            {
                "entity_type": record["entity_type"],
                "entities": int(record["entities"]),
            }
            for record in type_records
        ]

    return counts


def find_entity_documents(
    entity_id: str,
    limit: int = 50,
) -> list[dict]:
    """Return documents linked to an entity through MENTIONS edges."""

    if not entity_id or not entity_id.strip():
        raise ValueError("entity_id cannot be empty.")

    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100.")

    query = """
    MATCH (document:Document)-[:MENTIONS]->(entity:Entity {entity_id: $entity_id})
    RETURN
        document.document_id AS document_id,
        document.title AS title,
        document.source AS source
    ORDER BY document.title
    LIMIT $limit
    """

    with get_driver().session(database=NEO4J_DATABASE) as session:
        records = session.run(
            query,
            entity_id=entity_id.strip(),
            limit=limit,
        )

        return [dict(record) for record in records]


def get_entity_neighborhood(
    entity_id: str,
    limit: int = 25,
) -> dict:
    """Return one entity and its bounded incoming/outgoing edges."""

    if not entity_id or not entity_id.strip():
        raise ValueError("entity_id cannot be empty.")

    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100.")

    query = """
    MATCH (center:Entity {entity_id: $entity_id})
    CALL {
        WITH center
        OPTIONAL MATCH (center)-[relationship]->(neighbor:Entity)
        RETURN collect(
            CASE WHEN relationship IS NULL THEN null ELSE {
                entity_id: neighbor.entity_id,
                name: neighbor.name,
                entity_type: neighbor.entity_type,
                relationship: type(relationship)
            } END
        )[0..$limit] AS outgoing_edges
    }
    CALL {
        WITH center
        OPTIONAL MATCH (neighbor:Entity)-[relationship]->(center)
        RETURN collect(
            CASE WHEN relationship IS NULL THEN null ELSE {
                entity_id: neighbor.entity_id,
                name: neighbor.name,
                entity_type: neighbor.entity_type,
                relationship: type(relationship)
            } END
        )[0..$limit] AS incoming_edges
    }
    RETURN
        center.entity_id AS center_id,
        center.name AS center_name,
        center.entity_type AS center_type,
        outgoing_edges,
        incoming_edges
    """

    with get_driver().session(database=NEO4J_DATABASE) as session:
        record = session.run(
            query,
            entity_id=entity_id.strip(),
            limit=limit,
        ).single()

    if record is None:
        return {"entity": None, "edges": []}

    entity = {
        "entity_id": record["center_id"],
        "name": record["center_name"],
        "entity_type": record["center_type"],
    }
    edges = []
    seen_edges = set()

    for direction, candidates in (
        ("outgoing", record["outgoing_edges"] or []),
        ("incoming", record["incoming_edges"] or []),
    ):
        for candidate in candidates:
            if not candidate or not candidate.get("entity_id"):
                continue

            if direction == "outgoing":
                edge = {
                    "source_id": entity["entity_id"],
                    "source": entity["name"],
                    "relationship": candidate["relationship"],
                    "target_id": candidate["entity_id"],
                    "target": candidate["name"],
                    "direction": direction,
                }
            else:
                edge = {
                    "source_id": candidate["entity_id"],
                    "source": candidate["name"],
                    "relationship": candidate["relationship"],
                    "target_id": entity["entity_id"],
                    "target": entity["name"],
                    "direction": direction,
                }

            key = (
                edge["source_id"],
                edge["relationship"],
                edge["target_id"],
            )
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append(edge)

    return {"entity": entity, "edges": edges}


def get_entity_relationships(
    entity_id: str,
    document_id: str | None = None,
):
    """Return entity relationships, optionally limited to one document."""

    query = """
    MATCH (entity:Entity {entity_id: $entity_id})

    OPTIONAL MATCH (entity)-[outgoing]->(target:Entity)
    WHERE $document_id IS NULL
       OR $document_id IN coalesce(outgoing.document_ids, [])
    WITH entity,
         collect(
             CASE
                 WHEN outgoing IS NULL THEN null
                 ELSE {
                     source: entity.name,
                     relationship: type(outgoing),
                     target: target.name,
                     direction: 'outgoing'
                 }
             END
         ) AS outgoing_relationships

    OPTIONAL MATCH (source:Entity)-[incoming]->(entity)
    WHERE $document_id IS NULL
       OR $document_id IN coalesce(incoming.document_ids, [])
    WITH outgoing_relationships,
         collect(
             CASE
                 WHEN incoming IS NULL THEN null
                 ELSE {
                     source: source.name,
                     relationship: type(incoming),
                     target: entity.name,
                     direction: 'incoming'
                 }
             END
         ) AS incoming_relationships

    RETURN outgoing_relationships, incoming_relationships
    """

    # Fetch relationships in both directions.
    with get_driver().session(database=NEO4J_DATABASE) as session:
        record = session.run(
            query,
            entity_id=entity_id,
            document_id=document_id,
        ).single()

        # Return an empty list when the entity does not exist.
        if record is None:
            return []

        outgoing = record["outgoing_relationships"]
        incoming = record["incoming_relationships"]

        # Combine both directions and discard empty optional matches.
        relationships = [
            item for item in outgoing + incoming
            if item is not None
        ]

        return relationships


# --------------------------------------------------
# Document / case ingestion and graph retrieval
# --------------------------------------------------
# Stable identifiers: documents are keyed by the document_id generated
# by the ingestion service, entities by a deterministic
# "TYPE:normalized-name" id, so re-ingestion merges instead of
# duplicating nodes. All Cypher queries use parameters.

# Bounds so one large document cannot flood the graph in a single
# ingestion run.
MAX_ENTITIES_PER_DOCUMENT = 20
MAX_RELATIONS_PER_DOCUMENT = 40

def _is_usable_entity_name(name: str) -> bool:
    """Backwards-compatible alias for the name-quality validator.

    The rules live in ``entity_validation`` so name quality and type
    plausibility stay separate from masking detection.
    """

    return is_usable_entity_name(name)


def _entity_node_id(entity_type: str, entity_name: str) -> str:
    """Build the deterministic, stable entity id used by MERGE.

    Built from the shared comparison key, so "Dr. Smith" and
    "dr smith" resolve to the same node instead of two near-duplicates.
    """

    normalized = entity_comparison_key(entity_name)

    slug = "".join(
        character
        if character.isalnum() or character in " -"
        else "_"
        for character in normalized
    ).strip("_")[:120]

    return f"{entity_type.lower()}:{slug}"


def create_document_node(
    document_id: str,
    title: str,
    source: str | None = None,
    case_id: str | None = None,
):
    """Create or reuse a Document node linked to its stable id.

    When ``case`` is supplied, a Case node is created or reused and the
    document is attached to it with PART_OF (re-ingestion is idempotent).
    """

    if not document_id.strip():
        raise ValueError("document_id cannot be empty.")

    if not title.strip():
        raise ValueError("Document title cannot be empty.")

    query = """
    MERGE (document:Document {document_id: $document_id})
    SET document.title = $title,
        document.source = $source
    WITH document
    FOREACH (_ IN CASE WHEN $case_id IS NULL THEN [] ELSE [1] END |
        MERGE (case_node:Case {case_id: $case_id})
        ON CREATE SET case_node.name = $case_id
        MERGE (document)-[:PART_OF]->(case_node)
    )
    RETURN
        document.document_id AS document_id,
        document.title AS title,
        document.source AS source
    """

    with get_driver().session(database=NEO4J_DATABASE) as session:
        record = session.run(
            query,
            document_id=document_id.strip(),
            title=title.strip(),
            source=source or None,
            case_id=case_id,
        ).single()

        return dict(record)


def link_document_entity(
    document_id: str,
    entity_id: str,
    relationship_type: str = "MENTIONS",
):
    """Create or reuse a relationship from a Document to an Entity."""

    if relationship_type not in {"MENTIONS", "CONCERNS"}:
        raise ValueError(
            f"Unsupported document relationship type: "
            f"{relationship_type}"
        )

    query = f"""
    MATCH (document:Document {{document_id: $document_id}})
    MATCH (entity:Entity {{entity_id: $entity_id}})
    MERGE (document)-[:{relationship_type}]->(entity)
    RETURN document.document_id AS document_id,
           $relationship_type AS relationship,
           entity.entity_id AS target
    """

    with get_driver().session(database=NEO4J_DATABASE) as session:
        record = session.run(
            query,
            document_id=document_id,
            entity_id=entity_id,
            relationship_type=relationship_type,
        ).single()

        if record is None:
            raise ValueError(
                "Document or entity was not found in the graph."
            )

        return dict(record)


def _spacy_extraction(
    content: str,
    *,
    document_id: str,
    pages: list[dict] | None = None,
):
    """Original spaCy NER extraction, kept as an explicit opt-in path.

    Only selected via ``extractor="spacy"`` or
    ``GRAPH_EXTRACTION_BACKEND=spacy``; it is never chosen silently.
    """

    from app.context_engine.llm_graph_transformer import (
        ExtractedNode,
        ExtractedRelationship,
        ExtractionResult,
        _relationship_evidence_is_supported,
        build_entity_id,
    )
    from app.masking.ner_service import get_nlp

    source_units = [
        page
        for page in (pages or [])
        if str(page.get("content") or "").strip()
    ]
    if not source_units:
        source_units = [{"content": content}]

    raw_sentences: list[dict] = []

    for unit in source_units:
        page_number = unit.get("page_number")
        source_unit_id = unit.get("source_unit_id")
        source_text = str(unit.get("content") or "")[:20000]
        source_ref = (
            f"{document_id}#page={page_number}#unit={source_unit_id}"
            if page_number is not None and source_unit_id
            else (
                f"{document_id}#page={page_number}"
                if page_number is not None
                else document_id
            )
        )
        doc = get_nlp()(source_text)

        # Keep each sentence and its source-unit metadata together.
        for sentence in doc.sents:
            candidates: list[tuple[str, str]] = []

            for entity in sentence.ents:
                if entity.label_ not in _ENTITY_LABELS:
                    continue

                name = normalize_entity_name(entity.text)
                if is_usable_entity_name(name):
                    candidates.append((name, entity.label_))

            raw_sentences.append(
                {
                    "text": str(getattr(sentence, "text", "") or ""),
                    "source_text": source_text,
                    "page_number": page_number,
                    "source_ref": source_ref,
                    "candidates": candidates,
                }
            )

    # Phase 2: resolve one type per name across the whole document, then
    # validate the resolved type. Contradictory labels for the same name
    # (for example "Nagpur" as both GPE and PERSON) become one node with
    # the better-supported type instead of two duplicate nodes.
    resolved = resolve_entity_types(
        [
            candidate
            for sentence in raw_sentences
            for candidate in sentence["candidates"]
        ]
    )

    accepted = {
        key: entry
        for key, entry in resolved.items()
        if is_plausible_entity(entry["name"], entry["entity_type"])
    }

    # Build entities and candidate pairs per sentence.
    nodes: dict[str, ExtractedNode] = {}
    sentence_entities: list[tuple[dict, list[str]]] = []

    for sentence in raw_sentences:
        sentence_ids: list[str] = []

        for name, _label in sentence["candidates"]:
            entry = accepted.get(entity_name_key(name))

            if entry is None:
                continue

            entity_id = build_entity_id(
                entry["entity_type"],
                entry["name"],
            )

            node = nodes.get(entity_id)
            if node is None:
                node = ExtractedNode(
                    entity_id=entity_id,
                    name=entry["name"],
                    entity_type=entry["entity_type"],
                )
                nodes[entity_id] = node

            if sentence["page_number"] is not None:
                node.page_numbers.append(sentence["page_number"])
            if sentence["source_ref"] not in node.source_refs:
                node.source_refs.append(sentence["source_ref"])

            if entity_id not in sentence_ids:
                sentence_ids.append(entity_id)

        sentence_entities.append((sentence, sentence_ids))

    # Co-occurrence alone is not a relationship. Only create a generic edge
    # when the sentence contains both entities and an explicit relation cue.
    relationships: dict[tuple[str, str, str], ExtractedRelationship] = {}
    dropped = []

    for sentence, sentence_ids in sentence_entities:
        for index, source_id in enumerate(sentence_ids):
            for target_id in sentence_ids[index + 1:]:
                if source_id == target_id:
                    continue

                source_node = nodes[source_id]
                target_node = nodes[target_id]
                evidence = sentence["text"]
                if not _relationship_evidence_is_supported(
                    "RELATES_TO",
                    evidence,
                    source_node.name,
                    target_node.name,
                    sentence["source_text"],
                ):
                    dropped.append(
                        {
                            "name": (
                                f"{source_node.name}->{target_node.name}"
                            ),
                            "reason": "unsupported_relationship_evidence",
                        }
                    )
                    continue

                key = (source_id, target_id, "RELATES_TO")
                relationship = relationships.get(key)
                if relationship is None:
                    relationship = ExtractedRelationship(
                        source_id=source_id,
                        target_id=target_id,
                        relationship_type="RELATES_TO",
                    )
                    relationships[key] = relationship
                if sentence["page_number"] is not None:
                    relationship.page_numbers.append(
                        sentence["page_number"]
                    )
                if sentence["source_ref"] not in relationship.source_refs:
                    relationship.source_refs.append(sentence["source_ref"])

    for node in nodes.values():
        node.document_id = document_id
        node.page_numbers = sorted(set(node.page_numbers))

    for relationship in relationships.values():
        relationship.document_id = document_id
        relationship.page_numbers = sorted(set(relationship.page_numbers))

    return ExtractionResult(
        nodes=list(nodes.values()),
        relationships=list(relationships.values()),
        dropped=dropped,
        backend="spacy",
    )


def _resolve_extraction_backend(extractor: str | None) -> str:
    """Pick the extraction backend, reporting rather than guessing."""

    from app.context_engine import llm_graph_transformer

    if extractor:
        selected = extractor.strip().casefold()
    elif llm_graph_transformer.llm_extraction_enabled():
        selected = "llm"
    else:
        selected = "spacy"

    if selected not in {"llm", "spacy"}:
        raise ValueError(f"Unsupported graph extractor: {extractor}")

    return selected


def ingest_document_graph(
    document_id: str,
    title: str,
    content: str,
    case: str | None = None,
    pages: list[dict] | None = None,
    extractor: str | None = None,
    transformer=None,
    on_stage: Callable[[str, str, float, str | None], None] | None = None,
) -> dict:
    """Ingest a document and its entities into the Neo4j knowledge graph.

    Entities are extracted with LangChain's LLMGraphTransformer by default
    (see ``llm_graph_transformer``), validated and deduplicated before any
    write, then merged into Neo4j together with their page provenance.

    ``extractor="spacy"`` selects the legacy NER path explicitly; it is
    never used silently when the LLM path is the configured one.
    """
    if not content or not content.strip():
        raise ValueError("Document content is empty; nothing to ingest.")

    # Extract + validate before opening a write transaction. A model or
    # validation failure must not leave a partial Document node behind.
    backend = _resolve_extraction_backend(extractor)

    extraction_started = time.perf_counter()
    if on_stage is not None:
        on_stage("graph_extraction", "running", 0.0, None)
    try:
        if backend == "spacy":
            extraction = _spacy_extraction(
                content,
                document_id=document_id,
                pages=pages,
            )
        else:
            from app.context_engine.llm_graph_transformer import extract_graph

            extraction = extract_graph(
                content,
                document_id=document_id,
                transformer=transformer,
                pages=pages,
            )
    except Exception as error:
        if on_stage is not None:
            on_stage(
                "graph_extraction",
                "failed",
                (time.perf_counter() - extraction_started) * 1000,
                type(error).__name__,
            )
        raise
    if on_stage is not None:
        on_stage(
            "graph_extraction",
            "completed",
            (time.perf_counter() - extraction_started) * 1000,
            None,
        )

    # Limit the number of entities/relationships stored for one document.
    accepted_nodes = extraction.nodes[:MAX_ENTITIES_PER_DOCUMENT]
    accepted_ids = {node.entity_id for node in accepted_nodes}

    accepted_relationships = [
        relationship
        for relationship in extraction.relationships
        if relationship.source_id in accepted_ids
        and relationship.target_id in accepted_ids
    ][:MAX_RELATIONS_PER_DOCUMENT]

    # 3. Create/reuse entity nodes, linking each to the document.
    # The document link carries page and source-reference provenance.
    entity_upsert = """
    MERGE (entity:Entity {entity_id: $entity_id})
    ON CREATE SET
        entity.name = $name,
        entity.canonical_name = $name,
        entity.aliases = [],
        entity.entity_type = $entity_type
    ON MATCH SET
        entity.canonical_name = coalesce(
            entity.canonical_name,
            entity.name,
            $name
        ),
        entity.aliases = CASE
            WHEN $name = coalesce(entity.canonical_name, entity.name)
              OR $name IN coalesce(entity.aliases, [])
            THEN coalesce(entity.aliases, [])
            ELSE coalesce(entity.aliases, []) + [$name]
        END
    WITH entity
    WHERE entity.entity_type = $entity_type
    MATCH (document:Document {document_id: $document_id})
    MERGE (document)-[mention:MENTIONS]->(entity)
    SET mention.document_id = $document_id,
        mention.page_numbers = $page_numbers,
        mention.source_refs = $source_refs
    RETURN entity.entity_id AS entity_id
    """

    relationship_upsert = """
    MATCH (source:Entity {entity_id: $source_id})
    MATCH (target:Entity {entity_id: $target_id})
    {relationship_merge}
    SET relationship.document_ids = reduce(
                unique_values = [],
                value IN coalesce(relationship.document_ids, []) + [$document_id] |
                CASE
                    WHEN value IN unique_values THEN unique_values
                    ELSE unique_values + [value]
                END
        ),
        relationship.source_refs = reduce(
                unique_values = [],
                value IN coalesce(relationship.source_refs, []) + $source_refs |
                CASE
                    WHEN value IN unique_values THEN unique_values
                    ELSE unique_values + [value]
                END
        ),
        relationship.page_numbers = reduce(
                unique_values = [],
                value IN coalesce(relationship.page_numbers, []) + $page_numbers |
                CASE
                    WHEN value IN unique_values THEN unique_values
                    ELSE unique_values + [value]
                END
        )
    RETURN source.entity_id AS source_id
    """

    document_upsert = """
    MERGE (document:Document {document_id: $document_id})
    SET document.title = $title,
        document.source = $source
    WITH document
    FOREACH (_ IN CASE WHEN $case_id IS NULL THEN [] ELSE [1] END |
        MERGE (case_node:Case {case_id: $case_id})
        ON CREATE SET case_node.name = $case_id
        MERGE (document)-[:PART_OF]->(case_node)
    )
    RETURN document.document_id AS document_id
    """
    stale_mentions = """
    MATCH (document:Document {document_id: $document_id})
          -[mention:MENTIONS]->(entity:Entity)
    WITH collect(mention) AS mentions,
         collect(entity.entity_id) AS entity_ids
    FOREACH (mention IN mentions | DELETE mention)
    RETURN entity_ids
    """
    stale_orphan_entities = """
    UNWIND $entity_ids AS entity_id
    MATCH (entity:Entity {entity_id: entity_id})
    OPTIONAL MATCH (entity)-[relationship]-()
    WITH entity, count(relationship) AS relationship_count
    WHERE relationship_count = 0
    DELETE entity
    """
    stale_relationships = """
    MATCH (:Entity)-[relationship]->(:Entity)
    WHERE $document_id IN coalesce(relationship.document_ids, [])
    WITH relationship,
         [value IN coalesce(relationship.document_ids, [])
          WHERE value <> $document_id] AS remaining_document_ids,
         [ref IN coalesce(relationship.source_refs, [])
          WHERE ref <> $document_id
            AND NOT ref STARTS WITH $document_id + "#page="]
             AS remaining_source_refs
    WITH relationship, remaining_document_ids, remaining_source_refs,
         [ref IN remaining_source_refs
          WHERE ref CONTAINS "#page="
          | toInteger(split(split(ref, "#page=")[1], "#")[0])]
             AS remaining_page_numbers
    FOREACH (_ IN CASE WHEN size(remaining_document_ids) = 0
                       THEN [1] ELSE [] END |
        DELETE relationship
    )
    FOREACH (_ IN CASE WHEN size(remaining_document_ids) > 0
                       THEN [1] ELSE [] END |
        SET relationship.document_ids = remaining_document_ids,
            relationship.source_refs = remaining_source_refs,
            relationship.page_numbers = reduce(
                pages = [],
                page IN remaining_page_numbers |
                    CASE
                        WHEN page IN pages THEN pages
                        ELSE pages + [page]
                    END
            )
    )
    """

    def write_transaction(transaction):
        created_entities = 0
        created_relations = 0

        document_record = transaction.run(
            document_upsert,
            document_id=document_id,
            title=title,
            source="document-upload",
            case_id=case,
        ).single()
        if document_record is None:
            raise RuntimeError("The graph document could not be created.")

        # Refresh only edges owned by this document. Shared entity nodes
        # and relationships from other documents remain untouched.
        stale_mention_record = transaction.run(
            stale_mentions,
            document_id=document_id,
        ).single()
        stale_entity_ids = (
            stale_mention_record.get("entity_ids", [])
            if stale_mention_record is not None
            else []
        )
        transaction.run(stale_relationships, document_id=document_id)
        transaction.run(
            stale_orphan_entities,
            entity_ids=[
                entity_id
                for entity_id in stale_entity_ids
                if entity_id not in accepted_ids
            ],
        )

        for node in accepted_nodes:
            record = transaction.run(
                entity_upsert,
                entity_id=node.entity_id,
                name=node.name,
                entity_type=node.entity_type,
                document_id=document_id,
                page_numbers=list(node.page_numbers),
                source_refs=list(node.source_refs),
            ).single()
            if record is None:
                raise RuntimeError(
                    "Graph entity type conflicts with its stable entity id."
                )
            created_entities += 1

        for relationship in accepted_relationships:
            if relationship.relationship_type not in ALLOWED_RELATIONSHIPS:
                raise ValueError(
                    "Unsupported relationship type in validated graph."
                )

            merge_query = (
                "MERGE (source)-[relationship:"
                f"{relationship.relationship_type}]->(target)"
            )
            transaction.run(
                relationship_upsert.replace(
                    "{relationship_merge}", merge_query
                ),
                source_id=relationship.source_id,
                target_id=relationship.target_id,
                document_id=document_id,
                page_numbers=list(relationship.page_numbers),
                source_refs=list(relationship.source_refs),
            )
            created_relations += 1

        return created_entities, created_relations

    persistence_started = time.perf_counter()
    if on_stage is not None:
        on_stage("neo4j_persistence", "running", 0.0, None)
    try:
        with get_driver().session(database=NEO4J_DATABASE) as session:
            created_entities, created_relations = session.execute_write(
                write_transaction
            )
    except Exception as error:
        if on_stage is not None:
            on_stage(
                "neo4j_persistence",
                "failed",
                (time.perf_counter() - persistence_started) * 1000,
                type(error).__name__,
            )
        raise
    if on_stage is not None:
        on_stage(
            "neo4j_persistence",
            "completed",
            (time.perf_counter() - persistence_started) * 1000,
            None,
        )

    return {
        "document_id": document_id,
        "entities_ingested": created_entities,
        "entity_relations_ingested": created_relations,
        "case": case,
        "extraction_backend": extraction.backend,
        "extraction_provider": extraction.provider or extraction.backend,
        "entities_dropped": len(extraction.dropped),
    }


def find_document_graph(document_id: str, limit: int = 200) -> dict:
    """Return a document's graph: its entities and the edges between them.

    Only relationships whose source and target are both mentioned by
    the same document are returned, so the view never pulls in another
    document's subgraph. Read-only: MATCH/RETURN only.
    """

    if not document_id.strip():
        raise ValueError("document_id cannot be empty.")

    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500.")

    entities = find_document_entities(document_id)

    edge_query = """
    MATCH (document:Document {document_id: $document_id})
          -[:MENTIONS]->(source:Entity)
    MATCH (source)-[relationship]->(target:Entity)
    MATCH (document)-[:MENTIONS]->(target)
    WHERE $document_id IN coalesce(relationship.document_ids, [])
    RETURN
        source.entity_id AS source_id,
        source.name AS source,
        type(relationship) AS relationship,
        target.entity_id AS target_id,
        target.name AS target
    ORDER BY source.name, target.name
    LIMIT $limit
    """

    with get_driver().session(database=NEO4J_DATABASE) as session:
        records = session.run(
            edge_query,
            document_id=document_id.strip(),
            limit=limit,
        )

        edges = [dict(record) for record in records]

    return {
        "document_id": document_id.strip(),
        "entities": entities,
        "edges": edges,
    }


def find_document_entities(document_id: str) -> list[dict]:
    """Return the entities one document MENTIONS in the graph."""

    if not document_id.strip():
        raise ValueError("document_id cannot be empty.")

    query = """
    MATCH (document:Document {document_id: $document_id})
          -[:MENTIONS]->(entity:Entity)
    RETURN
        entity.entity_id AS entity_id,
        entity.name AS name,
        entity.entity_type AS entity_type
    ORDER BY entity.name
    """

    with get_driver().session(database=NEO4J_DATABASE) as session:
        records = session.run(query, document_id=document_id.strip())

        return [dict(record) for record in records]


def list_documents(limit: int = 25) -> list[dict]:
    """List ingested Document nodes, newest first."""

    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100.")

    query = """
    MATCH (document:Document)
    RETURN
        document.document_id AS document_id,
        document.title AS title,
        document.source AS source
    ORDER BY document.document_id
    LIMIT $limit
    """

    with get_driver().session(database=NEO4J_DATABASE) as session:
        records = session.run(query, limit=limit)

        return [dict(record) for record in records]


def close_graph_connection():
    """Close the Neo4j driver when the application shuts down."""

    global driver

    if driver is not None:
        driver.close()
        driver = None
