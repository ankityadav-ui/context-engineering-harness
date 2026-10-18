
# Import libraries for environment variables, paths, and Neo4j.
import os
from pathlib import Path
from dotenv import load_dotenv
import certifi
from neo4j import GraphDatabase, TrustCustomCAs

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


def list_entities(limit: int = 25):
    """Return entities from the graph, ordered by name."""

    # Keep a single request bounded.
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100.")

    query = """
    MATCH (entity:Entity)
    RETURN
        entity.entity_id AS entity_id,
        entity.name AS name,
        entity.entity_type AS entity_type
    ORDER BY entity.name
    LIMIT $limit
    """

    # Return matching entities as ordinary Python dictionaries.
    with get_driver().session(database=NEO4J_DATABASE) as session:
        records = session.run(query, limit=limit)

        return [dict(record) for record in records]


def find_entities(search_text: str):
    """Find entities whose names contain the supplied text."""

    # Reject empty searches.
    if not search_text or not search_text.strip():
        raise ValueError("Search text cannot be empty.")

    query = """
    MATCH (entity:Entity)
    WHERE toLower(entity.name) CONTAINS toLower($search_text)
    RETURN
        entity.entity_id AS entity_id,
        entity.name AS name,
        entity.entity_type AS entity_type
    ORDER BY entity.name
    LIMIT 20
    """

    # Return matching entities as ordinary Python dictionaries.
    with get_driver().session(database=NEO4J_DATABASE) as session:
        records = session.run(
            query,
            search_text=search_text.strip(),
        )

        return [dict(record) for record in records]


def get_entity_relationships(entity_id: str):
    """Return incoming and outgoing relationships for one entity."""

    query = """
    MATCH (entity:Entity {entity_id: $entity_id})

    OPTIONAL MATCH (entity)-[outgoing]->(target:Entity)
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


def ingest_document_graph(
    document_id: str,
    title: str,
    content: str,
    case: str | None = None,
) -> dict:
    """Ingest a document and its entities into the Neo4j knowledge graph.

    Supported entities are extracted with the shared spaCy NER loader,
    filtered with graph-specific quality checks, and linked to the document.
    Entity pairs found in the same sentence receive RELATES_TO edges.
    """
    if not content or not content.strip():
        raise ValueError("Document content is empty; nothing to ingest.")

    # 1. Create or reuse the document and optional case node.
    create_document_node(
        document_id=document_id,
        title=title,
        source="document-upload",
        case_id=case,
    )

    # 2. Extract entities with the shared spaCy NER loader. Detection is
    # shared with masking; only the validation below is graph-specific.
    from app.masking.ner_service import get_nlp

    doc = get_nlp()(content[:20000])

    # Phase 1: collect raw candidates, preserving sentence boundaries.
    raw_sentences: list[list[tuple[str, str]]] = []

    for sentence in doc.sents:
        candidates: list[tuple[str, str]] = []

        for entity in sentence.ents:
            # Keep only categories supported by the graph.
            if entity.label_ not in _ENTITY_LABELS:
                continue

            name = normalize_entity_name(entity.text)

            # Name-quality validation (placeholders, OCR debris, etc.).
            if not is_usable_entity_name(name):
                continue

            candidates.append((name, entity.label_))

        raw_sentences.append(candidates)

    # Phase 2: resolve one type per name across the whole document, then
    # validate the resolved type. Contradictory labels for the same name
    # (for example "Nagpur" as both GPE and PERSON) become one node with
    # the better-supported type instead of two duplicate nodes.
    resolved = resolve_entity_types(
        [
            candidate
            for candidates in raw_sentences
            for candidate in candidates
        ]
    )

    accepted = {
        key: entry
        for key, entry in resolved.items()
        if is_plausible_entity(entry["name"], entry["entity_type"])
    }

    # Phase 3: build per-sentence entity ids for relationship extraction.
    seen_entities: dict[str, str] = {}
    sentences_entities: list[list[str]] = []

    for candidates in raw_sentences:
        sentence_ids: list[str] = []

        for name, _label in candidates:
            entry = accepted.get(entity_name_key(name))

            if entry is None:
                continue

            entity_id = _entity_node_id(
                entry["entity_type"],
                entry["name"],
            )

            seen_entities.setdefault(entity_id, entry["name"])

            if entity_id not in sentence_ids:
                sentence_ids.append(entity_id)

        sentences_entities.append(sentence_ids)

    # Limit the number of entities stored for one document.
    capped_entities = dict(
        list(seen_entities.items())[:MAX_ENTITIES_PER_DOCUMENT]
    )

    # 3. Create/reuse entity nodes and link them to the document.
    entity_upsert = """
    MERGE (entity:Entity {entity_id: $entity_id})
    SET entity.name = $name,
        entity.entity_type = $entity_type
    WITH entity
    MATCH (document:Document {document_id: $document_id})
    MERGE (document)-[:MENTIONS]->(entity)
    RETURN entity.entity_id AS entity_id
    """

    with get_driver().session(database=NEO4J_DATABASE) as session:
        created_entities = 0

        for entity_id, name in capped_entities.items():
            entity_type = entity_id.split(":", 1)[0].upper()
            session.run(
                entity_upsert,
                entity_id=entity_id,
                name=name,
                entity_type=entity_type,
                document_id=document_id,
            )
            created_entities += 1

        # 4. Link entity pairs that occur in the same sentence.
        created_relations = 0

        for sentence_ids in sentences_entities:
            for index, source_id in enumerate(sentence_ids):
                for target_id in sentence_ids[index + 1:]:
                    # Only create relationships for retained entities.
                    if (
                        source_id not in capped_entities
                        or target_id not in capped_entities
                    ):
                        continue

                    session.run(
                        "MATCH (source:Entity {entity_id: $source_id}) "
                        "MATCH (target:Entity {entity_id: $target_id}) "
                        "MERGE (source)-[:RELATES_TO]->(target)",
                        source_id=source_id,
                        target_id=target_id,
                    )
                    created_relations += 1

                    if created_relations >= MAX_RELATIONS_PER_DOCUMENT:
                        break
                if created_relations >= MAX_RELATIONS_PER_DOCUMENT:
                    break
            if created_relations >= MAX_RELATIONS_PER_DOCUMENT:
                break

    return {
        "document_id": document_id,
        "entities_ingested": created_entities,
        "entity_relations_ingested": created_relations,
        "case": case,
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
