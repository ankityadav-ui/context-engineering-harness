
# Import libraries for environment variables, paths, and Neo4j.
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


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
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )


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


def close_graph_connection():
    """Close the Neo4j driver when the application shuts down."""

    global driver

    if driver is not None:
        driver.close()
        driver = None
