
import _bootstrap  # noqa: F401 - adds backend/ to sys.path
from app.context_engine.graph_service import (
    verify_graph_connection,
    create_entity,
    create_relationship,
    find_entities,
    get_entity_relationships,
    close_graph_connection,
)


def main():
    # Manual integration script: require explicit opt-in before writing.
    from manual_guard import require_live_opt_in

    require_live_opt_in(
        "create test entities and relationships in Neo4j"
    )

    try:
        # Verify the database connection before testing.
        verify_graph_connection()
        print("PASS: Neo4j connection succeeded.")

        # Create three fictional entities for this test.
        document = create_entity(
            entity_id="test-doc-001",
            name="Test Document",
            entity_type="Document",
        )

        case = create_entity(
            entity_id="test-case-001",
            name="Test Case",
            entity_type="Case",
        )

        patient = create_entity(
            entity_id="test-patient-001",
            name="Test Patient",
            entity_type="Patient",
        )

        # Connect the document to the case.
        create_relationship(
            source_id=document["entity_id"],
            target_id=case["entity_id"],
            relationship_type="RELATES_TO",
        )

        # Connect the case to the patient.
        create_relationship(
            source_id=case["entity_id"],
            target_id=patient["entity_id"],
            relationship_type="CONCERNS",
        )

        # Search for an entity by its name.
        search_results = find_entities("Test Case")

        assert any(
            item["entity_id"] == "test-case-001"
            for item in search_results
        )

        # Retrieve the case's incoming and outgoing relationships.
        relationships = get_entity_relationships(
            "test-case-001"
        )

        print("\nRelationships:")
        for relationship in relationships:
            print(relationship)

        # The case should connect to both the document and patient.
        assert len(relationships) >= 2

        print("\nPASS: Knowledge graph test succeeded.")

    finally:
        # Close the driver even if the test fails.
        close_graph_connection()


if __name__ == "__main__":
    # Execute the test directly.
    main()
