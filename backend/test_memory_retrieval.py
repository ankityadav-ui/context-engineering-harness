
from app.context_engine.database import SessionLocal
from app.context_engine.memory_service import (
    create_memory,
    retrieve_relevant_memories,
)


def main():
    # Use a dedicated owner so test data stays isolated.
    test_user_id = "memory-retrieval-test"

    # Open a PostgreSQL session.
    with SessionLocal() as db:
        # Store a memory directly relevant to the query.
        relevant_memory = create_memory(
            db=db,
            user_id=test_user_id,
            memory_type="fact",
            content=(
                "PostgreSQL stores conversation history "
                "for the Context Engineering project."
            ),
            importance=4,
        )

        # Store a memory with little relevance to the query.
        unrelated_memory = create_memory(
            db=db,
            user_id=test_user_id,
            memory_type="note",
            content="The favorite weekend activity is gardening.",
            importance=2,
        )

        # Retrieve memories relevant to the current question.
        results = retrieve_relevant_memories(
            db=db,
            user_id=test_user_id,
            query="How do I store conversation history in PostgreSQL?",
            limit=5,
        )

        # Print scores and matched keywords for inspection.
        print("Retrieved memories:\n")

        for memory in results:
            print("Content:", memory["content"])
            print("Score:", memory["score"])
            print("Matched:", memory["matched_keywords"])
            print()

        # Confirm that both test records were considered.
        result_ids = [memory["id"] for memory in results]

        assert relevant_memory.id in result_ids
        assert unrelated_memory.id in result_ids

        # Confirm that the relevant memory ranks higher.
        relevant_position = result_ids.index(relevant_memory.id)
        unrelated_position = result_ids.index(unrelated_memory.id)

        assert relevant_position < unrelated_position

        print("PASS: Memory retrieval test succeeded.")


if __name__ == "__main__":
    # Execute the test when this file is run directly.
    main()
