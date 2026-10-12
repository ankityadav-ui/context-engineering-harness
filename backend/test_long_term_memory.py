
from app.context_engine.database import SessionLocal
from app.context_engine.memory_service import (
    create_memory,
    list_memories,
    delete_memory,
)


def main():
    # Use a dedicated test owner to isolate test data.
    test_user_id = "ltm-test-user"

    # Open a PostgreSQL session.
    with SessionLocal() as db:
        # Save a fact.
        fact = create_memory(
            db=db,
            user_id=test_user_id,
            memory_type="fact",
            content="The project backend uses FastAPI.",
            importance=4,
        )

        # Save a preference.
        preference = create_memory(
            db=db,
            user_id=test_user_id,
            memory_type="preference",
            content="Explain technical concepts step by step.",
            importance=5,
        )

        # Save a goal.
        goal = create_memory(
            db=db,
            user_id=test_user_id,
            memory_type="goal",
            content="Complete the Context Engineering Harness.",
            importance=5,
        )

        # Retrieve this test user's memories.
        memories = list_memories(
            db=db,
            user_id=test_user_id,
        )

        # Print the stored memories.
        print("Stored memories:")

        for memory in memories:
            print(
                f"{memory['memory_type']} | "
                f"Importance: {memory['importance']} | "
                f"{memory['content']}"
            )

        # Verify that all three memory categories were saved.
        assert len(memories) >= 3

        saved_ids = {
            memory["id"]
            for memory in memories
        }

        assert fact.id in saved_ids
        assert preference.id in saved_ids
        assert goal.id in saved_ids

        # Verify that deleting a memory works.
        deleted = delete_memory(
            db=db,
            user_id=test_user_id,
            memory_id=fact.id,
        )

        assert deleted is True

        print("\nPASS: Long-term memory CRUD test succeeded.")


if __name__ == "__main__":
    # Run this test when the script is executed directly.
    main()
