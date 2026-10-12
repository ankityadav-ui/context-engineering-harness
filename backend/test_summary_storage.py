
from app.context_engine.database import SessionLocal
from app.context_engine.conversation_service import (
    create_conversation,
    save_message,
)
from app.context_engine.summary_service import (
    get_or_update_summary,
)


def main():
    # Open a PostgreSQL session for the test.
    with SessionLocal() as db:
        # Create an isolated test conversation.
        conversation = create_conversation(
            db,
            title="Summary Storage Test",
        )

        # Add eight messages so older messages will exist.
        for number in range(1, 9):
            role = (
                "user"
                if number % 2 == 1
                else "assistant"
            )

            save_message(
                db=db,
                conversation_id=conversation.id,
                role=role,
                content=(
                    f"Test conversation message {number}. "
                    "The project uses FastAPI and PostgreSQL."
                ),
            )

        # Generate and store a summary of messages outside N=2.
        query = "How does this project store conversation history?"

        first_result = get_or_update_summary(
            db=db,
            conversation_id=conversation.id,
            current_query=query,
            n=2,
        )

        print("First call:", first_result["status"])
        print("Summary:", first_result["summary"])

        # Repeat the exact request to test summary reuse.
        second_result = get_or_update_summary(
            db=db,
            conversation_id=conversation.id,
            current_query=query,
            n=2,
        )

        print("\nSecond call:", second_result["status"])

        # Verify that the stored summary was reused.
        assert first_result["summary"] == second_result["summary"]
        assert second_result["status"] == "reused"

        print("\nPASS: Summary storage and reuse succeeded.")


if __name__ == "__main__":
    # Run the integration test directly.
    main()
