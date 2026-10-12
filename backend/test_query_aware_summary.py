
from app.context_engine.database import SessionLocal
from app.context_engine.conversation_service import (
    create_conversation,
    save_message,
    get_conversation_message_window,
)
from app.context_engine.summarizer import (
    generate_query_aware_summary,
)


def main():
    # Open a database session for the integration test.
    with SessionLocal() as db:
        # Create a conversation with known information.
        conversation = create_conversation(
            db,
            title="Query-Aware Summary Test",
        )

        # Store older messages containing project details.
        messages = [
            (
                "user",
                "I am building a Context Engineering Harness.",
            ),
            (
                "assistant",
                "The backend uses FastAPI and the frontend uses React.",
            ),
            (
                "user",
                "I use PostgreSQL for persistent storage.",
            ),
            (
                "assistant",
                "We will store conversation messages in PostgreSQL.",
            ),
            (
                "user",
                "I want to understand each component before moving on.",
            ),
            (
                "assistant",
                "We will build the project one block at a time.",
            ),
            (
                "user",
                "I want to add conversation memory.",
            ),
            (
                "assistant",
                "Recent messages can be retrieved from the database.",
            ),
        ]

        # Save every message in its original order.
        for role, content in messages:
            save_message(
                db=db,
                conversation_id=conversation.id,
                role=role,
                content=content,
            )

        # Keep the latest two messages raw for this test.
        window = get_conversation_message_window(
            db=db,
            conversation_id=conversation.id,
            n=2,
        )

        # Summarize only the older messages.
        summary = generate_query_aware_summary(
            older_messages=window["older_messages"],
            current_query=(
                "How should I store conversation history "
                "in my Context Engineering Harness?"
            ),
        )

        # Display the inputs and the generated summary.
        print("Older message count:", window["older_count"])
        print("\nGenerated summary:\n")
        print(summary)

        # Ensure the model returned usable text.
        assert isinstance(summary, str)
        assert summary.strip()

        print("\nPASS: Query-aware summary test succeeded.")


if __name__ == "__main__":
    # Execute the test when this file is run directly.
    main()
