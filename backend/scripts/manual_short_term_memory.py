
import _bootstrap  # noqa: F401 - adds backend/ to sys.path
from app.context_engine.database import SessionLocal
from app.context_engine.conversation_service import (
    create_conversation,
    save_message,
    get_last_n_messages,
)


def main():
    # Manual integration script: require explicit opt-in before writing.
    from manual_guard import require_live_opt_in

    require_live_opt_in(
        "write conversation and message rows in PostgreSQL"
    )

    # Open a database session for this test.
    with SessionLocal() as db:
        # Create a fresh conversation for testing.
        conversation = create_conversation(
            db,
            title="Short-Term Memory Test",
        )

        # Insert ten messages into the conversation.
        for number in range(1, 11):
            # Alternate between user and assistant messages.
            role = (
                "user"
                if number % 2 == 1
                else "assistant"
            )

            # Save each numbered message to PostgreSQL.
            save_message(
                db=db,
                conversation_id=conversation.id,
                role=role,
                content=f"This is test message {number}.",
            )

        # Retrieve only the latest four messages.
        recent_messages = get_last_n_messages(
            db=db,
            conversation_id=conversation.id,
            n=4,
        )

        # Display the conversation ID for later testing.
        print("Conversation ID:", conversation.id)

        # Display the messages returned by STM.
        print("\nRetrieved recent messages:")

        for message in recent_messages:
            print(
                f"{message['id']} | "
                f"{message['role']} | "
                f"{message['content']}"
            )

        # Verify that exactly messages 7 through 10 were returned.
        expected = [
            "This is test message 7.",
            "This is test message 8.",
            "This is test message 9.",
            "This is test message 10.",
        ]

        actual = [
            message["content"]
            for message in recent_messages
        ]

        assert actual == expected, (
            f"Unexpected messages returned: {actual}"
        )

        print("\nPASS: Short-term memory test succeeded.")


if __name__ == "__main__":
    # Run the test only when this file is executed directly.
    main()
