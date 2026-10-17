
import _bootstrap  # noqa: F401 - adds backend/ to sys.path
from app.context_engine.database import SessionLocal
from app.context_engine.conversation_service import (
    create_conversation,
    save_message,
    get_conversation_message_window,
)


def main():
    # Manual integration script: require explicit opt-in before writing.
    from manual_guard import require_live_opt_in

    require_live_opt_in(
        "write conversation and message rows in PostgreSQL"
    )

    # Open a database session for this test.
    with SessionLocal() as db:
        # Create an isolated test conversation.
        conversation = create_conversation(
            db,
            title="Message Window Test",
        )

        # Save ten messages in chronological order.
        for number in range(1, 11):
            role = (
                "user"
                if number % 2 == 1
                else "assistant"
            )

            save_message(
                db=db,
                conversation_id=conversation.id,
                role=role,
                content=f"Test message {number}",
            )

        # Split the conversation using N = 4.
        window = get_conversation_message_window(
            db=db,
            conversation_id=conversation.id,
            n=4,
        )

        # Print a summary of the two groups.
        print("Total:", window["total_messages"])
        print("Older:", window["older_count"])
        print("Recent:", window["recent_count"])

        # Show the older messages.
        print("\nOlder messages:")
        for message in window["older_messages"]:
            print(message["content"])

        # Show the recent messages.
        print("\nRecent messages:")
        for message in window["recent_messages"]:
            print(message["content"])

        # Confirm that the split is correct.
        assert window["total_messages"] == 10
        assert window["older_count"] == 6
        assert window["recent_count"] == 4

        assert [
            message["content"]
            for message in window["older_messages"]
        ] == [
            f"Test message {number}"
            for number in range(1, 7)
        ]

        assert [
            message["content"]
            for message in window["recent_messages"]
        ] == [
            f"Test message {number}"
            for number in range(7, 11)
        ]

        print("\nPASS: Message window test succeeded.")


if __name__ == "__main__":
    # Run the test when this script is executed directly.
    main()
