from unittest.mock import patch

import _bootstrap  # noqa: F401 - adds backend/ to sys.path
from app.context_engine.database import SessionLocal
from app.context_engine.conversation_service import (
    create_conversation,
    save_message,
)
from app.context_engine.summary_service import (
    get_or_update_summary,
)


def main():
    # Manual integration script: require explicit opt-in before writing.
    from manual_guard import require_live_opt_in

    require_live_opt_in(
        "write conversation, message, and summary rows in PostgreSQL"
    )

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

        # Patch the summarizer so the test does not call the LLM and so
        # we can observe whether a stored summary is reused.
        with patch(
            "app.context_engine.summary_service"
            ".generate_query_aware_summary",
            return_value="Stored summary text.",
        ) as mock_summary:
            first_result = get_or_update_summary(
                db=db,
                conversation_id=conversation.id,
                current_query=query,
                n=2,
            )

            # Repeat the exact request to test summary reuse.
            second_result = get_or_update_summary(
                db=db,
                conversation_id=conversation.id,
                current_query=query,
                n=2,
            )

        print("First call:", first_result)
        print("Second call:", second_result)

        # The service returns the summary text as a plain string.
        assert isinstance(first_result, str)
        assert first_result == "Stored summary text."

        # Both calls return the same stored summary.
        assert second_result == first_result

        # The summarizer ran only once; the second call reused the
        # stored summary instead of regenerating it.
        assert mock_summary.call_count == 1

        print("\nPASS: Summary storage and reuse succeeded.")


if __name__ == "__main__":
    # Run the integration test directly.
    main()
