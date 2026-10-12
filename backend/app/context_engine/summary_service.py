
from sqlalchemy.orm import Session

from app.context_engine.models import (
    ConversationSummary,
)
from app.context_engine.conversation_service import (
    get_conversation_message_window,
)
from app.context_engine.summarizer import (
    generate_query_aware_summary,
)


def get_or_update_summary(
    db: Session,
    conversation_id: str,
    current_query: str,
    n: int = 6,
) -> dict:
    """
    Retrieve a reusable summary or generate and persist
    a new summary for the older conversation messages.
    """

    # Reject empty questions.
    if not current_query.strip():
        raise ValueError("Current query cannot be empty.")

    # Split the conversation into older and recent messages.
    window = get_conversation_message_window(
        db=db,
        conversation_id=conversation_id,
        n=n,
    )

    older_messages = window["older_messages"]

    # Find the last older message covered by this summary.
    covered_through_id = (
        older_messages[-1]["id"]
        if older_messages
        else 0
    )

    # Find the existing summary for this conversation.
    stored_summary = (
        db.query(ConversationSummary)
        .filter(
            ConversationSummary.conversation_id
            == conversation_id
        )
        .first()
    )

    # If no older messages exist, there is nothing to summarize.
    if not older_messages:
        return {
            "conversation_id": conversation_id,
            "summary": "",
            "covered_through_message_id": 0,
            "status": "no_older_messages",
        }

    # Reuse the summary only when its coverage and query match.
    if (
        stored_summary is not None
        and stored_summary.covered_through_message_id
        == covered_through_id
        and stored_summary.last_query == current_query
    ):
        return {
            "conversation_id": conversation_id,
            "summary": stored_summary.summary_text,
            "covered_through_message_id": covered_through_id,
            "status": "reused",
        }

    # Generate a fresh query-aware summary.
    # This prioritizes accuracy when the question changes.
    summary_text = generate_query_aware_summary(
        older_messages=older_messages,
        current_query=current_query,
    )

    # Create a summary record if this is the first summary.
    if stored_summary is None:
        stored_summary = ConversationSummary(
            conversation_id=conversation_id,
            summary_text=summary_text,
            covered_through_message_id=covered_through_id,
            last_query=current_query,
        )

        db.add(stored_summary)

    else:
        # Update the existing summary instead of creating duplicates.
        stored_summary.summary_text = summary_text
        stored_summary.covered_through_message_id = (
            covered_through_id
        )
        stored_summary.last_query = current_query

    # Persist the generated or updated summary.
    db.commit()
    db.refresh(stored_summary)

    return {
        "conversation_id": conversation_id,
        "summary": stored_summary.summary_text,
        "covered_through_message_id": (
            stored_summary.covered_through_message_id
        ),
        "status": "created_or_updated",
    }
