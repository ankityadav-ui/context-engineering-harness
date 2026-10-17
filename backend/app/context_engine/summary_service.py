
# Import the database model for stored conversation summaries.
from app.context_engine.models import ConversationSummary

# Import the conversation window service.
from app.context_engine.conversation_service import (
    get_conversation_message_window,
)

# Import the query-aware summarization function.
from app.context_engine.summarizer import (
    generate_query_aware_summary,
)


def get_or_update_summary(
    db,
    conversation_id: str,
    current_query: str,
    n: int = 6,
) -> str:
    """Retrieve or regenerate a summary of older conversation messages.

    Returns the summary text as a plain string. This matches how the
    Context Engine consumes the value in
    ``app.context_engine.engine._get_conversation_context`` (stored under
    the "summary" key and rendered by ``format_context``).
    """

    # Validate the current query.
    if not current_query or not current_query.strip():
        raise ValueError("The current query cannot be empty.")

    # Retrieve older messages and the most recent N messages.
    window = get_conversation_message_window(
        db,
        conversation_id,
        n=n,
    )

    # Get older messages that need summarization.
    older_messages = window.get("older_messages", [])

    # No older messages means no summary is required.
    if not older_messages:
        return ""

    # Determine the latest message covered by this summary.
    covered_through_message_id = older_messages[-1]["id"]

    # Look for an existing summary for this conversation.
    stored_summary = (
        db.query(ConversationSummary)
        .filter(
            ConversationSummary.conversation_id == conversation_id
        )
        .first()
    )

    # Reuse the saved summary only when its coverage and query match.
    if (
        stored_summary is not None
        and stored_summary.covered_through_message_id
        == covered_through_message_id
        and stored_summary.last_query == current_query.strip()
    ):
        return stored_summary.summary_text or ""

    # Generate a query-aware summary of the older messages.
    summary_text = generate_query_aware_summary(
        older_messages=older_messages,
        current_query=current_query.strip(),
    )

    # Reject an empty summary rather than saving invalid content.
    if not summary_text or not summary_text.strip():
        raise ValueError("The summarizer returned an empty summary.")

    # Create a new database record or update the existing one.
    if stored_summary is None:
        stored_summary = ConversationSummary(
            conversation_id=conversation_id,
            summary_text=summary_text.strip(),
            covered_through_message_id=covered_through_message_id,
            last_query=current_query.strip(),
        )

        db.add(stored_summary)
    else:
        stored_summary.summary_text = summary_text.strip()
        stored_summary.covered_through_message_id = (
            covered_through_message_id
        )
        stored_summary.last_query = current_query.strip()

    # Persist the summary and return its text.
    db.commit()
    db.refresh(stored_summary)

    return stored_summary.summary_text
