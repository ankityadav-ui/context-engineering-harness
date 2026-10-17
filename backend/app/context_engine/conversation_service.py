
from uuid import uuid4

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.context_engine.models import Conversation, Message


def create_conversation(
    db: Session,
    title: str | None = None,
) -> Conversation:
    # Generate a unique ID for the new conversation.
    conversation_id = str(uuid4())

    # Create the conversation database record.
    conversation = Conversation(
        id=conversation_id,
        title=title,
    )

    # Add the record to the current database transaction.
    db.add(conversation)

    # Save the conversation to PostgreSQL.
    db.commit()

    # Refresh the object to retrieve database-generated values.
    db.refresh(conversation)

    return conversation


def get_conversation(
    db: Session,
    conversation_id: str,
) -> Conversation | None:
    """Return one conversation or None when it does not exist."""

    if not conversation_id or not conversation_id.strip():
        raise ValueError("conversation_id cannot be empty.")

    return db.get(Conversation, conversation_id.strip())


def list_conversations(
    db: Session,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """List conversations, newest first, with their message counts."""

    # Validate the requested page before querying the database.
    if not 1 <= limit <= 200:
        raise ValueError("limit must be between 1 and 200.")

    if offset < 0:
        raise ValueError("offset cannot be negative.")

    conversations = (
        db.query(Conversation)
        .order_by(
            Conversation.created_at.desc(),
            Conversation.id.desc(),
        )
        .offset(offset)
        .limit(limit)
        .all()
    )

    if not conversations:
        return []

    # Count messages once for every conversation on this page.
    conversation_ids = [
        conversation.id for conversation in conversations
    ]

    message_counts = dict(
        db.query(
            Message.conversation_id,
            func.count(Message.id),
        )
        .filter(Message.conversation_id.in_(conversation_ids))
        .group_by(Message.conversation_id)
        .all()
    )

    return [
        {
            "id": conversation.id,
            "title": conversation.title,
            "created_at": (
                conversation.created_at.isoformat()
                if conversation.created_at
                else None
            ),
            "message_count": message_counts.get(
                conversation.id,
                0,
            ),
        }
        for conversation in conversations
    ]


def save_message(
    db: Session,
    conversation_id: str,
    role: str,
    content: str,
) -> Message:
    # Allow only the roles supported by our chat application.
    if role not in {"user", "assistant"}:
        raise ValueError(
            "Role must be 'user' or 'assistant'."
        )

    # Reject messages that contain no meaningful content.
    if not content.strip():
        raise ValueError("Message content cannot be empty.")

    # Verify that the conversation exists before saving.
    conversation = db.get(Conversation, conversation_id)

    if conversation is None:
        raise ValueError("Conversation not found.")

    # Create a message linked to the selected conversation.
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
    )

    # Add the message to the database transaction.
    db.add(message)

    # Persist the message in PostgreSQL.
    db.commit()

    # Retrieve generated fields such as the message ID.
    db.refresh(message)

    return message


def get_last_n_messages(
    db: Session,
    conversation_id: str,
    n: int = 6,
) -> list[dict]:
    # Ensure the requested number of messages is valid.
    if n < 1:
        raise ValueError("n must be at least 1.")

    # Fetch the newest N messages first.
    # Descending IDs mean the most recently inserted messages
    # are selected before older messages.
    messages = (
        db.query(Message)
        .filter(
            Message.conversation_id == conversation_id
        )
        .order_by(Message.id.desc())
        .limit(n)
        .all()
    )

    # Reverse the selected messages into chronological order.
    messages.reverse()

    # Return plain dictionaries for the Context Engine.
    return [
        {
            "id": message.id,
            "conversation_id": message.conversation_id,
            "role": message.role,
            "content": message.content,
            "created_at": message.created_at.isoformat(),
        }
        for message in messages
    ]


def get_conversation_message_window(
    db: Session,
    conversation_id: str,
    n: int = 6,
) -> dict:
    """
    Separate a conversation into older messages
    and the latest N recent messages.
    """

    # Reject invalid window sizes.
    if n < 1:
        raise ValueError("n must be at least 1.")

    # Verify that the requested conversation exists.
    conversation = db.get(
        Conversation,
        conversation_id,
    )

    if conversation is None:
        raise ValueError("Conversation not found.")

    # Load messages in chronological order.
    all_messages = (
        db.query(Message)
        .filter(
            Message.conversation_id == conversation_id
        )
        .order_by(Message.id.asc())
        .all()
    )

    # Split at the boundary between older and recent messages.
    split_index = max(0, len(all_messages) - n)

    older_messages = all_messages[:split_index]
    recent_messages = all_messages[split_index:]

    # Convert database objects into simple dictionaries.
    def serialize_messages(messages):
        return [
            {
                "id": message.id,
                "conversation_id": message.conversation_id,
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at.isoformat(),
            }
            for message in messages
        ]

    # Return both groups without changing stored messages.
    return {
        "conversation_id": conversation_id,
        "total_messages": len(all_messages),
        "recent_count": len(recent_messages),
        "older_count": len(older_messages),
        "older_messages": serialize_messages(
            older_messages
        ),
        "recent_messages": serialize_messages(
            recent_messages
        ),
    }
