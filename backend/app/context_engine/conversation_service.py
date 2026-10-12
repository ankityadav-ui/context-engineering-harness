
from uuid import uuid4

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
