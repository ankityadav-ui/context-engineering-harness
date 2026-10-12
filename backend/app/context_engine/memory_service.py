
from sqlalchemy.orm import Session

from app.context_engine.models import Memory


# Restrict memory records to the five supported categories.
VALID_MEMORY_TYPES = {
    "fact",
    "preference",
    "goal",
    "context",
    "note",
}


def create_memory(
    db: Session,
    user_id: str,
    memory_type: str,
    content: str,
    importance: int = 3,
    conversation_id: str | None = None,
) -> Memory:
    # Ensure that the memory has a valid owner.
    if not user_id.strip():
        raise ValueError("user_id cannot be empty.")

    # Normalize and validate the memory category.
    memory_type = memory_type.strip().lower()

    if memory_type not in VALID_MEMORY_TYPES:
        raise ValueError("Unsupported memory type.")

    # Reject empty information.
    if not content.strip():
        raise ValueError("Memory content cannot be empty.")

    # Keep importance within the allowed range.
    if not 1 <= importance <= 5:
        raise ValueError("Importance must be between 1 and 5.")

    # Verify an optional source conversation exists.
    if conversation_id is not None:
        from app.context_engine.models import Conversation

        conversation = db.get(
            Conversation,
            conversation_id,
        )

        if conversation is None:
            raise ValueError("Conversation not found.")

    # Build the memory record.
    memory = Memory(
        user_id=user_id.strip(),
        memory_type=memory_type,
        content=content.strip(),
        importance=importance,
        conversation_id=conversation_id,
    )

    # Persist the new memory in PostgreSQL.
    db.add(memory)
    db.commit()
    db.refresh(memory)

    return memory


def list_memories(
    db: Session,
    user_id: str,
    limit: int = 100,
) -> list[dict]:
    # Validate the requested number of results.
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500.")

    # Retrieve only memories belonging to this user.
    memories = (
        db.query(Memory)
        .filter(Memory.user_id == user_id)
        .order_by(
            Memory.importance.desc(),
            Memory.id.desc(),
        )
        .limit(limit)
        .all()
    )

    # Return simple dictionaries for other Context Engine modules.
    return [
        {
            "id": memory.id,
            "user_id": memory.user_id,
            "memory_type": memory.memory_type,
            "content": memory.content,
            "importance": memory.importance,
            "conversation_id": memory.conversation_id,
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
        }
        for memory in memories
    ]


def delete_memory(
    db: Session,
    user_id: str,
    memory_id: int,
) -> bool:
    # Match both the memory ID and owner to prevent
    # deleting another user's record.
    memory = (
        db.query(Memory)
        .filter(
            Memory.id == memory_id,
            Memory.user_id == user_id,
        )
        .first()
    )

    # Return False if the memory does not exist for this user.
    if memory is None:
        return False

    # Delete the selected memory and save the change.
    db.delete(memory)
    db.commit()

    return True
