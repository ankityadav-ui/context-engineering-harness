
from sqlalchemy.orm import Session

from app.context_engine.models import Memory

import re
from datetime import datetime, timezone


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


def retrieve_relevant_memories(
    db: Session,
    user_id: str,
    query: str,
    limit: int = 5,
) -> list[dict]:
    """
    Retrieve the most relevant long-term memories using
    keyword overlap, importance, and recency.
    """

    # Reject invalid input before searching the database.
    if not user_id.strip():
        raise ValueError("user_id cannot be empty.")

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50.")

    # Ignore common words that provide little retrieval value.
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were",
        "do", "does", "did", "how", "what", "when",
        "where", "why", "which", "who", "can", "could",
        "should", "would", "i", "me", "my", "we", "our",
        "in", "on", "at", "to", "for", "of", "and",
        "or", "with", "this", "that", "it", "be",
        "have", "has", "had", "use", "using",
    }

    # Convert text into lowercase keyword sets.
    def extract_keywords(text: str) -> set[str]:
        words = set(re.findall(r"\b[a-zA-Z0-9_+#.-]+\b", text.lower()))
        return words - stop_words

    # Extract the meaningful terms from the current question.
    query_keywords = extract_keywords(query)

    # Retrieve only this user's memories.
    memories = (
        db.query(Memory)
        .filter(Memory.user_id == user_id)
        .all()
    )

    # Store each memory together with its calculated score.
    scored_memories = []

    # Calculate the relevance score for each candidate.
    for memory in memories:
        memory_keywords = extract_keywords(memory.content)

        # Count the keywords shared with the current question.
        matched_keywords = query_keywords & memory_keywords
        keyword_score = len(matched_keywords) * 2

        # Add the memory's manually assigned importance.
        importance_score = memory.importance

        # Give recently updated memories a small bonus.
        updated_at = memory.updated_at

        # Normalize naive timestamps if the database returns one.
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(
                tzinfo=timezone.utc
            )

        age_days = (
            datetime.now(timezone.utc) - updated_at
        ).total_seconds() / 86400

        if age_days <= 7:
            recency_score = 1.0
        elif age_days <= 30:
            recency_score = 0.5
        else:
            recency_score = 0.0

        # Combine the three retrieval signals.
        total_score = (
            keyword_score
            + importance_score
            + recency_score
        )

        # Preserve the score and matched keywords for debugging.
        scored_memories.append({
            "id": memory.id,
            "user_id": memory.user_id,
            "memory_type": memory.memory_type,
            "content": memory.content,
            "importance": memory.importance,
            "conversation_id": memory.conversation_id,
            "score": total_score,
            "matched_keywords": sorted(matched_keywords),
            "updated_at": memory.updated_at.isoformat(),
        })

    # Sort by score first, then importance, then newest update.
    scored_memories.sort(
        key=lambda item: (
            item["score"],
            item["importance"],
            item["updated_at"],
        ),
        reverse=True,
    )

    # Return only the requested number of results.
    return scored_memories[:limit]
