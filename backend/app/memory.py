from datetime import datetime

from sqlalchemy import desc
from sqlalchemy.orm import Session

from .config import (
    MEMORY_DEFAULT_USER_ID,
    MEMORY_MAX_RETRIEVED,
    MEMORY_MAX_CONTEXT_CHARS,
    MEMORY_AUTO_EXTRACT_ENABLED,
    MEMORY_EXTRACT_MAX_PER_MESSAGE,
    MEMORY_EXTRACT_MIN_WORDS,
)
from .models import Memory


# ============================================================
# ALLOWED MEMORY TYPES
# ============================================================

ALLOWED_MEMORY_TYPES = {
    "fact",
    "preference",
    "context",
    "note",
    "goal",
}


# ============================================================
# CREATE MEMORY
# ============================================================


def create_memory(
    content: str,
    memory_type: str = "fact",
    user_id: str = MEMORY_DEFAULT_USER_ID,
    case_id: int | None = None,
    importance: float = 0.5,
    db: Session | None = None,
) -> Memory:
    if db is None:
        raise ValueError("Database session is required")

    memory = Memory(
        user_id=user_id,
        content=content.strip(),
        memory_type=memory_type,
        case_id=case_id,
        importance=importance,
    )

    db.add(memory)
    db.commit()
    db.refresh(memory)
    return memory


# ============================================================
# LIST MEMORIES
# ============================================================


def list_memories(
    user_id: str = MEMORY_DEFAULT_USER_ID,
    case_id: int | None = None,
    memory_type: str | None = None,
    db: Session | None = None,
) -> list[Memory]:
    if db is None:
        return []

    query = db.query(Memory).filter(Memory.user_id == user_id)

    if case_id is not None:
        query = query.filter(
            (Memory.case_id == None) | (Memory.case_id == case_id)
        )
    else:
        query = query.filter(Memory.case_id == None)

    if memory_type is not None:
        query = query.filter(Memory.memory_type == memory_type)

    return query.order_by(desc(Memory.importance), desc(Memory.updated_at)).all()


# ============================================================
# GET MEMORY
# ============================================================


def get_memory(
    memory_id: int,
    user_id: str = MEMORY_DEFAULT_USER_ID,
    db: Session | None = None,
) -> Memory | None:
    if db is None:
        return None
    return (
        db.query(Memory)
        .filter(Memory.id == memory_id, Memory.user_id == user_id)
        .first()
    )


# ============================================================
# UPDATE MEMORY
# ============================================================


def update_memory(
    memory_id: int,
    content: str | None = None,
    memory_type: str | None = None,
    importance: float | None = None,
    user_id: str = MEMORY_DEFAULT_USER_ID,
    db: Session | None = None,
) -> Memory | None:
    if db is None:
        return None

    memory = get_memory(memory_id=memory_id, user_id=user_id, db=db)
    if memory is None:
        return None

    if content is not None:
        memory.content = content.strip()
    if memory_type is not None:
        memory.memory_type = memory_type
    if importance is not None:
        memory.importance = importance

    memory.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(memory)
    return memory


# ============================================================
# DELETE MEMORY
# ============================================================


def delete_memory(
    memory_id: int,
    user_id: str = MEMORY_DEFAULT_USER_ID,
    db: Session | None = None,
) -> bool:
    if db is None:
        return False

    memory = get_memory(memory_id=memory_id, user_id=user_id, db=db)
    if memory is None:
        return False

    db.delete(memory)
    db.commit()
    return True


# ============================================================
# RETRIEVE RELEVANT MEMORIES
# ============================================================

STOP_WORDS = {
    "what", "am", "i", "are", "is", "the", "a", "an", "my", "me",
    "currently", "now", "do", "does", "did", "how", "why", "when",
    "where", "which", "who", "tell", "about", "you", "your",
}

# Synonym expansion: maps words to their semantic equivalents
# so "studying" matches "learning", etc.
SYNONYM_MAP: dict[str, set[str]] = {
    "learn": {"study", "studying", "learned", "learning", "mastering"},
    "study": {"learn", "learning", "studying", "studied", "mastering"},
    "like": {"love", "enjoy", "prefer", "liking"},
    "love": {"like", "enjoy", "prefer"},
    "enjoy": {"like", "love", "prefer"},
    "prefer": {"like", "love", "enjoy"},
    "want": {"need", "desire", "wish", "plan"},
    "need": {"want", "desire", "wish", "plan"},
    "goal": {"objective", "target", "aim"},
    "problem": {"issue", "difficulty", "challenge"},
    "fix": {"repair", "resolve", "solve"},
    "focus": {"concentrate", "centering", "focusing", "learning", "studying"},
    "concentrate": {"focus", "focusing"},
    "trying": {"attempting", "working", "learning"},
    "work": {"working", "doing", "building", "creating"},
    "working": {"work", "doing", "building", "creating"},
    "subject": {"topic", "course", "field", "area"},
    "topic": {"subject", "course", "field", "area"},
}


def _expand_with_synonyms(words: set[str]) -> set[str]:
    """Expand a set of words with their synonyms.

    Uses removesuffix (not rstrip) to properly strip
    suffixes like 'ing', 'ed', 's' without removing
    individual characters.
    """
    expanded = set(words)
    for word in words:
        # Try the word directly first
        if word in SYNONYM_MAP:
            expanded |= SYNONYM_MAP[word]

        # Try stripping common suffixes to find base form
        for suffix in ("ing", "ed", "s"):
            base = word.removesuffix(suffix)
            if base != word and base in SYNONYM_MAP:
                expanded |= SYNONYM_MAP[base]
                break

    return expanded


def _normalize_text(text: str) -> set[str]:
    """Lowercase, strip punctuation, return word set."""
    return {
        w
        for w in text.lower().replace("?", "").replace(",", "")
        .replace(".", "").replace("!", "").split()
    }


def retrieve_relevant_memories(
    query: str,
    user_id: str = MEMORY_DEFAULT_USER_ID,
    case_id: int | None = None,
    db: Session | None = None,
) -> list[dict]:
    if db is None:
        return []

    # Normalize query words (remove stop words, short words)
    all_query_words = _normalize_text(query)
    query_words = {
        w for w in all_query_words
        if len(w) > 2 and w not in STOP_WORDS
    }

    # Expand query words with synonyms for better matching
    expanded_query_words = _expand_with_synonyms(query_words)

    # Retrieve user memories
    all_memories = (
        db.query(Memory)
        .filter(
            Memory.user_id == user_id,
            (Memory.case_id == None) | (Memory.case_id == case_id),
        )
        .all()
    )

    # No meaningful query words: return top by importance + recency
    if not query_words:
        memories = sorted(
            all_memories,
            key=lambda m: (-m.importance, -m.updated_at.timestamp()),
        )[:MEMORY_MAX_RETRIEVED]
    else:
        scored = []
        now = datetime.utcnow()

        for mem in all_memories:
            mem_words = _normalize_text(mem.content)
            # Use expanded synonyms for matching, but weight original
            # words higher than synonyms
            exact_overlap = len(query_words & mem_words)
            synonym_only = len(expanded_query_words & mem_words) - exact_overlap
            overlap = exact_overlap + synonym_only * 0.8

            # Recency bonus: max 0.5, decays over ~30 days
            age_days = max((now - mem.updated_at).total_seconds() / 86400, 0)
            recency_bonus = max(0.5 * (1.0 - age_days / 30.0), 0)

            score = overlap * 2.0 + float(mem.importance) + recency_bonus

            # Include if keyword/synonym matches or importance is high
            has_any_match = len(expanded_query_words & mem_words) > 0
            if overlap > 0 or has_any_match or mem.importance >= 0.7:
                scored.append((score, mem))

        scored.sort(
            key=lambda item: (-item[0], -item[1].importance, -item[1].updated_at.timestamp())
        )
        memories = [mem for _, mem in scored[:MEMORY_MAX_RETRIEVED]]

    return [
        {
            "id": mem.id,
            "content": mem.content,
            "memory_type": mem.memory_type,
            "case_id": mem.case_id,
            "importance": mem.importance,
            "created_at": mem.created_at.isoformat(),
            "updated_at": mem.updated_at.isoformat(),
        }
        for mem in memories
    ]


# ============================================================
# BUILD MEMORY CONTEXT
# ============================================================


def build_memory_context(
    query: str,
    user_id: str = MEMORY_DEFAULT_USER_ID,
    case_id: int | None = None,
    db: Session | None = None,
) -> str:
    memories = retrieve_relevant_memories(
        query=query, user_id=user_id, case_id=case_id, db=db,
    )
    if not memories:
        return ""

    parts = []
    total_chars = 0

    for memory in memories:
        entry = f"- [{memory['memory_type']}] {memory['content']}"
        if total_chars + len(entry) > MEMORY_MAX_CONTEXT_CHARS:
            break
        parts.append(entry)
        total_chars += len(entry)

    if not parts:
        return ""

    return "USER MEMORIES:\n" + "\n".join(parts)


# ============================================================
# STORE EXTRACTED MEMORIES (Auto-extract from conversations)
# ============================================================


def store_extracted_memories(
    user_message: str,
    assistant_response: str,
    user_id: str = MEMORY_DEFAULT_USER_ID,
    case_id: int | None = None,
    db: Session | None = None,
) -> list[Memory]:
    if db is None:
        return []
    if not MEMORY_AUTO_EXTRACT_ENABLED:
        return []
    if not user_message or not user_message.strip():
        return []

    words = user_message.strip().split()
    if len(words) < MEMORY_EXTRACT_MIN_WORDS:
        return []

    extracted = []
    lower_msg = user_message.lower().strip()

    goal_patterns = [
        "i want to", "i will", "my goal", "i plan to",
        "i am going to", "i need to", "i intend to",
    ]
    learning_patterns = [
        "i am learning", "i am studying", "i am working on",
        "i am practicing", "learning about", "studying",
    ]
    preference_patterns = [
        "i prefer", "i like", "i love", "i enjoy", "i hate", "i dislike",
    ]
    context_patterns = ["i am a", "i work as", "currently", "right now"]
    fact_patterns = ["i am", "i have", "i know", "i use"]

    def _check(message, patterns):
        return any(p in message for p in patterns)

    content = user_message.strip()

    if _check(lower_msg, goal_patterns):
        extracted.append(("goal", content, 0.6))
    elif _check(lower_msg, learning_patterns):
        extracted.append(("fact", content, 0.5))
    elif _check(lower_msg, preference_patterns):
        extracted.append(("preference", content, 0.5))
    elif _check(lower_msg, context_patterns):
        extracted.append(("context", content, 0.4))
    elif _check(lower_msg, fact_patterns):
        extracted.append(("fact", content, 0.3))

    extracted = extracted[:MEMORY_EXTRACT_MAX_PER_MESSAGE]

    memories = []
    for memory_type, content, importance in extracted:
        # Check for duplicate
        recent = (
            db.query(Memory)
            .filter(Memory.user_id == user_id, Memory.content == content.strip())
            .order_by(desc(Memory.created_at))
            .first()
        )
        if recent is not None:
            continue

        memory = create_memory(
            content=content,
            memory_type=memory_type,
            user_id=user_id,
            case_id=case_id,
            importance=importance,
            db=db,
        )
        memories.append(memory)

    return memories
