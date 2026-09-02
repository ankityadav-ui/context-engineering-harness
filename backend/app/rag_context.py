import logging

from sqlalchemy.orm import Session

from .config import (
    RAG_DEDUP_THRESHOLD,
    RAG_DISTANCE_THRESHOLD,
    RAG_MAX_CONTEXT_CHARS,
    RAG_TOKENS_PER_CHAR,
)
from .models import Document
from .vector_store import search_chunks

logger = logging.getLogger(__name__)


# ============================================================
# STAGE 1: RETRIEVE
# ============================================================


def retrieve_chunks(
    query: str,
    case_id: int,
    top_k: int = 5,
) -> list[dict]:
    results = search_chunks(
        query=query,
        case_id=case_id,
        top_k=top_k,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    chunks = []

    for index, text in enumerate(documents):
        metadata = metadatas[index]
        chunks.append({
            "text": text,
            "case_id": metadata.get("case_id"),
            "document_id": metadata.get("document_id"),
            "chunk_index": metadata.get("chunk_index"),
            "distance": distances[index],
        })

    return chunks


# ============================================================
# STAGE 2: RELEVANCE FILTER
# ============================================================


def filter_by_relevance(
    chunks: list[dict],
    threshold: float = RAG_DISTANCE_THRESHOLD,
) -> list[dict]:
    return [
        chunk
        for chunk in chunks
        if chunk["distance"] <= threshold
    ]


# ============================================================
# STAGE 3: RANK
# ============================================================


def rank_chunks(chunks: list[dict]) -> list[dict]:
    return sorted(chunks, key=lambda c: c["distance"])


# ============================================================
# STAGE 4: DEDUPLICATE
# ============================================================


def _jaccard_similarity(a: str, b: str) -> float:
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def deduplicate_chunks(
    chunks: list[dict],
    threshold: float = RAG_DEDUP_THRESHOLD,
) -> list[dict]:
    if not chunks:
        return []

    kept = [chunks[0]]

    for chunk in chunks[1:]:
        is_duplicate = False
        for existing in kept:
            similarity = _jaccard_similarity(
                chunk["text"],
                existing["text"],
            )
            if similarity >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(chunk)

    return kept


# ============================================================
# STAGE 5: CONTEXT BUDGET
# ============================================================


def apply_context_budget(
    chunks: list[dict],
    max_chars: int = RAG_MAX_CONTEXT_CHARS,
) -> list[dict]:
    if not chunks:
        return []

    used = []
    total_chars = 0

    for chunk in chunks:
        chunk_len = len(chunk["text"])
        if total_chars + chunk_len <= max_chars:
            used.append(chunk)
            total_chars += chunk_len
        else:
            continue

    return used


# ============================================================
# STAGE 6: BUILD STRUCTURED CONTEXT
# ============================================================


def build_structured_context(chunks: list[dict]) -> str:
    if not chunks:
        return ""

    parts = []

    for index, chunk in enumerate(chunks):
        filename = chunk.get("filename", "unknown")
        distance = chunk.get("distance", 0)
        chunk_index = chunk.get("chunk_index", "?")
        lines = [
            f"--- SOURCE {index + 1} ---",
            f"Document: {filename}",
            f"Chunk: {chunk_index}",
            f"Relevance Distance: {distance:.4f}",
            "",
            chunk["text"],
        ]
        parts.append("\n".join(lines))

    return "\n\n".join(parts)


# ============================================================
# STAGE 7: RESOLVE FILENAMES
# ============================================================


def resolve_filenames(
    chunks: list[dict],
    db: Session,
) -> list[dict]:
    if not chunks or db is None:
        return chunks

    document_ids = list(
        set(c["document_id"] for c in chunks if c.get("document_id"))
    )

    if not document_ids:
        return chunks

    documents = (
        db.query(Document)
        .filter(Document.id.in_(document_ids))
        .all()
    )

    filename_map = {doc.id: doc.filename for doc in documents}

    for chunk in chunks:
        chunk["filename"] = filename_map.get(
            chunk.get("document_id"), "unknown"
        )

    return chunks


# ============================================================
# FULL PIPELINE
# ============================================================


def build_rag_context(
    query: str,
    case_id: int,
    top_k: int = 5,
    db: Session = None,
) -> dict:
    retrieved = retrieve_chunks(
        query=query,
        case_id=case_id,
        top_k=top_k,
    )

    filtered = filter_by_relevance(retrieved)
    ranked = rank_chunks(filtered)
    deduplicated = deduplicate_chunks(ranked)
    deduplicated = resolve_filenames(deduplicated, db)
    final_chunks = apply_context_budget(deduplicated)
    context = build_structured_context(final_chunks)

    logger.info(
        "[RAG] case_id=%s query=%s retrieved=%d filtered=%d dedup=%d final=%d threshold=%.2f",
        case_id,
        query[:80],
        len(retrieved),
        len(filtered),
        len(deduplicated),
        len(final_chunks),
        RAG_DISTANCE_THRESHOLD,
    )

    if retrieved and not filtered:
        distances = [c["distance"] for c in retrieved]
        logger.warning(
            "[RAG] All %d chunks filtered out (distances: %s, threshold: %.2f)",
            len(retrieved),
            [f"{d:.4f}" for d in distances],
            RAG_DISTANCE_THRESHOLD,
        )
    elif not retrieved:
        logger.warning(
            "[RAG] No chunks retrieved from vector store for case_id=%s (top_k=%d)",
            case_id,
            top_k,
        )

    context_chars = sum(len(c["text"]) for c in final_chunks)

    metadata = {
        "chunks_retrieved": len(retrieved),
        "chunks_after_filter": len(filtered),
        "chunks_after_dedup": len(deduplicated),
        "chunks_used": len(final_chunks),
        "context_character_count": context_chars,
        "context_token_estimate": int(
            context_chars * RAG_TOKENS_PER_CHAR
        ),
        "distance_threshold": RAG_DISTANCE_THRESHOLD,
        "max_context_chars": RAG_MAX_CONTEXT_CHARS,
        "dedup_threshold": RAG_DEDUP_THRESHOLD,
    }

    return {
        "query": query,
        "case_id": case_id,
        "chunks": final_chunks,
        "context": context,
        "metadata": metadata,
        "retrieved_chunks": retrieved,
        "filtered_chunks": filtered,
        "deduplicated_chunks": deduplicated,
    }

