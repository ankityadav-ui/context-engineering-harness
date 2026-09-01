import json

from sqlalchemy.orm import Session

from .models import EvalQuery, EvalResult, Document
from .rag_context import (
    retrieve_chunks,
    filter_by_relevance,
    rank_chunks,
    deduplicate_chunks,
)


# ============================================================
# METRICS CALCULATION
# ============================================================


def calculate_hit_at_k(
    retrieved_doc_ids: list[int],
    expected_doc_ids: list[int],
    top_k: int,
) -> bool:
    """
    Hit@K: Is at least one expected document
    in the top K retrieved results?
    """

    retrieved_set = set(retrieved_doc_ids[:top_k])
    expected_set = set(expected_doc_ids)

    return bool(retrieved_set & expected_set)


def calculate_precision_at_k(
    retrieved_doc_ids: list[int],
    expected_doc_ids: list[int],
    top_k: int,
) -> float:
    """
    Precision@K: Fraction of top K retrieved
    documents that are relevant (expected).
    """

    if top_k == 0:
        return 0.0

    retrieved_set = set(retrieved_doc_ids[:top_k])
    expected_set = set(expected_doc_ids)

    relevant_retrieved = len(retrieved_set & expected_set)

    return relevant_retrieved / top_k


def calculate_recall_at_k(
    retrieved_doc_ids: list[int],
    expected_doc_ids: list[int],
    top_k: int,
) -> float:
    """
    Recall@K: Fraction of expected documents
    found in top K retrieved results.
    """

    if not expected_doc_ids:
        return 1.0

    retrieved_set = set(retrieved_doc_ids[:top_k])
    expected_set = set(expected_doc_ids)

    relevant_retrieved = len(retrieved_set & expected_set)

    return relevant_retrieved / len(expected_set)


def calculate_reciprocal_rank(
    retrieved_doc_ids: list[int],
    expected_doc_ids: list[int],
) -> float:
    """
    Reciprocal Rank: 1/rank of the first
    relevant document in the retrieved list.
    """

    expected_set = set(expected_doc_ids)

    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in expected_set:
            return 1.0 / rank

    return 0.0


# ============================================================
# SINGLE QUERY EVALUATION
# ============================================================


def evaluate_single_query(
    eval_query: EvalQuery,
    top_k: int,
    db: Session,
) -> dict:
    """
    Run the RAG pipeline for a single evaluation query
    and compute all metrics.
    """

    # Parse expected document IDs
    expected_doc_ids = json.loads(
        eval_query.expected_document_ids
    )

    # Parse expected chunk indices (optional)
    expected_chunk_indices = None
    if eval_query.expected_chunk_indices:
        expected_chunk_indices = json.loads(
            eval_query.expected_chunk_indices
        )

    # --------------------------------------------------------
    # Run retrieval pipeline
    # --------------------------------------------------------

    retrieved = retrieve_chunks(
        query=eval_query.query,
        case_id=eval_query.case_id,
        top_k=top_k,
    )

    filtered = filter_by_relevance(retrieved)
    ranked = rank_chunks(filtered)
    deduplicated = deduplicate_chunks(ranked)

    # --------------------------------------------------------
    # Extract retrieved document IDs and distances
    # --------------------------------------------------------

    retrieved_doc_ids = [
        c.get("document_id")
        for c in deduplicated
        if c.get("document_id") is not None
    ]

    retrieved_distances = [
        c.get("distance", 0.0)
        for c in deduplicated
    ]

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    hit = calculate_hit_at_k(
        retrieved_doc_ids, expected_doc_ids, top_k
    )

    precision = calculate_precision_at_k(
        retrieved_doc_ids, expected_doc_ids, top_k
    )

    recall = calculate_recall_at_k(
        retrieved_doc_ids, expected_doc_ids, top_k
    )

    rr = calculate_reciprocal_rank(
        retrieved_doc_ids, expected_doc_ids
    )

    # --------------------------------------------------------
    # Determine pass/fail
    # --------------------------------------------------------

    # Pass if any expected document is retrieved
    passed = hit

    # --------------------------------------------------------
    # Resolve filenames for retrieved chunks
    # --------------------------------------------------------

    all_doc_ids = list(set(
        c.get("document_id")
        for c in deduplicated
        if c.get("document_id")
    ))

    filename_map = {}
    if all_doc_ids:
        docs = (
            db.query(Document)
            .filter(Document.id.in_(all_doc_ids))
            .all()
        )
        filename_map = {d.id: d.filename for d in docs}

    for chunk in deduplicated:
        chunk["filename"] = filename_map.get(
            chunk.get("document_id"), "unknown"
        )

    # --------------------------------------------------------
    # Build result
    # --------------------------------------------------------

    avg_distance = (
        sum(retrieved_distances) / len(retrieved_distances)
        if retrieved_distances
        else None
    )

    return {
        "query": eval_query.query,
        "description": eval_query.description,
        "expected_document_ids": expected_doc_ids,
        "expected_chunk_indices": expected_chunk_indices,
        "retrieved_doc_ids": retrieved_doc_ids,
        "retrieved_distances": retrieved_distances,
        "retrieved_chunks": deduplicated,
        "hit_at_k": hit,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "reciprocal_rank": rr,
        "passed": passed,
        "avg_distance": avg_distance,
    }


# ============================================================
# AGGREGATE METRICS
# ============================================================


def calculate_aggregate(results: list[dict]) -> dict:
    """
    Calculate aggregate metrics across all evaluation queries.
    """

    total = len(results)

    if total == 0:
        return {
            "total_queries": 0,
            "queries_with_results": 0,
            "total_passed": 0,
            "total_failed": 0,
            "hit_at_k": 0.0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "avg_distance": None,
        }

    total_passed = sum(1 for r in results if r["passed"])
    total_failed = total - total_passed

    avg_hit = sum(r["hit_at_k"] for r in results) / total
    avg_precision = sum(r["precision_at_k"] for r in results) / total
    avg_recall = sum(r["recall_at_k"] for r in results) / total
    avg_rr = sum(r["reciprocal_rank"] for r in results) / total

    avg_distances = [
        r["avg_distance"]
        for r in results
        if r["avg_distance"] is not None
    ]

    avg_distance = (
        sum(avg_distances) / len(avg_distances)
        if avg_distances
        else None
    )

    return {
        "total_queries": total,
        "queries_with_results": total,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "hit_at_k": round(avg_hit, 4),
        "precision_at_k": round(avg_precision, 4),
        "recall_at_k": round(avg_recall, 4),
        "mrr": round(avg_rr, 4),
        "avg_distance": round(avg_distance, 4) if avg_distance is not None else None,
    }
