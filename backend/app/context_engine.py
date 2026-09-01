from sqlalchemy.orm import Session

from .rag_context import build_rag_context
from .graph_rag import graph_rag


def build_context(
    query: str,
    case_id: int | None = None,
    conversation_history: list[dict] | None = None,
    memory_context: str = "",
    top_k: int = 5,
    db: Session | None = None,
) -> dict:
    """
    Central Context Engineering layer.

    Combines:

        1. Short-term memory
        2. Long-term memory
        3. Vector RAG / document context
        4. Graph RAG / knowledge graph context

    The LLM is NOT called here.

    This layer is responsible only for preparing
    high-quality context for the LLM.
    """

    conversation_history = conversation_history or []

    # ========================================================
    # DOCUMENT / VECTOR RAG
    # ========================================================

    rag_result = {
        "context": "",
        "chunks": [],
        "metadata": {},
    }

    if case_id is not None:
        rag_result = build_rag_context(
            query=query,
            case_id=case_id,
            top_k=top_k,
            db=db,
        )

    document_context = rag_result.get(
        "context",
        "",
    )

    # ========================================================
    # GRAPH RAG
    # ========================================================

    graph_result = {
        "context": "",
        "results": [],
    }

    try:
        graph_result = graph_rag.build_graph_context(
            query
        )
    except Exception as e:
        # Graph RAG should not break normal vector RAG.
        print(
            f"[GraphRAG] Retrieval failed: {e}"
        )

    graph_context = graph_result.get(
        "context",
        "",
    )

    graph_results = graph_result.get(
        "results",
        [],
    )

    # ========================================================
    # SHORT-TERM MEMORY
    # ========================================================

    short_term_context = ""

    if conversation_history:

        parts = []

        for message in conversation_history:

            role = message.get(
                "role",
                "unknown",
            )

            content = message.get(
                "content",
                "",
            )

            if not content:
                continue

            parts.append(
                f"{role.upper()}: {content}"
            )

        short_term_context = "\n".join(
            parts
        )

    # ========================================================
    # LONG-TERM MEMORY
    # ========================================================

    long_term_context = (
        memory_context or ""
    )

    # ========================================================
    # BUILD FINAL CONTEXT
    # ========================================================

    sections = []

    # --------------------------------------------------------
    # SHORT-TERM
    # --------------------------------------------------------

    if short_term_context:

        sections.append(
            "=== SHORT-TERM CONVERSATION ===\n"
            + short_term_context
        )

    # --------------------------------------------------------
    # LONG-TERM
    # --------------------------------------------------------

    if long_term_context:

        sections.append(
            "=== LONG-TERM MEMORY ===\n"
            + long_term_context
        )

    # --------------------------------------------------------
    # VECTOR RAG
    # --------------------------------------------------------

    if document_context:

        sections.append(
            "=== CASE DOCUMENTS ===\n"
            + document_context
        )

    # --------------------------------------------------------
    # GRAPH RAG
    # --------------------------------------------------------

    if graph_context:

        sections.append(
            "=== KNOWLEDGE GRAPH ===\n"
            + graph_context
        )

    final_context = "\n\n".join(
        sections
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {

        # Memory
        "short_term_messages": len(
            conversation_history
        ),

        "long_term_memory_used": bool(
            long_term_context
        ),

        # Vector RAG
        "document_chunks_used": len(
            rag_result.get(
                "chunks",
                [],
            )
        ),

        "document_context_chars": len(
            document_context
        ),

        # Graph RAG
        "graph_results_used": len(
            graph_results
        ),

        "graph_context_chars": len(
            graph_context
        ),

        # Everything
        "total_context_chars": len(
            final_context
        ),
    }

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "query": query,

        "case_id": case_id,

        "context": final_context,

        "sources": {

            "short_term_memory":
                conversation_history,

            "long_term_memory":
                long_term_context,

            "documents":
                rag_result.get(
                    "chunks",
                    [],
                ),

            "graph":
                graph_results,
        },

        "metadata": metadata,

        # Keep vector RAG metadata available
        # for debugging / UI.
        "rag_metadata":
            rag_result.get(
                "metadata",
                {},
            ),
    }