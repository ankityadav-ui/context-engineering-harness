
import logging

# Import the existing RAG retrieval function.
from app.rag.rag_service import retrieve_context
from app.rag.pinecone_service import search_similar_content
from app.rag.embedding_service import generate_text_embedding


logger = logging.getLogger("app.context_engine.engine")

# Bounds that keep one assembled context predictable and affordable for
# the model, however many sources respond.
MAX_RAG_RESULTS = 20
MAX_MEMORIES = 5
MAX_GRAPH_RELATIONSHIPS = 10

# Retrieved text is truncated per unit and in total before it is sent to
# the model. Truncation is reported in the formatted context so the model
# never sees a silently cut-off sentence presented as complete evidence.
MAX_UNIT_CHARS = 4000
MAX_CONTEXT_CHARS = 24000
TRUNCATION_NOTICE = "[context truncated: additional retrieved text omitted]"


def _dedupe_rag_results(results: list[dict]) -> list[dict]:
    """Drop repeated retrieval results, keeping the highest score.

    Overlapping document chunks and repeated table text can be returned
    as separate near-identical matches; sending the same evidence twice
    wastes context and can make an answer look better supported than it is.
    """

    best: dict[tuple, dict] = {}

    for result in results:
        key = (
            result.get("document_id"),
            result.get("content_type"),
            result.get("page_number"),
            " ".join(str(result.get("content", "")).split()).casefold(),
            result.get("content_ref") or "",
        )

        existing = best.get(key)

        if existing is None:
            best[key] = result
            continue

        if (result.get("score") or 0) > (existing.get("score") or 0):
            best[key] = result

    return list(best.values())


def _dedupe_relationships(relationships: list[dict]) -> list[dict]:
    """Drop duplicated graph relationships, preserving order."""

    seen: set[tuple] = set()
    unique: list[dict] = []

    for relationship in relationships:
        key = (
            relationship.get("matched_entity"),
            relationship.get("source"),
            relationship.get("relationship"),
            relationship.get("target"),
            relationship.get("direction"),
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(relationship)

    return unique


# Define a helper that retrieves conversation information.
def _get_conversation_context(
    conversation_id: str,
    query: str,
    recent_message_count: int = 6,
) -> dict:
    # Import database and conversation services only when needed.
    from app.context_engine.database import SessionLocal
    from app.context_engine.conversation_service import (
        get_conversation_message_window,
    )
    from app.context_engine.summary_service import (
        get_or_update_summary,
    )

    # Open a database session and close it automatically afterward.
    with SessionLocal() as db:
        # Split the conversation into older and recent messages.
        window = get_conversation_message_window(
            db=db,
            conversation_id=conversation_id,
            n=recent_message_count,
        )

        # Generate or reuse a summary of the older messages.
        summary = get_or_update_summary(
            db=db,
            conversation_id=conversation_id,
            current_query=query,
            n=recent_message_count,
        )

        # Return the summary and recent messages together, with the
        # real window counts reported by the conversation service
        # (total_messages / older_count / recent_count).
        return {
            "summary": summary,
            "recent_messages": window.get(
                "recent_messages", []
            ),
            "total_messages": window.get("total_messages", 0),
            "older_count": window.get("older_count", 0),
            "recent_count": window.get("recent_count", 0),
        }


# Define a helper that retrieves long-term memories.
def _get_memory_context(
    query: str,
    user_id: str,
    limit: int = MAX_MEMORIES,
) -> list[dict]:
    # Import the memory service and database session factory.
    from app.context_engine.database import SessionLocal
    from app.context_engine.memory_service import (
        retrieve_relevant_memories,
    )

    # Open a database session for memory retrieval.
    with SessionLocal() as db:
        # Retrieve only memories belonging to this user.
        return retrieve_relevant_memories(
            db=db,
            user_id=user_id,
            query=query,
            limit=limit,
        )


# Define a helper that retrieves related graph information.
def _get_graph_context(query: str) -> list[dict]:
    # Import graph functions here so graph failures do not prevent
    # the Context Engine module itself from loading.
    from app.context_engine.graph_service import (
        find_entities,
        get_entity_relationships,
    )

    # Search for entities whose names match the current question.
    entities = find_entities(query)

    # Collect relationships for the matching entities.
    relationships = []

    for entity in entities:
        # Fetch both incoming and outgoing relationships.
        entity_relationships = get_entity_relationships(
            entity["entity_id"]
        )

        # Add the entity name to help explain each relationship.
        for relationship in entity_relationships:
            relationships.append({
                "matched_entity": entity["name"],
                **relationship,
            })

    # Return a consistent list, even when no entities match.
    return relationships


# Define the single authoritative context assembly function.
def assemble_context(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
) -> dict:
    # Reject a blank question before contacting external services.
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    # Initialize every context source with a predictable value.
    context = {
        "query": query.strip(),
        "conversation": None,
        "memory": [],
        "rag": [],
        "knowledge_graph": [],
        "source_errors": [],
    }

    # Retrieve document evidence from Pinecone. top_k is bounded by the
    # API request models and re-bounded here so an internal caller cannot
    # request an unbounded context either.
    try:
        bounded_top_k = max(1, min(int(top_k), MAX_RAG_RESULTS))

        context["rag"] = _dedupe_rag_results(
            retrieve_context(
                query=query.strip(),
                top_k=bounded_top_k,
                document_id=document_id,
            )
        )
    except Exception as error:
        # Preserve other sources if document retrieval fails.
        context["source_errors"].append({
            "source": "rag",
            "error": str(error),
        })

    # Short resumes and image-based PDFs are indexed by full extracted
    # text rather than vectors. When retrieval found nothing for the
    # requested document, build that full-text block ourselves.
    if document_id is not None and not context["rag"]:
        try:
            from app.rag.document_service import (
                build_full_text,
                read_masked_text_artifact,
                resolve_stored_document_path,
                run_pdf_text_pipeline,
            )
            from app.rag.text_extractor import is_pdf_openable

            full_text = ""
            source = "full_text"

            # A masked upload deliberately keeps no raw PDF. Its
            # safety-validated text is the only fallback, so raw PII is
            # never re-extracted and sent to the model.
            masked_text = read_masked_text_artifact(document_id)

            if masked_text:
                full_text = masked_text
                source = "masked_full_text"
            else:
                stored_path = resolve_stored_document_path(document_id)

                if stored_path.is_file() and is_pdf_openable(stored_path):
                    # run_pdf_text_pipeline applies the same OCR fallback
                    # used at upload time, so a scanned resume is also
                    # summarizable when the vector index returns nothing.
                    text_units, _pages, _ = run_pdf_text_pipeline(
                        document_path=stored_path,
                        document_id=document_id,
                    )

                    full_text = build_full_text(
                        [
                            {**unit, "content_type": "text"}
                            for unit in text_units
                        ]
                    )

            if full_text.strip():
                context["rag"] = _dedupe_rag_results(
                    [
                        {
                            "content_type": "text",
                            "content": full_text,
                            "document_id": document_id,
                            "page_number": 1,
                            "source": source,
                        }
                    ]
                )
        except Exception as error:
            # A full-text fallback failure is a source error, not a
            # hard failure: the conversation, memory and graph sources
            # still contribute to the answer.
            context["source_errors"].append({
                "source": "full_text",
                "error": str(error),
            })

    # Retrieve conversation summary and recent messages when
    # a specific conversation has been supplied.
    if conversation_id is not None:
        try:
            context["conversation"] = _get_conversation_context(
                conversation_id=conversation_id,
                query=query.strip(),
            )
        except Exception as error:
            # Record the failure instead of losing all context.
            context["source_errors"].append({
                "source": "conversation",
                "error": str(error),
            })

    # Retrieve long-term memories only for a known user.
    if user_id:
        try:
            context["memory"] = _get_memory_context(
                query=query.strip(),
                user_id=user_id,
            )
        except Exception as error:
            # Keep the request usable if memory retrieval fails.
            context["source_errors"].append({
                "source": "memory",
                "error": str(error),
            })

    # Search the knowledge graph for related entities.
    try:
        context["knowledge_graph"] = _dedupe_relationships(
            _get_graph_context(query.strip())
        )[:MAX_GRAPH_RELATIONSHIPS]
    except Exception as error:
        # Neo4j may be unavailable or not configured yet.
        context["source_errors"].append({
            "source": "knowledge_graph",
            "error": str(error),
        })

    # Inspectable assembly summary: counts and sizes only. No query text,
    # memory content, document text, or entity value is logged.
    logger.info(
        "Assembled context: query_chars=%d rag=%d memory=%d graph=%d "
        "conversation=%s source_errors=%d",
        len(context["query"]),
        len(context["rag"]),
        len(context["memory"]),
        len(context["knowledge_graph"]),
        "yes" if context["conversation"] else "no",
        len(context["source_errors"]),
    )

    # Return the combined result to the calling endpoint.
    return context


def _truncate_text(text: str, limit: int) -> str:
    """Truncate one text block to ``limit`` characters."""

    if text is None:
        return ""

    text = str(text)

    if len(text) <= limit:
        return text

    return text[:limit].rstrip() + " ..."


# Define a function that turns structured context into readable text.
def format_context(
    context: dict,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    # Store each non-empty section before joining them together.
    sections = []

    # Add the current question for traceability.
    query = context.get("query", "")
    if query:
        sections.append(
            "CURRENT QUESTION:\n" + str(query)
        )

    # Add the older conversation summary and recent messages.
    conversation = context.get("conversation")
    if conversation:
        summary = conversation.get("summary", "")

        if summary:
            sections.append(
                "CONVERSATION SUMMARY:\n" + str(summary)
            )

        recent_messages = conversation.get(
            "recent_messages", []
        )

        if recent_messages:
            message_lines = []

            for message in recent_messages:
                # Support both dictionaries and serialized messages.
                role = message.get("role", "unknown")
                content = message.get("content", "")
                message_lines.append(
                    f"{str(role).upper()}: {content}"
                )

            sections.append(
                "RECENT CONVERSATION:\n"
                + "\n".join(message_lines)
            )

    # Add relevant long-term memories.
    memories = context.get("memory", [])
    if memories:
        memory_lines = []

        for memory in memories:
            memory_type = memory.get(
                "memory_type", "memory"
            )
            content = _truncate_text(
                memory.get("content", ""),
                MAX_UNIT_CHARS,
            )
            memory_lines.append(
                f"- [{memory_type}] {content}"
            )

        sections.append(
            "RELEVANT LONG-TERM MEMORIES:\n"
            + "\n".join(memory_lines)
        )


    # Add relationships retrieved from the knowledge graph.
    graph_results = context.get("knowledge_graph", [])
    if graph_results:
        graph_lines = []

        for relationship in graph_results:
            # Format each relationship as source → relation → target.
            source = relationship.get("source", "Unknown")
            relation = relationship.get(
                "relationship", "RELATED_TO"
            )
            target = relationship.get("target", "Unknown")

            graph_lines.append(
                f"{source} --{relation}--> {target}"
            )

        sections.append(
            "KNOWLEDGE GRAPH RELATIONSHIPS:\n"
            + "\n".join(graph_lines)
        )

    # Add retrieved document, table, and image-reference results.
    rag_results = context.get("rag", [])
    if rag_results:
        rag_lines = []

        for index, result in enumerate(rag_results, start=1):
            # Identify the kind of content returned by retrieval.
            content_type = result.get(
                "content_type", "text"
            )
            content = _truncate_text(
                result.get("content", ""),
                MAX_UNIT_CHARS,
            )
            page_number = result.get("page_number")
            document_id = result.get("document_id")

            # Include available source metadata for traceability.
            source_details = []

            if document_id:
                source_details.append(
                    f"document_id={document_id}"
                )

            if page_number is not None:
                source_details.append(
                    f"page={page_number}"
                )

            metadata_text = (
                ", ".join(source_details)
                if source_details
                else "source metadata unavailable"
            )

            rag_lines.append(
                f"[{index}] TYPE={content_type}; "
                f"{metadata_text}\n{content}"
            )

        sections.append(
            "RELEVANT DOCUMENT EVIDENCE:\n"
            + "\n\n".join(rag_lines)
        )

    # Join all available sources into one context string.
    formatted = "\n\n".join(sections)

    # Hard ceiling on the context sent to the model. The notice makes the
    # truncation visible instead of presenting a cut-off transcript as
    # complete evidence. A non-positive limit disables truncation.
    if max_chars and max_chars > 0 and len(formatted) > max_chars:
        logger.info(
            "Context truncated: %d chars -> %d chars",
            len(formatted),
            max_chars,
        )

        formatted = (
            formatted[:max_chars].rstrip()
            + "\n\n"
            + TRUNCATION_NOTICE
        )

    return formatted
