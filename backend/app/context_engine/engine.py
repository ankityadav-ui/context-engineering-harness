
import logging
import re

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

# A whole-document summary cannot be grounded in a handful of top-k
# chunks: they are disjoint, so the model sees fragments (and masked
# placeholders) rather than the document and reports that it cannot
# summarise. Such requests take the full-text path when a document is
# scoped.
SUMMARY_REQUEST_PATTERN = re.compile(
    r"\b(summar(?:y|ise|ize|ised|ized|ising|izing)|tl;dr)\b",
    re.IGNORECASE,
)


def is_summary_request(query: str) -> bool:
    """Report whether the question asks for a document summary."""

    return bool(SUMMARY_REQUEST_PATTERN.search(query or ""))


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


def _safe_failure(source: str, error: Exception) -> dict:
    """Return a fixed, non-sensitive diagnostic for a failed source."""

    source_name = source.strip().casefold()
    unavailable = isinstance(error, (ConnectionError, TimeoutError, OSError))

    if isinstance(error, RuntimeError):
        message = str(error).casefold()
        unavailable = any(
            marker in message
            for marker in (
                "not configured",
                "unavailable",
                "connection",
                "timed out",
            )
        )

    messages = {
        "rag": (
            "Document retrieval failed. "
            "The vector index may be unavailable."
        ),
        "memory": "Long-term memory could not be read.",
        "conversation": "Conversation context could not be read.",
        "knowledge_graph": "The knowledge graph could not be queried.",
    }

    return {
        "source": source_name,
        "category": "unavailable" if unavailable else "internal",
        "message": messages.get(source_name, "This context source failed."),
    }


def _record_source_failure(
    context: dict,
    source: str,
    error: Exception,
) -> None:
    diagnostic_source = "rag" if source == "full_text" else source
    safe_failure = _safe_failure(diagnostic_source, error)
    context["source_errors"].append(
        {
            "source": diagnostic_source,
            "error": safe_failure["message"],
        }
    )
    context["diagnostic_errors"].append(safe_failure)


def build_context_report(
    context: dict,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> dict:
    """Describe the exact formatted context without duplicating its content."""

    formatted_context = format_context(context, max_chars=max_chars)
    headers = (
        ("conversation_summary", "CONVERSATION SUMMARY:\n"),
        ("recent_conversation", "RECENT CONVERSATION:\n"),
        ("memory", "RELEVANT LONG-TERM MEMORIES:\n"),
        ("knowledge_graph", "KNOWLEDGE GRAPH RELATIONSHIPS:\n"),
        ("document_evidence", "RELEVANT DOCUMENT EVIDENCE:\n"),
    )
    found = sorted(
        (
            (formatted_context.find(header), name, header)
            for name, header in headers
            if formatted_context.find(header) >= 0
        ),
        key=lambda item: item[0],
    )
    sections = []

    for index, (start, name, _header) in enumerate(found):
        end = (
            found[index + 1][0]
            if index + 1 < len(found)
            else len(formatted_context)
        )
        section_text = formatted_context[start:end].strip()
        section_text = section_text.split(
            f"\n\n{TRUNCATION_NOTICE}", maxsplit=1
        )[0]
        if section_text:
            sections.append({
                "name": name,
                "chars": len(section_text),
            })

    return {
        "formatted_context": formatted_context,
        "chars": len(formatted_context),
        "empty": not sections,
        "truncated": TRUNCATION_NOTICE in formatted_context,
        "sections": sections,
    }


def build_context_diagnostics(
    context: dict,
    *,
    conversation_requested: bool,
    memory_requested: bool,
    duration_ms: int,
    assembly: dict | None = None,
) -> dict:
    """Summarize source outcomes and the assembled context for preview."""

    failures = list(context.get("diagnostic_errors", []))
    failed_sources = {failure["source"] for failure in failures}
    conversation = context.get("conversation") or {}
    recent_messages = conversation.get("recent_messages", [])
    older_messages = int(conversation.get("older_count", 0) or 0)
    summary = conversation.get("summary") or ""
    memory_items = context.get("memory", [])
    rag_items = context.get("rag", [])
    graph_items = context.get("knowledge_graph", [])

    short_term_status = (
        "not_requested"
        if not conversation_requested
        else (
            "error"
            if "conversation" in failed_sources
            else ("ok" if conversation else "empty")
        )
    )
    summary_status = (
        "not_requested"
        if not conversation_requested or older_messages == 0
        else (
            "error"
            if "conversation" in failed_sources
            else ("ok" if summary else "empty")
        )
    )
    memory_status = (
        "not_requested"
        if not memory_requested
        else (
            "error"
            if "memory" in failed_sources
            else ("ok" if memory_items else "empty")
        )
    )
    document_ids = sorted(
        {
            str(item["document_id"])
            for item in rag_items
            if item.get("document_id")
        }
    )
    content_types = sorted(
        {str(item.get("content_type", "text")) for item in rag_items}
    )
    graph_entities = sorted(
        {
            str(item["matched_entity"])
            for item in graph_items
            if item.get("matched_entity")
        }
    )
    if not graph_entities:
        graph_entities = sorted(
            {
                str(value)
                for item in graph_items
                for value in (item.get("source"), item.get("target"))
                if value
            }
        )

    report = assembly or build_context_report(context)
    rag_status = (
        "error"
        if "rag" in failed_sources
        else ("ok" if rag_items else "empty")
    )
    graph_status = (
        "error"
        if "knowledge_graph" in failed_sources
        else ("ok" if graph_items else "empty")
    )

    return {
        "status": "partial" if failures else "ok",
        "duration_ms": max(0, int(duration_ms)),
        "errors": failures,
        "conversation_requested": conversation_requested,
        "memory_requested": memory_requested,
        "sources": {
            "short_term": {
                "status": short_term_status,
                "messages": len(recent_messages),
            },
            "conversation_summary": {
                "status": summary_status,
                "older_messages": older_messages,
                "chars": len(str(summary)),
            },
            "memory": {
                "status": memory_status,
                "items": len(memory_items),
            },
            "rag": {
                "status": rag_status,
                "chunks": len(rag_items),
                "documents": len(document_ids),
                "document_ids": document_ids,
                "content_types": content_types,
            },
            "knowledge_graph": {
                "status": graph_status,
                "relationships": len(graph_items),
                "entities": len(graph_entities),
            },
        },
        "assembly": report,
    }


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
def _get_graph_context(
    query: str,
    document_id: str | None = None,
) -> list[dict]:
    # Import graph functions here so graph failures do not prevent
    # the Context Engine module itself from loading.
    from app.context_engine.graph_service import (
        find_entities,
        find_document_entities,
        get_entity_relationships,
    )

    if document_id is None:
        entities = find_entities(query)
    else:
        # A selected document must never inherit an edge from another
        # document just because the same entity name appears in both.
        query_text = " ".join(query.split())
        entities = [
            entity
            for entity in find_document_entities(document_id)
            if isinstance(entity.get("name"), str)
            and entity["name"].strip()
            and re.search(
                rf"(?<!\w){re.escape(entity['name'].strip())}(?!\w)",
                query_text,
                flags=re.IGNORECASE,
            )
        ]

    # Collect relationships for the matching entities.
    relationships = []

    for entity in entities:
        # Fetch both incoming and outgoing relationships.
        entity_relationships = (
            get_entity_relationships(entity["entity_id"])
            if document_id is None
            else get_entity_relationships(
                entity["entity_id"],
                document_id=document_id,
            )
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
        "diagnostic_errors": [],
    }

    document_summary_mode = (
        document_id is not None and is_summary_request(query)
    )

    if document_summary_mode:
        context["document_summary_info"] = {
            "document_id": document_id,
            "document_found": None,
            "page_count": None,
            "pages_without_text": [],
            "units_truncated": False,
        }
        from app.rag.document_service import (
            read_masked_content_units,
            read_masked_text_artifact,
            resolve_stored_document_path,
            run_pdf_text_pipeline,
        )
        from app.rag.pinecone_service import get_document_searchable_units
        from app.rag.text_extractor import (
            get_pdf_page_count,
            is_pdf_openable,
        )

        try:
            indexed = get_document_searchable_units(document_id)
        except Exception as error:
            indexed = None
            _record_source_failure(context, "document_summary", error)

        if indexed is not None:
            context["document_summary_info"].update(
                {
                    "document_found": indexed.get("document_found"),
                    "page_count": indexed.get("page_count"),
                    "pages_without_text": indexed.get(
                        "pages_without_text",
                        [],
                    ),
                    "units_truncated": indexed.get(
                        "units_truncated",
                        False,
                    ),
                }
            )
            seen_source_text: set[tuple] = set()
            for unit in indexed.get("units", []):
                if (
                    not isinstance(unit, dict)
                    or unit.get("document_id") != document_id
                    or unit.get("content_type") not in {"text", "table"}
                    or not isinstance(unit.get("content"), str)
                    or not unit["content"].strip()
                ):
                    continue
                key = (
                    unit.get("page_number"),
                    unit.get("content_type"),
                    " ".join(unit["content"].split()).casefold(),
                )
                if key in seen_source_text:
                    continue
                seen_source_text.add(key)
                context["rag"].append(unit)

        if not context["rag"]:
            try:
                masked_units = read_masked_content_units(document_id)
                if masked_units is not None:
                    context["document_summary_info"].update(
                        {
                            "document_found": True,
                            "page_count": masked_units.get("page_count"),
                            "pages_without_text": masked_units.get(
                                "pages_without_text",
                                [],
                            ),
                        }
                    )
                    context["rag"] = masked_units["units"]
                else:
                    # Old plain-text artifacts have lost their original
                    # page/chunk boundaries and cannot ground a summary.
                    legacy_masked_text = read_masked_text_artifact(
                        document_id
                    )
                    if legacy_masked_text:
                        context["document_summary_info"].update(
                            {
                                "document_found": True,
                                "source_provenance_unavailable": True,
                            }
                        )
                    else:
                        stored_path = resolve_stored_document_path(
                            document_id
                        )
                        if (
                            stored_path.is_file()
                            and is_pdf_openable(stored_path)
                        ):
                            text_units, _ocr_pages, _ = (
                                run_pdf_text_pipeline(
                                    document_path=stored_path,
                                    document_id=document_id,
                                )
                            )
                            page_count = get_pdf_page_count(stored_path)
                            extracted_pages = {
                                unit["page_number"]
                                for unit in text_units
                                if unit.get("content", "").strip()
                            }
                            context["document_summary_info"].update(
                                {
                                    "document_found": True,
                                    "page_count": page_count,
                                    "pages_without_text": [
                                        page
                                        for page in range(1, page_count + 1)
                                        if page not in extracted_pages
                                    ],
                                }
                            )
                            context["rag"] = [
                                {
                                    "unit_id": (
                                        unit.get("unit_id")
                                        or unit.get("id")
                                        or (
                                            f"{document_id}:page:"
                                            f"{unit['page_number']}:"
                                            f"{index}"
                                        )
                                    ),
                                    "document_id": document_id,
                                    "content_type": "text",
                                    "page_number": unit.get("page_number"),
                                    "content": unit["content"],
                                    "source": unit.get(
                                        "source",
                                        "pdf_text",
                                    ),
                                }
                                for index, unit in enumerate(text_units)
                                if unit.get("content", "").strip()
                            ]
            except Exception as error:
                _record_source_failure(context, "document_fallback", error)

        # A document-only summary must not acquire conversation, memory, or
        # graph facts, even when the caller also supplies a conversation/user.
        logger.info(
            "Assembled document summary: document_id=%s units=%d "
            "source_errors=%d",
            document_id,
            len(context["rag"]),
            len(context["source_errors"]),
        )
        return context

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
        _record_source_failure(context, "rag", error)

    # For an ordinary document question with no semantic matches, try the
    # local full-text fallback. Whole-document summaries use the isolated,
    # page-aware path above instead.
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
            _record_source_failure(context, "full_text", error)

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
            _record_source_failure(context, "conversation", error)

    # Retrieve long-term memories only for a known user.
    if user_id:
        try:
            context["memory"] = _get_memory_context(
                query=query.strip(),
                user_id=user_id,
            )
        except Exception as error:
            # Keep the request usable if memory retrieval fails.
            _record_source_failure(context, "memory", error)

    # Search the knowledge graph for related entities.
    try:
        graph_context = (
            _get_graph_context(query.strip())
            if document_id is None
            else _get_graph_context(query.strip(), document_id=document_id)
        )
        context["knowledge_graph"] = _dedupe_relationships(
            graph_context
        )[:MAX_GRAPH_RELATIONSHIPS]
    except Exception as error:
        # Neo4j may be unavailable or not configured yet.
        _record_source_failure(context, "knowledge_graph", error)

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

            unit_id = result.get("unit_id") or result.get("id")
            if unit_id:
                source_details.append(f"chunk_id={unit_id}")

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
