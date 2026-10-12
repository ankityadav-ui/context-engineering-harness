
# Import the existing RAG retrieval function.
from app.rag.rag_service import retrieve_context


# Define a helper that retrieves conversation information.
def _get_conversation_context(
    conversation_id: int,
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

        # Return the summary and recent messages together.
        return {
            "summary": summary,
            "recent_messages": window.get(
                "recent_messages", []
            ),
            "older_message_count": window.get(
                "older_message_count", 0
            ),
        }


# Define a helper that retrieves long-term memories.
def _get_memory_context(
    query: str,
    user_id: str,
    limit: int = 5,
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
    conversation_id: int | None = None,
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

    # Retrieve document evidence from Pinecone.
    try:
        context["rag"] = retrieve_context(
            query=query.strip(),
            top_k=top_k,
            document_id=document_id,
        )
    except Exception as error:
        # Preserve other sources if document retrieval fails.
        context["source_errors"].append({
            "source": "rag",
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
        context["knowledge_graph"] = _get_graph_context(
            query.strip()
        )
    except Exception as error:
        # Neo4j may be unavailable or not configured yet.
        context["source_errors"].append({
            "source": "knowledge_graph",
            "error": str(error),
        })

    # Return the combined result to the calling endpoint.
    return context


# Define a function that turns structured context into readable text.
def format_context(context: dict) -> str:
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
            content = memory.get("content", "")
            memory_lines.append(
                f"- [{memory_type}] {content}"
            )

        sections.append(
            "RELEVANT LONG-TERM MEMORIES:\n"
            + "\n".join(memory_lines)
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
            content = result.get("content", "")
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

    # Join all available sources into one context string.
    return "\n\n".join(sections)
