
# Import the two Context Engine functions we need to test.
import _bootstrap  # noqa: F401 - adds backend/ to sys.path
from app.context_engine.engine import (
    assemble_context,
    format_context,
)


# Run a live integration check against configured services.
def main():
    # Manual integration script: require explicit opt-in before use.
    from manual_guard import require_live_opt_in

    require_live_opt_in(
        "read from PostgreSQL, Pinecone, and Neo4j"
    )

    query = "Summarize the most relevant information in my documents."

    # Assemble context using the central Context Engine.
    context = assemble_context(
        query=query,
        top_k=5,
    )

    # Format the structured context for the LLM.
    formatted_context = format_context(context)

    # Report context source counts.
    print("\n--- CONTEXT SOURCE COUNTS ---")
    print("RAG results:", len(context.get("rag", [])))
    print("Memories:", len(context.get("memory", [])))
    print(
        "Graph relationships:",
        len(context.get("knowledge_graph", [])),
    )
    print(
        "Conversation loaded:",
        context.get("conversation") is not None,
    )

    # Report source errors.
    print("\n--- SOURCE ERRORS ---")
    source_errors = context.get("source_errors", [])

    if source_errors:
        for error in source_errors:
            print(f"{error['source']}: {error['error']}")
    else:
        print("No source errors were reported.")

    # Display a limited context preview.
    print("\n--- FORMATTED CONTEXT PREVIEW ---")
    print(
        formatted_context[:3000]
        or "(No context was assembled.)"
    )

    # Validate the expected structure.
    required_keys = {
        "query",
        "conversation",
        "memory",
        "rag",
        "knowledge_graph",
        "source_errors",
    }

    assert required_keys.issubset(context.keys()), (
        "The Context Engine result is missing expected keys."
    )

    assert isinstance(formatted_context, str), (
        "format_context() must return a string."
    )

    # Report completion.
    print("\nPASS: Context Engine integration call completed.")
    print(
        "Check the source counts and errors above to verify "
        "which services actually worked."
    )


# Run the integration test directly.
if __name__ == "__main__":
    main()
