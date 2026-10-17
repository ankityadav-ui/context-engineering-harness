"""Manual check: semantic search against the live Pinecone index.

Calls the paid Gemini embedding API and queries Pinecone, so it only
runs on demand.
Run from backend/:  python scripts/manual_retrieval.py
"""

import _bootstrap  # noqa: F401 - adds backend/ to sys.path
from app.rag.embedding_service import generate_embedding
from app.rag.pinecone_service import search_similar_content

from manual_guard import require_live_opt_in


def search_sample_query() -> None:
    # Define the question we want to search for.
    query = "What medicine was prescribed?"

    # Convert the question into an embedding vector.
    query_embedding = generate_embedding(
        query
    )

    # Search Pinecone for the most relevant content.
    results = search_similar_content(
        query_embedding=query_embedding,
        top_k=5,
    )

    # Display every retrieved result.
    for result in results:
        # Print the similarity score.
        print(
            "Score:",
            result["score"],
        )

        # Print the content type.
        print(
            "Type:",
            result["content_type"],
        )

        # Print the page number.
        print(
            "Page:",
            result["page_number"],
        )

        # Print the retrieved content.
        print(
            "Content:",
            result["content"],
        )

        # Print the original content reference if available.
        print(
            "Reference:",
            result["content_ref"],
        )

        # Print a separator between results.
        print("-" * 50)


if __name__ == "__main__":
    require_live_opt_in("embed a query and search the live Pinecone index")

    search_sample_query()
