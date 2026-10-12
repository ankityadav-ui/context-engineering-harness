from app.rag.embedding_service import (
    generate_text_embedding,
)

from app.rag.pinecone_service import (
    search_similar_content,
)


def retrieve_context(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
) -> list[dict]:
    """
    Convert the user's text query into a multimodal
    embedding and search Pinecone.
    """
    query_embedding = generate_text_embedding(
        query
    )

    results = search_similar_content(
        query_embedding=query_embedding,
        top_k=top_k,
        document_id=document_id,
    )

    return results