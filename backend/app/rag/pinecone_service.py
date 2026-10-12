import os

from dotenv import load_dotenv
from pinecone import Pinecone


# Load environment variables from the project's .env file.
load_dotenv()


# Read the Pinecone API key.
PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY"
)


# Read the Pinecone index name.
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "context-engineering-harness",
)


# Keep new multimodal vectors isolated from old vectors.
PINECONE_NAMESPACE = "multimodal-v2"


# Stop the application if the API key is missing.
if not PINECONE_API_KEY:
    raise RuntimeError(
        "PINECONE_API_KEY is not configured."
    )


# Create the Pinecone client.
pinecone_client = Pinecone(
    api_key=PINECONE_API_KEY
)


# Connect to the existing Pinecone index.
index = pinecone_client.Index(
    PINECONE_INDEX_NAME
)


def upsert_content_units(
    embedded_units: list[dict],
) -> int:
    """
    Store multimodal embeddings in the dedicated
    Pinecone namespace.
    """
    vectors = []

    for unit in embedded_units:
        metadata = {
            "document_id": unit["document_id"],
            "content_type": unit["content_type"],
            "page_number": unit["page_number"],
            "content": unit.get("content", ""),
            "content_ref": unit.get("content_ref") or "",
            "source": unit["metadata"].get(
                "source",
                "unknown",
            ),
        }

        if unit["content_type"] == "table":
            metadata["table_number"] = (
                unit["metadata"].get(
                    "table_number",
                    0,
                )
            )

        if unit["content_type"] == "image":
            metadata["image_id"] = (
                unit["metadata"].get(
                    "image_id",
                    "",
                )
            )

        vector = {
            "id": unit["id"],
            "values": unit["embedding"],
            "metadata": metadata,
        }

        vectors.append(vector)

    if not vectors:
        return 0

    index.upsert(
        vectors=vectors,
        namespace=PINECONE_NAMESPACE,
    )

    return len(vectors)


def search_similar_content(
    query_embedding: list[float],
    top_k: int = 5,
    document_id: str | None = None,
) -> list[dict]:
    """
    Search only the new multimodal namespace.
    """
    search_filter = None

    if document_id:
        search_filter = {
            "document_id": {
                "$eq": document_id
            }
        }

    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=PINECONE_NAMESPACE,
        filter=search_filter,
    )

    results = []

    for match in response.matches:
        metadata = match.metadata or {}

        result = {
            "id": match.id,
            "score": match.score,
            "document_id": metadata.get(
                "document_id"
            ),
            "content_type": metadata.get(
                "content_type"
            ),
            "page_number": metadata.get(
                "page_number"
            ),
            "content": metadata.get(
                "content",
                "",
            ),
            "content_ref": metadata.get(
                "content_ref",
                "",
            ),
            "source": metadata.get(
                "source",
                "unknown",
            ),
        }

        if "table_number" in metadata:
            result["table_number"] = (
                metadata["table_number"]
            )

        if "image_id" in metadata:
            result["image_id"] = (
                metadata["image_id"]
            )

        results.append(result)

    return results