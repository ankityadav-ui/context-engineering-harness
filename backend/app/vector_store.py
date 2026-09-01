import chromadb

from .embeddings import create_embeddings


# Persistent ChromaDB client
client = chromadb.PersistentClient(
    path="./chroma_db"
)


# Collection containing document chunks
collection = client.get_or_create_collection(
    name="document_chunks"
)


def add_chunks(
    chunks: list[dict],
):
    """
    Add multiple document chunks to ChromaDB.

    Each chunk must contain:
        id
        text
        case_id
        document_id
        chunk_index
    """

    if not chunks:
        return

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = create_embeddings(texts)

    ids = [
        str(chunk["id"])
        for chunk in chunks
    ]

    metadatas = [
        {
            "case_id": chunk["case_id"],
            "document_id": chunk["document_id"],
            "chunk_index": chunk["chunk_index"],
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def delete_chunks_by_document(
    document_id: int,
):
    """
    Delete all vectors belonging to a specific document.
    """

    results = collection.get(
        where={"document_id": document_id},
    )

    if results["ids"]:
        collection.delete(
            ids=results["ids"],
        )


def get_collection_count():
    """
    Return the number of vectors stored.
    """

    return collection.count()

def search_chunks(
    query: str,
    case_id: int,
    top_k: int = 5,
):
    """
    Search for the most relevant chunks inside a case.
    """

    query_embedding = create_embeddings([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={
            "case_id": case_id
        },
    )

    return results