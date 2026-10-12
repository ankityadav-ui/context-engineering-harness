# Import the embedding function.
from app.rag.embedding_service import generate_embedding

# Import the Pinecone indexing function.
from app.rag.pinecone_service import upsert_content_units


# Create a sample content unit.
content_unit = {
    "id": "test-unit-001",
    "document_id": "test-document-001",
    "content_type": "text",
    "page_number": 1,
    "content": "The patient was prescribed Drug A.",
    "content_ref": None,
    "metadata": {
        "source": "test",
    },
}


# Generate an embedding for the sample content.
embedding = generate_embedding(
    content_unit["content"]
)


# Attach the generated embedding to the content unit.
embedded_unit = {
    **content_unit,
    "embedding": embedding,
}


# Store the vector in Pinecone.
count = upsert_content_units(
    [embedded_unit]
)


# Print the number of vectors stored.
print("Vectors indexed:", count)