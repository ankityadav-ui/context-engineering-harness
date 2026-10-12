# Import the embedding function from the RAG embedding service.
from app.rag.embedding_service import generate_embedding


# Generate an embedding for a test sentence.
embedding = generate_embedding(
    "The patient was prescribed Drug A."
)


# Print the number of dimensions in the generated vector.
print("Embedding dimensions:", len(embedding))


# Print the first five values to verify that a vector was generated.
print("First five values:", embedding[:5])