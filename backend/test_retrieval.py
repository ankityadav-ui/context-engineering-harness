# Import the function that converts text into an embedding.
from app.rag.embedding_service import generate_embedding

# Import the Pinecone similarity search function.
from app.rag.pinecone_service import search_similar_content


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