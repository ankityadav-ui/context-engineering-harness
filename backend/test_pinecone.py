# Import the Pinecone index connection.
from app.rag.pinecone_service import index


# Request statistics from the Pinecone index.
stats = index.describe_index_stats()


# Print the index statistics.
print("Pinecone index statistics:")
print(stats)