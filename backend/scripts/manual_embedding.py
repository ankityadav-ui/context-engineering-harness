"""Manual check: generate a Gemini embedding for a sample sentence.

Calls the paid Gemini embedding API, so it only runs on demand.
Run from backend/:  python scripts/manual_embedding.py
"""

import _bootstrap  # noqa: F401 - adds backend/ to sys.path
from app.rag.embedding_service import generate_embedding

from manual_guard import require_live_opt_in


def generate_sample_embedding() -> None:
    # Generate an embedding for a test sentence.
    embedding = generate_embedding(
        "The patient was prescribed Drug A."
    )

    # Print the number of dimensions in the generated vector.
    print("Embedding dimensions:", len(embedding))

    # Print the first five values to verify a vector was generated.
    print("First five values:", embedding[:5])


if __name__ == "__main__":
    require_live_opt_in("call the Gemini embedding API")

    generate_sample_embedding()
