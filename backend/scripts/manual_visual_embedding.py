"""Manual check of the embedding path used for extracted images.

IMPORTANT LIMITATION
--------------------
This project does NOT implement true visual (image) embeddings.
There is no ``generate_image_embedding`` function. Extracted images are
embedded through a generated *text description* produced by
``app.rag.embedding_service.get_embedding_text`` (for example,
"Image <id> extracted from page <n> of the document.").

This script demonstrates that supported text-description embedding path
and keeps the limitation explicit, rather than pretending image
embeddings exist.

Calls the paid Gemini embedding API, so it only runs on demand.
Run from backend/:  python scripts/manual_visual_embedding.py
"""

import _bootstrap  # noqa: F401 - adds backend/ to sys.path
from app.rag.embedding_service import (
    generate_embedding,
    get_embedding_text,
)

from manual_guard import require_live_opt_in


def embed_image_description() -> None:
    # Build a representative image content unit.
    image_id = input(
        "Enter an image content-unit id (or press Enter for a default): "
    ).strip() or "image-1"

    content_unit = {
        "content_type": "image",
        "page_number": 1,
        "metadata": {
            "image_id": image_id,
        },
    }

    # Build the same description that is used during indexing.
    embedding_text = get_embedding_text(content_unit)

    print(
        "\nNOTE: this is a TEXT embedding of the image description, "
        "not a true visual embedding."
    )
    print("Embedding text:", embedding_text)

    # Generate the embedding for the description text.
    embedding = generate_embedding(embedding_text)

    print(
        "\nEmbedding generated successfully."
    )
    print(
        f"Embedding dimensions: {len(embedding)}"
    )
    print("First 10 values:")
    print(embedding[:10])


if __name__ == "__main__":
    require_live_opt_in("call the Gemini embedding API")

    embed_image_description()
