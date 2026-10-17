"""Manual integration check for the multimodal RAG pipeline.

Verifies:
1. Pinecone retrieval works.
2. Text and table content can be retrieved.
3. Image content can be retrieved.
4. Retrieved image references resolve to real files.
5. The multimodal LLM can receive the actual images.

Requires live services (Pinecone, Gemini, and the vision model).
Run from backend/:  python scripts/manual_multimodal_rag.py
"""

from pathlib import Path

import _bootstrap  # noqa: F401 - adds backend/ to sys.path
from app.context_engine.engine import assemble_context
from app.rag.embedding_service import resolve_image_path
from app.llm import generate_multimodal_response

from manual_guard import require_live_opt_in


def retrieve(document_id: str) -> list[dict]:
    """Retrieve multimodal content from Pinecone for a document."""

    context = assemble_context(
        query="What information is shown in this document?",
        top_k=10,
        document_id=document_id,
    )

    results = context["rag"]

    print("\n=== RETRIEVAL RESULTS ===")

    for result in results:
        print(
            f"Type: {result['content_type']} | "
            f"Page: {result['page_number']} | "
            f"Score: {result['score']}"
        )

    return results


def resolve_images(results: list[dict]) -> list[str]:
    """Resolve image content units to existing files on disk.

    ``resolve_image_path`` expects the full content-unit dictionary and
    returns None for missing, invalid, or unsafe references.
    """

    print("\n=== IMAGE RESOLUTION ===")

    image_paths = []

    for result in results:
        if result.get("content_type") != "image":
            continue

        # Pass the content unit itself, not a bare reference string.
        image_path = resolve_image_path(result)

        if image_path and Path(image_path).is_file():
            image_paths.append(image_path)
            print(f"Image: {image_path}")
        else:
            print(
                "Skipping unresolvable image reference: "
                f"{result.get('content_ref')!r}"
            )

    print(f"Resolved images: {len(image_paths)}")

    return image_paths


def run_multimodal_llm(image_paths: list[str]) -> None:
    """Send resolved images and a question to the vision model."""

    if not image_paths:
        print(
            "\nNo images were resolved. "
            "Skipping the multimodal LLM check."
        )
        return

    print("\n=== MULTIMODAL LLM TEST ===")

    answer = generate_multimodal_response(
        message=(
            "Look at the provided document image. "
            "Describe what you can see in the image. "
            "Do not invent information."
        ),
        image_paths=image_paths[:3],
    )

    print("\nVision model response:")
    print(answer)

    assert answer.strip()


if __name__ == "__main__":
    require_live_opt_in(
        "query Pinecone and send images to the vision model"
    )

    document_id = input(
        "Enter document ID: "
    ).strip()

    if not document_id:
        raise ValueError(
            "Document ID cannot be empty."
        )

    results = retrieve(
        document_id=document_id
    )

    resolved_images = resolve_images(
        results=results
    )

    run_multimodal_llm(
        image_paths=resolved_images
    )

    print("\n=== ALL CHECKS COMPLETED ===")
