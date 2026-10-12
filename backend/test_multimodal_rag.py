"""
Manual integration tests for the multimodal RAG pipeline.

These tests verify:
1. Pinecone retrieval works.
2. Text and table content can be retrieved.
3. Image content can be retrieved.
4. Retrieved image paths can be resolved.
5. The multimodal LLM can receive the actual image.
"""

from pathlib import Path

from app.context_engine.engine import (
    assemble_context,
    resolve_image_path,
)
from app.llm import generate_multimodal_response


def test_retrieval(document_id: str):
    """
    Retrieve multimodal content from Pinecone
    using a natural-language query.
    """
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


def test_image_resolution(results: list[dict]):
    """
    Verify that every retrieved image reference
    points to a real file on disk.
    """
    print("\n=== IMAGE RESOLUTION ===")

    image_count = 0

    for result in results:
        if result["content_type"] != "image":
            continue

        image_count += 1

        image_path = resolve_image_path(
            result["content_ref"]
        )

        exists = Path(image_path).exists()

        print(f"Image: {image_path}")
        print(f"Exists: {exists}")

        assert exists, (
            f"Image file does not exist: {image_path}"
        )

    print(f"Images found: {image_count}")


def test_multimodal_llm(results: list[dict]):
    """
    Send retrieved images together with a question
    to the vision-capable Qwen model.
    """
    image_paths = []

    for result in results:
        if result["content_type"] != "image":
            continue

        image_path = resolve_image_path(
            result["content_ref"]
        )

        if Path(image_path).exists():
            image_paths.append(image_path)

    if not image_paths:
        print(
            "\nNo image was retrieved. "
            "Skipping multimodal LLM test."
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

    print("\nQwen response:")
    print(answer)

    assert answer.strip()


if __name__ == "__main__":
    """
    Ask for the document ID interactively so that
    we can test any previously uploaded document.
    """
    document_id = input(
        "Enter document ID: "
    ).strip()

    if not document_id:
        raise ValueError(
            "Document ID cannot be empty."
        )

    results = test_retrieval(
        document_id=document_id
    )

    test_image_resolution(
        results=results
    )

    test_multimodal_llm(
        results=results
    )

    print("\n=== ALL TESTS COMPLETED ===")