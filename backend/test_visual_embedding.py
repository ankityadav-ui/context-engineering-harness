from pathlib import Path

from app.rag.embedding_service import (
    generate_image_embedding,
)


def main():
    """
    Ask the user for an image path and generate
    a true visual embedding for that image.
    """
    image_path = input(
        "Enter image path: "
    ).strip()

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image does not exist: {path}"
        )

    embedding = generate_image_embedding(
        str(path)
    )

    print(
        "\nVisual embedding generated successfully."
    )

    print(
        f"Embedding dimensions: {len(embedding)}"
    )

    print(
        "First 10 values:"
    )

    print(
        embedding[:10]
    )


if __name__ == "__main__":
    main()