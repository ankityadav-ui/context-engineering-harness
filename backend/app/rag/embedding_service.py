
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not configured."
    )

client = genai.Client(
    api_key=GEMINI_API_KEY
)

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 1024

# backend/app/rag/embedding_service.py -> backend/
BACKEND_DIR = Path(__file__).resolve().parents[2]


# --------------------------------------------------
# Generate embeddings
# --------------------------------------------------

def generate_embedding(text: str) -> list[float]:
    """
    Generate a 1024-dimensional Gemini embedding.
    """
    if not text or not text.strip():
        raise ValueError(
            "Cannot generate an embedding for empty text."
        )

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSION
        ),
    )

    if not response.embeddings:
        raise RuntimeError(
            "Gemini returned no embedding."
        )

    values = response.embeddings[0].values

    if values is None or len(values) != EMBEDDING_DIMENSION:
        raise RuntimeError(
            "Gemini returned an embedding with an unexpected dimension."
        )

    return list(values)


def generate_text_embedding(text: str) -> list[float]:
    """
    Generate a query embedding for semantic RAG retrieval.
    """
    return generate_embedding(text)


# --------------------------------------------------
# Convert content units to embedding text
# --------------------------------------------------

def get_embedding_text(content_unit: dict) -> str:
    """
    Convert a text, table, or image content unit into
    text suitable for semantic embedding.
    """
    content_type = content_unit.get("content_type")

    if content_type == "text":
        content = content_unit.get("content", "")

        if not content.strip():
            raise ValueError(
                "Text content unit has empty content."
            )

        return content

    if content_type == "table":
        content = content_unit.get("content", "")

        if not content.strip():
            raise ValueError(
                "Table content unit has empty content."
            )

        return content

    if content_type == "image":
        page_number = content_unit.get(
            "page_number", "unknown"
        )

        metadata = content_unit.get("metadata") or {}

        image_id = metadata.get(
            "image_id", "unknown-image"
        )

        return (
            f"Image {image_id} extracted from "
            f"page {page_number} of the document."
        )

    raise ValueError(
        f"Unsupported content type: {content_type}"
    )


# --------------------------------------------------
# Embed content units
# --------------------------------------------------

def embed_content_units(
    content_units: list[dict],
) -> list[dict]:
    """
    Generate embeddings for content units while preserving
    their original metadata and content references.
    """
    embedded_units = []

    for content_unit in content_units:
        embedding_text = get_embedding_text(content_unit)
        embedding = generate_embedding(embedding_text)

        embedded_unit = {
            **content_unit,
            "embedding": embedding,
            "embedding_text": embedding_text,
        }

        embedded_units.append(embedded_unit)

    return embedded_units


# --------------------------------------------------
# Resolve extracted image paths
# --------------------------------------------------

def resolve_image_path(
    content_unit: dict,
) -> str | None:
    """
    Resolve an image content_ref such as
    storage/images/<document_id>/page_5_img_1.png
    to an existing local file.

    Returns None if the reference is missing, invalid,
    or points outside the backend directory.
    """
    if content_unit.get("content_type") != "image":
        return None

    content_ref = content_unit.get("content_ref")

    if not isinstance(content_ref, str) or not content_ref.strip():
        return None

    # Reject absolute paths and Windows drive paths.
    reference = Path(content_ref)

    if reference.is_absolute() or reference.drive:
        return None

    try:
        resolved_path = (
            BACKEND_DIR / reference
        ).resolve()

        # Ensure the resolved path stays inside backend/.
        resolved_path.relative_to(BACKEND_DIR.resolve())

    except (ValueError, OSError):
        return None

    if not resolved_path.is_file():
        return None

    return str(resolved_path)
