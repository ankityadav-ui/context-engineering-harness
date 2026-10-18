
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# Formats the vision path supports. Kept in sync with
# app.rag.image_extractor.SUPPORTED_IMAGE_EXTENSIONS.
SUPPORTED_IMAGE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


# --------------------------------------------------
# Configuration
# --------------------------------------------------

# Load the repository-root .env explicitly, so configuration does not
# depend on the process working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

load_dotenv(PROJECT_ROOT / ".env")

# Hold the Gemini client until it is actually needed.
# Building it at import time would stop unrelated API routes from
# starting when Gemini is not configured.
_client = None


def is_configured() -> bool:
    """Report whether GEMINI_API_KEY is set, without connecting."""

    return bool(os.getenv("GEMINI_API_KEY"))


def get_client():
    """Return the Gemini client, creating it on first use."""

    global _client

    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. Set it in the "
                "root .env file to use embedding features."
            )

        _client = genai.Client(api_key=api_key)

    return _client


def __getattr__(name):
    # Preserve the previous module-level name for callers that import
    # `client` directly, while still connecting lazily.
    if name == "client":
        return get_client()

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 1024

# How much of the surrounding page context is embedded with an image.
MAX_IMAGE_CONTEXT_CHARS = 800

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

    response = get_client().models.embed_content(
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

        image_format = metadata.get("image_format")

        parts = [
            f"Image {image_id} extracted from "
            f"page {page_number} of the document."
        ]

        if image_format:
            parts.append(f"Format: {image_format}.")

        # The surrounding page text is what makes the image findable
        # by semantic queries about its content (charts, diagrams,
        # figures). It stays inside the same 1024-dimension text
        # embedding space, so the existing Pinecone index and query
        # path are untouched.
        caption = str(metadata.get("caption") or "").strip()

        if caption:
            parts.append(
                "Surrounding page context: "
                + caption[:MAX_IMAGE_CONTEXT_CHARS]
            )

        return "\n".join(parts)

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
    Resolve stored image references to existing files.

    Supports references such as:
    - images/<document_id>/page_1_img_1.png
    - storage/images/<document_id>/page_1_img_1.png

    Returns None for invalid references, missing files,
    or paths that escape the backend directory.
    """
    if content_unit.get("content_type") != "image":
        return None

    content_ref = content_unit.get("content_ref")

    if not isinstance(content_ref, str) or not content_ref.strip():
        return None

    # Treat stored references as relative paths.
    reference = Path(content_ref)

    # Reject absolute paths and Windows drive paths.
    if reference.is_absolute() or reference.drive:
        return None

    # Normalize references from the retrieval index.
    parts = reference.parts

    if parts and parts[0] == "images":
        reference = Path("storage", *parts)
    elif not parts or parts[0] != "storage":
        return None

    try:
        backend_root = BACKEND_DIR.resolve()
        resolved_path = (backend_root / reference).resolve()

        # Prevent directory traversal outside backend/.
        resolved_path.relative_to(backend_root)

    except (ValueError, OSError):
        return None

    if not resolved_path.is_file():
        return None

    # Only formats the vision path can open are resolvable; a corrupt
    # or truncated file must not reach the multimodal model.
    if resolved_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        return None

    try:
        if resolved_path.stat().st_size == 0:
            return None

        # Cheap integrity check: Pillow verifies the file structure
        # without decoding the full image.
        with Image.open(resolved_path) as image:
            image.verify()

    except Exception:
        return None

    return str(resolved_path)
