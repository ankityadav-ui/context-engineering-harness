import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


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


def generate_embedding(text: str) -> list[float]:
    """
    Generate a 1024-dimensional embedding for text.
    """
    if not text.strip():
        raise ValueError(
            "Cannot generate an embedding for empty text."
        )

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=1024
        ),
    )

    return response.embeddings[0].values


def generate_text_embedding(text: str) -> list[float]:
    """
    Generate a text embedding for RAG retrieval.
    """
    return generate_embedding(text)


def get_embedding_text(content_unit: dict) -> str:
    """
    Convert a content unit into the text that will be embedded.
    """
    content_type = content_unit["content_type"]

    if content_type == "text":
        return content_unit["content"]

    if content_type == "table":
        return content_unit["content"]

    if content_type == "image":
        page_number = content_unit["page_number"]

        image_id = content_unit["metadata"].get(
            "image_id",
            "unknown-image",
        )

        return (
            f"Image {image_id} extracted from "
            f"page {page_number} of the document."
        )

    raise ValueError(
        f"Unsupported content type: {content_type}"
    )


def embed_content_units(
    content_units: list[dict],
) -> list[dict]:
    """
    Generate embeddings for all content units.
    """
    embedded_units = []

    for content_unit in content_units:
        embedding_text = get_embedding_text(
            content_unit
        )

        embedding = generate_embedding(
            embedding_text
        )

        embedded_unit = {
            **content_unit,
            "embedding": embedding,
            "embedding_text": embedding_text,
        }

        embedded_units.append(
            embedded_unit
        )

    return embedded_units