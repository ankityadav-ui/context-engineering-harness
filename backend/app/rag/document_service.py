# This file handles document storage and processing.

from pathlib import Path
from uuid import uuid4
import shutil

from fastapi import UploadFile
# Import the image extraction function.
from app.rag.image_extractor import (
    extract_images_from_pdf,
)
from app.rag.text_extractor import (
    extract_text_from_pdf,
)

# Import the table extraction function.
from app.rag.table_extractor import (
    extract_tables_from_pdf,
)
from app.rag.ocr import (
    extract_text_with_ocr,
)
# Import the function that generates embeddings.
from app.rag.embedding_service import embed_content_units

# Import the function that stores vectors in Pinecone.
from app.rag.pinecone_service import upsert_content_units

# Import the function that converts extracted content into unified units.
from app.rag.chunker import build_content_units


# Find the backend directory.
BASE_DIR = Path(__file__).resolve().parents[2]


# Define the original document storage directory.
DOCUMENT_STORAGE_DIR = (
    BASE_DIR / "storage" / "documents"
)


# Define the extracted image storage directory.
IMAGE_STORAGE_DIR = (
    BASE_DIR / "storage" / "images"
)


def save_document(
    file: UploadFile,
) -> tuple[str, str, Path]:
    # Make sure the storage directory exists.
    DOCUMENT_STORAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Get the original filename.
    original_filename = file.filename or "unknown.pdf"

    # Extract the extension.
    extension = Path(
        original_filename
    ).suffix.lower()

    # Only accept PDFs.
    if extension != ".pdf":
        raise ValueError(
            "Only PDF documents are supported."
        )

    # Generate a unique document ID.
    document_id = str(uuid4())

    # Use the generated ID as the stored filename.
    stored_filename = f"{document_id}.pdf"

    # Create the final storage path.
    destination = (
        DOCUMENT_STORAGE_DIR / stored_filename
    )

    # Save the uploaded PDF.
    with destination.open("wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer,
        )

    # Return document information.
    return (
        document_id,
        original_filename,
        destination,
    )


# Process a PDF through extraction, chunking, embedding, and indexing.
def process_document(
    document_path: Path,
    document_id: str,
) -> tuple[list[dict], list[int], list[dict], list[dict]]:
    # Extract normal PDF text and identify pages that need OCR.
    text_units, pages_needing_ocr = extract_text_from_pdf(
        document_path=document_path,
        document_id=document_id,
    )

    # Run OCR only on pages where normal text extraction was insufficient.
    if pages_needing_ocr:
        ocr_units = extract_text_with_ocr(
            document_path=document_path,
            document_id=document_id,
            pages=pages_needing_ocr,
        )

        # Add OCR-generated text to the extracted text collection.
        text_units.extend(ocr_units)

    # Extract structured tables from the PDF.
    table_units = extract_tables_from_pdf(
        document_path=document_path,
        document_id=document_id,
    )

    # Extract embedded images from the PDF.
    image_units = extract_images_from_pdf(
        document_path=document_path,
        document_id=document_id,
        image_storage_dir=IMAGE_STORAGE_DIR,
    )

    # Convert text, tables, and images into unified content units.
    content_units = build_content_units(
        text_units=text_units,
        table_units=table_units,
        image_units=image_units,
    )

    # Generate embeddings for all content units.
    embedded_units = embed_content_units(
        content_units
    )

    # Store the generated vectors and metadata in Pinecone.
    upsert_content_units(
        embedded_units
    )

    # Return the processed content and extraction statistics.
    return (
        content_units,
        pages_needing_ocr,
        table_units,
        image_units,
    )