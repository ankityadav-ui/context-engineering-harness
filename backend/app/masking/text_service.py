
# Import Path to validate the PDF file location.
from pathlib import Path

# Reuse the existing RAG document extraction pipeline.
from app.rag.document_service import process_document


def extract_text_for_masking(
    document_path: str,
    document_id: str,
) -> dict:
    # Convert the supplied file path into a Path object.
    pdf_path = Path(document_path)

    # Stop if the document does not exist.
    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"Document not found: {pdf_path}"
        )

    # Reuse normal text extraction and OCR fallback.
    # Note: the existing process_document function also
    # extracts tables/images and indexes content in Pinecone.
    content_units, ocr_pages, table_units, image_units = (
        process_document(
            document_path=pdf_path,
            document_id=document_id,
        )
    )

    # Keep only text units for the masking pipeline.
    text_units = [
        unit
        for unit in content_units
        if unit.get("content_type") == "text"
        and unit.get("content", "").strip()
    ]

    # Sort extracted text in page order.
    text_units.sort(
        key=lambda unit: unit.get("page_number", 0)
    )

    # Combine page text while preserving page boundaries.
    combined_text = "\n\n".join(
        f"[Page {unit.get('page_number', 'Unknown')}]\n"
        f"{unit['content']}"
        for unit in text_units
    )

    # Return text and basic extraction metadata.
    return {
        "document_id": document_id,
        "text": combined_text,
        "pages_with_ocr": ocr_pages,
        "text_unit_count": len(text_units),
        "table_unit_count": len(table_units),
        "image_unit_count": len(image_units),
    }
