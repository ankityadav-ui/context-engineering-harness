# This file extracts selectable PDF text
# and identifies pages that need OCR.

import fitz


def is_meaningful_text(text: str) -> bool:
    # Remove unnecessary whitespace.
    cleaned_text = text.strip()

    # Reject very short extracted text.
    if len(cleaned_text) < 20:
        return False

    # Count useful alphanumeric characters.
    alphanumeric_count = sum(
        character.isalnum()
        for character in cleaned_text
    )

    # Accept text when enough meaningful characters exist.
    return alphanumeric_count >= 10


def extract_text_from_pdf(
    document_path,
    document_id: str,
) -> tuple[list[dict], list[int]]:
    # Store successfully extracted text units.
    text_units = []

    # Store page numbers that need OCR.
    pages_needing_ocr = []

    # Open the PDF.
    pdf_document = fitz.open(document_path)

    try:
        # Process every page.
        for page_index, page in enumerate(pdf_document):

            # Convert zero-based index into human page number.
            page_number = page_index + 1

            # Extract selectable text.
            text = page.get_text("text")

            # Clean the extracted text.
            text = text.strip()

            # Check whether useful text was found.
            if is_meaningful_text(text):

                # Create a normal PDF text unit.
                text_unit = {
                    "document_id": document_id,
                    "content_type": "text",
                    "page_number": page_number,
                    "content": text,
                    "source": "pdf_text",
                }

                # Store the normal text unit.
                text_units.append(text_unit)

            else:
                # Mark this page for OCR.
                pages_needing_ocr.append(page_number)

    finally:
        # Close the PDF.
        pdf_document.close()

    # Return both normal text and pages requiring OCR.
    return text_units, pages_needing_ocr