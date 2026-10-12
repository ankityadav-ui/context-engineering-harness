import io
import os
from pathlib import Path

import fitz
import pytesseract
from PIL import Image


TESSERACT_PATH = os.getenv(
    "TESSERACT_PATH",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def extract_text_with_ocr(
    document_path: Path,
    document_id: str,
    pages: list[int],
) -> list[dict]:
    # Store OCR-generated text units.
    text_units = []

    # Open the PDF.
    pdf_document = fitz.open(document_path)

    try:
        # Process only pages that need OCR.
        for page_number in pages:

            # Convert human page number into zero-based index.
            page_index = page_number - 1

            # Get the requested page.
            page = pdf_document[page_index]

            # Render the page as a higher-resolution image.
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(2, 2),
                alpha=False,
            )

            # Convert the rendered page into PNG bytes.
            image_bytes = pixmap.tobytes("png")

            # Convert PNG bytes into a Pillow image.
            image = Image.open(
                io.BytesIO(image_bytes)
            )

            # Run OCR.
            text = pytesseract.image_to_string(
                image
            )

            # Clean OCR output.
            text = text.strip()

            # Skip pages where OCR found nothing.
            if not text:
                continue

            # Create a normal text content unit.
            text_unit = {
                "document_id": document_id,
                "content_type": "text",
                "page_number": page_number,
                "content": text,
                "source": "ocr",
            }

            # Add the OCR result.
            text_units.append(text_unit)

    finally:
        # Close the PDF.
        pdf_document.close()

    # Return OCR-generated text units.
    return text_units