# This file extracts selectable PDF text and identifies pages that need OCR.

import re

import pymupdf as fitz  # noqa: F401 (API surface used as fitz)

# PyMuPDF is imported under its historical name `fitz`, which is the
# API used throughout this module.

class PDFProcessingError(RuntimeError):
    """Raised when a PDF cannot be opened or provides no usable text."""

# Required PDF header. A file may start with this and still not be a
# real PDF, so it is only the first gate; the final gate is openability.
PDF_HEADER = b"%PDF-"

OCR_UNIT_MAX_CHARS = 20000
OCR_UNIT_MIN_CHARS = 3

# OCR sometimes renders a link on its own: a footer, a template
# artifact, or a screenshot URL. A unit that is nothing but a URL is not
# document content, so it is dropped whole. A URL embedded in prose is
# left untouched.
_URL_ONLY_RE = re.compile(
    r"^(?:https?://|www\.)\S+$",
    re.IGNORECASE,
)


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


def _is_pdf_signature(data: bytes) -> bool:
    """Return True when bytes begin with a PDF header."""
    return bool(data.startswith(PDF_HEADER))


def _open_pdf_document(document_path) -> fitz.Document:
    """Open a PDF and reject files that are not real PDFs.

    ``PyMuPDF`` raises ``EmptyFileError``/`FileDataError` for empty or
    truncated payloads, and silently decodes some other stream types.
    We keep the signature check and the openability check separate so the
    caller can tell a renamed non-PDF from a valid PDF with no readable
    text.
    """
    with open(document_path, "rb") as handle:
        header = handle.read(16)

    if not _is_pdf_signature(header):
        raise PDFProcessingError(
            "Not a valid PDF: the file does not start with a PDF header."
        )

    try:
        document = fitz.open(document_path)
    except (fitz.EmptyFileError, fitz.FileDataError, ValueError) as error:
        raise PDFProcessingError(
            "Not a valid PDF: the file is empty or truncated."
        ) from error
    except Exception as error:
        raise PDFProcessingError(
            "Not a valid PDF: the file cannot be opened."
        ) from error

    # A signature with no actual PDF object table is not usable.
    try:
        catalog = document.pdf_catalog()
    except Exception as error:
        document.close()
        raise PDFProcessingError(
            "Not a valid PDF: the file cannot be opened."
        ) from error
    if catalog is None:
        document.close()
        raise PDFProcessingError(
            "Not a valid PDF: the file has an invalid PDF structure."
        )

    # A header that opens to zero pages is a truncated or decoy file, not
    # a real document. A genuine one-page blank PDF still opens here and
    # is reported later as having no retrievable content.
    if document.page_count == 0:
        document.close()
        raise PDFProcessingError(
            "Not a valid PDF: the file contains no pages."
        )

    return document


def is_pdf_openable(document_path) -> bool:
    """Report whether a file is a real, openable PDF without reading it.

    The signature check and the openability check are deliberately
    separate, so callers can distinguish a renamed file from a valid
    document and report an actionable error instead of a traceback.
    """
    try:
        with open(document_path, "rb") as handle:
            header = handle.read(16)
    except OSError:
        return False

    if not _is_pdf_signature(header):
        return False

    try:
        document = fitz.open(document_path)
    except (fitz.EmptyFileError, fitz.FileDataError, ValueError):
        return False
    except Exception:
        return False

    try:
        return (
            document.page_count > 0
            and document.pdf_catalog() is not None
        )
    finally:
        document.close()


def get_pdf_page_count(document_path) -> int:
    """Return the number of pages in a validated PDF."""

    document = _open_pdf_document(document_path)
    try:
        return document.page_count
    finally:
        document.close()


def _page_has_meaningful_text(page) -> bool:
    try:
        text = page.get_text("text") or ""
    except Exception:
        # Corrupt page content or an unsupported render: treat as
        # "no readable text" so the page can be handed to OCR.
        return False

    return is_meaningful_text(text)


def _sanitize_ocr_unit(page_number: int, text: str) -> dict | None:
    """Clean one page of OCR output, or return None when it is noise.

    Empty output, a bare URL, text shorter than the minimum unit size,
    and punctuation-only noise are all rejected so a page the OCR engine
    could not really read is never indexed as successful text.
    """

    if not isinstance(text, str):
        return None

    cleaned_text = text.strip()

    if not cleaned_text:
        return None

    # Reject headless / placeholder URLs the OCR layer sometimes emits
    # after template rendering or repeated PDF re-saves.
    if _URL_ONLY_RE.match(cleaned_text):
        return None

    if (
        len(cleaned_text) < OCR_UNIT_MIN_CHARS
        or len(cleaned_text) > OCR_UNIT_MAX_CHARS
    ):
        return None

    # Punctuation-only OCR noise is not retrievable content either.
    if not any(
        character.isalnum() for character in cleaned_text
    ):
        return None

    return {
        "document_id": "",
        "content_type": "text",
        "page_number": page_number,
        "content": cleaned_text,
        "source": "ocr",
    }


def extract_text_from_pdf(
    document_path,
    document_id: str,
) -> tuple[list[dict], list[int]]:
    # Store successfully extracted text units.
    text_units = []

    # Store page numbers that need OCR.
    pages_needing_ocr = []

    # Open the PDF and reject renamed or truncated payloads.
    pdf_document = _open_pdf_document(document_path)

    try:
        # Process every page.
        for page_index, page in enumerate(pdf_document):
            # Convert zero-based index into human page number.
            page_number = page_index + 1

            if _page_has_meaningful_text(page):
                # Extract selectable text.
                text = page.get_text("text").strip()

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
                # Not enough readable text: hand this page to OCR. The
                # sanitizer applies to OCR output, not to this decision,
                # so a page with a completely empty text layer (for
                # example a scan or a solid-color image) is still queued
                # for OCR instead of being skipped as "no text".
                pages_needing_ocr.append(page_number)
    finally:
        # Always close the PDF.
        pdf_document.close()

    # Return both normal text and pages requiring OCR.
    return text_units, pages_needing_ocr
