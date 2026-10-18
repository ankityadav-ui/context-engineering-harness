"""Tesseract OCR for scanned PDF pages.

Configuration (environment, read at call time so tests and operators
can change it without restarting imports):

- ``TESSERACT_PATH``   absolute path of the tesseract executable.
- ``TESSERACT_TIMEOUT`` per-page OCR timeout in seconds (default 45).

Failure model:

- ``PDFProcessingError`` - the PDF is not a real, openable PDF. Raised
  before any page work; callers convert it to a 422 and stop the upload.
- ``OCRUnavailableError`` - the Tesseract binary is missing or not
  executable. Raised before page processing, with a clear actionable
  message (configured path + environment variable name). Callers treat
  this as a service error, not as a silent skip.
- Per-page failures (timeout, unrenderable page, corrupt content) are
  logged without page text and skipped, so one bad page never crashes
  an entire ingestion run.

Temporary files: PyMuPDF renders pages in memory and pytesseract uses
its own NamedTemporary files which it removes; this module never writes
extra files itself.
"""

import io
import logging
import os
from pathlib import Path

# PyMuPDF is imported under its historical name `fitz`, which is the
# API used throughout this module.
import pymupdf as fitz
import pytesseract
from PIL import Image

# Import the same exception used by the PDF ingestion path so OCR and
# text extraction surface identical, client-safe errors.
from app.rag.text_extractor import (
    PDFProcessingError,
    _sanitize_ocr_unit,
)

logger = logging.getLogger("app.rag.ocr")

# Fallback for environments that never set TESSERACT_PATH: the default
# Windows installer location.
DEFAULT_TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

DEFAULT_OCR_TIMEOUT_SECONDS = 45


class OCRError(RuntimeError):
    """Base class for OCR failures with client-safe messages."""


class OCRUnavailableError(OCRError):
    """The Tesseract binary is missing or not executable."""


def get_tesseract_path() -> str:
    """Return the configured Tesseract executable path.

    Read from the environment on every call so the path can be
    overridden per process (tests, deployments) without re-importing.
    """

    return (
        os.getenv("TESSERACT_PATH", "").strip()
        or DEFAULT_TESSERACT_PATH
    )


def get_ocr_timeout_seconds() -> int:
    """Return the per-page OCR timeout, clamped to a sane range."""

    raw = os.getenv("TESSERACT_TIMEOUT", "").strip()

    try:
        timeout = int(raw) if raw else DEFAULT_OCR_TIMEOUT_SECONDS
    except ValueError:
        timeout = DEFAULT_OCR_TIMEOUT_SECONDS

    return max(5, min(timeout, 600))


# Kept for backwards compatibility with code that reads the module
# attribute directly; get_tesseract_path() is authoritative.
TESSERACT_PATH = get_tesseract_path()

pytesseract.pytesseract.tesseract_cmd = get_tesseract_path()


def is_configured() -> bool:
    """Report whether the configured Tesseract executable exists.

    Only the presence of the binary is checked here; OCR itself is only
    attempted for pages that need it during document ingestion.
    """

    return Path(get_tesseract_path()).is_file()


def _ensure_available() -> None:
    """Raise a clear error when OCR is requested without the binary."""

    path = get_tesseract_path()

    if not Path(path).is_file():
        raise OCRUnavailableError(
            "Tesseract OCR is not available: no executable was found "
            "at the configured path. Set TESSERACT_PATH in the root "
            ".env file to the full path of tesseract.exe "
            f"(current configuration points to: {path})."
        )


def extract_text_with_ocr(
    document_path: Path,
    document_id: str,
    pages: list[int],
) -> list[dict]:
    # Fail fast with a clear message instead of one error per page.
    try:
        _ensure_available()
    except OCRUnavailableError:
        # Re-raise so callers always see the same exception type, even
        # when the OCR module is the first thing anyone touches.
        raise
    except PDFProcessingError as error:
        # A non-PDF or unreadable PDF is a document problem, not an OCR
        # problem. Surface the ingestion error directly.
        raise error

    pytesseract.pytesseract.tesseract_cmd = get_tesseract_path()

    timeout = get_ocr_timeout_seconds()

    # Store OCR-generated text units.
    text_units = []

    # Open the PDF.
    pdf_document = fitz.open(document_path)

    try:
        # Process only pages that need OCR.
        for page_number in pages:

            # Convert human page number into zero-based index.
            page_index = page_number - 1

            if not 0 <= page_index < pdf_document.page_count:
                logger.warning(
                    "OCR skipped page %s of document %s: page index "
                    "out of range.",
                    page_number,
                    document_id,
                )
                continue

            # Get the requested page.
            page = pdf_document[page_index]

            try:
                # Render the page as a higher-resolution image.
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    alpha=False,
                )

                # Convert the rendered page into PNG bytes.
                image_bytes = pixmap.tobytes("png")

                # Convert PNG bytes into a Pillow image.
                image = Image.open(io.BytesIO(image_bytes))

                # Run OCR with a per-page timeout so a pathological
                # page cannot hang the whole ingestion run.
                text = pytesseract.image_to_string(
                    image,
                    timeout=timeout,
                )

            except pytesseract.TesseractNotFoundError as error:
                # The binary disappeared between the check and the run
                # (or is not executable). Fail loudly and clearly.
                raise OCRUnavailableError(
                    "Tesseract OCR became unavailable during "
                    "extraction. Verify TESSERACT_PATH points to a "
                    "working tesseract executable."
                ) from error

            except RuntimeError:
                # pytesseract raises RuntimeError on timeout.
                logger.warning(
                    "OCR timed out after %ss on page %s of document "
                    "%s; the page was skipped.",
                    timeout,
                    page_number,
                    document_id,
                )
                continue

            except Exception:
                # Corrupt page content, unsupported render, broken
                # image: log without the page text and continue.
                logger.warning(
                    "OCR failed on page %s of document %s; the page "
                    "was skipped.",
                    page_number,
                    document_id,
                    exc_info=True,
                )
                continue

            # Clean OCR output. A page where the engine found nothing,
            # only a bare URL, or only punctuation yields no unit, so a
            # blank scan is never indexed as successful text.
            text_unit = _sanitize_ocr_unit(page_number, text)

            if text_unit is None:
                continue

            text_unit["document_id"] = document_id

            # Add the OCR result.
            text_units.append(text_unit)

    finally:
        # Close the PDF.
        pdf_document.close()

    # Return OCR-generated text units.
    return text_units
