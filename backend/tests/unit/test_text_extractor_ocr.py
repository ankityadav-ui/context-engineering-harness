import io

import pytest
import pymupdf
from PIL import Image

from app.rag import ocr
from app.rag.text_extractor import PDFProcessingError, _sanitize_ocr_unit

from tests.unit.test_pdf_pipeline import (  # noqa: F401
    build_image_first_page_pdf,
)


def build_solid_color_pdf(path):
    """Write a one-page PDF whose only content is a solid-color image."""

    buffer = io.BytesIO()
    Image.new("RGB", (200, 120), color=(10, 200, 10)).save(
        buffer, format="PNG"
    )

    document = pymupdf.open()
    page = document.new_page()
    page.insert_image(
        pymupdf.Rect(72, 72, 400, 300),
        stream=buffer.getvalue(),
    )
    document.save(path)
    document.close()


class TestSanitizeOcrUnit:
    def test_rejects_headless_and_placeholder_urls(self):
        cases = [
            "https://ui.example.com/start",
            "http://app.example.com:8501/",
            "https://example.com/_template/placeholder.pdf",
            "https://example.com/api/report/12345.pdf",
        ]
        for text in cases:
            assert _sanitize_ocr_unit(1, text) is None

    def test_accepts_plain_resume_text(self):
        unit = _sanitize_ocr_unit(
            3,
            "Quarterly summary prepared by Priya Sharma. Contact priya.sharma@example.com.",
        )
        assert unit is not None
        assert unit["page_number"] == 3
        assert unit["source"] == "ocr"

    def test_preserves_length_and_page_number(self):
        unit = _sanitize_ocr_unit(7, "A" * 5000)
        assert unit is not None
        assert unit["page_number"] == 7
        assert len(unit["content"]) == 5000

    def test_drops_oversized_units(self):
        assert _sanitize_ocr_unit(1, "A" * 30000) is None

    def test_drops_undersized_units(self):
        assert _sanitize_ocr_unit(1, "ab") is None


class TestTextExtractorOcrIntegration:
    def test_image_only_page_is_queued_for_ocr(self, tmp_path):
        # The first page is an image with no text layer and the second
        # page has selectable text, so only page two is returned as text
        # and page one is handed to the OCR fallback.
        pdf_path = tmp_path / "scanned.pdf"
        build_image_first_page_pdf(pdf_path)

        from app.rag.text_extractor import extract_text_from_pdf

        text_units, pages_needing_ocr = extract_text_from_pdf(
            str(pdf_path), "doc-scanned"
        )

        assert [unit["page_number"] for unit in text_units] == [2]
        assert pages_needing_ocr == [1]

    @pytest.mark.skipif(
        not ocr.is_configured(),
        reason="Tesseract is not installed in this environment",
    )
    def test_solid_color_page_yields_no_ocr_units(self, tmp_path):
        # Live check against the local Tesseract binary: a blank image
        # must not be reported as successful OCR text.
        pdf_path = tmp_path / "solid.pdf"
        build_solid_color_pdf(pdf_path)

        units = ocr.extract_text_with_ocr(
            document_path=str(pdf_path),
            document_id="doc-solid",
            pages=[1],
        )

        assert units == []
