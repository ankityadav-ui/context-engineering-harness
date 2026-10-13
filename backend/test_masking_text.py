
from unittest.mock import patch

import pytest

from app.masking.text_service import extract_text_for_masking


def test_extract_text_for_masking_combines_pages_in_order(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-test")

    content_units = [
        {
            "content_type": "text",
            "content": "Text from page two.",
            "page_number": 2,
        },
        {
            "content_type": "image",
            "content": "An image description.",
            "page_number": 1,
        },
        {
            "content_type": "text",
            "content": "Text from page one.",
            "page_number": 1,
        },
        {
            "content_type": "text",
            "content": "   ",
            "page_number": 3,
        },
    ]

    with patch(
        "app.masking.text_service.process_document"
    ) as mock_process:
        mock_process.return_value = (
            content_units,
            [1],
            [{"content_type": "table"}],
            [{"content_type": "image"}],
        )

        result = extract_text_for_masking(
            document_path=str(pdf_path),
            document_id="sample-document",
        )

    assert result["document_id"] == "sample-document"
    assert result["text_unit_count"] == 2
    assert result["table_unit_count"] == 1
    assert result["image_unit_count"] == 1
    assert result["pages_with_ocr"] == [1]

    assert result["text"].index("Text from page one.") < (
        result["text"].index("Text from page two.")
    )
    assert "[Page 1]" in result["text"]
    assert "[Page 2]" in result["text"]
    assert "An image description." not in result["text"]


def test_extract_text_for_masking_rejects_missing_file(tmp_path):
    missing_pdf = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        extract_text_for_masking(
            document_path=str(missing_pdf),
            document_id="missing-document",
        )
