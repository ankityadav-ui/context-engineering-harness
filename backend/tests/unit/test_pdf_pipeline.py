# End-to-end document pipeline over a real PDF.
#
# A real (tiny, generated) PDF is parsed by PyMuPDF, pdfplumber, and the
# image extractor. Only the paid/remote stages are replaced: Gemini
# embeddings and the Pinecone upsert. Neo4j is reported as unconfigured.
#
# No external service is contacted and no real document is used.
#

from types import SimpleNamespace

import io

import pymupdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_module
from app.main import app
from app.rag import document_service, embedding_service

PAGE_ONE_TEXT = (
    "Quarterly summary prepared by Priya Sharma. "
    "Contact priya.sharma@example.com for the appendix."
)
PAGE_TWO_TEXT = (
    "Second page content with enough characters to be meaningful text."
)


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 24), color=(200, 40, 40)).save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


def build_pdf(path: str) -> None:
    """Write a two-page PDF whose first page has selectable text and a table."""
    document = pymupdf.open()

    page_one = document.new_page()
    page_one.insert_text((72, 72), PAGE_ONE_TEXT)
    page_one.insert_image(
        pymupdf.Rect(72, 120, 200, 190),
        stream=_png_bytes(),
    )

    page_two = document.new_page()
    page_two.insert_text((72, 72), PAGE_TWO_TEXT)

    columns = [72, 190, 300]
    rows = [160, 190, 220]
    shape = page_two.new_shape()

    for x in columns:
        shape.draw_line((x, rows[0]), (x, rows[-1]))
    for y in rows:
        shape.draw_line((columns[0], y), (columns[-1], y))

    shape.finish(color=(0, 0, 0), width=0.7)
    shape.commit()

    cells = [
        (78, 176, "Region"),
        (196, 176, "Units"),
        (78, 206, "North"),
        (196, 206, "42"),
    ]

    for x, y, text in cells:
        page_two.insert_text((x, y), text)

    document.save(path)
    document.close()


def build_image_first_page_pdf(path: str) -> None:
    """Write a PDF whose first page is an image and whose second page is text.

    The first page has no selectable text layer, so the extractor must hand
    it to the OCR fallback instead of returning it as normal text.
    """
    document = pymupdf.open()

    page_one = document.new_page()
    page_one.insert_image(
        pymupdf.Rect(72, 72, 200, 180),
        stream=_png_bytes(),
    )

    page_two = document.new_page()
    page_two.insert_text((72, 72), PAGE_TWO_TEXT)

    document.save(path)
    document.close()


@pytest.fixture()
def pipeline(tmp_path, monkeypatch):
    """Redirect storage and stub the paid stages of the pipeline."""
    storage_root = tmp_path / "storage"
    document_dir = storage_root / "documents"
    image_dir = storage_root / "images"
    document_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)

    monkeypatch.setattr(main_module, "DOCUMENT_DIR", document_dir)
    monkeypatch.setattr(document_service, "DOCUMENT_STORAGE_DIR", document_dir)
    monkeypatch.setattr(document_service, "IMAGE_STORAGE_DIR", image_dir)
    monkeypatch.setattr(document_service, "BASE_DIR", tmp_path)

    # Image references are resolved relative to the backend directory.
    monkeypatch.setattr(embedding_service, "BACKEND_DIR", tmp_path)

    # Neo4j must not be written to from a test run.
    monkeypatch.setattr(
        main_module.graph_service, "is_configured", lambda: False
    )

    record = {"embedded": [], "upserted": 0, "units": []}

    def fake_embed(content_units):
        record["units"] = content_units
        record["embedded"] = [unit["content"] for unit in content_units]
        return [{**unit, "embedding": [0.0, 0.0]} for unit in content_units]

    def fake_upsert(embedded_units):
        record["upserted"] = len(embedded_units)
        return len(embedded_units)

    monkeypatch.setattr(document_service, "embed_content_units", fake_embed)
    monkeypatch.setattr(document_service, "upsert_content_units", fake_upsert)

    return {
        "record": record,
        "document_dir": document_dir,
        "image_dir": image_dir,
    }


def upload(pdf_path, mask_content="false"):
    with TestClient(app) as client:
        with open(pdf_path, "rb") as handle:
            accepted = client.post(
                "/documents/upload",
                files={
                    "file": (
                        "sample.pdf",
                        handle.read(),
                        "application/pdf",
                    )
                },
                data={"mask_content": mask_content},
            )
        if accepted.status_code != 202:
            return accepted

        job = client.get(
            f"/documents/jobs/{accepted.json()['job_id']}"
        ).json()
        if job["status"] != "completed":
            error_code = (job.get("error") or {}).get("code")
            return SimpleNamespace(
                status_code={
                    "invalid_pdf": 400,
                    "no_retrievable_content": 422,
                    "masking_rejected": 422,
                    "embedding_failed": 500,
                }.get(error_code, 500),
                text=str(job),
                json=lambda: {"detail": job["error"]["message"]},
            )
        return SimpleNamespace(
            status_code=200,
            text=accepted.text,
            json=lambda: {
                "document_id": job["document_id"],
                **job["result"],
            },
        )


def test_real_pdf_is_extracted_embedded_and_indexed(tmp_path, pipeline):
    pdf_path = tmp_path / "sample.pdf"
    build_pdf(pdf_path)

    response = upload(pdf_path)

    assert response.status_code == 200, response.text

    body = response.json()

    assert body["content_units"] >= 2
    assert body["images_extracted"] >= 1
    assert body["ocr_pages"] == []
    assert body["masking_applied"] is False
    assert body["knowledge_graph"]["ingested"] is False

    record = pipeline["record"]

    # The page text, the table text, and the image description were all
    # converted into embedding input.
    joined = "\n".join(
        embedding_service.get_embedding_text(unit)
        for unit in record["units"]
    )
    assert "Quarterly summary" in joined
    assert "Region: North" in joined
    assert "Image" in joined
    assert record["upserted"] == body["content_units"]

    # Extracted image units really resolve to a stored file, and the
    # reference cannot escape the storage directory.
    image_units = [
        unit
        for unit in record["units"]
        if unit["content_type"] == "image"
    ]
    assert image_units

    for unit in image_units:
        resolved = embedding_service.resolve_image_path(unit)
        assert resolved is not None
        assert resolved.startswith(str(pipeline["image_dir"].parent))

    # A traversal attempt and an absolute path are both refused.
    for unsafe_ref in (
        "storage/images/../../secrets.txt",
        "../../secrets.txt",
        "C:/Windows/secret.txt",
        "/etc/passwd",
    ):
        assert embedding_service.resolve_image_path(
            {"content_type": "image", "content_ref": unsafe_ref}
        ) is None

    # A missing file resolves to None instead of raising.
    assert embedding_service.resolve_image_path(
        {
            "content_type": "image",
            "content_ref": "storage/images/doc-missing/page_1_img_1.png",
        }
    ) is None

    # The original PDF is kept for the unmasked flow.
    assert len(list(pipeline["document_dir"].glob("*.pdf"))) == 1


def test_real_pdf_is_ocr_checked_and_tables_are_captured(tmp_path, pipeline):
    pdf_path = tmp_path / "sample.pdf"
    build_pdf(pdf_path)

    response = upload(pdf_path)

    assert response.status_code == 200

    body = response.json()

    # Both pages carry selectable text, so OCR was not required.
    assert body["ocr_pages"] == []

    # The ruled table on page two is captured as its own content type.
    content_types = {
        unit["content_type"] for unit in pipeline["record"]["units"]
    }
    assert "text" in content_types
    assert "image" in content_types


def test_image_page_is_handed_to_ocr(tmp_path, pipeline, monkeypatch):
    pdf_path = tmp_path / "image_first.pdf"
    build_image_first_page_pdf(pdf_path)

    calls = {}

    def fake_extract_text_with_ocr(document_path, document_id, pages):
        calls["pages"] = list(pages)
        return [
            {
                "document_id": document_id,
                "content_type": "text",
                "page_number": pages[0],
                "content": (
                    "Scanned resume text with enough characters to be "
                    "meaningful."
                ),
                "source": "ocr",
            }
        ]

    monkeypatch.setattr(
        document_service,
        "extract_text_with_ocr",
        fake_extract_text_with_ocr,
    )

    response = upload(pdf_path)

    assert response.status_code == 200, response.text

    body = response.json()

    # The image-only page was handed to OCR instead of silently dropped.
    assert body["ocr_pages"] == [1]
    assert calls["pages"] == [1]

    # The OCR output was folded into the embedded unit set. The
    # extractor's source label travels in the unit metadata.
    record = pipeline["record"]
    ocr_units = [
        unit
        for unit in record["units"]
        if unit.get("metadata", {}).get("source") == "ocr"
    ]
    assert ocr_units, "OCR fallback did not produce any units"


def test_ocr_with_no_text_does_not_fabricate_a_unit(
    tmp_path, pipeline, monkeypatch
):
    """A page OCR cannot read must not become a successful text unit.

    The OCR engine is stubbed to return no text, which is what a real
    Tesseract run returns for a solid-color image. The page is still
    reported as OCR-checked, but no content unit is created for it.
    """
    pdf_path = tmp_path / "solid.pdf"
    build_image_first_page_pdf(pdf_path)

    calls = {}

    def fake_extract_text_with_ocr(document_path, document_id, pages):
        calls["pages"] = list(pages)
        return []

    monkeypatch.setattr(
        document_service,
        "extract_text_with_ocr",
        fake_extract_text_with_ocr,
    )

    response = upload(pdf_path)

    assert response.status_code == 200, response.text

    body = response.json()

    # The page was checked, but no OCR text was fabricated.
    assert body["ocr_pages"] == [1]
    assert calls["pages"] == [1]
    assert not [
        unit
        for unit in pipeline["record"]["units"]
        if unit.get("metadata", {}).get("source") == "ocr"
    ]


def test_empty_pdf_upload_fails_with_actionable_error(tmp_path):
    empty_path = tmp_path / "empty.pdf"
    empty_path.write_bytes(b"")

    response = upload(empty_path)

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_pdf_with_no_pages_fails_with_client_error(tmp_path, pipeline):
    # A file with a PDF header but no object table or pages is not a
    # usable document; it must not surface as a server error.
    truncated_path = tmp_path / "truncated.pdf"
    truncated_path.write_bytes(b"%PDF-1.7\n%%EOF\n")

    response = upload(truncated_path)

    assert response.status_code == 400
    assert "not a valid pdf" in response.json()["detail"].lower()


def test_blank_pdf_upload_fails_with_no_retrievable_content(
    tmp_path, pipeline
):
    # A genuine one-page PDF with no text, tables, or images is a valid
    # document with nothing to index: that is a 422, distinct from an
    # empty or corrupted upload.
    blank_path = tmp_path / "blank.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(blank_path)
    document.close()

    response = upload(blank_path)

    assert response.status_code == 422
    assert "No retrievable text or content" in response.json()["detail"]

    # A rejected upload leaves no raw PDF or extracted images behind.
    assert list(pipeline["document_dir"].glob("*.pdf")) == []
    assert list(pipeline["image_dir"].glob("*")) == []


def test_renamed_non_pdf_upload_fails_with_actionable_error(tmp_path):
    renamed_path = tmp_path / "renamed.pdf"
    renamed_path.write_bytes(b"MZ\x90\x00 not a pdf")

    response = upload(renamed_path)

    assert response.status_code == 400
    assert "not a valid PDF" in response.json()["detail"]
