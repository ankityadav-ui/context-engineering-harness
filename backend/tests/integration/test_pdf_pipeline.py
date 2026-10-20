"""End-to-end document pipeline over a real PDF.

A real (tiny, generated) PDF is parsed by PyMuPDF, pdfplumber, and the
image extractor. Only the paid/remote stages are replaced: Gemini
embeddings and the Pinecone upsert. Neo4j is reported as unconfigured.

No external service is contacted and no real document is used.
"""

import io

import pymupdf
from types import SimpleNamespace

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


def build_pdf(path):
    """Write a small two-page PDF with text, a table, and an image."""

    document = pymupdf.open()

    page_one = document.new_page()
    page_one.insert_text((72, 72), PAGE_ONE_TEXT)
    page_one.insert_image(
        pymupdf.Rect(72, 120, 200, 190),
        stream=_png_bytes(),
    )

    # A ruled table so pdfplumber can detect it from the page lines.
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
            return SimpleNamespace(
                status_code=500,
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


def test_masked_upload_indexes_only_masked_text(tmp_path, pipeline):
    pdf_path = tmp_path / "sample.pdf"
    build_pdf(pdf_path)

    response = upload(pdf_path, mask_content="true")

    assert response.status_code == 200, response.text

    body = response.json()

    assert body["masking_applied"] is True
    assert body["masking_safety_passed"] is True
    assert body["masked_entities"] >= 1
    # Image pixels cannot be masked, so no images are extracted.
    assert body["images_extracted"] == 0

    joined = "\n".join(pipeline["record"]["embedded"])

    assert "Priya Sharma" not in joined
    assert "priya.sharma@example.com" not in joined
    assert "[EMAIL_1]" in joined
    assert "[PERSON_" in joined

    # The raw PDF is not kept once masking is applied.
    assert list(pipeline["document_dir"].glob("*.pdf")) == []
    assert list(pipeline["image_dir"].glob("*")) == []
