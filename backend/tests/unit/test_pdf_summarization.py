"""Question summarization regression tests.

A normal text-based PDF uploads and is indexed; a scanned PDF falls back
to OCR; empty/unreadable PDFs fail with actionable errors; and a
document-grounded ``summarise this`` answer is assembled from the
uploaded resume instead of the garbled-text fallback.
"""

import io

import pymupdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_module
from app.main import app
from app.rag import document_service, embedding_service


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 24), color=(200, 40, 40)).save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


def build_pdf(path):
    """Write a small two-page PDF with selectable text, a table, and an image."""
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


PAGE_ONE_TEXT = (
    "Quarterly summary prepared by Priya Sharma. "
    "Contact priya.sharma@example.com for the appendix."
)
PAGE_TWO_TEXT = "Second page content with enough characters to be meaningful text."


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
            return client.post(
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


def test_normal_pdf_upload_indexes_and_returns_content_units(tmp_path, pipeline):
    pdf_path = tmp_path / "sample.pdf"
    build_pdf(pdf_path)

    response = upload(pdf_path)

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["content_units"] >= 2
    assert body["images_extracted"] >= 1
    assert body["ocr_pages"] == []
    assert body["masking_applied"] is False


def test_empty_pdf_upload_fails_with_actionable_error(tmp_path):
    empty_path = tmp_path / "empty.pdf"
    empty_path.write_bytes(b"")

    response = upload(empty_path)

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_renamed_non_pdf_upload_fails_with_actionable_error(tmp_path):
    renamed_path = tmp_path / "renamed.pdf"
    renamed_path.write_bytes(b"MZ\x90\x00 not a pdf")

    response = upload(renamed_path)

    assert response.status_code == 400
    assert "not a valid PDF" in response.json()["detail"]


def test_document_indexing_failure_is_500_with_no_document_left(tmp_path, pipeline, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    build_pdf(pdf_path)

    def burst_embedding(content_units):
        raise RuntimeError("embedding host unreachable")

    monkeypatch.setattr(document_service, "embed_content_units", burst_embedding)

    response = upload(pdf_path)

    assert response.status_code == 500
    assert "No content was indexed" in response.json()["detail"]
    assert list(pipeline["document_dir"].glob("*.pdf")) == []
    assert list(pipeline["image_dir"].glob("*")) == []


def test_summarise_this_after_upload_uses_stored_full_text(
    tmp_path, pipeline, monkeypatch
):
    """`summarise this` must be grounded in the uploaded resume.

    Retrieval is stubbed to return nothing, so the Context Engine's
    full-text fallback is what feeds the model. Embeddings and the vector
    upsert are stubbed by the pipeline fixture; the rest is the real
    upload, storage, extraction, and assembly path.
    """
    import app.context_engine.engine as engine_module

    monkeypatch.setattr(
        engine_module, "retrieve_context", lambda **kwargs: []
    )

    # One short line keeps the whole resume on page one at the default
    # font size, so nothing is clipped by the page width.
    resume_text = (
        "Sarah Chen builds recruitment platforms and logistics "
        "automation."
    )

    pdf_path = tmp_path / "resume.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), resume_text)
    document.save(pdf_path)
    document.close()

    upload_response = upload(pdf_path)

    assert upload_response.status_code == 200, upload_response.text
    document_id = upload_response.json()["document_id"]

    prompts = []

    def fake_generate_response(prompt, system=None):
        prompts.append(prompt)

        if "Sarah Chen" in prompt:
            return (
                "Sarah Chen builds recruitment platforms and "
                "logistics automation."
            )

        return "I can help with that."

    monkeypatch.setattr(
        main_module, "generate_response", fake_generate_response
    )

    response = TestClient(app).post(
        "/documents/ask",
        json={"query": "summarise this", "document_id": document_id},
    )

    assert response.status_code == 200, response.text
    body = response.json()

    # No garbled-text claim: the answer is grounded in the resume.
    assert "garbled" not in body["answer"].lower()
    assert "fragmented" not in body["answer"].lower()
    assert "illegible" not in body["answer"].lower()
    assert "Sarah Chen" in body["answer"]

    # The assembled prompt contains the actual resume content.
    assembled_context = " ".join(prompts)
    assert "Sarah Chen" in assembled_context
    assert "recruitment platforms" in assembled_context
