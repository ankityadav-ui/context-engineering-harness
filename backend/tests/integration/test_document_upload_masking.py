"""Tests for masking inside the document upload pipeline.

Extraction, embedding, indexing, and the masking pipeline are all
replaced with recording stubs, so no external service is contacted and
no real PDF is parsed. The tests verify what actually reaches the
downstream embedding/indexing path.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.rag import document_service


RAW_TEXT = "Contact John Doe at john@example.com for details."


def masked_text_stub(text, document_id, registry=None):
    """Stand-in for mask_and_validate with a predictable replacement.

    Accepts the optional shared placeholder registry that the upload
    pipeline passes to every unit.
    """

    masked = (
        text.replace("john@example.com", "[EMAIL_1]")
        .replace("John Doe", "[PERSON_1]")
        .replace("Jane Smith", "[PERSON_2]")
    )

    return {
        "result": SimpleNamespace(
            document_id=document_id,
            original_text=text,
            masked_text=masked,
            entities=[SimpleNamespace(entity_type="EMAIL")],
            safety_passed=True,
        ),
        "safety": {"passed": True, "reasons": []},
    }


def failing_mask_stub(text, document_id, registry=None):
    """Stand-in where the secondary safety validation fails."""

    return {
        "result": SimpleNamespace(
            document_id=document_id,
            original_text=text,
            masked_text=text,
            entities=[],
            safety_passed=False,
        ),
        "safety": {
            "passed": False,
            "reasons": ["Possible unmasked EMAIL detected."],
        },
    }


@pytest.fixture()
def pipeline_recorder(monkeypatch, tmp_path):
    """Replace every external stage of the pipeline with a recorder."""

    record = {
        "embedded_contents": [],
        "upserted": None,
        "mask_calls": [],
        "registries": [],
        "image_calls": 0,
    }

    document_dir = tmp_path / "documents"
    document_dir.mkdir()
    monkeypatch.setattr(main_module, "DOCUMENT_DIR", document_dir)

    def fake_text_extraction(document_path, document_id):
        return (
            [
                {
                    "document_id": document_id,
                    "page_number": 1,
                    "content": RAW_TEXT,
                    "source": "pdf_text",
                }
            ],
            [2],  # page 2 needs OCR
        )

    def fake_ocr(document_path, document_id, pages):
        return [
            {
                "document_id": document_id,
                "page_number": 2,
                "content": "Scanned note for Jane Smith.",
                "source": "pdf_ocr",
            }
        ]

    def fake_tables(document_path, document_id):
        return [
            {
                "document_id": document_id,
                "page_number": 3,
                "content": "Patient | Doctor\nAlice | Dr. House",
                "source": "pdf_table",
                "table_number": 1,
                "headers": ["Patient", "Doctor"],
                "rows": [["Alice", "Dr. House"]],
            }
        ]

    def fake_images(document_path, document_id, image_storage_dir):
        record["image_calls"] += 1
        return [
            {
                "document_id": document_id,
                "page_number": 1,
                "content_ref": "storage/images/x.png",
                "source": "pdf_image",
                "image_id": "img_1",
            }
        ]

    def fake_embed(content_units):
        record["embedded_contents"] = [
            unit["content"] for unit in content_units
        ]
        return [
            {**unit, "embedding": [0.0, 0.0]}
            for unit in content_units
        ]

    def fake_upsert(embedded_units):
        record["upserted"] = [
            unit["content"] for unit in embedded_units
        ]
        return len(embedded_units)

    def fake_mask(text, document_id, registry=None):
        record["mask_calls"].append(text)
        record.setdefault("registries", []).append(registry)
        return masked_text_stub(text, document_id, registry)

    monkeypatch.setattr(document_service, "extract_text_from_pdf", fake_text_extraction)
    monkeypatch.setattr(document_service, "extract_text_with_ocr", fake_ocr)
    monkeypatch.setattr(document_service, "extract_tables_from_pdf", fake_tables)
    monkeypatch.setattr(document_service, "extract_images_from_pdf", fake_images)
    monkeypatch.setattr(document_service, "embed_content_units", fake_embed)
    monkeypatch.setattr(document_service, "upsert_content_units", fake_upsert)
    monkeypatch.setattr(document_service, "mask_and_validate", fake_mask)

    return record


def upload(client, mask_content=None):
    data = {"mask_content": mask_content} if mask_content else None

    return client.post(
        "/documents/upload",
        files={"file": ("sample.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data=data,
    )


def test_masked_upload_indexes_only_masked_content(pipeline_recorder):
    with TestClient(app) as client:
        response = upload(client, mask_content="true")

    assert response.status_code == 200

    body = response.json()

    assert body["masking_applied"] is True
    assert body["masking_safety_passed"] is True
    # Two text-bearing units were masked (page text + OCR text + table
    # content): three masking calls in total.
    assert body["masked_entities"] == 3

    # Everything embedded and upserted is sanitized; the raw email and
    # names never reach the vector index.
    assert pipeline_recorder["upserted"] is not None
    joined = " ".join(pipeline_recorder["upserted"])

    assert "john@example.com" not in joined
    assert "John Doe" not in joined
    assert "Jane Smith" not in joined
    assert "[EMAIL_1]" in joined

    # OCR output went through the masking stage as well.
    assert any("Scanned note" in text for text in pipeline_recorder["mask_calls"])

    # Every unit shared one placeholder registry, so repeated values keep
    # one stable placeholder across the whole document.
    registries = pipeline_recorder["registries"]
    assert len(registries) == len(pipeline_recorder["mask_calls"])
    assert registries[0] is not None
    assert all(registry is registries[0] for registry in registries)

    # Image extraction is skipped when masking: pixels cannot be sanitized.
    assert pipeline_recorder["image_calls"] == 0
    assert body["images_extracted"] == 0

    # The raw PDF is not kept on disk.
    document_dir = main_module.DOCUMENT_DIR
    assert list(document_dir.glob("*.pdf")) == []


def test_masked_upload_fails_closed_when_safety_check_fails(
    pipeline_recorder,
    monkeypatch,
):
    monkeypatch.setattr(
        document_service,
        "mask_and_validate",
        failing_mask_stub,
    )

    with TestClient(app) as client:
        response = upload(client, mask_content="true")

    assert response.status_code == 422
    assert "safety validation" in response.json()["detail"]

    # Nothing was embedded or indexed, and no raw file remains.
    assert pipeline_recorder["embedded_contents"] == []
    assert pipeline_recorder["upserted"] is None

    document_dir = main_module.DOCUMENT_DIR
    assert list(document_dir.glob("*.pdf")) == []

    # The error must never echo the original text.
    assert "john@example.com" not in response.text


def test_default_upload_behavior_is_unchanged(pipeline_recorder):
    with TestClient(app) as client:
        response = upload(client)

    assert response.status_code == 200

    body = response.json()

    assert body["masking_applied"] is False
    assert body["masking_safety_passed"] is None

    # No masking calls; the raw content is embedded as before, and
    # images are still extracted.
    assert pipeline_recorder["mask_calls"] == []
    assert RAW_TEXT in pipeline_recorder["upserted"]
    assert pipeline_recorder["image_calls"] == 1
    assert body["images_extracted"] == 1

    # The raw PDF is still stored, as in the unmasked flow.
    document_dir = main_module.DOCUMENT_DIR
    assert len(list(document_dir.glob("*.pdf"))) == 1
