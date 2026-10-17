"""Upload validation: file type, content signature, and size limit.

The processing pipeline is never reached in these tests, so no PDF is
parsed and no external service is contacted.
"""

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app


@pytest.fixture()
def upload_dir(tmp_path, monkeypatch):
    """Keep uploads inside a temporary directory."""

    document_dir = tmp_path / "documents"
    document_dir.mkdir()
    monkeypatch.setattr(main_module, "DOCUMENT_DIR", document_dir)
    return document_dir


def post_upload(filename, payload):
    with TestClient(app) as client:
        return client.post(
            "/documents/upload",
            files={"file": (filename, payload, "application/pdf")},
        )


def test_rejects_a_non_pdf_extension(upload_dir):
    response = post_upload("notes.txt", b"%PDF-1.4 content")

    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]
    assert list(upload_dir.iterdir()) == []


def test_rejects_a_renamed_non_pdf_payload(upload_dir):
    # A .pdf name is not proof of PDF content.
    response = post_upload("malicious.pdf", b"MZ\x90\x00 not a pdf")

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "The uploaded file is not a valid PDF."
    )
    assert list(upload_dir.iterdir()) == []


def test_rejects_an_empty_upload(upload_dir):
    response = post_upload("empty.pdf", b"")

    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded file is empty."
    assert list(upload_dir.iterdir()) == []


def test_rejects_an_upload_above_the_size_limit(upload_dir, monkeypatch):
    # A small limit keeps the test fast; the real limit is 25 MiB.
    monkeypatch.setattr(main_module, "MAX_UPLOAD_BYTES", 2048)

    payload = b"%PDF-1.4 " + b"x" * 4096

    response = post_upload("big.pdf", payload)

    assert response.status_code == 413
    assert "larger than" in response.json()["detail"]

    # Nothing was written and no content was indexed.
    assert list(upload_dir.iterdir()) == []


def test_upload_limit_default_is_documented_in_bytes():
    assert main_module.MAX_UPLOAD_BYTES == 25 * 1024 * 1024
    assert main_module.UPLOAD_CHUNK_BYTES < main_module.MAX_UPLOAD_BYTES
