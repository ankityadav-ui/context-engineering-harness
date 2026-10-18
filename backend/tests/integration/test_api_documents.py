"""Tests for the document listing, detail, and deletion endpoints.

Pinecone is replaced with a fake index and local storage is redirected to
a temporary directory, so no external service is contacted and no real
document is touched.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag import document_service, pinecone_service


class FakeListPage:
    def __init__(self, ids):
        self.vectors = [SimpleNamespace(id=value) for value in ids]


class FakeVector:
    def __init__(self, id, metadata):
        self.id = id
        self.metadata = metadata


class FakePagination:
    def __init__(self, next_token=None):
        self.next = next_token


class FakeIndex:
    """A small stand-in for the parts of a Pinecone index we use."""

    def __init__(self, vectors, page_size=100):
        self.vectors = dict(vectors)
        self.page_size = page_size
        self.delete_calls = []

    def list(self, limit=None, namespace=None):
        size = limit or self.page_size
        ids = list(self.vectors)

        for start in range(0, len(ids), size):
            yield FakeListPage(ids[start:start + size])

    def fetch(self, ids, namespace=None):
        return SimpleNamespace(
            vectors={
                vector_id: self.vectors[vector_id]
                for vector_id in ids
                if vector_id in self.vectors
            }
        )

    def fetch_by_metadata(
        self,
        filter,
        namespace=None,
        limit=None,
        pagination_token=None,
    ):
        document_id = filter["document_id"]["$eq"]

        matched = [
            vector_id
            for vector_id, vector in self.vectors.items()
            if (vector.metadata or {}).get("document_id")
            == document_id
        ]

        start = int(pagination_token or 0)
        window = matched[start:start + (limit or self.page_size)]
        next_start = start + len(window)

        next_token = (
            str(next_start) if next_start < len(matched) else None
        )

        return SimpleNamespace(
            vectors={
                vector_id: self.vectors[vector_id]
                for vector_id in window
            },
            pagination=(
                FakePagination(next_token) if next_token else None
            ),
        )

    def delete(
        self,
        filter=None,
        namespace=None,
        ids=None,
        delete_all=False,
    ):
        self.delete_calls.append(filter)

        document_id = filter["document_id"]["$eq"]

        for vector_id in [
            vector_id
            for vector_id, vector in self.vectors.items()
            if (vector.metadata or {}).get("document_id")
            == document_id
        ]:
            del self.vectors[vector_id]


def build_index():
    return FakeIndex(
        {
            "unit-1": FakeVector(
                "unit-1",
                {
                    "document_id": "doc-alpha",
                    "content_type": "text",
                    "page_number": 1,
                    "source": "pdf_text",
                    "content": "Confidential patient record.",
                },
            ),
            "unit-2": FakeVector(
                "unit-2",
                {
                    "document_id": "doc-alpha",
                    "content_type": "table",
                    "page_number": 2,
                    "source": "pdf_table",
                    "content": "Chamber | Function",
                    "table_number": 1,
                },
            ),
            "unit-3": FakeVector(
                "unit-3",
                {
                    "document_id": "doc-beta",
                    "content_type": "image",
                    "page_number": 1,
                    "source": "pdf_image",
                    "content": "",
                    "content_ref": (
                        "storage/images/doc-beta/page_1_img_1.png"
                    ),
                    "image_id": "page_1_img_1",
                },
            ),
        }
    )


@pytest.fixture()
def fake_index(monkeypatch):
    index = build_index()
    monkeypatch.setattr(pinecone_service, "get_index", lambda: index)
    return index


def test_listing_documents_groups_indexed_vectors(fake_index):
    with TestClient(app) as client:
        response = client.get("/documents")

    assert response.status_code == 200

    body = response.json()

    assert body["scanned_vectors"] == 3
    assert body["scan_truncated"] is False

    documents = {
        document["document_id"]: document
        for document in body["documents"]
    }

    assert set(documents) == {"doc-alpha", "doc-beta"}
    assert documents["doc-alpha"]["content_units"] == 2
    assert documents["doc-alpha"]["content_types"] == [
        "table",
        "text",
    ]
    assert documents["doc-alpha"]["pages"] == [1, 2]
    assert documents["doc-beta"]["content_types"] == ["image"]

    # Listing must not leak the indexed document text.
    assert "Confidential patient record." not in response.text


def test_document_details_describe_units_without_returning_text(
    fake_index,
):
    with TestClient(app) as client:
        response = client.get("/documents/doc-alpha")

    assert response.status_code == 200

    body = response.json()

    assert body["document_id"] == "doc-alpha"
    assert body["content_units"] == 2
    assert body["units_truncated"] is False

    assert [
        unit["content_type"] for unit in body["units"]
    ] == ["text", "table"]

    first_unit = body["units"][0]

    assert first_unit["page_number"] == 1
    assert first_unit["content_length"] == len(
        "Confidential patient record."
    )
    assert "content" not in first_unit
    assert "Confidential patient record." not in response.text


def test_unknown_document_returns_404(fake_index):
    with TestClient(app) as client:
        response = client.get("/documents/doc-missing")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_delete_document_removes_vectors_and_stored_files(
    fake_index,
    monkeypatch,
    tmp_path,
):
    document_dir = tmp_path / "documents"
    image_dir = tmp_path / "images"
    document_dir.mkdir()
    image_dir.mkdir()

    pdf_path = document_dir / "doc-alpha.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    stored_images = image_dir / "doc-alpha"
    stored_images.mkdir()
    (stored_images / "page_1_img_1.png").write_bytes(b"image")

    monkeypatch.setattr(
        document_service,
        "DOCUMENT_STORAGE_DIR",
        document_dir,
    )
    monkeypatch.setattr(
        document_service,
        "IMAGE_STORAGE_DIR",
        image_dir,
    )

    with TestClient(app) as client:
        response = client.delete("/documents/doc-alpha")

    assert response.status_code == 200

    body = response.json()

    assert body["index_delete_requested"] is True
    assert body["matched_content_units_on_first_page"] == 2

    assert fake_index.delete_calls == [
        {"document_id": {"$eq": "doc-alpha"}}
    ]

    # Only the requested document was removed from the index.
    assert {
        vector.metadata["document_id"]
        for vector in fake_index.vectors.values()
    } == {"doc-beta"}

    assert not pdf_path.exists()
    assert not stored_images.exists()
    assert len(body["removed_files"]) == 2


def test_delete_unknown_document_returns_404(fake_index):
    with TestClient(app) as client:
        response = client.delete("/documents/doc-missing")

    assert response.status_code == 404
    assert fake_index.delete_calls == []
    assert len(fake_index.vectors) == 3


def test_document_routes_reject_unsafe_ids(fake_index):
    with TestClient(app) as client:
        deleted = client.delete("/documents/bad..id")
        fetched = client.get("/documents/bad..id")

    assert deleted.status_code == 400
    assert fetched.status_code == 400
    assert fake_index.delete_calls == []


def test_documents_report_missing_pinecone_configuration(monkeypatch):
    monkeypatch.setattr(pinecone_service, "_client", None)
    monkeypatch.setattr(pinecone_service, "_index", None)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)

    with TestClient(app) as client:
        response = client.get("/documents")

    assert response.status_code == 503
    assert "PINECONE_API_KEY" in response.json()["detail"]
