"""Tests for the image-aware retrieval pipeline.

Covers: image extraction with page-context captions, content-addressed
assets, deterministic content-unit ids (idempotent re-ingestion), the
image embedding text, path safety of resolve_image_path, and the
Pinecone dimension guard. No external service is contacted.
"""

import io

import pymupdf
import pytest
from PIL import Image

from app.rag import chunker, embedding_service, pinecone_service
from app.rag.image_extractor import extract_images_from_pdf

PAGE_TEXT = (
    "Figure 1 shows quarterly revenue by region for Acme Corp. "
    "The chart below summarizes the annual performance review."
)


def _png_bytes(color=(30, 90, 200)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (48, 32), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _build_pdf(path, *, with_image=True, second_color=None):
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 60), PAGE_TEXT)

    if with_image:
        page.insert_image(
            pymupdf.Rect(72, 100, 220, 180),
            stream=_png_bytes(),
        )

    if second_color is not None:
        page_two = document.new_page()
        page_two.insert_text((72, 60), "Second page without images.")
        page_two.insert_image(
            pymupdf.Rect(72, 100, 220, 180),
            stream=_png_bytes(color=second_color),
        )

    document.save(path)
    document.close()
    return path


@pytest.fixture()
def image_dirs(tmp_path):
    storage = tmp_path / "storage"
    image_dir = storage / "images"
    image_dir.mkdir(parents=True)
    return image_dir


# ------------------------------------------------------- extraction


def test_extracted_image_carries_caption_hash_and_safe_ref(
    tmp_path, image_dirs
):
    pdf = _build_pdf(tmp_path / "doc.pdf")

    units = extract_images_from_pdf(
        document_path=pdf,
        document_id="doc-test-1",
        image_storage_dir=image_dirs,
    )

    assert len(units) == 1

    unit = units[0]

    # Source attribution: document, page, image id, storage ref.
    assert unit["document_id"] == "doc-test-1"
    assert unit["page_number"] == 1
    assert unit["image_id"] == "doc-test-1_image_1"
    assert unit["content_ref"].startswith(
        "images/doc-test-1/"
    )

    # The caption is the surrounding page text (normalized excerpt).
    assert "quarterly revenue by region" in unit["caption"]

    # Content hash of the real image bytes.
    assert len(unit["image_sha256"]) == 64

    # The reference resolves to an existing file inside the storage
    # directory (the resolver anchors at BACKEND_DIR, which the test
    # points at tmp_path).
    original_root = embedding_service.BACKEND_DIR
    embedding_service.BACKEND_DIR = tmp_path

    try:
        resolved = embedding_service.resolve_image_path(
            {
                "content_type": "image",
                "content_ref": unit["content_ref"],
            }
        )
    finally:
        embedding_service.BACKEND_DIR = original_root

    assert resolved is not None


def test_repeated_ingestion_does_not_duplicate_image_files(
    tmp_path, image_dirs
):
    pdf = _build_pdf(tmp_path / "doc.pdf")

    first = extract_images_from_pdf(
        document_path=pdf,
        document_id="doc-dup",
        image_storage_dir=image_dirs,
    )
    second = extract_images_from_pdf(
        document_path=pdf,
        document_id="doc-dup",
        image_storage_dir=image_dirs,
    )

    assert [unit["content_ref"] for unit in first] == [
        unit["content_ref"] for unit in second
    ]
    assert [unit["image_sha256"] for unit in first] == [
        unit["image_sha256"] for unit in second
    ]

    stored_files = list((image_dirs / "doc-dup").iterdir())

    # Content-addressed names: the second run overwrote, never
    # duplicated.
    assert len(stored_files) == len(first) == 1


def test_extraction_rejects_an_unsafe_document_id(tmp_path, image_dirs):
    pdf = _build_pdf(tmp_path / "doc.pdf")

    with pytest.raises(ValueError):
        extract_images_from_pdf(
            document_path=pdf,
            document_id="../escape",
            image_storage_dir=image_dirs,
        )


def test_pdf_without_images_yields_no_image_units(
    tmp_path, image_dirs
):
    pdf = _build_pdf(tmp_path / "doc.pdf", with_image=False)

    units = extract_images_from_pdf(
        document_path=pdf,
        document_id="doc-text-only",
        image_storage_dir=image_dirs,
    )

    assert units == []


# -------------------------------------------- deterministic unit ids


def _text_units(document_id="doc-1"):
    return [
        {
            "document_id": document_id,
            "page_number": 1,
            "content": "Return window: 30 days",
            "source": "pdf_text",
        }
    ]


def test_unit_ids_are_deterministic_across_runs():
    first = chunker.build_content_units(
        text_units=_text_units(), table_units=[], image_units=[]
    )
    second = chunker.build_content_units(
        text_units=_text_units(), table_units=[], image_units=[]
    )

    assert first[0]["id"] == second[0]["id"]

    # Different content on the same page produces a different id.
    changed = chunker.build_content_units(
        text_units=[
            {
                "document_id": "doc-1",
                "page_number": 1,
                "content": "Return window: 60 days",
                "source": "pdf_text",
            }
        ],
        table_units=[],
        image_units=[],
    )

    assert changed[0]["id"] != first[0]["id"]


def test_image_unit_ids_are_deterministic_and_caption_survives():
    image_unit = {
        "document_id": "doc-1",
        "page_number": 2,
        "content_ref": "storage/images/doc-1/page_2_img_1.png",
        "image_id": "doc-1_image_1",
        "image_format": "png",
        "caption": "Revenue chart by region",
        "image_sha256": "abc123",
        "source": "pdf_image",
    }

    first = chunker.build_content_units(
        text_units=[], table_units=[], image_units=[image_unit]
    )
    second = chunker.build_content_units(
        text_units=[], table_units=[], image_units=[image_unit]
    )

    assert first[0]["id"] == second[0]["id"]
    assert first[0]["metadata"]["caption"] == (
        "Revenue chart by region"
    )
    assert first[0]["metadata"]["image_sha256"] == "abc123"


# ------------------------------------------------- embedding text


def test_image_embedding_text_includes_surrounding_context():
    text = embedding_service.get_embedding_text(
        {
            "content_type": "image",
            "page_number": 3,
            "metadata": {
                "image_id": "img-9",
                "image_format": "png",
                "caption": "Bar chart of quarterly revenue",
            },
        }
    )

    assert "Image img-9" in text
    assert "page 3" in text
    assert "Format: png." in text
    assert (
        "Surrounding page context: Bar chart of quarterly revenue"
        in text
    )


def test_image_embedding_text_without_caption_stays_valid():
    text = embedding_service.get_embedding_text(
        {
            "content_type": "image",
            "page_number": 1,
            "metadata": {"image_id": "img-1"},
        }
    )

    assert text.startswith("Image img-1")
    assert "Surrounding page context" not in text


def test_image_embedding_text_is_bounded():
    caption = "chart " * 5000

    text = embedding_service.get_embedding_text(
        {
            "content_type": "image",
            "page_number": 1,
            "metadata": {"image_id": "img", "caption": caption},
        }
    )

    assert len(text) < embedding_service.MAX_IMAGE_CONTEXT_CHARS + 500


# --------------------------------------------------- path safety


def _write_ref(image_dirs, name, payload: bytes):
    document_dir = image_dirs / "doc-safe"
    document_dir.mkdir(exist_ok=True)
    path = document_dir / name
    path.write_bytes(payload)
    return f"storage/images/doc-safe/{name}"


def test_resolve_image_path_rejects_unsafe_and_broken_files(
    tmp_path, image_dirs
):
    monkeypatch_root = tmp_path

    # The resolver anchors at BACKEND_DIR; tests point it at tmp_path.
    original = embedding_service.BACKEND_DIR
    embedding_service.BACKEND_DIR = monkeypatch_root

    try:
        storage = monkeypatch_root / "storage" / "images"
        storage.mkdir(parents=True, exist_ok=True)

        good_ref = _write_ref(storage, "good.png", _png_bytes())
        empty_ref = _write_ref(storage, "empty.png", b"")
        corrupt_ref = _write_ref(storage, "corrupt.png", b"notpng")
        unsupported_ref = _write_ref(storage, "note.txt", b"hello")

        def resolve(ref):
            return embedding_service.resolve_image_path(
                {"content_type": "image", "content_ref": ref}
            )

        assert resolve(good_ref) is not None
        assert resolve(empty_ref) is None
        assert resolve(corrupt_ref) is None
        assert resolve(unsupported_ref) is None
        assert resolve("storage/images/../../etc/passwd") is None
        assert resolve("../etc/passwd") is None
        assert resolve("C:/Windows/system32/config") is None
        assert resolve("/etc/passwd") is None
    finally:
        embedding_service.BACKEND_DIR = original


# --------------------------------------------- dimension guard


class _Field:
    def __init__(self, dimension):
        self.read_parameters = {"dimension": dimension}


class _Schema:
    def __init__(self, dimension):
        self.fields = {"text": _Field(dimension)}


class _ModelBackedIndexModel:
    """Mirrors the real schema-backed index description."""

    def __init__(self, dimension):
        self.dimension = None  # attribute raises on the real object
        self.schema = _Schema(dimension)


class _ClassicIndexModel:
    def __init__(self, dimension):
        self.dimension = dimension


class _FakeUpsertIndex:
    """The data-plane handle: upsert only, no describe_index.

    The real Pinecone 10.x ``Index`` object has no ``describe_index``;
    index administration lives on the ``Pinecone`` client. Reproducing
    that shape here is what catches the production AttributeError.
    """

    def __init__(self):
        self.upserted = []

    def upsert(self, vectors, namespace):
        self.upserted.extend(vectors)


def _unit(embedding):
    return {
        "id": "unit-1",
        "document_id": "doc-1",
        "content_type": "text",
        "page_number": 1,
        "content": "hello",
        "content_ref": None,
        "metadata": {"source": "pdf_text"},
        "embedding": embedding,
    }


@pytest.fixture(autouse=True)
def _reset_dimension_cache():
    pinecone_service.reset_index_cache()
    yield
    pinecone_service.reset_index_cache()


def test_upsert_rejects_vectors_with_the_wrong_dimension(
    monkeypatch,
):
    index = _FakeUpsertIndex()
    monkeypatch.setattr(pinecone_service, "get_index", lambda: index)
    monkeypatch.setattr(
        pinecone_service,
        "describe_index_model",
        lambda name=None: _ClassicIndexModel(1024),
    )

    with pytest.raises(ValueError) as error:
        pinecone_service.upsert_content_units(
            [_unit([0.0] * 768)]
        )

    # The refusal is clear and nothing was written.
    assert "768" in str(error.value)
    assert "1024" in str(error.value)
    assert index.upserted == []


def test_upsert_accepts_matching_dimensions_and_reads_schema(
    monkeypatch,
):
    index = _FakeUpsertIndex()
    monkeypatch.setattr(pinecone_service, "get_index", lambda: index)
    monkeypatch.setattr(
        pinecone_service,
        "describe_index_model",
        lambda name=None: _ModelBackedIndexModel(1024),
    )

    written = pinecone_service.upsert_content_units(
        [_unit([0.0] * 1024)]
    )

    assert written == 1
    assert len(index.upserted) == 1
    # Cached: a second call does not describe the index again.
    assert pinecone_service.get_index_dimension() == 1024


def test_dimension_lookup_uses_the_control_plane_client(monkeypatch):
    """Regression: the data-plane index has no describe_index.

    This is the exact production failure: calling ``describe_index`` on
    the object returned by ``get_index()`` raised AttributeError and
    every real upload failed before it could verify the vector
    dimension.
    """

    class _ControlClient:
        def __init__(self, model):
            self.model = model
            self.described = []

        def describe_index(self, name):
            self.described.append(name)
            return self.model

    data_plane = _FakeUpsertIndex()

    # The real data-plane handle really does lack the method.
    assert not hasattr(data_plane, "describe_index")

    control = _ControlClient(_ModelBackedIndexModel(1024))

    monkeypatch.setattr(pinecone_service, "get_index", lambda: data_plane)
    monkeypatch.setattr(
        pinecone_service, "get_client", lambda: control
    )

    assert pinecone_service.get_index_dimension() == 1024
    assert control.described == [pinecone_service.PINECONE_INDEX_NAME]


def test_search_results_expose_image_captions(monkeypatch):
    class _Match:
        id = "unit-1"
        score = 0.9
        metadata = {
            "document_id": "doc-1",
            "content_type": "image",
            "page_number": 2,
            "content": "",
            "content_ref": "storage/images/doc-1/x.png",
            "source": "pdf_image",
            "image_id": "doc-1_image_1",
            "caption": "Revenue chart",
        }

    class _Response:
        matches = [_Match()]

    class _QueryIndex:
        def query(self, **_kwargs):
            return _Response()

    monkeypatch.setattr(
        pinecone_service, "get_index", lambda: _QueryIndex()
    )

    results = pinecone_service.search_similar_content(
        query_embedding=[0.0] * 1024
    )

    assert results[0]["image_id"] == "doc-1_image_1"
    assert results[0]["caption"] == "Revenue chart"
