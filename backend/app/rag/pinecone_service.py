import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone


# Load environment variables from the repository-root .env explicitly,
# so configuration does not depend on the process working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

load_dotenv(PROJECT_ROOT / ".env")


# Read the Pinecone index name.
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "context-engineering-harness",
)


# Keep new multimodal vectors isolated from old vectors.
PINECONE_NAMESPACE = "multimodal-v2"


# Bound how much of the namespace a single document listing reads, so the
# endpoint stays predictable even on a large index.
MAX_DOCUMENT_SCAN_VECTORS = 1000

# Maximum vectors sent in one Pinecone upsert request. 100 vectors of
# 1024 float32 values plus metadata stay well under the 2 MB request cap.
UPSERT_BATCH_SIZE = 100
DOCUMENT_FETCH_BATCH_SIZE = 100

# Bound how many content units one document lookup or delete reads.
MAX_DOCUMENT_UNITS = 1000
DOCUMENT_UNIT_PAGE_SIZE = 100


# Hold the control-plane client and the data-plane index handle until
# they are actually needed. Creating them at import time would stop
# unrelated API routes from starting when Pinecone is not configured.
_client: Pinecone | None = None
_index = None

# Cached index dimension, read once from the live index description so
# an incompatible vector can never be upserted (see
# get_index_dimension). None means "not read yet".
_index_dimension = None


def get_client() -> Pinecone:
    """Return the Pinecone control-plane client (created on first use).

    Index administration (describing an index) lives on this client.
    ``get_index()`` returns a different object: the data-plane handle
    used for upsert/query, which has no ``describe_index`` in the
    current Pinecone SDK.
    """

    global _client

    if _client is None:
        api_key = os.getenv("PINECONE_API_KEY")

        if not api_key:
            raise RuntimeError(
                "PINECONE_API_KEY is not configured. Set it in the "
                "root .env file to use document retrieval and indexing."
            )

        _client = Pinecone(api_key=api_key)

    return _client


def describe_index_model(name: str | None = None):
    """Return the index description from the control-plane client.

    The data-plane index handle returned by :func:`get_index` has no
    ``describe_index``; calling it there raised ``AttributeError`` on
    every real upload and made the dimension check fail closed.
    """

    return get_client().describe_index(name or PINECONE_INDEX_NAME)


def get_index_dimension() -> int:
    """Return the configured index dimension, read from Pinecone.

    Supports both classic indexes (``model.dimension``) and
    model-backed indexes whose dimension lives in the schema's field
    read parameters. The value is cached per process. The index is
    never recreated or reconfigured by this call.
    """

    global _index_dimension

    if _index_dimension is not None:
        return _index_dimension

    model = describe_index_model()

    dimension = getattr(model, "dimension", None)

    if not isinstance(dimension, int) or dimension <= 0:
        # Model-backed index: the dense dimension lives in the field
        # schema (for example llama-text-embed-v2 at 1024).
        schema = getattr(model, "schema", None)
        fields = getattr(schema, "fields", None) or {}

        for field in getattr(fields, "values", lambda: [])():
            read_parameters = (
                getattr(field, "read_parameters", None) or {}
            )
            candidate = read_parameters.get("dimension")

            if isinstance(candidate, int) and candidate > 0:
                dimension = candidate
                break

    if not isinstance(dimension, int) or dimension <= 0:
        raise RuntimeError(
            "The Pinecone index dimension could not be determined; "
            "refusing to upsert vectors that cannot be verified "
            "against the index."
        )

    _index_dimension = dimension

    return dimension


def reset_index_cache() -> None:
    """Forget the cached client, index handle, and dimension."""

    global _client, _index, _index_dimension

    _client = None
    _index = None
    _index_dimension = None


def is_configured() -> bool:
    """Report whether PINECONE_API_KEY is set, without connecting."""

    return bool(os.getenv("PINECONE_API_KEY"))


def get_index():
    """Return the index handle for upsert/query, created on first use."""

    global _index

    if _index is None:
        _index = get_client().Index(PINECONE_INDEX_NAME)

    return _index


def __getattr__(name):
    # Preserve the previous module-level name for callers that import
    # `index` directly, while still connecting lazily.
    if name == "index":
        return get_index()

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def upsert_content_units(
    embedded_units: list[dict],
) -> int:
    """
    Store multimodal embeddings in the dedicated
    Pinecone namespace.

    Every vector is verified against the live index dimension before
    it is sent; a mismatch aborts the whole upsert instead of mixing
    incompatible vectors into the index.
    """
    vectors = []

    expected_dimension = (
        get_index_dimension() if embedded_units else None
    )

    for unit in embedded_units:
        embedding = unit.get("embedding") or []

        if (
            expected_dimension is not None
            and len(embedding) != expected_dimension
        ):
            raise ValueError(
                "Refusing to upsert a vector with "
                f"{len(embedding)} dimensions: the configured index "
                f"expects {expected_dimension}. Nothing was written."
            )

        metadata = {
            "document_id": unit["document_id"],
            "content_type": unit["content_type"],
            "page_number": unit["page_number"],
            "content": unit.get("content", ""),
            "content_ref": unit.get("content_ref") or "",
            "source": unit["metadata"].get(
                "source",
                "unknown",
            ),
        }
        for key in ("page_count", "pages_without_text"):
            if key in unit["metadata"]:
                metadata[key] = unit["metadata"][key]

        if unit["content_type"] == "table":
            metadata["table_number"] = (
                unit["metadata"].get(
                    "table_number",
                    0,
                )
            )

        if unit["content_type"] == "image":
            metadata["image_id"] = (
                unit["metadata"].get(
                    "image_id",
                    "",
                )
            )

            # Caption / surrounding page context makes the image
            # retrievable by content, and the content hash lets equal
            # assets be recognized across documents.
            metadata["caption"] = (
                unit["metadata"].get("caption") or ""
            )

            image_hash = unit["metadata"].get("image_sha256")

            if image_hash:
                metadata["image_sha256"] = image_hash

        vector = {
            "id": unit["id"],
            "values": unit["embedding"],
            "metadata": metadata,
        }

        vectors.append(vector)

    if not vectors:
        return 0

    # Pinecone caps an upsert request by size (2 MB on serverless). A
    # large PDF produces hundreds of 1024-dimension vectors, so sending
    # them in one request was rejected by the API and surfaced as a
    # generic upload failure. Batches keep each request small.
    for start in range(0, len(vectors), UPSERT_BATCH_SIZE):
        get_index().upsert(
            vectors=vectors[start:start + UPSERT_BATCH_SIZE],
            namespace=PINECONE_NAMESPACE,
        )

    return len(vectors)


def search_similar_content(
    query_embedding: list[float],
    top_k: int = 5,
    document_id: str | None = None,
) -> list[dict]:
    """
    Search only the new multimodal namespace.
    """
    search_filter = None

    if document_id:
        search_filter = {
            "document_id": {
                "$eq": document_id
            }
        }

    response = get_index().query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=PINECONE_NAMESPACE,
        filter=search_filter,
    )

    results = []

    for match in response.matches:
        metadata = match.metadata or {}

        result = {
            "id": match.id,
            "score": match.score,
            "document_id": metadata.get(
                "document_id"
            ),
            "content_type": metadata.get(
                "content_type"
            ),
            "page_number": metadata.get(
                "page_number"
            ),
            "content": metadata.get(
                "content",
                "",
            ),
            "content_ref": metadata.get(
                "content_ref",
                "",
            ),
            "source": metadata.get(
                "source",
                "unknown",
            ),
        }

        if "table_number" in metadata:
            result["table_number"] = (
                metadata["table_number"]
            )

        if "image_id" in metadata:
            result["image_id"] = (
                metadata["image_id"]
            )

        if "caption" in metadata:
            result["caption"] = metadata["caption"]

        results.append(result)

    return results


def _validate_document_id(document_id: str) -> str:
    """Reject an empty document identifier before querying Pinecone."""

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id cannot be empty.")

    return document_id.strip()


def list_indexed_documents(
    max_vectors: int = MAX_DOCUMENT_SCAN_VECTORS,
) -> dict:
    """
    Group the indexed vectors by document_id.

    The namespace stores one vector per content unit, so documents are
    reconstructed from the vector metadata rather than from a separate
    document table. Only metadata is read; vector values are not used.
    """

    index = get_index()

    vector_ids = []
    scan_truncated = False

    for page in index.list(
        limit=DOCUMENT_FETCH_BATCH_SIZE,
        namespace=PINECONE_NAMESPACE,
    ):
        for item in page.vectors:
            # A listed entry can carry no id; it cannot be fetched.
            if item.id is None:
                continue

            if len(vector_ids) >= max_vectors:
                scan_truncated = True
                break

            vector_ids.append(item.id)

        if scan_truncated:
            break

    documents: dict[str, dict] = {}

    for start in range(0, len(vector_ids), DOCUMENT_FETCH_BATCH_SIZE):
        batch = vector_ids[
            start:start + DOCUMENT_FETCH_BATCH_SIZE
        ]

        response = index.fetch(
            ids=batch,
            namespace=PINECONE_NAMESPACE,
        )

        for vector in response.vectors.values():
            metadata = vector.metadata or {}
            document_id = metadata.get("document_id")

            if not document_id:
                continue

            entry = documents.setdefault(
                str(document_id),
                {
                    "document_id": str(document_id),
                    "content_units": 0,
                    "content_types": set(),
                    "pages": set(),
                    "sources": set(),
                },
            )

            entry["content_units"] += 1

            content_type = metadata.get("content_type")

            if content_type:
                entry["content_types"].add(str(content_type))

            page_number = metadata.get("page_number")

            if page_number is not None:
                entry["pages"].add(page_number)

            source = metadata.get("source")

            if source:
                entry["sources"].add(str(source))

    documents_list = [
        {
            "document_id": entry["document_id"],
            "content_units": entry["content_units"],
            "content_types": sorted(entry["content_types"]),
            "pages": sorted(
                page
                for page in entry["pages"]
                if isinstance(page, int)
            ),
            "sources": sorted(entry["sources"]),
        }
        for entry in documents.values()
    ]

    documents_list.sort(
        key=lambda document: document["document_id"]
    )

    return {
        "documents": documents_list,
        "scanned_vectors": len(vector_ids),
        "scan_truncated": scan_truncated,
    }


def get_document_content_units(
    document_id: str,
    max_units: int = MAX_DOCUMENT_UNITS,
) -> dict:
    """
    Describe one document's content units.

    Returns unit metadata only (type, page, source, reference, and the
    length of the stored text). The stored text itself is deliberately
    not returned, because indexed content has not been masked.
    """

    document_id = _validate_document_id(document_id)

    index = get_index()

    filter_expression = {
        "document_id": {"$eq": document_id}
    }

    units = []
    units_truncated = False
    pagination_token = None

    while True:
        page = index.fetch_by_metadata(
            filter=filter_expression,
            namespace=PINECONE_NAMESPACE,
            limit=DOCUMENT_UNIT_PAGE_SIZE,
            pagination_token=pagination_token,
        )

        for vector in page.vectors.values():
            metadata = vector.metadata or {}
            content = metadata.get("content") or ""

            unit = {
                "unit_id": vector.id,
                "content_type": metadata.get("content_type"),
                "page_number": metadata.get("page_number"),
                "source": metadata.get("source"),
                "content_length": (
                    len(content)
                    if isinstance(content, str)
                    else 0
                ),
            }

            for key in (
                "table_number",
                "image_id",
                "content_ref",
                "caption",
            ):
                if metadata.get(key) is not None:
                    unit[key] = metadata[key]

            units.append(unit)

            if len(units) >= max_units:
                units_truncated = True
                break

        if units_truncated:
            break

        pagination = page.pagination

        pagination_token = (
            pagination.next
            if pagination is not None
            else None
        )

        if not pagination_token:
            break

    # Present the units in document order.
    units.sort(
        key=lambda unit: (
            unit.get("page_number") is None,
            unit.get("page_number") or 0,
            str(unit.get("content_type") or ""),
        )
    )

    return {
        "document_id": document_id,
        "content_units": len(units),
        "units": units,
        "units_truncated": units_truncated,
    }


def get_document_searchable_units(
    document_id: str,
    max_units: int = MAX_DOCUMENT_UNITS,
) -> dict:
    """Read text-bearing indexed units for exactly one document.

    Unlike semantic search, this returns the complete bounded source set
    with its original unit id, page, and document metadata. It is intended
    for grounded whole-document summaries, not user-facing document detail.
    """

    document_id = _validate_document_id(document_id)
    index = get_index()
    filter_expression = {"document_id": {"$eq": document_id}}
    units: list[dict] = []
    seen_unit_ids: set[str] = set()
    pages_without_text: list[int] = []
    page_count = None
    document_found = False
    pagination_token = None
    truncated = False

    while True:
        page = index.fetch_by_metadata(
            filter=filter_expression,
            namespace=PINECONE_NAMESPACE,
            limit=DOCUMENT_UNIT_PAGE_SIZE,
            pagination_token=pagination_token,
        )
        for vector in page.vectors.values():
            metadata = vector.metadata or {}
            if str(metadata.get("document_id") or "") != document_id:
                continue
            document_found = True

            if page_count is None:
                candidate_page_count = metadata.get("page_count")
                if isinstance(candidate_page_count, int):
                    page_count = candidate_page_count
                candidate_missing_pages = metadata.get(
                    "pages_without_text"
                )
                if isinstance(candidate_missing_pages, list):
                    pages_without_text = [
                        page_number
                        for page_number in candidate_missing_pages
                        if isinstance(page_number, int)
                    ]

            if metadata.get("content_type") not in {"text", "table"}:
                continue
            content = metadata.get("content")
            if not isinstance(content, str) or not content.strip():
                continue

            unit_id = str(vector.id)
            if unit_id in seen_unit_ids:
                continue

            if len(units) >= max_units:
                truncated = True
                break

            seen_unit_ids.add(unit_id)
            units.append(
                {
                    "unit_id": unit_id,
                    "document_id": document_id,
                    "content_type": metadata["content_type"],
                    "page_number": metadata.get("page_number"),
                    "content": content,
                    "source": metadata.get("source", "unknown"),
                }
            )

        if truncated:
            break
        pagination = page.pagination
        pagination_token = (
            pagination.next if pagination is not None else None
        )
        if not pagination_token:
            break

    units.sort(
        key=lambda unit: (
            unit["page_number"] is None,
            unit["page_number"] or 0,
            unit["unit_id"],
        )
    )
    return {
        "document_id": document_id,
        "document_found": document_found,
        "units": units,
        "page_count": page_count,
        "pages_without_text": sorted(set(pages_without_text)),
        "units_truncated": truncated,
    }


def delete_document_from_index(document_id: str) -> dict:
    """
    Delete every vector stored for one document.

    The document is looked up first so a missing document is reported
    instead of silently deleting nothing. Pinecone deletes by metadata
    filter, and a successful call is not proof that anything was removed,
    so the number of units seen before the delete is reported back.
    """

    document_id = _validate_document_id(document_id)

    index = get_index()

    filter_expression = {
        "document_id": {"$eq": document_id}
    }

    first_page = index.fetch_by_metadata(
        filter=filter_expression,
        namespace=PINECONE_NAMESPACE,
        limit=DOCUMENT_UNIT_PAGE_SIZE,
    )

    matched_on_first_page = len(first_page.vectors)

    if matched_on_first_page == 0:
        return {
            "document_id": document_id,
            "matched_content_units_on_first_page": 0,
            "index_delete_requested": False,
        }

    index.delete(
        filter=filter_expression,
        namespace=PINECONE_NAMESPACE,
    )

    return {
        "document_id": document_id,
        "matched_content_units_on_first_page": matched_on_first_page,
        "index_delete_requested": True,
    }