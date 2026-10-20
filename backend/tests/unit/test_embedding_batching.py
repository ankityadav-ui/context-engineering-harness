"""Batching tests for Gemini embeddings and Pinecone upserts.

A large PDF produces hundreds of content units. Sending one embedding
request per unit hit the rate limit, and one oversized upsert request was
rejected, both surfacing as a generic upload failure. These tests pin the
batched behavior without touching any live service.
"""

from types import SimpleNamespace

import pytest

from app.rag import embedding_service, pinecone_service


class FakeEmbeddingClient:
    """Records batch sizes and returns ordered fake vectors."""

    def __init__(self, error_code=None):
        self.calls: list[list[str]] = []
        self.error_code = error_code
        self.models = SimpleNamespace(embed_content=self._embed)

    def _embed(self, *, model, contents, config):
        self.calls.append(list(contents))

        if self.error_code is not None:
            error = RuntimeError(f"upstream error {self.error_code}")
            error.code = self.error_code
            raise error

        return SimpleNamespace(
            embeddings=[
                SimpleNamespace(
                    values=[float(text)]
                    * embedding_service.EMBEDDING_DIMENSION
                )
                for text in contents
            ]
        )


def test_embeddings_are_batched_and_keep_input_order(monkeypatch):
    client = FakeEmbeddingClient()
    monkeypatch.setattr(embedding_service, "get_client", lambda: client)

    texts = [str(index) for index in range(120)]

    vectors = embedding_service.generate_embeddings(texts)

    # 120 units -> 50 + 50 + 20, never one request per unit.
    assert [len(batch) for batch in client.calls] == [50, 50, 20]
    assert len(vectors) == 120
    assert [vector[0] for vector in vectors] == [
        float(index) for index in range(120)
    ]


def test_a_rate_limited_batch_is_retried(monkeypatch):
    client = FakeEmbeddingClient(error_code=429)
    monkeypatch.setattr(embedding_service, "get_client", lambda: client)
    monkeypatch.setattr(embedding_service.time, "sleep", lambda _s: None)

    class FlakyClient(FakeEmbeddingClient):
        def _embed(self, *, model, contents, config):
            if len(self.calls) == 0:
                self.calls.append(list(contents))
                error = RuntimeError("rate limited")
                error.code = 429
                raise error

            return super()._embed(
                model=model, contents=contents, config=config
            )

    flaky = FlakyClient()
    flaky.models = SimpleNamespace(embed_content=flaky._embed)
    monkeypatch.setattr(embedding_service, "get_client", lambda: flaky)

    vectors = embedding_service.generate_embeddings(["7"])

    assert vectors[0][0] == 7.0
    # One failed attempt plus the successful retry.
    assert len(flaky.calls) == 2


def test_a_permanent_embedding_error_is_not_retried(monkeypatch):
    client = FakeEmbeddingClient(error_code=400)
    monkeypatch.setattr(embedding_service, "get_client", lambda: client)

    sleeps: list[float] = []
    monkeypatch.setattr(
        embedding_service.time, "sleep", lambda seconds: sleeps.append(seconds)
    )

    with pytest.raises(RuntimeError):
        embedding_service.generate_embeddings(["7"])

    assert len(client.calls) == 1
    assert sleeps == []


def test_empty_embedding_input_is_rejected(monkeypatch):
    client = FakeEmbeddingClient()
    monkeypatch.setattr(embedding_service, "get_client", lambda: client)

    with pytest.raises(ValueError):
        embedding_service.generate_embeddings(["ok", "   "])

    assert client.calls == []


def _unit(index: int) -> dict:
    return {
        "id": f"unit-{index}",
        "document_id": "doc-batch",
        "content_type": "text",
        "page_number": 1,
        "content": f"unit {index}",
        "content_ref": None,
        "embedding": [0.0] * 1024,
        "metadata": {"source": "pdf_text"},
    }


def test_large_upserts_are_split_into_batches(monkeypatch):
    class RecordingIndex:
        def __init__(self):
            self.batches: list[int] = []

        def upsert(self, *, vectors, namespace):
            self.batches.append(len(vectors))

    index = RecordingIndex()

    monkeypatch.setattr(pinecone_service, "get_index", lambda: index)
    monkeypatch.setattr(
        pinecone_service, "get_index_dimension", lambda: 1024
    )

    written = pinecone_service.upsert_content_units(
        [_unit(index) for index in range(250)]
    )

    assert written == 250
    assert index.batches == [100, 100, 50]


def test_a_single_unit_still_uses_one_upsert(monkeypatch):
    class RecordingIndex:
        def __init__(self):
            self.batches: list[int] = []

        def upsert(self, *, vectors, namespace):
            self.batches.append(len(vectors))

    index = RecordingIndex()

    monkeypatch.setattr(pinecone_service, "get_index", lambda: index)
    monkeypatch.setattr(
        pinecone_service, "get_index_dimension", lambda: 1024
    )

    assert pinecone_service.upsert_content_units([_unit(0)]) == 1
    assert index.batches == [1]
