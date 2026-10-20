import json

import pytest

from app.context_engine import summarizer
from app.llm import LLMError


def _unit(unit_id, page, text, document_id="doc-selected"):
    return {
        "unit_id": unit_id,
        "document_id": document_id,
        "content_type": "text",
        "page_number": page,
        "content": text,
        "source": "pdf_text",
    }


def test_summary_uses_only_verifiable_excerpts_from_selected_document(
    monkeypatch,
):
    first = _unit(
        "chunk-1",
        1,
        "The device measures temperature during the manufacturing process.",
    )
    second = _unit(
        "chunk-2",
        4,
        "The garden contains native plants and a seasonal pond.",
    )
    unrelated = _unit(
        "chunk-other",
        2,
        "Person E owns Organization B near Place A.",
        document_id="doc-other",
    )
    prompts = []

    def fake_generate(prompt):
        prompts.append(prompt)
        return json.dumps(
            [
                {
                    "unit_id": "chunk-1",
                    "quote": (
                        "The device measures temperature during the "
                        "manufacturing process."
                    ),
                },
                {
                    "unit_id": "chunk-2",
                    "quote": "The garden contains native plants",
                },
                {
                    "unit_id": "chunk-1",
                    "quote": (
                        "The device measures temperature and owns the "
                        "garden."
                    ),
                },
                {
                    "unit_id": "chunk-other",
                    "quote": "Person E owns Organization B near Place A.",
                },
            ]
        )

    monkeypatch.setattr(summarizer, "generate_response", fake_generate)

    result = summarizer.summarize_document_units(
        [first, second, first, unrelated],
        document_id="doc-selected",
    )

    assert result.count("- **Page") == 2
    assert "Page 1" in result
    assert "Page 4" in result
    assert "device measures temperature" in result
    assert "native plants" in result
    assert "owns the garden" not in result
    assert "Person E" not in result
    assert len(prompts) == 1
    assert "doc-other" not in prompts[0]


def test_existing_placeholder_alias_is_validated_without_unmasking(
    monkeypatch,
):
    source = _unit(
        "chunk-1",
        3,
        "The report was prepared by [PERSON_1] for the review committee.",
    )
    monkeypatch.setattr(
        summarizer,
        "generate_response",
        lambda _prompt: json.dumps(
            [
                {
                    "unit_id": "chunk-1",
                    "quote": (
                        "The report was prepared by Person A for the "
                        "review committee."
                    ),
                }
            ]
        ),
    )

    result = summarizer.summarize_document_units(
        [source],
        document_id="doc-selected",
    )

    assert "Person A" in result
    assert "[PERSON_1]" not in result


def test_missing_ocr_text_is_reported_without_llm_call(monkeypatch):
    def unexpected_call(_prompt):
        raise AssertionError("empty source must not be sent to the LLM")

    monkeypatch.setattr(summarizer, "generate_response", unexpected_call)

    result = summarizer.summarize_document_units(
        [],
        document_id="doc-selected",
        pages_without_text=[2, 5],
    )

    assert "page(s) 2, 5" in result
    assert "no content has been inferred" in result


def test_summarization_failure_is_explicit(monkeypatch):
    monkeypatch.setattr(
        summarizer,
        "generate_response",
        lambda _prompt: "not json",
    )

    with pytest.raises(LLMError, match="invalid source excerpts"):
        summarizer.summarize_document_units(
            [_unit("chunk-1", 1, "A sufficiently long source passage.")],
            document_id="doc-selected",
        )


def test_large_documents_are_summarized_in_source_batches(monkeypatch):
    monkeypatch.setattr(summarizer, "SUMMARY_BATCH_CHARS", 30)
    sources = [
        _unit("chunk-1", 1, "The first independent topic is about sensors."),
        _unit("chunk-2", 2, "The second independent topic is about gardens."),
    ]
    prompts = []

    def fake_generate(prompt):
        prompts.append(prompt)
        parsed_sources = prompt.split("Sources:\n", 1)[1]
        source_list = json.loads(parsed_sources)
        source = source_list[0]
        quote = source["text"]
        return json.dumps(
            [{"unit_id": source["unit_id"], "quote": quote}]
        )

    monkeypatch.setattr(summarizer, "generate_response", fake_generate)

    result = summarizer.summarize_document_units(
        sources,
        document_id="doc-selected",
    )

    assert len(prompts) == 2
    assert "sensors" in result
    assert "gardens" in result


def test_source_size_limit_is_explicit(monkeypatch):
    monkeypatch.setattr(summarizer, "SUMMARY_MAX_SOURCE_CHARS", 10)
    monkeypatch.setattr(
        summarizer,
        "generate_response",
        lambda _prompt: (_ for _ in ()).throw(
            AssertionError("over-limit sources must not be sent")
        ),
    )

    with pytest.raises(LLMError, match="safe whole-document summary size"):
        summarizer.summarize_document_units(
            [_unit("chunk-1", 1, "This source exceeds the limit.")],
            document_id="doc-selected",
        )
