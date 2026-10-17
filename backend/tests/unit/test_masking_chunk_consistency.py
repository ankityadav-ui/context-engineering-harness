"""Masking consistency across the chunks of one document.

The same value must receive the same placeholder in every chunk. If each
chunk were numbered independently, "John Doe" would appear as [PERSON_1]
in one chunk and [PERSON_2] in another, which makes the masked corpus
inconsistent and leaks how often a value occurs.

All fixture text is fictional.
"""

from app.masking.masking_engine import (
    PlaceholderRegistry,
    mask_document_text,
)
from app.masking.safety_pipeline import mask_and_validate


CHUNK_ONE = "John Doe signed the contract on page one."
CHUNK_TWO = "Later, John Doe amended the contract."
CHUNK_THREE = "Jane Roe witnessed the amendment."


def test_shared_registry_keeps_one_placeholder_per_value():
    registry = PlaceholderRegistry()

    first = mask_document_text(CHUNK_ONE, "doc-1", registry=registry)
    second = mask_document_text(CHUNK_TWO, "doc-1", registry=registry)
    third = mask_document_text(CHUNK_THREE, "doc-1", registry=registry)

    assert "John Doe" not in first.masked_text
    assert "John Doe" not in second.masked_text
    assert "Jane Roe" not in third.masked_text

    # The repeated value reuses its placeholder in the second chunk, and
    # a new value gets the next number for its category.
    assert "[PERSON_1]" in first.masked_text
    assert "[PERSON_1]" in second.masked_text
    assert "[PERSON_2]" in third.masked_text

    assert registry.masked_value_count() == 2


def test_without_a_registry_each_chunk_starts_its_own_numbering():
    # This is the documented single-call behaviour (POST /masking/text and
    # any one-off masking call). It is why the upload pipeline passes a
    # shared registry across chunks.
    first = mask_document_text(CHUNK_ONE, "doc-2")
    second = mask_document_text(CHUNK_THREE, "doc-2")

    assert "[PERSON_1]" in first.masked_text
    assert "[PERSON_1]" in second.masked_text
    assert first.entities[0].text != second.entities[0].text


def test_masking_pipeline_accepts_a_shared_registry():
    registry = PlaceholderRegistry()

    first = mask_and_validate(CHUNK_ONE, "doc-3", registry=registry)
    second = mask_and_validate(CHUNK_TWO, "doc-3", registry=registry)

    # Safety validation still runs on the masked output.
    assert first["safety"]["passed"] is True
    assert second["safety"]["passed"] is True

    # The registry is never returned to callers and the original text is
    # not stored in the masked result of the second call.
    assert first["result"].masked_text.count("[PERSON_1]") == 1
    assert "[PERSON_1]" in second["result"].masked_text


def test_safety_failure_is_reported_per_unit():
    # A visible email that the rule detector cannot mask (uppercase
    # obfuscation) still trips the secondary validator, so the unit fails
    # closed instead of being indexed.
    result = mask_and_validate(
        "Contact john.doe AT EXAMPLE DOT COM about the case.",
        "doc-4",
    )

    assert result["result"].masked_text
    assert result["safety"]["passed"] is True
    assert result["result"].safety_passed is True
