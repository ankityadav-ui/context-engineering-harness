
# Import the existing entity detectors.
from app.masking.ner_service import detect_entities
from app.masking.rule_service import detect_rule_entities

# Import the resolver and shared result schema.
from app.masking.entity_resolver import resolve_entities
from app.masking.schemas import MaskingResult


def mask_document_text(
    text: str,
    document_id: str,
) -> MaskingResult:
    # Reject empty input instead of producing a misleading result.
    if not text or not text.strip():
        raise ValueError("Document text cannot be empty.")

    # Detect names, organizations, locations, and dates using NLP.
    ner_entities = detect_entities(text)

    # Detect emails, phones, and identifier patterns using rules.
    rule_entities = detect_rule_entities(text)

    # Combine both detector results into a single collection.
    all_entities = ner_entities + rule_entities

    # Remove duplicate detections and resolve overlapping spans.
    entities = resolve_entities(
        entities=all_entities,
        source_text=text,
    )

    # Map each unique entity value and type to one placeholder.
    # Repeated occurrences of the same value reuse the placeholder.
    placeholders = {}
    counters = {}

    # Store replacement spans separately from the original text.
    replacements = []

    for entity in entities:
        # Create a key that distinguishes entity types.
        entity_key = (
            entity.entity_type,
            entity.text.casefold(),
        )

        # Assign a placeholder only the first time this value appears.
        if entity_key not in placeholders:
            # Increment the counter for this entity category.
            counters[entity.entity_type] = (
                counters.get(entity.entity_type, 0) + 1
            )

            # Create a readable placeholder such as [PERSON_1].
            placeholders[entity_key] = (
                f"[{entity.entity_type}_"
                f"{counters[entity.entity_type]}]"
            )

        # Record the exact span and its replacement.
        replacements.append(
            (
                entity.start,
                entity.end,
                placeholders[entity_key],
            )
        )

    # Apply replacements from right to left.
    # Earlier character offsets remain valid as we edit the string.
    masked_text = text
    for start, end, placeholder in sorted(
        replacements,
        key=lambda item: item[0],
        reverse=True,
    ):
        masked_text = (
            masked_text[:start]
            + placeholder
            + masked_text[end:]
        )

    # Return the internal result for controlled processing.
    # Do not expose original_text or raw entity values to the LLM.
    return MaskingResult(
        document_id=document_id,
        original_text=text,
        masked_text=masked_text,
        entities=entities,
        safety_passed=False,
    )
