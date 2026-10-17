
# Import the existing entity detectors.
from app.masking.ner_service import detect_entities
from app.masking.rule_service import detect_rule_entities

# Import the resolver and shared result schema.
from app.masking.entity_resolver import resolve_entities
from app.masking.schemas import MaskingResult


class PlaceholderRegistry:
    """Allocates stable placeholders across many text units.

    One registry is shared by every chunk of a document so the same
    value always receives the same placeholder. Without it, "John Doe"
    in chunk 1 and in chunk 4 would be numbered independently
    ([PERSON_1] and [PERSON_2]), which makes the masked corpus
    inconsistent and leaks how often a value occurs.

    The registry never stores or logs the original values.
    """

    def __init__(self):
        self._placeholders: dict[tuple[str, str], str] = {}
        self._counters: dict[str, int] = {}

    def placeholder_for(self, entity_type: str, text: str) -> str:
        """Return the placeholder for one detected value."""

        key = (entity_type, text.casefold())

        placeholder = self._placeholders.get(key)

        if placeholder is not None:
            return placeholder

        self._counters[entity_type] = (
            self._counters.get(entity_type, 0) + 1
        )

        placeholder = (
            f"[{entity_type}_{self._counters[entity_type]}]"
        )
        self._placeholders[key] = placeholder

        return placeholder

    def masked_value_count(self) -> int:
        """Return how many distinct values have been masked."""

        return len(self._placeholders)


def mask_document_text(
    text: str,
    document_id: str,
    registry: PlaceholderRegistry | None = None,
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

    # Map each unique entity value and type to one placeholder. When a
    # registry is supplied (multi-chunk indexing) numbering continues
    # across units, so repeated values keep one stable placeholder.
    active_registry = registry or PlaceholderRegistry()

    # Store replacement spans separately from the original text.
    replacements = []

    for entity in entities:
        # Record the exact span and its replacement.
        replacements.append(
            (
                entity.start,
                entity.end,
                active_registry.placeholder_for(
                    entity.entity_type,
                    entity.text,
                ),
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
