
# Import regular expressions for the document-header name rule.
import re

# Import the existing entity detectors.
from app.masking.ner_service import detect_entities
from app.masking.rule_service import detect_rule_entities

# Import the resolver and shared result schema.
from app.masking.entity_resolver import resolve_entities
from app.masking.schemas import SensitiveEntity, MaskingResult


# A document header (the first line of a resume or letter) is the most
# common place a person's real name appears. ``en_core_web_sm`` misses
# many non-western names entirely (for example "Priya Nair" or
# "Ankit Yadav" are labelled as nothing at all), so the masked pipeline
# used to send the full name to the model untouched. The rule below is
# deliberately narrow: it only applies to the FIRST non-empty line of
# the first text unit, and only when that line looks like a bare name.
HEADER_NAME_MIN_TOKENS = 2
HEADER_NAME_MAX_TOKENS = 4

# One capitalized, alphabetic token (initials and hyphens/apostrophes
# allowed), with no digits.
_HEADER_NAME_TOKEN_RE = re.compile(r"^[A-Z][A-Za-z'\u2019.\-]*$")

# Structural headings that must never be mistaken for a person name.
# Generic document-section vocabulary, not document-specific entries.
_HEADER_SECTION_WORDS = frozenset({
    "about", "academics", "achievements", "additional", "awards",
    "certifications", "contact", "declaration", "details", "education",
    "employment", "experience", "interests", "internship", "languages",
    "objective", "personal", "profile", "project", "projects",
    "publications", "references", "skills", "summary", "technical",
    "training", "work",
})


def _detect_header_name(text: str) -> SensitiveEntity | None:
    """Return the leading name line as a PERSON entity, when it is one.

    Only the first non-empty line is considered. The line must be 2-4
    capitalized alphabetic tokens and must not be a known section
    heading ("Work Experience", "Technical Skills"). A line that fails
    any check stops the search: this never reaches into the body.
    """

    offset = 0

    for raw_line in text.splitlines(keepends=True):
        line = raw_line.strip()

        if not line:
            offset += len(raw_line)
            continue

        tokens = line.split()

        if not (
            HEADER_NAME_MIN_TOKENS
            <= len(tokens)
            <= HEADER_NAME_MAX_TOKENS
        ):
            return None

        if not all(
            _HEADER_NAME_TOKEN_RE.match(token) for token in tokens
        ):
            return None

        if any(
            token.casefold().strip(".") in _HEADER_SECTION_WORDS
            for token in tokens
        ):
            return None

        start = offset + (len(raw_line) - len(raw_line.lstrip()))

        return SensitiveEntity(
            text=line,
            entity_type="PERSON",
            start=start,
            end=start + len(line),
            confidence=0.6,
        )

    return None


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


def _line_boundary_text(text: str) -> tuple[str, list[int]]:
    """Insert sentence boundaries at line breaks and map offsets back.

    PDF extraction returns one line per text row. spaCy treats the newline
    as plain whitespace, so a resume heading and the following line are
    read as one span (``"Education\\nExample University"`` becomes a
    single ORG). Masking that span deletes the heading and turns an
    otherwise readable resume into disconnected placeholders.

    Replacing each line break with ``". "`` lets the detector end a
    sentence at the line break without changing any other character. The
    returned ``origin`` maps every index of the rewritten text back to
    the original string, so replacements still use the real offsets.
    """

    characters: list[str] = []
    origin: list[int] = []
    index = 0

    while index < len(text):
        character = text[index]

        if character in {"\r", "\n"}:
            # Treat CRLF as one line break so no empty sentence appears.
            if (
                character == "\r"
                and index + 1 < len(text)
                and text[index + 1] == "\n"
            ):
                index += 1

            characters.extend((".", " "))
            origin.extend((index, index))
        else:
            characters.append(character)
            origin.append(index)

        index += 1

    return "".join(characters), origin


def _detect_ner_entities(text: str) -> list[SensitiveEntity]:
    """Detect named entities without letting a span cross a line break."""

    boundary_text, origin = _line_boundary_text(text)
    entities: list[SensitiveEntity] = []

    for entity in detect_entities(boundary_text):
        if entity.start >= len(origin) or entity.end <= entity.start:
            continue

        start = origin[entity.start]
        end = origin[entity.end - 1] + 1

        # The span boundary is a line break: use the original text for
        # the entity so span validation and replacement stay exact.
        span_text = text[start:end]

        if not span_text.strip():
            continue

        entities.append(
            SensitiveEntity(
                text=span_text,
                entity_type=entity.entity_type,
                start=start,
                end=end,
                confidence=entity.confidence,
            )
        )

    return entities


def mask_document_text(
    text: str,
    document_id: str,
    registry: PlaceholderRegistry | None = None,
    include_header_name: bool = False,
) -> MaskingResult:
    # Reject empty input instead of producing a misleading result.
    if not text or not text.strip():
        raise ValueError("Document text cannot be empty.")

    # Detect names, organizations, locations, and dates using NLP. Line
    # boundaries are preserved as sentence boundaries so a section heading
    # is never merged into the entity on the next line.
    ner_entities = _detect_ner_entities(text)

    # The first text unit's leading line is the document header. spaCy
    # misses many real names there, so a narrow layout rule adds it as a
    # PERSON so a masked upload never sends the name unmasked.
    if include_header_name:
        header_entity = _detect_header_name(text)

        if header_entity is not None:
            ner_entities.append(header_entity)

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
