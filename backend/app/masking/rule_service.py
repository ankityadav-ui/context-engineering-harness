
# Import regex support.
import re

# Import the shared sensitive-entity schema.
from app.masking.schemas import SensitiveEntity


# Define patterns for common sensitive information.
PATTERNS = {
    # Match email addresses.
    "EMAIL": (
        r"\b[A-Za-z0-9._%+-]+@"
        r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),

    # Match common 10-digit Indian mobile numbers.
    "PHONE": (
        r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"
    ),

    # Match the standard format of an Indian PAN identifier.
    "PAN": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",

    # Match 12 consecutive digits as a possible identifier.
    # This does not verify that the number is an Aadhaar number.
    "ID_NUMBER": r"(?<!\d)\d{12}(?!\d)",
}


def detect_rule_entities(
    text: str,
) -> list[SensitiveEntity]:
    # Return an empty list for empty input.
    if not text or not text.strip():
        return []

    # Store every match found by the rules.
    entities = []

    # Check each sensitive-data pattern.
    for entity_type, pattern in PATTERNS.items():
        # Find all non-overlapping matches with character positions.
        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE if entity_type == "EMAIL" else 0,
        ):
            # Save the matched text and its exact offsets.
            entities.append(
                SensitiveEntity(
                    text=match.group(),
                    entity_type=entity_type,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.99,
                )
            )

    # Return the detected entities without changing the source text.
    return entities
