
# Import the shared entity schema.
from app.masking.schemas import SensitiveEntity


# Define deterministic priorities for overlapping detections.
ENTITY_PRIORITY = {
    "EMAIL": 100,
    "PHONE": 90,
    "PAN": 90,
    "ID_NUMBER": 90,
    "PERSON": 70,
    "ORG": 60,
    "GPE": 50,
    "LOC": 50,
    "DATE": 40,
}


def resolve_entities(
    entities: list[SensitiveEntity],
    source_text: str,
) -> list[SensitiveEntity]:
    # Keep only entities with valid positions in the source text.
    valid_entities = [
        entity
        for entity in entities
        if (
            0 <= entity.start < entity.end <= len(source_text)
            and source_text[entity.start:entity.end] == entity.text
        )
    ]

    # Remove exact duplicate detections.
    unique_entities = {}
    for entity in valid_entities:
        # Use span and type as the identity of a detection.
        key = (
            entity.start,
            entity.end,
            entity.entity_type,
        )

        # Retain the highest-confidence duplicate.
        existing = unique_entities.get(key)
        if (
            existing is None
            or entity.confidence > existing.confidence
        ):
            unique_entities[key] = entity

    # Prefer longer spans, then category priority, then earlier offsets.
    candidates = sorted(
        unique_entities.values(),
        key=lambda entity: (
            -(entity.end - entity.start),
            -ENTITY_PRIORITY.get(entity.entity_type, 0),
            entity.start,
        ),
    )

    # Select non-overlapping entities.
    resolved = []
    for candidate in candidates:
        # Two spans overlap when their character ranges intersect.
        overlaps = any(
            candidate.start < chosen.end
            and chosen.start < candidate.end
            for chosen in resolved
        )

        # Skip a candidate if a higher-ranked span already covers it.
        if not overlaps:
            resolved.append(candidate)

    # Return entities in the same order they appear in the text.
    return sorted(
        resolved,
        key=lambda entity: entity.start,
    )
