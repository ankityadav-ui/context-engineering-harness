
from app.masking.schemas import SensitiveEntity
from app.masking.entity_resolver import resolve_entities


def test_resolve_entities_removes_duplicates_and_overlaps():
    text = "Contact rahul@example.com today."

    entities = [
        SensitiveEntity(
            text="rahul@example.com",
            entity_type="EMAIL",
            start=8,
            end=25,
            confidence=0.99,
        ),
        SensitiveEntity(
            text="rahul@example.com",
            entity_type="EMAIL",
            start=8,
            end=25,
            confidence=0.80,
        ),
        SensitiveEntity(
            text="rahul",
            entity_type="PERSON",
            start=8,
            end=13,
            confidence=0.75,
        ),
    ]

    resolved = resolve_entities(
        entities=entities,
        source_text=text,
    )

    assert len(resolved) == 1
    assert resolved[0].entity_type == "EMAIL"
    assert resolved[0].text == "rahul@example.com"
    assert text[resolved[0].start:resolved[0].end] == resolved[0].text
