
# Import the schema and resolver.
from app.masking.schemas import SensitiveEntity
from app.masking.entity_resolver import resolve_entities


def main():
    # Prepare text containing an email address.
    text = "Contact rahul@example.com today."

    # Simulate detections from two different detectors.
    entities = [
        SensitiveEntity(
            text="rahul@example.com",
            entity_type="EMAIL",
            start=8,
            end=25,
            confidence=0.99,
        ),
        # Simulate a duplicate email detection.
        SensitiveEntity(
            text="rahul@example.com",
            entity_type="EMAIL",
            start=8,
            end=25,
            confidence=0.80,
        ),
        # Simulate a conflicting entity inside the email.
        SensitiveEntity(
            text="rahul",
            entity_type="PERSON",
            start=8,
            end=13,
            confidence=0.75,
        ),
    ]

    # Resolve duplicates and overlapping spans.
    resolved = resolve_entities(
        entities=entities,
        source_text=text,
    )

    # The email should win over its overlapping partial name.
    assert len(resolved) == 1
    assert resolved[0].entity_type == "EMAIL"
    assert resolved[0].text == "rahul@example.com"

    # Verify that the selected span matches the original text.
    entity = resolved[0]
    assert text[entity.start:entity.end] == entity.text

    # Report successful completion.
    print("PASS: Duplicate and overlapping entities resolved.")


if __name__ == "__main__":
    main()
