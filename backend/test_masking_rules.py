
# Import the regex-based detection function.
from app.masking.rule_service import detect_rule_entities


def main():
    # Prepare sample text with common sensitive patterns.
    text = (
        "Contact rahul@example.com or call 9876543210. "
        "PAN: ABCDE1234F. Reference: 123456789012."
    )

    # Run the rule-based detector.
    entities = detect_rule_entities(text)

    # Print the detected values and their categories.
    for entity in entities:
        print(
            f"{entity.entity_type}: {entity.text} "
            f"({entity.start}, {entity.end})"
        )

    # Collect the categories that were detected.
    detected_types = {
        entity.entity_type for entity in entities
    }

    # Verify that every expected category was detected.
    assert "EMAIL" in detected_types
    assert "PHONE" in detected_types
    assert "PAN" in detected_types
    assert "ID_NUMBER" in detected_types

    # Verify every span matches the original text.
    for entity in entities:
        assert text[entity.start:entity.end] == entity.text

    # Report successful completion.
    print("\nPASS: Regex and rule-based detection works.")


if __name__ == "__main__":
    main()
