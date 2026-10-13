
from app.masking.rule_service import detect_rule_entities


def test_rule_detector_finds_sensitive_patterns():
    text = (
        "Contact rahul@example.com or call 9876543210. "
        "PAN: ABCDE1234F. Reference: 123456789012."
    )

    entities = detect_rule_entities(text)
    detected_types = {entity.entity_type for entity in entities}

    assert "EMAIL" in detected_types
    assert "PHONE" in detected_types
    assert "PAN" in detected_types
    assert "ID_NUMBER" in detected_types

    for entity in entities:
        assert text[entity.start:entity.end] == entity.text
