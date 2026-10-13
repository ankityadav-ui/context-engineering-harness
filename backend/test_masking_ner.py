
from app.masking.ner_service import detect_entities


def test_ner_detects_person_and_valid_offsets():
    text = (
        "Rahul Sharma visited City Hospital "
        "in Nagpur on 12 October 2026."
    )

    entities = detect_entities(text)

    detected_types = {
        entity.entity_type for entity in entities
    }

    assert "PERSON" in detected_types

    for entity in entities:
        assert text[entity.start:entity.end] == entity.text
        assert entity.start >= 0
        assert entity.end <= len(text)
