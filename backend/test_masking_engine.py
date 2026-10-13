
from app.masking.masking_engine import mask_document_text


def test_mask_document_text_masks_sensitive_data():
    text = (
        "Rahul Sharma can be contacted at rahul@example.com. "
        "Rahul Sharma's phone number is 9876543210."
    )

    result = mask_document_text(
        text=text,
        document_id="test-document-001",
    )

    assert "Rahul Sharma" not in result.masked_text
    assert "rahul@example.com" not in result.masked_text
    assert "9876543210" not in result.masked_text
    assert "[EMAIL_1]" in result.masked_text
    assert "[PHONE_1]" in result.masked_text
    assert result.safety_passed is False

    person_placeholders = {
        entity.entity_type
        for entity in result.entities
        if entity.entity_type == "PERSON"
    }
    assert "PERSON" in person_placeholders
