
import pytest
from pydantic import ValidationError

from app.masking.schemas import SensitiveEntity, MaskingResult


def test_masking_schemas_store_valid_data():
    entity = SensitiveEntity(
        text="rahul@example.com",
        entity_type="EMAIL",
        start=7,
        end=25,
        confidence=0.98,
    )

    result = MaskingResult(
        document_id="test-document-001",
        original_text="Email: rahul@example.com",
        masked_text="Email: [EMAIL_1]",
        entities=[entity],
    )

    assert result.entities[0].entity_type == "EMAIL"
    assert result.masked_text == "Email: [EMAIL_1]"
    assert result.safety_passed is False


def test_masking_schema_rejects_invalid_confidence():
    with pytest.raises(ValidationError):
        SensitiveEntity(
            text="rahul@example.com",
            entity_type="EMAIL",
            start=7,
            end=25,
            confidence=1.5,
        )
