
from app.masking.safety_service import validate_masked_text


def test_safety_validator_accepts_masked_text():
    result = validate_masked_text(
        "Patient [PERSON_1], email [EMAIL_1]."
    )

    assert result["passed"] is True


def test_safety_validator_rejects_visible_email():
    result = validate_masked_text(
        "Patient [PERSON_1], email rahul@example.com."
    )

    assert result["passed"] is False
    assert any(
        "EMAIL" in reason
        for reason in result["reasons"]
    )
