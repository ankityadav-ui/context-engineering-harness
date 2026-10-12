
# Import the schemas created in this block.
from app.masking.schemas import SensitiveEntity, MaskingResult


def main():
    # Create an example detected email entity.
    entity = SensitiveEntity(
        text="rahul@example.com",
        entity_type="EMAIL",
        start=7,
        end=25,
        confidence=0.98,
    )

    # Create a masking result using the detected entity.
    result = MaskingResult(
        document_id="test-document-001",
        original_text="Email: rahul@example.com",
        masked_text="Email: [EMAIL_1]",
        entities=[entity],
    )

    # Verify that the values were stored correctly.
    assert result.entities[0].entity_type == "EMAIL"
    assert result.masked_text == "Email: [EMAIL_1]"
    assert result.safety_passed is False

    # Confirm that this block's schema test passed.
    print("PASS: Masking schemas work correctly.")


if __name__ == "__main__":
    main()
