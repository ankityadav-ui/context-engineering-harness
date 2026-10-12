
# Import the masking engine.
from app.masking.masking_engine import mask_document_text


def main():
    # Use synthetic data rather than real personal information.
    text = (
        "Rahul Sharma can be contacted at rahul@example.com. "
        "Rahul Sharma's phone number is 9876543210."
    )

    # Run detection, resolution, and masking.
    result = mask_document_text(
        text=text,
        document_id="test-document-001",
    )

    # Display only the masked text and entity categories.
    print("Masked text:")
    print(result.masked_text)

    print("\nDetected categories:")
    print(sorted({
        entity.entity_type
        for entity in result.entities
    }))

    # Verify that the original name was replaced.
    assert "Rahul Sharma" not in result.masked_text

    # Verify that the email was replaced.
    assert "rahul@example.com" not in result.masked_text

    # Verify that the phone number was replaced.
    assert "9876543210" not in result.masked_text

    # Verify that repeated names use the same placeholder.
    assert result.masked_text.count("[PERSON_1]") == 1
    assert result.masked_text.count("[PERSON_2]") == 1
    assert "[EMAIL_1]" in result.masked_text
    assert "[PHONE_1]" in result.masked_text

    # Confirm safety validation has not been performed yet.
    assert result.safety_passed is False

    # Report successful completion.
    print("\nPASS: Masking engine test succeeded.")


if __name__ == "__main__":
    main()
