
# Import the primary masking engine.
from app.masking.masking_engine import mask_document_text

# Import the secondary safety validator.
from app.masking.safety_service import validate_masked_text


def mask_and_validate(
    text: str,
    document_id: str,
) -> dict:
    # Run primary detection and masking first.
    result = mask_document_text(
        text=text,
        document_id=document_id,
    )

    # Validate only the masked output.
    # Never send original_text or raw entity values to a checker.
    validation = validate_masked_text(
        result.masked_text
    )

    # Update the result's safety status.
    result.safety_passed = validation["passed"]

    # Return the result for controlled internal processing.
    return {
        "result": result,
        "safety": validation,
    }
