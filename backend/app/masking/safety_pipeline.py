
# Import the primary masking engine.
from app.masking.masking_engine import mask_document_text

# Import the secondary safety validator.
from app.masking.safety_service import validate_masked_text


def mask_and_validate(
    text: str,
    document_id: str,
    registry=None,
    include_header_name: bool = False,
) -> dict:
    """Mask one text unit and validate the masked output.

    ``registry`` is an optional shared
    :class:`~app.masking.masking_engine.PlaceholderRegistry`. Passing one
    keeps placeholders identical for the same value across every chunk of
    a document. ``include_header_name`` enables the document-header name
    rule for the first text unit only.
    """

    # Run primary detection and masking first.
    result = mask_document_text(
        text=text,
        document_id=document_id,
        registry=registry,
        include_header_name=include_header_name,
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
