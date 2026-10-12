
# Import Pydantic tools for validating our data.
from pydantic import BaseModel, Field


class SensitiveEntity(BaseModel):
    # Store the exact sensitive value found in the document.
    text: str

    # Store its category, such as PERSON, EMAIL, or PHONE.
    entity_type: str

    # Store the character position where the value starts.
    start: int = Field(ge=0)

    # Store the character position immediately after the value.
    end: int = Field(gt=0)

    # Store the confidence score supplied by the detector.
    confidence: float = Field(ge=0.0, le=1.0)


class MaskingResult(BaseModel):
    # Store the document's identifier.
    document_id: str

    # Store the original extracted text for controlled processing.
    original_text: str

    # Store the text after sensitive values are replaced.
    masked_text: str

    # Store the detected entities for auditing and validation.
    entities: list[SensitiveEntity] = Field(default_factory=list)

    # Indicate whether safety validation has passed.
    safety_passed: bool = False
