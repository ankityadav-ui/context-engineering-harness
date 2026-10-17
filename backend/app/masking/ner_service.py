
# Import the shared entity schema.
from app.masking.schemas import SensitiveEntity


# Hold the loaded spaCy model until NER is actually needed.
# Loading it at import time would stop unrelated API routes from
# starting when the model or the spaCy package is unavailable.
_nlp = None


def get_nlp():
    """Return the English spaCy model, loading it on first use."""

    global _nlp

    if _nlp is None:
        try:
            import spacy
        except ImportError as error:
            raise RuntimeError(
                "spaCy is not installed. Install it with "
                "'python -m pip install -r requirements.txt'."
            ) from error

        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError as error:
            raise RuntimeError(
                "The spaCy model 'en_core_web_sm' is not installed. "
                "Install it with: python -m spacy download en_core_web_sm"
            ) from error

    return _nlp


def detect_entities(text: str) -> list[SensitiveEntity]:
    # Reject empty input because there is nothing to analyze.
    if not text or not text.strip():
        return []

    # Analyze the document and identify named entities.
    doc = get_nlp()(text)

    # Store the detected entities in our common schema.
    entities = []

    # Process each entity identified by spaCy.
    for entity in doc.ents:
        # Keep only categories relevant to document privacy.
        if entity.label_ not in {
            "PERSON",
            "ORG",
            "GPE",
            "LOC",
            "DATE",
        }:
            continue

        # Convert spaCy's entity into our project schema.
        entities.append(
            SensitiveEntity(
                text=entity.text,
                entity_type=entity.label_,
                start=entity.start_char,
                end=entity.end_char,
                confidence=0.75,
            )
        )

    # Return the detected entities without modifying the text.
    return entities
