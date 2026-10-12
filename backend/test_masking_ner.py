
# Import the NER detection function.
from app.masking.ner_service import detect_entities


def main():
    # Use a simple example containing several entity types.
    text = (
        "Rahul Sharma visited City Hospital "
        "in Nagpur on 12 October 2026."
    )

    # Detect named entities in the example.
    entities = detect_entities(text)

    # Print the entity values, labels, and character positions.
    for entity in entities:
        print(
            f"Text: {entity.text} | "
            f"Type: {entity.entity_type} | "
            f"Start: {entity.start} | "
            f"End: {entity.end}"
        )

    # Collect the labels found by the model.
    detected_types = {
        entity.entity_type for entity in entities
    }

    # Confirm that the model recognized a person's name.
    assert "PERSON" in detected_types, (
        "NER did not detect the expected person."
    )

    # Confirm that entity spans point to the correct text.
    for entity in entities:
        assert text[entity.start:entity.end] == entity.text

    # Report successful completion.
    print("\nPASS: NER detection and character offsets work.")


if __name__ == "__main__":
    main()
