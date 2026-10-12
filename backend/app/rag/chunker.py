# Import uuid4 so every content unit receives a unique identifier.
from uuid import uuid4


# Create a standardized content unit from an extracted item.
def create_content_unit(
    document_id: str,
    content_type: str,
    page_number: int,
    content: str = "",
    content_ref: str | None = None,
    metadata: dict | None = None,
) -> dict:
    # Generate a unique identifier for this content unit.
    unit_id = str(uuid4())

    # Create the common structure used by all content types.
    return {
        "id": unit_id,
        "document_id": document_id,
        "content_type": content_type,
        "page_number": page_number,
        "content": content,
        "content_ref": content_ref,
        "metadata": metadata or {},
    }


# Convert extracted text, tables, and images into unified content units.
def build_content_units(
    text_units: list[dict],
    table_units: list[dict],
    image_units: list[dict],
) -> list[dict]:
    # Store all standardized units in one list.
    content_units = []

    # Convert every extracted text item into a content unit.
    for text_unit in text_units:
        content_unit = create_content_unit(
            document_id=text_unit["document_id"],
            content_type="text",
            page_number=text_unit["page_number"],
            content=text_unit["content"],
            metadata={
                "source": text_unit.get("source", "unknown"),
            },
        )

        # Add the text unit to the common collection.
        content_units.append(content_unit)

    # Convert every extracted table into a content unit.
    for table_unit in table_units:
        content_unit = create_content_unit(
            document_id=table_unit["document_id"],
            content_type="table",
            page_number=table_unit["page_number"],
            content=table_unit["content"],
            metadata={
                "source": table_unit.get("source", "unknown"),
                "table_number": table_unit.get("table_number"),
                "headers": table_unit.get("headers", []),
                "rows": table_unit.get("rows", []),
            },
        )

        # Add the table unit to the common collection.
        content_units.append(content_unit)

    # Convert every extracted image into a content unit.
    for image_unit in image_units:
        content_unit = create_content_unit(
            document_id=image_unit["document_id"],
            content_type="image",
            page_number=image_unit["page_number"],
            content="",
            content_ref=image_unit["content_ref"],
            metadata={
                "source": image_unit.get("source", "unknown"),
                "image_id": image_unit.get("image_id"),
                "image_format": image_unit.get("image_format"),
            },
        )

        # Add the image unit to the common collection.
        content_units.append(content_unit)

    # Sort units by page number so the original document order is preserved.
    content_units.sort(
        key=lambda unit: (
            unit["page_number"],
            unit["content_type"],
        )
    )

    # Return the complete multimodal collection.
    return content_units