# Import uuid4 so every content unit receives a unique identifier.
from uuid import uuid4


def _dedupe_key(content_unit: dict) -> tuple:
    """Return the identity used to drop duplicate content units.

    Image units are identified by their stored reference. Text and table
    units are identified by page, content type, and normalized content,
    so a repeated chunk is indexed once while a table keeps its own
    structured unit (its searchable text has a different shape from the
    page text layer, and its headers/rows are preserved metadata).
    """

    if content_unit["content_type"] == "image":
        return (
            "image",
            content_unit.get("content_ref")
            or content_unit.get("metadata", {}).get("image_id"),
        )

    normalized = " ".join(
        str(content_unit.get("content", "")).split()
    ).casefold()

    return (
        content_unit["content_type"],
        content_unit["page_number"],
        normalized,
    )


def deduplicate_content_units(
    content_units: list[dict],
) -> list[dict]:
    """Return the units in order, dropping exact duplicates.

    Units without any identity (an empty image reference) are kept, so
    deduplication never silently removes unidentifiable content.
    """

    seen: set[tuple] = set()
    unique: list[dict] = []

    for unit in content_units:
        key = _dedupe_key(unit)

        if key[-1] in ("", None):
            unique.append(unit)
            continue

        if key in seen:
            continue

        seen.add(key)
        unique.append(unit)

    return unique


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

    # Drop duplicated content (for example table text already present in
    # the page text layer, or the same image reference extracted twice)
    # so the index does not store near-identical vectors.
    content_units = deduplicate_content_units(content_units)

    # Sort units by page number so the original document order is preserved.
    content_units.sort(
        key=lambda unit: (
            unit["page_number"],
            unit["content_type"],
        )
    )

    # Return the complete multimodal collection.
    return content_units