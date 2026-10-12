# This file extracts embedded images from PDF documents.

from pathlib import Path

import fitz


def extract_images_from_pdf(
    document_path: Path,
    document_id: str,
    image_storage_dir: Path,
) -> list[dict]:
    # Store information about all extracted images.
    image_units = []

    # Create a directory dedicated to this document.
    document_image_dir = (
        image_storage_dir / document_id
    )

    # Make sure the directory exists.
    document_image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Open the PDF.
    pdf_document = fitz.open(document_path)

    # Keep track of images inside the document.
    image_counter = 0

    try:
        # Process every page.
        for page_number, page in enumerate(
            pdf_document,
            start=1,
        ):

            # Get all images embedded on this page.
            images = page.get_images(
                full=True
            )

            # Process every image.
            for image_index, image_info in enumerate(
                images,
                start=1,
            ):

                # Get the internal PDF image reference.
                xref = image_info[0]

                # Extract the image bytes and extension.
                image_data = pdf_document.extract_image(
                    xref
                )

                # Get the actual image bytes.
                image_bytes = image_data["image"]

                # Get the image format.
                image_extension = image_data["ext"]

                # Generate a unique image number.
                image_counter += 1

                # Create the image filename.
                image_filename = (
                    f"page_{page_number}"
                    f"_img_{image_index}"
                    f".{image_extension}"
                )

                # Build the image storage path.
                image_path = (
                    document_image_dir
                    / image_filename
                )

                # Save the image to disk.
                image_path.write_bytes(
                    image_bytes
                )

                # Create a reference that can be stored
                # in our content metadata.
                content_ref = str(
                    image_path.relative_to(
                        image_storage_dir.parent
                    )
                ).replace("\\", "/")

                # Create the image content unit.
                image_unit = {
                    "document_id": document_id,
                    "content_type": "image",
                    "page_number": page_number,
                    "image_id": (
                        f"{document_id}_"
                        f"image_{image_counter}"
                    ),
                    "content_ref": content_ref,
                    "image_format": image_extension,
                    "source": "pdf_image",
                }

                # Add the image unit to the result.
                image_units.append(
                    image_unit
                )

    finally:
        # Always close the PDF.
        pdf_document.close()

    # Return all extracted image units.
    return image_units