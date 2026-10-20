# This file extracts structured tables from PDF documents.

from pathlib import Path

import pdfplumber


def clean_cell(cell) -> str:
    # Convert empty cells to an empty string.
    if cell is None:
        return ""

    # Convert the cell to text and remove unnecessary whitespace.
    return " ".join(str(cell).split())


def table_to_searchable_text(
    headers: list[str],
    rows: list[list[str]],
) -> str:
    # Store searchable versions of all rows.
    searchable_rows = []

    # Process every table row.
    for row in rows:

        # Store header-value pairs.
        values = []

        # Process every column.
        for index, header in enumerate(headers):

            # Get the corresponding cell.
            value = (
                row[index]
                if index < len(row)
                else ""
            )

            # Preserve the relationship between
            # the column name and its value.
            values.append(
                f"{header}: {value}"
            )

        # Convert the row into searchable text.
        searchable_rows.append(
            " | ".join(values)
        )

    # Combine all rows.
    return "\n".join(searchable_rows)


def extract_tables_from_pdf(
    document_path: Path,
    document_id: str,
) -> list[dict]:
    # Store extracted table units.
    table_units = []

    # Keep track of tables across the entire document.
    table_counter = 0

    # Open the PDF.
    with pdfplumber.open(document_path) as pdf:

        # Process every page.
        for page_number, page in enumerate(
            pdf.pages,
            start=1,
        ):

            # Extract tables from the current page.
            tables = page.extract_tables()

            # Process every detected table.
            for table in tables:

                # Ignore empty tables.
                if not table:
                    continue

                # Remove empty rows.
                rows = [
                    row
                    for row in table
                    if row
                    and any(
                        cell is not None
                        and str(cell).strip()
                        for cell in row
                    )
                ]

                # Ignore tables with no useful content.
                if not rows:
                    continue

                # Clean every cell.
                cleaned_rows = [
                    [
                        clean_cell(cell)
                        for cell in row
                    ]
                    for row in rows
                ]

                # Treat the first row as headers.
                headers = cleaned_rows[0]

                # Treat remaining rows as data.
                data_rows = cleaned_rows[1:]

                # Create searchable table text.
                searchable_text = table_to_searchable_text(
                    headers=headers,
                    rows=data_rows,
                )

                # A header-only table (or one whose remaining rows carry
                # no values) has nothing retrievable to index. Keeping it
                # produced an empty content unit, which the embedding
                # stage rejects and the upload fails as a whole.
                if not searchable_text.strip():
                    continue

                # Generate a document-level table number.
                table_counter += 1

                # Create the multimodal content unit.
                table_unit = {
                    "document_id": document_id,
                    "content_type": "table",
                    "page_number": page_number,
                    "table_number": table_counter,
                    "headers": headers,
                    "rows": data_rows,
                    "content": searchable_text,
                    "source": "pdf_table",
                }

                # Store the table.
                table_units.append(table_unit)

    # Return all tables.
    return table_units