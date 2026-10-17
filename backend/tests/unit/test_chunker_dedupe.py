"""Content-unit deduplication.

A PDF's table text is often also present in the page text layer, and the
same image reference can be produced twice. Duplicates would become
near-identical embeddings in the vector index, so they are collapsed.
"""

from app.rag.chunker import build_content_units, deduplicate_content_units


def _text_unit(page, content, source="pdf_text"):
    return {
        "document_id": "doc-1",
        "page_number": page,
        "content": content,
        "source": source,
    }


def test_duplicate_text_units_are_collapsed():
    units = build_content_units(
        text_units=[
            _text_unit(1, "Return window: 30 days"),
            _text_unit(1, "Return window: 30 days"),
        ],
        table_units=[],
        image_units=[],
    )

    # Two identical text units, so only one survives.
    assert len(units) == 1
    assert units[0]["content"] == "Return window: 30 days"


def test_duplicate_table_units_are_collapsed():
    table_unit = {
        "document_id": "doc-1",
        "page_number": 1,
        "content": "Return window: 30 days",
        "source": "pdf_table",
        "table_number": 1,
        "headers": ["Return window"],
        "rows": [["30 days"]],
    }

    units = build_content_units(
        text_units=[],
        table_units=[table_unit, dict(table_unit)],
        image_units=[],
    )

    assert len(units) == 1


def test_a_table_next_to_page_text_is_kept():
    # The table's searchable text and its structured metadata are a
    # different unit from the page text layer, so it is preserved.
    units = build_content_units(
        text_units=[_text_unit(1, "Return window: 30 days")],
        table_units=[
            {
                "document_id": "doc-1",
                "page_number": 1,
                "content": "Return window: 30 days",
                "source": "pdf_table",
                "table_number": 1,
                "headers": ["Return window"],
                "rows": [["30 days"]],
            }
        ],
        image_units=[],
    )

    assert sorted(unit["content_type"] for unit in units) == [
        "table",
        "text",
    ]


def test_whitespace_and_case_differences_are_treated_as_duplicates():
    units = deduplicate_content_units(
        [
            {
                "content_type": "text",
                "page_number": 1,
                "content": "Repeat  Heading",
                "content_ref": None,
                "metadata": {},
            },
            {
                "content_type": "text",
                "page_number": 1,
                "content": "repeat heading",
                "content_ref": None,
                "metadata": {},
            },
        ]
    )

    assert len(units) == 1


def test_distinct_pages_and_content_are_kept():
    units = deduplicate_content_units(
        [
            {
                "content_type": "text",
                "page_number": 1,
                "content": "Page one",
                "content_ref": None,
                "metadata": {},
            },
            {
                "content_type": "text",
                "page_number": 2,
                "content": "Page two",
                "content_ref": None,
                "metadata": {},
            },
        ]
    )

    assert len(units) == 2


def test_same_image_reference_is_collapsed():
    units = deduplicate_content_units(
        [
            {
                "content_type": "image",
                "page_number": 1,
                "content": "",
                "content_ref": "storage/images/doc-1/page_1_img_1.png",
                "metadata": {"image_id": "img_1"},
            },
            {
                "content_type": "image",
                "page_number": 1,
                "content": "",
                "content_ref": "storage/images/doc-1/page_1_img_1.png",
                "metadata": {"image_id": "img_1"},
            },
        ]
    )

    assert len(units) == 1


def test_unit_without_an_identity_is_never_dropped():
    units = deduplicate_content_units(
        [
            {
                "content_type": "image",
                "page_number": 1,
                "content": "",
                "content_ref": None,
                "metadata": {},
            },
            {
                "content_type": "image",
                "page_number": 1,
                "content": "",
                "content_ref": "",
                "metadata": {},
            },
        ]
    )

    assert len(units) == 2
