"""Table extraction tests.

A table whose only non-empty row is the header produced an empty content
unit. The embedding stage rejects empty content, so a single such table
failed an entire PDF upload ("Document processing failed unexpectedly").
"""

from app.rag import table_extractor


class FakePage:
    def __init__(self, tables):
        self._tables = tables

    def extract_tables(self):
        return self._tables


class FakePdf:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _extract(monkeypatch, tables):
    page = FakePage(tables)

    monkeypatch.setattr(
        table_extractor.pdfplumber,
        "open",
        lambda _path: FakePdf([page]),
    )

    return table_extractor.extract_tables_from_pdf(
        document_path="ignored.pdf",
        document_id="doc-1",
    )


def test_header_only_table_is_dropped(monkeypatch):
    # One row holding only headers: no data rows, so no searchable values.
    tables = _extract(monkeypatch, [[["Question", "Marks"]]])

    assert tables == []


def test_table_with_data_rows_is_kept(monkeypatch):
    tables = _extract(
        monkeypatch,
        [[["Question", "Marks"], ["Q1", "5"]]],
    )

    assert len(tables) == 1
    assert tables[0]["content"] == "Question: Q1 | Marks: 5"
    assert tables[0]["table_number"] == 1


def test_blank_rows_are_ignored(monkeypatch):
    tables = _extract(
        monkeypatch,
        [[["Question", "Marks"], [None, ""], ["", None]]],
    )

    assert tables == []


def test_table_numbers_stay_contiguous_when_an_empty_table_is_skipped(
    monkeypatch,
):
    tables = _extract(
        monkeypatch,
        [
            [["Header"]],
            [["Header"], ["value"]],
        ],
    )

    assert [table["table_number"] for table in tables] == [1]
    assert tables[0]["content"] == "Header: value"
