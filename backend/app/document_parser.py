from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader


def extract_text(file_path: str) -> str:
    """
    Extract text from supported document types.

    Supported:
    - PDF
    - DOCX
    - TXT
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    if extension == ".pdf":
        return extract_pdf(path)

    if extension == ".docx":
        return extract_docx(path)

    if extension == ".txt":
        return extract_txt(path)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )


def extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n\n".join(pages)


def extract_docx(path: Path) -> str:
    document = DocxDocument(str(path))

    paragraphs = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            paragraphs.append(paragraph.text)

    return "\n\n".join(paragraphs)


def extract_txt(path: Path) -> str:
    return path.read_text(
        encoding="utf-8",
        errors="ignore"
    )