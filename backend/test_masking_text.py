
from pathlib import Path

from app.masking.text_service import extract_text_for_masking


def main():
    # Locate the backend and document storage directory.
    backend_dir = Path(__file__).resolve().parent
    document_dir = backend_dir / "storage" / "documents"

    # Find available PDF files.
    pdf_files = sorted(
        document_dir.glob("*.pdf"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not pdf_files:
        print("FAIL: No PDF files found in storage/documents.")
        return

    # Select the most recently modified PDF.
    document_path = pdf_files[0]
    document_id = document_path.stem

    print("Selected document:", document_path.name)
    print("Document ID:", document_id)

    # Extract text for the masking pipeline.
    result = extract_text_for_masking(
        document_path=document_path,
        document_id=document_id,
    )

    print("\n--- MASKING EXTRACTION RESULT ---")
    print(result)

    print("\nPASS: Masking text extraction completed.")


if __name__ == "__main__":
    main()
