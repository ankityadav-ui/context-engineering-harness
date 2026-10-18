# PDF Resume / Document Ingestion and Context Assembly Fix

## Root cause

### 1. Upload failure: empty/unreadable PDFs produced no retrievable content

The frontend error `No content was indexed.` comes from the
`POST /documents/upload` 422 branch, which checks `not content_units`
after `process_document()` has returned nothing retrievable:

```python
if not content_units:
    raise HTTPException(422, "No retrievable content was extracted from this PDF.")
```

Reaching and swallowing that branch is the first problem. An empty or
unreadable PDF raises deep inside the extraction stack, and `process_document`
propagates it as an unhandled exception. The upload endpoint catches every
`Exception` and converts it to a generic, non-actionable HTTP 500:

```python
raise HTTPException(
    status_code=500,
    detail="Document processing failed unexpectedly. No content was indexed.",
)
```

That means empty PDFs, truncated PDFs, and PDFs whose text layer is fully
blank were never cleanly reported as 422. They also produced a misleading
`content_units: 0` record if `process_document` had returned instead of
raising.

### 2. No OCR fallback for scanned pages

`POST /documents/upload` calls `process_document()`, which extracts selectable
text and marks pages that need OCR, but the current request path never calls
`extract_text_with_ocr()`. The OCR module exists, is configured, and is
unit-tested, but the real upload path did not use it. On a scanned/resume
image PDF, every page fell back to "no meaningful text" and still no units
were produced.

### 3. `page.get_text("text")` is not a valid PDF verification step

The upload endpoint checks the `%PDF-` header, which rejects renamed
non-PDFs. But for a valid PDF with no extractable text, the extractor
silently returns empty. There was no device-level handling for a scanned
document.

### 4. Chat summarization did not fetch document evidence

When a user asked `summarise this` after uploading a resume, the chat endpoint
assembled context through the Context Engine:

- `assemble_context()` called `retrieve_context()`, which embedded the question
  and queried Pinecone.
- For an image-based PDF, uploaded images are never embedded in Pinecone as
  vector content.
- `format_context()` then fell back to the raw question.
- For a scanned image PDF, `retrieve_context()` returned no useful text.

The LLM therefore received no grounded evidence and answered that the
document contained fragmented, illegible, garbled characters.

## Fixes

### A. PDF ingestion

- `text_extractor.py`: add device/type probing. Determine whether a file is a
  real PDF (`%PDF-` signature), whether PyMuPDF can open it, and whether every
  page has extractable text.
- Empty, truncated, or unopenable PDFs now fail with a specific `PDFProcessingError`.
- Pages without meaningful text are still routed to OCR when Tesseract is
  available.
- If OCR cannot run, the page is reported as OCR-unsupported rather than
  silently empty.
- The upload endpoint treats "no text and no OCR content" as a hard 422 and
  removes stored artifacts.

### B. Summary retrieval

- `documents/ask`: try to retrieve document evidence by `document_id` first.
  If none is returned and `document_id` was supplied, fall back to the full
  extracted text for short resumes.
- `engine.py`: format that fallback as the same retrieved-evidence block, not
  the generic raw question, so the answer is grounded in the actual resume
  text.

### C. Chat / no-context behavior

- `documents/ask` and `/documents/ask`: if no document/conversation is
  selected and no content is retrievable, return a clear, actionable client
  error instead of fabricating a summary.
- Continue using the Context Engine as the authoritative assembly path.

## Files changed

- `backend/app/rag/text_extractor.py`: PDF device/type probing, empty/unreadable
  handling, OCR fallback, page-level content checks.
- `backend/app/rag/ocr.py`: `OCRUnavailableError` naming and alignment.
- `backend/app/rag/document_service.py`: guarded status, and a
  `PDFProcessingError` convenience path.
- `backend/app/rag/embedding_service.py`: module renamed for consistency.
- `backend/app/main.py`: upload endpoint normalization; use the full extracted
  text when no vector results exist but a document was uploaded; treat
  no-content as a hard 422; no fabricated summaries when nothing is found.
- `backend/app/context_engine/engine.py`: fallback formatted context for
  no-retrivable-documents; clearly separate `rag`/`no-document` sections.
- `backend/app/context_engine/summary_service.py`: no summary when there is no
  conversation/history to summarize.
- `backend/tests/integration/test_pdf_pipeline.py`: new end-to-end tests for
  normal PDFs, OCR fallback, empty/unreadable PDFs, and indexing failures.
- `backend/tests/integration/test_api_context_preview.py`: no-context behavior
  returns a clear guidance error rather than a fabricated summary.

## Test results

Everything in this repository passes cleanly.

```
287 passed, 2 warnings in 45.06s
```

## Live-integration limitations

The automated suite never touches live services: Pinecone, Gemini, Neo4j,
Ollama, PostgreSQL, or a real Tesseract binary.

- OCR fallback is exercised only where Tesseract is not installed in this
  environment. The code path is correct, but the benefit for a scanned resume
  was not measured against a real `tesseract` binary.
- Embeddings, Pinecone upsert, and document retrieval were mocked in
  `test_pdf_pipeline.py`; the code under test is the real `process_document`
  path, but vectors were fabricated by the test.
- Backend dependencies and test runner: Python 3.14.3 in `backend/.venv`, pytest
  9.1.1, run from `backend/` as `python -m pytest`.

In other words: the old failure mode is fixed and reproduced in tests where a
real OCR engine is unavailable; the remaining live checks require a real
Tesseract and a real vector store to verify that a scanned resume is actually
indexed and returned by `summarise this`.
