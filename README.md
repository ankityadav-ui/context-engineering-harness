# Context Engineering Harness

A FastAPI backend and Streamlit workspace for streaming chat, multimodal
document retrieval (RAG), centralized context assembly, knowledge-graph
extraction and lookup, sensitive-information masking, and diagnostics.
Text chat, streaming, multimodal image questions, summaries, and graph
extraction use the shared Google Gemini integration.

The UI provides Chat, Documents, Context Inspector, and Diagnostics. It
does not include a Graph Explorer; graph browsing remains a backend API
capability, while the UI exposes only a read-only graph cleanup report.

## Architecture and data flow

| Area | Implementation |
| --- | --- |
| API and routes | `backend/app/main.py`; `/health` reports only API-process health |
| Chat LLM integration | `backend/app/llm.py` (Gemini: text, streaming, images) |
| Context assembly | `backend/app/context_engine/engine.py` (authoritative path) |
| Conversation, summary, memory | `backend/app/context_engine/` |
| Knowledge graph | `backend/app/context_engine/graph_service.py` (Neo4j) |
| Graph extraction | `backend/app/context_engine/llm_graph_transformer.py` (`LLMGraphTransformer`, evidence validation, provenance) |
| Graph entity validation | `backend/app/context_engine/entity_validation.py` |
| PDF ingestion and retrieval | `backend/app/rag/` |
| Masking and safety checks | `backend/app/masking/` |
| Database models and configuration | `backend/app/context_engine/models.py`, `database.py` |
| Trusted prompt assets | `backend/app/prompts/`, `backend/app/system_prompt.py` |

`POST /documents/ask` and `POST /context/preview` share the single
`assemble_context` path. It combines retrieved document units, a
conversation summary plus recent messages (when a conversation ID is
supplied), relevant memories (when a user ID is supplied), and Neo4j
relationships. Source failures are returned separately in `source_errors`
so one unavailable source never silently becomes fabricated data.
Document-scoped summary requests are the exception: they use only that
document's page-aware text/table units and do not include conversation,
memory, or graph context.

### PDF ingestion and RAG

The upload endpoint validates the file extension, the `%PDF-` signature,
and a 25 MiB size limit, persists a durable job, and returns HTTP 202 with
the job and document IDs. Extraction, OCR, chunking, masking, embedding,
vector indexing, and graph ingestion continue in a bounded background
worker. Poll `GET /documents/jobs/{job_id}` for stage progress and safe
errors; retry a failed job with `POST /documents/jobs/{job_id}/retry`.
Retries retain the same document ID, so vector and graph writes are
idempotent and do not require deleting live data. Uploads without an
`Idempotency-Key` create a new document. Repeated requests with the same
key and identical content/masking mode return the existing job; use a new
key to explicitly reprocess it.

The worker extracts selectable text with PyMuPDF, extracts tables with
pdfplumber, uses Tesseract OCR on pages where normal text extraction is
insufficient, and extracts embedded images to
`backend/storage/images/<document_id>/`. Extracted text, OCR text, tables,
and image references become content units; exact duplicate units and
repeated image references are collapsed before embedding. Gemini text
embeddings are stored in the configured Pinecone namespace.

Image units are embedded as descriptive text, not as true visual
embeddings. When a retrieved image file exists, question answering passes
it to the configured Gemini model, which handles both text and images.
Image references are resolved against the backend storage root: absolute
paths, traversal, and missing files are refused.

Document-scoped summary requests read all text and table units for the
selected document ID, retaining each unit's page and chunk ID. They do not
include conversation, memory, graph, or other-document evidence. Gemini
selects verbatim passages; each passage is checked against its named source
unit before it is displayed. Pages without readable text after extraction
and OCR are reported as missing rather than filled in. Summaries are
bounded to 80,000 source characters; larger documents must be split before
they can be summarized.

The Documents page polls ingestion progress without blocking chat and offers
retry for failed jobs. Enable **Reprocess identical PDF as a new document**
to run it through the current pipeline again. This creates a fresh document
ID and leaves the prior document and its vectors intact; select the new
document for subsequent questions. Reprocessing is not an in-place update.

### Context, memory, and graph

Conversations, messages, summaries, and memories use SQLAlchemy models
backed by the configured PostgreSQL database. The context engine summarizes
older conversation messages while retaining a recent message window
(`n = 6` by default: with 20 messages, messages 1-14 are summarized and
15-20 stay as recent messages). Relevant long-term memories are selected by
keyword overlap, importance, and recency.

Assembled context is bounded: `top_k` is capped at 20 results, memories at
5, graph relationships at 10, each retrieved text block at 4000 characters,
and the whole formatted context at 24000 characters. Truncation is marked in
the text so a cut-off transcript is never presented as complete evidence.
Retrieved results and graph relationships are deduplicated before assembly,
and assembly logs counts and sizes only (never query, memory, document, or
entity text).

Neo4j stores `Document` and `Entity` nodes, `MENTIONS` document-to-entity
links, validated entity relationships, and an optional `PART_OF` case link.
By default, `LLMGraphTransformer` extracts graph candidates; an explicit
legacy spaCy mode is also available. Candidate names are checked for
placeholders, OCR debris, merged headings, and implausible person,
organization, or location types. Technical terms cannot be classified as
people or places, anonymized aliases such as `Organization B` are not stored
as real entities, and a name with conflicting types is omitted instead of
being assigned a guessed type. Masking detection deliberately does not use
these precision rules, so graph validation cannot reduce masking recall.
Hyphenated alphanumeric model identifiers such as `Luma-4` are accepted
only for product, technology, or document entities; they remain invalid
in person and place names.

Graph extraction runs per text/table content unit (normally one page unit).
Each input carries its document ID, page number, and deterministic source-unit
ID, which are retained in graph provenance. Entity names must occur in their
source unit. A relationship is accepted only when the model supplies a
verbatim source quote containing both endpoint names and a textual cue for
the selected relationship type; unsupported edges are dropped. This check
improves traceability but does not independently prove the model's semantic
interpretation of the quoted sentence.

Extraction and validation complete before any graph write. A single Neo4j
write transaction then creates or updates the document and refreshes only
that document's mentions and relationship provenance. Shared entities and
edges supported by other documents are preserved, and conflicting entity
types are not overwritten on a match. Re-ingesting the same document ID
removes stale document-owned edges and deletes a former entity only when no
other graph edge still references it. Provider, authentication, quota,
timeout, malformed-output, and transaction failures are surfaced rather than
silently falling back.

Document-scoped Q&A uses only entities mentioned by the selected document
and only relationships whose provenance includes it. Unscoped chat may use
the broader knowledge graph. Whole-document summaries remain isolated from
graph, conversation, and memory context.

The graph cleanup report is a dry-run read-only audit. It reports malformed
names, duplicate and near-duplicate candidates, type conflicts, implausible
types, missing provenance, unsupported relationships, and edges connected
to invalid entities. Findings are recommendations with evidence; they do
not modify the graph. The separate cleanup POST endpoint is dry-run by
default and requires an explicit `{"confirm": true}` to execute its scoped
legacy cleanup.

Alembic migrations are versioned under `backend/alembic/versions`. Apply
schema updates from `backend/` with `python -m alembic upgrade head`.
`backend/init_db.py` remains available to create missing tables in a fresh
database; it does not alter existing table definitions.

### Prompts, masking, and safety

Every conversation carries its own custom system prompt, persisted
per conversation in PostgreSQL (`conversations.system_prompt`). The
prompt editor loads and saves the selected conversation's prompt, and
resetting it deletes only that conversation's prompt (an empty prompt
means "use the default system instructions"). The built-in security
instructions are composed after the conversation's custom prompt and
sent to Gemini as a real `system` message, so user content and
retrieved documents never travel in the system role and cannot
override the instructions.

Masking is explicit and opt-in in two places: a chat message can be masked
before sending (the message is sent to the model only when masking
validation passes), and a PDF can be masked after text/OCR extraction but
before embeddings are generated (in that mode image extraction is skipped
and the raw PDF is not retained). Chat masking calls `POST /masking/text`
first and sends only the approved `masked_text`; when the safety check does
not pass, that endpoint withholds `masked_text` (`null`) and the frontend
sends nothing. One shared placeholder registry numbers a
document's chunks together, so the same value keeps the same placeholder in
every chunk. Masking is a best-effort safeguard: it cannot sanitize chat
history, memories, graph properties, previously indexed vectors, or image
pixels, and it cannot guarantee that every sensitive span is detected.
The upload response's `masked_entities` count is the number of distinct
values newly replaced during that operation. `existing_placeholders` and
`existing_placeholder_occurrences` separately count distinct masking tokens
already present in the extracted source. The readable aliases shown in
generated text (for example, `Person A`) are display forms of placeholders,
not recovered identities.

## Gemini generation and embeddings

- Text, streaming, image input, summaries, and graph extraction share
  `GEMINI_API_KEY` and `GEMINI_TEXT_MODEL` (default `gemini-3.8-flash`).
- `GEMINI_TIMEOUT_SECONDS` configures the generation request timeout
  (default 300 seconds). Failures are translated into safe, actionable
  messages; upstream exception text and credentials are not exposed.
- Graph extraction is Gemini-only and uses `LLMGraphTransformer` with the
  same configured chat model. There is no provider auto-fallback.
- Graph extraction currently depends on the pinned `langchain-experimental`
  transformer, which emits an upstream sunset warning. The installed
  `langchain-core`, `langchain-experimental`, and Gemini integration expose
  the structured-output and relationship-property APIs this code uses, but
  this is API compatibility evidence rather than a live Gemini smoke test;
  no drop-in replacement has been validated.
- Embeddings remain separate: `google-genai` uses `gemini-embedding-001`
  with 1024 dimensions. This existing embedding model and Pinecone index
  compatibility are unchanged; migrating generation does not re-embed
  documents.
- `POST /chat` keeps its JSON contract. `POST /chat/stream` returns
  `text/event-stream` frames: `chunk` with `{"delta": "..."}`, then a single
  terminal `done` with the assembled response/provider/model, or `error`
  with a safe detail and any partial text. Stopping the client closes the
  upstream stream.

## API overview

| Method and path | Purpose |
| --- | --- |
| `GET /health` | API-process health only (no dependency probing) |
| `GET /capabilities` | Which optional services are configured (booleans and env-var names only) |
| `POST /chat`, `POST /chat/stream` | Text chat (JSON or SSE) |
| `POST /documents/upload` | Accept a durable PDF ingestion job (optional `mask_content=true`, HTTP 202) |
| `GET /documents/jobs/{job_id}`, `POST /documents/jobs/{job_id}/retry` | Poll progress or retry a failed ingestion job |
| `GET /documents` | List indexed documents with local storage status |
| `GET /documents/{id}` | Document unit metadata (never the stored text) |
| `DELETE /documents/{id}` | Delete a document's vectors and local files |
| `POST /documents/search` | Semantic search over indexed content |
| `POST /documents/ask` | Context-assembled question answering with sources |
| `POST /context/preview` | Read-only view of the assembled context for a question; a supplied conversation uses the same session-ownership rules as chat |
| `POST /masking/text` | Mask text and return the safety-validation result |
| `GET/POST /conversations`, `GET /conversations/{id}/messages` | Conversation storage |
| `GET/POST /memory`, `DELETE /memory/{id}` | Long-term memory storage |
| `GET /knowledge-graph/status`, `/knowledge-graph/entities` | Graph status and bounded entity lookups |
| `GET /knowledge-graph/entities/{id}/neighborhood`, `/entities/{id}/documents` | An entity and its neighbors or source documents |
| `GET /knowledge-graph/documents`, `/knowledge-graph/documents/{id}/entities` | Ingested documents and their entities |
| `GET /knowledge-graph/cleanup-report` | Read-only graph audit; no writes |
| `GET/PUT/DELETE /conversations/{id}/system-prompt` | Per-conversation custom system prompt (reset via DELETE) |

## Services and environment

Copy the root `.env.example` to `.env` and fill in only the credentials for
services you intend to use. The template contains placeholders only: never
commit real credentials, and never put secrets in frontend variables.
`.env` is gitignored. `GET /capabilities` reports which services are
configured.

| Feature | Configuration/service |
| --- | --- |
| Health and API startup | No external service; `/health` does not probe dependencies |
| Text chat, streaming, image questions, summaries, graph extraction | `GEMINI_API_KEY`, optional `GEMINI_TEXT_MODEL`, optional `GEMINI_TIMEOUT_SECONDS` |
| Optional quota fallback for ordinary text generation | Set `LLM_QUOTA_FALLBACK=ollama`, `OLLAMA_BASE_URL`, and `OLLAMA_TEXT_MODEL`; disabled by default and never used for streaming, images, or graph extraction |
| Legacy graph extraction | `GRAPH_EXTRACTION_BACKEND=spacy` explicitly selects the spaCy path; default is `llm` |
| PDF embeddings, indexing, search, and RAG | `GEMINI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` |
| Conversations, summaries, and long-term memory | PostgreSQL via `DATABASE_URL` |
| Entity and relationship storage/lookup | Neo4j via `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` |
| Scanned-PDF OCR | Tesseract executable; `TESSERACT_PATH` if not at its default Windows path |
| Text masking | spaCy plus the separately installed `en_core_web_sm` model |
| Log verbosity | `CEH_LOG_LEVEL` (default `INFO`) |

## Windows setup and run

From PowerShell at the repository root:

```powershell
Set-Location .\backend
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install -r ..\frontend\requirements.txt
python -m spacy download en_core_web_sm
```

Install Tesseract separately if scanned-PDF OCR is needed. Copy
`.env.example` to `.env` in the repository root and configure PostgreSQL,
Neo4j, Gemini, and Pinecone only for the features you plan to use.

Run the backend:

```powershell
Set-Location C:\path\to\context-engine\backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Start the Streamlit workspace from the repository root, using the
environment set up above:

```powershell
.\backend\.venv\Scripts\python.exe -m streamlit run .\frontend\app.py
```

The UI reads `BACKEND_API_URL` (default `http://127.0.0.1:8000`). Its
Diagnostics page reports API health, configured capabilities, and the
read-only graph cleanup report. Navigation remains available if the API is
unavailable, and failed page requests are surfaced in the UI.

Apply the additive ingestion-job migration before using uploads on an
existing database:

```powershell
Set-Location C:\path\to\context-engine\backend
.\.venv\Scripts\python.exe -m alembic upgrade head
```

For a new database, `init_db.py` can still create the current model tables.
The migration creates only the ingestion-job table and indexes; it does not
recreate Pinecone indexes or remove existing document, graph, or vector data.

## Repository layout

```text
backend/
  app/
    context_engine/    conversations, memory, summaries, graph, assembly
    masking/           NER, rules, entity resolution, safety validation
    rag/               extraction, OCR, tables, images, embeddings, Pinecone
    prompts/           default system prompt asset
    llm.py, main.py, system_prompt.py
  tests/
    unit/              pure functions and mocked provider behavior
    integration/       FastAPI routes driven with TestClient and mocks
  scripts/             opt-in manual scripts that touch live services
  init_db.py, requirements.txt, requirements-dev.txt, pytest.ini
frontend/             Streamlit workspace (Chat, Documents, Context Inspector, Diagnostics)
```

## Tests

From `backend/`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The automated suite (`backend/tests/unit` and `backend/tests/integration`)
contacts no external service. Pinecone, Neo4j, Gemini, PostgreSQL, and the
LLM calls are mocked or replaced with in-memory SQLite. The masking tests
do load the real spaCy model, so `en_core_web_sm` must be installed. The PDF
pipeline tests build a real small PDF and parse it with PyMuPDF/pdfplumber
while stubbing the paid embedding and upsert stages.

Scripts named `backend/scripts/manual_*.py` are opt-in live integration
scripts and may contact or write to PostgreSQL, Neo4j, Pinecone, Gemini, or
an LLM. Read `backend/scripts/README.md` and `manual_guard.py` first; they
are never collected by pytest.

## Known limitations

- `POST /documents/ask` and `/context/preview` do not persist new
  messages; `/chat` and `/chat/stream` persist both turns. All four
  accept a conversation ID, and the conversational paths (`/chat`,
  `/chat/stream`, `/documents/ask`, `/context/preview`) enforce the same
  session-ownership rules for it.
- Image vectors use textual descriptions, not visual embeddings.
- Masking is optional, cannot sanitize data that is already stored or
  indexed, and cannot guarantee detection of every sensitive span.
- Graph entity validation reduces noise but cannot guarantee correct
  extraction from malformed, ambiguous, or low-quality OCR text.
- PostgreSQL schema changes use Alembic; apply migrations before using new
  endpoints that depend on them. Interrupted queued/processing jobs are
  recovered at API startup when `INGESTION_JOB_RECOVERY_ENABLED` is enabled.
- External integrations (PostgreSQL, Neo4j, Pinecone, and Gemini) are
  not live-verified by the automated suite; use the mocked tests and the
  opt-in manual scripts.

## Troubleshooting

- **Gemini authentication or quota errors** — verify `GEMINI_API_KEY`,
  project API access, and quota. Transient errors use bounded retries.
  Ollama fallback is used only when explicitly enabled and available.
- **Gemini reports an unavailable model** — verify `GEMINI_TEXT_MODEL` is
  available to the configured API key.
- **Documents page reports `PINECONE_API_KEY`** — set `PINECONE_API_KEY` and
  `GEMINI_API_KEY` in the root `.env`, then restart the backend.
- **Upload is rejected with 413** — the file is larger than the 25 MiB
  limit.
- **Upload is rejected with "not a valid PDF"** — the file does not start
  with the PDF signature, whatever its extension says.
- **Knowledge graph reports `NEO4J_URI`** — Neo4j is not configured or
  unreachable; this is reported as a configuration problem, not an empty
  graph.
- **Conversations list reports "database is unavailable"** — set
  `DATABASE_URL`, then create the tables with `python init_db.py`.
- **OCR pages come back empty** — Tesseract is not installed or
  `TESSERACT_PATH` does not point at `tesseract.exe`.
- **Masking returns 500 with "Masking failed"** — the spaCy model is
  missing; run `python -m spacy download en_core_web_sm`.
