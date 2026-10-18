# Context Engineering Harness

A FastAPI backend for streaming chat, multimodal document retrieval (RAG),
centralized context assembly, knowledge-graph lookups, sensitive-information
masking, and Ollama-based generation. A Streamlit frontend is in progress
under `frontend/`; it is not required to exercise the API.

Phase status: phases 1-5 (chat, multimodal RAG, context engine, masking,
Ollama integration and context controls) are implemented and covered by
automated tests. Phase 6 (end-to-end LLM evaluation) is not part of this
backend.

## Architecture and data flow

| Area | Implementation |
| --- | --- |
| API and routes | `backend/app/main.py`; `/health` reports only API-process health |
| LLM integration | `backend/app/llm.py` (Ollama only: text, streaming, images) |
| Context assembly | `backend/app/context_engine/engine.py` (authoritative path) |
| Conversation, summary, memory | `backend/app/context_engine/` |
| Knowledge graph | `backend/app/context_engine/graph_service.py` |
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

### PDF ingestion and RAG

The upload endpoint validates the file extension, the `%PDF-` signature,
and a 25 MiB size limit, then extracts selectable text with PyMuPDF,
extracts tables with pdfplumber, uses Tesseract OCR on pages where normal
text extraction is insufficient, and extracts embedded images to
`backend/storage/images/<document_id>/`. Extracted text, OCR text, tables,
and image references become content units; exact duplicate units and
repeated image references are collapsed before embedding. Gemini text
embeddings are stored in the configured Pinecone namespace.

Image units are embedded as descriptive text, not as true visual
embeddings. When a retrieved image file exists, question answering passes
it to the configured Ollama model, which handles both text and images.
Image references are resolved against the backend storage root: absolute
paths, traversal, and missing files are refused.

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

Neo4j stores `Document` nodes, `Entity` nodes, the `MENTIONS` document-to-
entity links, `RELATES_TO` sentence-level entity pairs, and an optional
`PART_OF` case link. Entity extraction reuses the shared spaCy NER loader
for detection, then applies graph-specific validation
(`entity_validation.py`): name-quality checks (placeholders, OCR debris,
merged headings, digit/letter mixtures), type-plausibility checks (people
are not concepts or organizations, organizations need a plausible shape,
laws need a legal term), and document-level type resolution so a name
labelled inconsistently (for example as both `GPE` and `PERSON`) becomes one
correctly typed node. Masking detection deliberately does not use these
rules, so graph precision work cannot reduce masking recall.

The repository has no versioned migration framework. `backend/init_db.py`
calls SQLAlchemy `metadata.create_all`; it creates missing tables but does
not migrate existing table definitions.

### Prompts, masking, and safety

Every conversation carries its own custom system prompt, persisted
per conversation in PostgreSQL (`conversations.system_prompt`). The
prompt editor loads and saves the selected conversation's prompt, and
resetting it deletes only that conversation's prompt (an empty prompt
means "use the default system instructions"). The built-in security
instructions are composed after the conversation's custom prompt and
sent to Ollama as a real `system` message, so user content and
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

## Ollama and streaming

- Ollama is the only LLM provider: text, streaming, and image input all use
  the configured model tag (`OLLAMA_TEXT_MODEL`, default `gemma4:31b`).
  `OLLAMA_VISION_MODEL` optionally overrides the model used for images and
  defaults to the text model.
- Host resolution: `OLLAMA_HOST` when set, otherwise Ollama Cloud
  (`https://ollama.com`) when an Ollama API key is present, otherwise the
  local server (`http://localhost:11434`).
- Generation requests use a 300 s timeout (`OLLAMA_TIMEOUT_SECONDS` to
  override). The startup model-availability probe uses its own 5 s-timeout
  client, so an unreachable host cannot delay startup indefinitely, and a
  definitive probe result is cached per configuration. A transient probe
  failure is reported as a warning and never blocks startup.
- Failures are translated into clear messages: missing configuration names
  the environment variable, an unreachable host names the host, an unknown
  model tag is reported instead of being substituted, and upstream exception
  text is never echoed to the client.
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
| `POST /documents/upload` | Validate, extract, embed, and index a PDF (optional `mask_content=true`) |
| `GET /documents` | List indexed documents with local storage status |
| `GET /documents/{id}` | Document unit metadata (never the stored text) |
| `DELETE /documents/{id}` | Delete a document's vectors and local files |
| `POST /documents/search` | Semantic search over indexed content |
| `POST /documents/ask` | Context-assembled question answering with sources |
| `POST /context/preview` | Read-only view of the assembled context for a question; a supplied conversation uses the same session-ownership rules as chat |
| `POST /masking/text` | Mask text and return the safety-validation result |
| `GET/POST /conversations`, `GET /conversations/{id}/messages` | Conversation storage |
| `GET/POST /memory`, `DELETE /memory/{id}` | Long-term memory storage |
| `GET /knowledge-graph/entities`, `GET /knowledge-graph/entities/{id}/relationships` | Graph lookups |
| `GET /knowledge-graph/documents`, `GET /knowledge-graph/documents/{id}/entities` | Ingested documents and their entities |
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
| Text chat, streaming, image questions | Ollama: `Ollama_API_KEY` (cloud) or a local server, `OLLAMA_TEXT_MODEL`, optional `OLLAMA_VISION_MODEL`, optional `OLLAMA_HOST`, optional `OLLAMA_TIMEOUT_SECONDS` |
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
python -m spacy download en_core_web_sm
```

Install Tesseract separately if scanned-PDF OCR is needed. Copy
`.env.example` to `.env` in the repository root and configure PostgreSQL,
Neo4j, Gemini, Pinecone, and/or Ollama only for the features you plan to
use. Install and start Ollama, or set an Ollama API key to use Ollama Cloud.

Run the backend:

```powershell
Set-Location C:\path\to\context-engine\backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Create the PostgreSQL tables once (non-destructive: it only creates missing
tables):

```powershell
Set-Location C:\path\to\context-engine\backend
.\.venv\Scripts\python.exe init_db.py
```

The Streamlit app under `frontend/` is unfinished; the API and the automated
tests do not require it.

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
frontend/             Streamlit app in progress (app.py, pages/, api_client.py)
```

## Tests

From `backend/`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The automated suite (`backend/tests/unit` and `backend/tests/integration`)
contacts no external service. Pinecone, Neo4j, Gemini, PostgreSQL, and the
LLM are mocked or replaced with in-memory SQLite, and the Ollama client is
never created in tests. The masking tests do load the real spaCy model, so
`en_core_web_sm` must be installed. The PDF pipeline tests build a real small
PDF and parse it with PyMuPDF/pdfplumber while stubbing the paid embedding
and upsert stages.

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
- Graph entity validation reduces noise but is not a general NER fix: a
  single-token name that spaCy labels `ORG` without any corroborating
  evidence in the document (for example a short OCR fragment) can still be
  stored.
- PostgreSQL schema initialization uses `create_all`, not versioned
  migrations.
- External integrations (PostgreSQL, Neo4j, Pinecone, Gemini, Ollama) are
  not live-verified by the automated suite; use the mocked tests and the
  opt-in manual scripts.
- The Streamlit frontend is incomplete.

## Troubleshooting

- **"Could not reach the Ollama backend at ..."** — Ollama is not running
  (or the host/API key is wrong). Start the local server, or set
  `Ollama_API_KEY` to use Ollama Cloud.
- **Startup logs "Model 'x' was not found"** — the configured
  `OLLAMA_TEXT_MODEL` is not available on the active backend. No substitute
  model is used.
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
