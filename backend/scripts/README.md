# Manual integration scripts

The `manual_*.py` files in this directory are **not automated tests**.
The pytest suite only collects `tests/` (`python_files = test_*.py`), so
these scripts are never executed by `python -m pytest`. Each one exercises
a real external service (PostgreSQL, Neo4j, Pinecone, or Gemini generation
and embeddings) and several of them **write**
data.

## Safety: opt-in is required

Every script calls `require_live_opt_in()` (from `manual_guard.py`) before
touching an external service, and several call it from inside `main()`.
By default the scripts refuse to run. To allow one, set
`CONTEXT_ENGINE_ALLOW_LIVE_TESTS=1`:

```bash
# Git Bash / macOS / Linux
cd backend
export CONTEXT_ENGINE_ALLOW_LIVE_TESTS=1
python scripts/manual_short_term_memory.py
```

```bat
:: Windows cmd
cd backend
set CONTEXT_ENGINE_ALLOW_LIVE_TESTS=1
python scripts\manual_short_term_memory.py
```

Run the scripts as `python scripts/<name>.py` from `backend/`. Each script
imports `_bootstrap`, which puts `backend/` on `sys.path` so `app.*` and
`manual_guard` resolve without installing the project or setting
`PYTHONPATH`.

## Scripts

| Script | What it does | Writes external data |
|---|---|---|
| `manual_embedding.py` | Generates a Gemini text embedding for a sample sentence | No (paid API call) |
| `manual_indexing.py` | Embeds and upserts one vector into Pinecone | Yes (Pinecone) |
| `manual_retrieval.py` | Semantic search against Pinecone | No |
| `manual_pinecone.py` | Prints Pinecone index statistics | No |
| `manual_multimodal_rag.py` | Retrieves multimodal content and sends images to the vision model | No |
| `manual_visual_embedding.py` | Demonstrates the image-description embedding path (true visual embeddings are not implemented) | No (paid API call) |
| `manual_short_term_memory.py` | Conversation + recent-message window in PostgreSQL | Yes (PostgreSQL) |
| `manual_message_window.py` | Older/recent message split in PostgreSQL | Yes (PostgreSQL) |
| `manual_long_term_memory.py` | Long-term memory create/list/delete in PostgreSQL | Yes (PostgreSQL) |
| `manual_memory_retrieval.py` | Memory relevance scoring in PostgreSQL | Yes (PostgreSQL) |
| `manual_query_aware_summary.py` | Query-aware summarization using the configured LLM | Yes (PostgreSQL + LLM) |
| `manual_summary_storage.py` | Stored-summary reuse in PostgreSQL | Yes (PostgreSQL) |
| `manual_context_engine_integration.py` | Assembles context from all sources and reports errors | No (reads services) |
| `manual_knowledge_graph.py` | Creates test entities/relationships in Neo4j | Yes (Neo4j) |

> Note: scripts that write data create records under dedicated test users
> / IDs, but they do not automatically clean up. Review the data after a
> run, or use a disposable database.
