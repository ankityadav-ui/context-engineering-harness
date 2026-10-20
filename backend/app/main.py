
# Import FastAPI's request validation tools.
import json
import hashlib
import math
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field

# Import the masking and safety pipeline.
from app.masking.safety_pipeline import mask_and_validate

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import os

from fastapi import (
    Depends,
    Header,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.exc import SQLAlchemyError

from app.llm import (
    describe_active_provider,
    generate_response,
    generate_multimodal_response,
    is_configured as llm_is_configured,
    reset_client as reset_llm_client,
    stream_response,
    validate_llm_config,
    LLMError,
)
from app.rag.document_service import (
    MaskingRejectedError,
    delete_stored_document_files,
    document_storage_status,
    process_document,
    validate_document_id,
    write_masked_text_artifact,
)
from app.rag.ocr import OCRUnavailableError
from app.rag.text_extractor import PDFProcessingError
from app.rag.embedding_service import (
    generate_text_embedding,
    resolve_image_path,
)
from app.rag.pinecone_service import (
    delete_document_from_index,
    get_document_content_units,
    list_indexed_documents,
    search_similar_content,
)

# Service modules whose configuration is reported by GET /capabilities.
# Importing them does not connect to anything: each client is created on
# first use (see the is_configured() helper in each module).
from app.context_engine import database as context_database
from app.context_engine import graph_cleanup, graph_service, ingestion_jobs
from app.rag import embedding_service, ocr, pinecone_service
from app.context_engine.engine import (
    assemble_context,
    build_context_diagnostics,
    build_context_report,
    format_context,
    is_summary_request,
)
from app.context_engine.summarizer import summarize_document_units
from app.context_engine.conversation_service import (
    clear_conversation,
    conversation_accessible,
    create_conversation,
    delete_conversation,
    get_conversation,
    get_conversation_system_prompt,
    get_last_n_messages,
    is_valid_conversation_id,
    list_conversation_messages,
    list_conversations,
    reset_conversation_system_prompt,
    save_conversation_system_prompt,
    save_message,
)
from app.context_engine.graph_service import (
    find_entity_documents,
    find_document_entities,
    find_document_graph,
    find_entities,
    get_entity_neighborhood,
    get_entity_relationships,
    graph_counts,
    ingest_document_graph,
    list_documents,
    list_entities,
    verify_graph_connection,
)
from app.context_engine.memory_service import (
    count_memories,
    create_memory,
    delete_memory,
    list_memories,
)
from app.system_prompt import (
    MAX_CUSTOM_PROMPT_LENGTH,
    SECURITY_INSTRUCTIONS,
    build_instructions,
)


class NoRetrievableContentError(PDFProcessingError):
    """Raised when a valid PDF yields no indexable content units."""


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[1]
DOCUMENT_DIR = BACKEND_DIR / "storage" / "documents"
DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)

# Uploads are read into memory in bounded chunks, so a single request
# cannot exhaust the process. 25 MiB also covers large scanned PDFs.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024

# PDF files start with this signature. Checking it rejects a renamed
# non-PDF payload before any parser touches the bytes.
PDF_SIGNATURE = b"%PDF-"


def _configure_logging() -> None:
    """Attach one stderr handler to the application logger.

    Uvicorn only configures its own loggers, so without this the
    application's INFO records would be dropped. Propagation is left on
    so an embedding host (or the test suite) can still capture records.
    """

    level = os.getenv("CEH_LOG_LEVEL", "INFO").strip().upper() or "INFO"

    app_logger = logging.getLogger("app")

    if not app_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s"
            )
        )
        app_logger.addHandler(handler)

    app_logger.setLevel(level)


_configure_logging()

logger = logging.getLogger("app.main")
_INGESTION_EXECUTOR = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="document-ingestion",
)
_SCHEDULED_INGESTION_JOBS: set[str] = set()
_SCHEDULED_INGESTION_LOCK = threading.Lock()


def _llm_http_exception(error: LLMError) -> HTTPException:
    headers = {
        "X-Error-Code": error.code,
        "X-Retryable": "true" if error.retryable else "false",
    }
    if error.retry_after_seconds is not None:
        headers["Retry-After"] = str(
            max(1, math.ceil(error.retry_after_seconds))
        )
    return HTTPException(
        status_code=503,
        detail=str(error),
        headers=headers,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validate the LLM configuration at startup. A missing API key is a
    # hard startup failure; an unreachable availability check or an
    # unavailable model tag is reported loudly but does not hide the
    # configured model behind a silent substitution.
    try:
        validation = validate_llm_config()
    except LLMError as error:
        logger.error("LLM configuration invalid at startup: %s", error)
        raise error

    if validation.get("warning"):
        logger.warning("LLM startup check: %s", validation["warning"])
    else:
        logger.info(
            "LLM startup check passed (model=%s)", validation["model"]
        )

    if (
        context_database.is_configured()
        and os.getenv("INGESTION_JOB_RECOVERY_ENABLED", "true")
        .strip()
        .casefold()
        in {"1", "true", "yes"}
    ):
        try:
            recovered_job_ids = ingestion_jobs.recover_interrupted_jobs()
            for recovered_job_id in recovered_job_ids:
                _schedule_ingestion_job(recovered_job_id)
            if recovered_job_ids:
                logger.info(
                    "Recovered interrupted ingestion jobs count=%s",
                    len(recovered_job_ids),
                )
        except Exception as error:
            logger.warning(
                "Ingestion job recovery unavailable code=%s",
                type(error).__name__,
            )

    try:
        yield

    finally:
        # Release every external resource the API may have opened. Each
        # cleanup is independent: one failing close must not prevent the
        # others, and nothing is raised out of shutdown.
        try:
            graph_service.close_graph_connection()
        except Exception:
            logger.warning("Neo4j connection cleanup failed.", exc_info=True)

        try:
            context_database.dispose_engine()
        except Exception:
            logger.warning(
                "Database connection cleanup failed.", exc_info=True
            )

        try:
            reset_llm_client()
        except Exception:
            logger.warning("LLM client cleanup failed.", exc_info=True)


app = FastAPI(
    title="Context Engineering Harness API",
    description="Chat, multimodal RAG, and unified context assembly.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Request models
# --------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)

    # Optional durable conversation. When present, both user and
    # assistant messages are persisted and the prompt is assembled by
    # the single authoritative context-assembly path (conversation
    # summary + recent messages + memory + RAG + graph). When absent,
    # the pre-existing stateless contract is preserved exactly.
    conversation_id: str | None = Field(default=None, max_length=36)

    # Session-ownership claim (no authentication system exists; this
    # mirrors the trusted user_id claim used by the memory API).
    session_id: str | None = Field(default=None, max_length=64)

    # Optional long-term-memory owner for context assembly.
    user_id: str | None = Field(default=None, max_length=100)

    # Optional scope for the uploaded/selected document. When present,
    # retrieval is restricted to that document and the Context Engine's
    # full-text fallback can ground "summarise this" even when vector
    # search returns nothing.
    document_id: str | None = Field(default=None, max_length=64)


class DocumentSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: str | None = None


class RAGRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: str | None = None
    conversation_id: str | None = Field(default=None, max_length=36)
    user_id: str | None = None

    # Session-ownership claim, mirroring ChatRequest: when a
    # conversation_id is supplied, the same session-ownership rules as
    # POST /chat apply to it.
    session_id: str | None = Field(default=None, max_length=64)


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Context Engineering Harness API is running."
    }


# --------------------------------------------------
# Health check
# --------------------------------------------------
# This endpoint only reports that the API process itself is up.
# It deliberately does NOT contact Pinecone, Gemini, Neo4j, or
# PostgreSQL, so it must not claim those services are healthy.

@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------
# Service capabilities
# --------------------------------------------------
# Reports which optional external services are configured, so the UI can
# label unavailable features honestly instead of failing at the point of
# use. Booleans and required environment-variable names only: no value,
# credential, or connection attempt is ever returned or performed.

@app.get("/capabilities")
def read_capabilities():
    """Report configured services (booleans only, no secrets)."""

    llm_active = describe_active_provider()

    return {
        "services": [
            {
                "key": "llm",
                "label": "Language model (Google Gemini)",
                "configured": llm_is_configured(),
                "model": llm_active["model"],
                "required_env": ["GEMINI_API_KEY"],
            },
            {
                "key": "embeddings",
                "label": "Gemini text embeddings",
                "configured": embedding_service.is_configured(),
                "required_env": ["GEMINI_API_KEY"],
            },
            {
                "key": "vector_index",
                "label": "Pinecone vector index",
                "configured": pinecone_service.is_configured(),
                "required_env": [
                    "PINECONE_API_KEY",
                    "PINECONE_INDEX_NAME",
                ],
            },
            {
                "key": "database",
                "label": "PostgreSQL (conversations, memory)",
                "configured": context_database.is_configured(),
                "required_env": ["DATABASE_URL"],
            },
            {
                "key": "knowledge_graph",
                "label": "Neo4j knowledge graph",
                "configured": graph_service.is_configured(),
                "required_env": ["NEO4J_URI", "NEO4J_PASSWORD"],
            },
            {
                "key": "ocr",
                "label": "Tesseract OCR binary",
                "configured": ocr.is_configured(),
                "required_env": ["TESSERACT_PATH"],
            },
        ]
    }


# --------------------------------------------------
# Normal chat
# --------------------------------------------------

def get_db_optional():
    """Database session for routes that persist only when asked.

    Yields a session when storage is configured, and None otherwise,
    so a stateless request (POST /chat without a conversation_id) keeps
    working without DATABASE_URL. A persistence request without
    configuration fails at the route with a clear 503 instead of a
    crash. Delegates to get_db for session handling, so tests that
    override get_db cover this dependency too (see conftest).
    """

    if not context_database.is_configured():
        yield None
        return

    yield from get_db()


def _normalize_conversation_id(
    conversation_id: str | None,
) -> str | None:
    """Trim an optional conversation id; None means stateless chat."""

    if not conversation_id or not conversation_id.strip():
        return None

    return conversation_id.strip()


def _validated_document_id(
    document_id: str | None,
) -> str | None:
    """Trim and validate an optional document scope.

    The id is later used to build storage paths, so an invalid value is
    rejected with a client error instead of being passed to a query or
    path helper.
    """

    if not document_id or not document_id.strip():
        return None

    try:
        return validate_document_id(document_id)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


def _load_accessible_conversation(
    db,
    conversation_id: str,
    session_id: str | None,
):
    """Return the conversation, or raise a client error.

    Unknown and not-owned conversations both return 404 so one
    session's conversation id can never be probed by another.
    """

    if db is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Conversation persistence requires storage; set "
                "DATABASE_URL in the root .env file."
            ),
        )

    if not is_valid_conversation_id(conversation_id):
        raise HTTPException(
            status_code=404,
            detail=f"Conversation {conversation_id!r} was not found.",
        )

    try:
        conversation = get_conversation(db, conversation_id)

    except SQLAlchemyError as error:
        raise _database_error() from error

    if (
        conversation is None
        or not conversation_accessible(conversation, session_id)
    ):
        raise HTTPException(
            status_code=404,
            detail=f"Conversation {conversation_id!r} was not found.",
        )

    return conversation


@app.post("/chat")
def chat(request: ChatRequest, db=Depends(get_db_optional)):
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    conversation_id = _normalize_conversation_id(
        request.conversation_id
    )

    # The document scope is what makes "summarise this" resolve to the
    # resume the user just uploaded. Without it, retrieval searches the
    # whole index and the full-text fallback can never run.
    target_document_id = _validated_document_id(request.document_id)

    # Stateless chat: the pre-existing contract, unchanged. A caller
    # that explicitly scopes a document still gets document evidence.
    if conversation_id is None and target_document_id is None:
        system_text, custom_prompt_applied = build_instructions()

        try:
            answer = generate_response(
                prompt=message,
                system=system_text,
            )
            return {
                "response": answer,
                "system_prompt_active": custom_prompt_applied,
            }

        except LLMError as error:
            raise _llm_http_exception(error) from error

    # Persistent chat: everything is assembled through the single
    # authoritative context path, which reads the conversation's
    # persisted history (summary of older messages + latest six).
    # A document-scoped stateless request still uses that same path,
    # just without conversation history.
    conversation = None

    if conversation_id is not None:
        conversation = _load_accessible_conversation(
            db,
            conversation_id,
            request.session_id,
        )

    # Trusted instructions (the selected conversation's custom prompt
    # plus the built-in security rules) travel as a separate system
    # message; the user's message stays user content.
    system_text, custom_prompt_applied = build_instructions(
        custom_prompt=(conversation.system_prompt or "") if conversation else ""
    )

    try:
        context = assemble_context(
            query=message,
            document_id=target_document_id,
            conversation_id=(
                conversation.id if conversation is not None else None
            ),
            user_id=request.user_id,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    document_summary = (
        target_document_id is not None and is_summary_request(message)
    )

    if (
        document_summary
        and not context.get("rag")
        and context.get("source_errors")
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "The selected document could not be read for a grounded "
                "summary. Check document retrieval and try again."
            ),
        )

    if context.get("document_summary_info", {}).get(
        "source_provenance_unavailable"
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "The stored masked text has no page/chunk provenance. "
                "Reprocess the PDF before requesting a grounded summary."
            ),
        )

    if (
        document_summary
        and context.get("document_summary_info", {}).get(
            "document_found"
        ) is False
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "The selected document has no indexed content. "
                "Reprocess the PDF or select a different document."
            ),
        )

    # A selected document with no retrievable content cannot support a
    # normal grounded answer. Summary requests can accurately report that
    # extraction/OCR found no text without asking the model to invent it.
    if (
        target_document_id is not None
        and not context.get("rag")
        and not document_summary
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "The selected document has no retrievable content, so "
                "it cannot be summarised. Re-upload the PDF or select a "
                "different document."
            ),
        )

    # An empty retrieval result without a document scope is a
    # legitimate outcome: chat works without uploaded documents.
    formatted_context = (
        None if document_summary else format_context(context)
    )

    # Persist the user turn only after the context was assembled, so
    # the current message is never duplicated inside its own context.
    if conversation is not None:
        _save_chat_message(db, conversation.id, "user", message)

    try:
        if document_summary:
            summary_info = context.get("document_summary_info", {})
            if summary_info.get("units_truncated"):
                raise LLMError(
                    "The document exceeds the safe whole-document "
                    "summary limit. Narrow the document or reprocess it "
                    "into smaller documents."
                )
            answer = summarize_document_units(
                units=context.get("rag", []),
                document_id=target_document_id,
                pages_without_text=summary_info.get(
                    "pages_without_text", []
                ),
            )
        else:
            answer = generate_response(
                prompt=formatted_context,
                system=system_text,
            )

    except LLMError as error:
        # Fail-closed persistence: the user turn is stored, but no
        # assistant answer is saved for a failed generation.
        raise _llm_http_exception(error) from error

    if conversation is None:
        return {
            "response": answer,
            "system_prompt_active": custom_prompt_applied,
        }

    _save_chat_message(db, conversation.id, "assistant", answer)

    return {
        "response": answer,
        "system_prompt_active": custom_prompt_applied,
        "conversation_id": conversation.id,
        "conversation_context": _describe_conversation_context(
            context
        ),
    }


def _save_chat_message(
    db,
    conversation_id: str,
    role: str,
    content: str,
) -> None:
    """Persist one chat message, converting failures to API errors."""

    try:
        save_message(
            db=db,
            conversation_id=conversation_id,
            role=role,
            content=content,
        )

    except ValueError as error:
        # The conversation disappeared between checks (or content was
        # empty, which the endpoint already rejected).
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except SQLAlchemyError as error:
        raise _database_error() from error


def _describe_conversation_context(context: dict) -> dict:
    """Counts-only description of the assembled conversation context."""

    conversation = context.get("conversation") or {}

    return {
        "total_messages": conversation.get("total_messages", 0),
        "older_count": conversation.get("older_count", 0),
        "recent_count": conversation.get("recent_count", 0),
        "summary_present": bool(conversation.get("summary")),
    }


def _summarize_document_context(context: dict, document_id: str) -> str:
    summary_info = context.get("document_summary_info", {})

    if summary_info.get("source_provenance_unavailable"):
        raise HTTPException(
            status_code=503,
            detail=(
                "The stored masked text has no page/chunk provenance. "
                "Reprocess the PDF before requesting a grounded summary."
            ),
        )

    if (
        summary_info.get("document_found") is False
        and not context.get("rag")
    ):
        raise HTTPException(
            status_code=404,
            detail="The selected document was not found in the index or storage.",
        )

    if summary_info.get("units_truncated"):
        raise LLMError(
            "The document exceeds the safe whole-document summary "
            "limit. Narrow the document or reprocess it into smaller "
            "documents."
        )

    if not context.get("rag") and context.get("source_errors"):
        raise HTTPException(
            status_code=503,
            detail=(
                "The selected document could not be read for a grounded "
                "summary. Check document retrieval and try again."
            ),
        )

    return summarize_document_units(
        units=context.get("rag", []),
        document_id=document_id,
        pages_without_text=summary_info.get("pages_without_text", []),
    )


# --------------------------------------------------
# Streaming chat (Server-Sent Events)
# --------------------------------------------------
# A separate endpoint keeps POST /chat's JSON contract untouched. The
# request body is identical, and the instructions are composed by the
# same build_instructions path as POST /chat (including the selected
# conversation's custom prompt), so streaming never bypasses the
# system-prompt composition or security instructions.
#
# Event protocol (one SSE event per frame):
#   event: chunk  data: {"delta": "..."}          incremental text
#   event: done   data: {"response": "...", "provider": "...",
#                        "model": "...", "system_prompt_active": bool}
#   event: error  data: {"detail": "...", "partial": "..."}
# The stream always ends with exactly one done or error event.


def _sse_event(event: str, data: dict) -> str:
    """Serialize one server-sent event frame (UTF-8 safe JSON)."""

    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    )


@app.post("/chat/stream")
def chat_stream(
    request: ChatRequest,
    db=Depends(get_db_optional),
):
    """Stream an assistant reply as server-sent events.

    Chunks are emitted as the provider generates them. Provider
    failures surface as a terminal `error` event (never leaking the
    prompt or credentials). When the client disconnects, Starlette
    closes this generator, which closes the provider's upstream
    stream and releases the connection.

    When a conversation_id is supplied, the prompt is assembled by the
    same authoritative context path as POST /chat, the user turn is
    persisted before streaming, and the FINAL assistant response is
    persisted exactly once after the stream completes successfully.
    No message is stored per SSE chunk, and a failed stream never
    stores a misleading assistant answer.
    """

    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    # Captured at request time so the completion event reports the
    # provider that actually served this stream.
    active = describe_active_provider()

    conversation_id = _normalize_conversation_id(
        request.conversation_id
    )
    target_document_id = _validated_document_id(request.document_id)
    conversation = None
    prompt_text = message

    # Same instruction-composition path as POST /chat: trusted
    # instructions are sent as the system message, and the user's
    # message is sent as user content. Without a conversation only the
    # default trusted instructions (built-in security rules) apply.
    system_text, custom_prompt_applied = build_instructions()

    if conversation_id is not None or target_document_id is not None:
        if conversation_id is not None:
            conversation = _load_accessible_conversation(
                db,
                conversation_id,
                request.session_id,
            )

            # The selected conversation's own custom prompt is composed
            # with the built-in security rules, exactly like POST /chat.
            system_text, custom_prompt_applied = build_instructions(
                custom_prompt=conversation.system_prompt or ""
            )

        try:
            context = assemble_context(
                query=message,
                document_id=target_document_id,
                conversation_id=(
                    conversation.id
                    if conversation is not None
                    else None
                ),
                user_id=request.user_id,
            )

        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=str(error),
            ) from error

        # A selected document with no retrievable content cannot
        # support a grounded answer; report it instead of streaming a
        # fabricated summary. Without a document scope, empty retrieval
        # is a legitimate outcome.
        if target_document_id is not None and not context.get("rag"):
            raise HTTPException(
                status_code=400,
                detail=(
                    "The selected document has no retrievable content, "
                    "so it cannot be summarised. Re-upload the PDF or "
                    "select a different document."
                ),
            )

        prompt_text = format_context(context)

        # Persist the user turn before streaming. The current message
        # is excluded from its own context because assembly ran first.
        if conversation is not None:
            _save_chat_message(db, conversation.id, "user", message)

    def event_stream():
        emitted: list[str] = []
        deltas = stream_response(
            prompt=prompt_text,
            system=system_text,
        )

        try:
            for delta in deltas:
                emitted.append(delta)
                yield _sse_event("chunk", {"delta": delta})

            answer = "".join(emitted)

            # Persist the FINAL response once, after the stream
            # completed successfully. A persistence failure is
            # reported as an error event with the generated answer,
            # instead of a done event that would falsely imply the
            # conversation stored it.
            if conversation is not None:
                try:
                    save_message(
                        db=db,
                        conversation_id=conversation.id,
                        role="assistant",
                        content=answer,
                    )

                except Exception:
                    logger.exception(
                        "Assistant message persistence failed for "
                        "conversation %s",
                        conversation.id,
                    )
                    yield _sse_event(
                        "error",
                        {
                            "detail": (
                                "The reply was generated but could "
                                "not be stored in the conversation."
                            ),
                            "partial": answer,
                        },
                    )
                    return

            done_payload = {
                "response": answer,
                "provider": active["provider"],
                "model": active["model"],
                "system_prompt_active": custom_prompt_applied,
            }

            if conversation is not None:
                done_payload["conversation_id"] = conversation.id

            yield _sse_event("done", done_payload)

        except LLMError as error:
            # The failure message comes from our own LLM layer: it
            # describes configuration/provider problems without the
            # prompt or any credentials. No assistant message is
            # persisted for a failed generation.
            yield _sse_event(
                "error",
                {
                    "detail": str(error),
                    "code": error.code,
                    "retryable": error.retryable,
                    "retry_after_seconds": error.retry_after_seconds,
                    "partial": "".join(emitted),
                },
            )

        except Exception:
            # Unknown failures never echo internals to the client.
            yield _sse_event(
                "error",
                {
                    "detail": (
                        "The response stream failed unexpectedly."
                    ),
                    "partial": "".join(emitted),
                },
            )

        finally:
            # Deterministic cancellation: when the client disconnects
            # (GeneratorExit) or the stream ends, close the LLM
            # generator so the provider releases its upstream
            # connection instead of waiting for garbage collection.
            deltas.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Disable proxy buffering so chunks leave immediately.
            "X-Accel-Buffering": "no",
        },
    )


# --------------------------------------------------
# Document upload and indexing
# --------------------------------------------------

def _parse_mask_flag(value: str | None) -> bool:
    """Parse the optional mask_content multipart field."""

    return (value or "").strip().lower() in {"true", "1", "yes"}


async def _read_upload_bytes(file: UploadFile) -> bytes:
    """Read an upload in bounded chunks.

    Raises 413 as soon as the configured limit is exceeded, so an
    oversized body is never buffered whole in memory.
    """

    chunks: list[bytes] = []
    total = 0

    while True:
        chunk = await file.read(UPLOAD_CHUNK_BYTES)

        if not chunk:
            break

        total += len(chunk)

        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    "The uploaded file is larger than the "
                    f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
                ),
            )

        chunks.append(chunk)

    return b"".join(chunks)


def _remove_upload_artifacts(
    document_id: str,
    document_path: Path,
    remove_indexed_content: bool = False,
) -> None:
    """Best-effort removal of a failed upload's stored artifacts.

    Removes the raw PDF and any extracted images so a failed (or
    rejected) upload never leaves document content on disk.
    """

    try:
        if document_path.is_file():
            document_path.unlink()

    except OSError:
        pass

    try:
        delete_stored_document_files(document_id)

    except Exception:
        logger.warning(
            "Could not remove local artifacts for failed document %s.",
            document_id,
        )

    if remove_indexed_content:
        try:
            delete_document_from_index(document_id)
        except Exception as error:
            # This id was generated for the in-flight upload, not an
            # existing document. Log cleanup failure without hiding the
            # original ingestion failure.
            logger.warning(
                "Could not remove partial vectors for failed document %s (%s).",
                document_id,
                type(error).__name__,
            )


def _record_ingestion_stage(
    job_id: str,
    document_id: str,
    stage: str,
    stage_status: str,
    duration_ms: float,
    error_code: str | None,
) -> None:
    ingestion_jobs.update_stage_timing(
        job_id,
        stage=stage,
        status=stage_status,
        duration_ms=duration_ms,
        error_code=error_code,
    )
    logger.info(
        "document_ingestion_stage job_id=%s document_id=%s stage=%s "
        "status=%s duration_ms=%.1f error_code=%s",
        job_id,
        document_id,
        stage,
        stage_status,
        duration_ms,
        error_code or "-",
    )


def _safe_ingestion_failure(stage: str, error: Exception) -> tuple[str, str]:
    if isinstance(error, NoRetrievableContentError):
        return (
            "no_retrievable_content",
            "No retrievable text or content was extracted from this PDF.",
        )
    if isinstance(error, PDFProcessingError):
        return (
            "invalid_pdf",
            "The file is not a valid PDF or has no readable pages. "
            "Check the file and retry.",
        )
    if isinstance(error, OCRUnavailableError):
        return "ocr_unavailable", "OCR is unavailable. Check TESSERACT_PATH and retry."
    if isinstance(error, MaskingRejectedError):
        return (
            "masking_rejected",
            "The document failed the masking safety validation.",
        )
    if isinstance(error, LLMError):
        return error.code, str(error)
    if stage == "graph_extraction":
        return "graph_extraction_failed", "Knowledge graph extraction failed. Retry this job."
    if stage == "neo4j_persistence":
        return "graph_persistence_failed", "Knowledge graph persistence failed. Retry this job."
    if stage == "embedding":
        return "embedding_failed", "Document embedding failed. Retry this job."
    if stage == "pinecone_indexing":
        return "vector_indexing_failed", "Document indexing failed. Retry this job."
    return (
        f"{stage or 'ingestion'}_failed",
        "Document processing failed at the reported stage. Retry this job.",
    )


def _execute_ingestion_job(job_id: str) -> None:
    """Run the durable ingestion job; retries reuse stable document IDs."""
    try:
        job = ingestion_jobs.get_job(job_id)
    except Exception as error:
        logger.error(
            "Could not load ingestion job job_id=%s error_code=%s",
            job_id,
            type(error).__name__,
        )
        return
    if job is None or job.status == "completed":
        return

    document_id = job.document_id
    document_path = Path(job.file_path)
    current_stage = "queued"
    ingestion_started = time.perf_counter()
    attempt = job.attempts + 1

    def stage_callback(
        stage: str,
        status: str,
        duration_ms: float,
        error_code: str | None,
    ) -> None:
        nonlocal current_stage
        current_stage = stage
        _record_ingestion_stage(
            job_id,
            document_id,
            stage,
            status,
            duration_ms,
            error_code,
        )

    try:
        ingestion_jobs.update_job(
            job_id,
            status="processing",
            stage="upload_validation",
            attempts=attempt,
            error_code=None,
            error_message=None,
        )
        logger.info(
            "document_ingestion_started job_id=%s document_id=%s attempt=%s",
            job_id,
            document_id,
            attempt,
        )
        if not document_path.is_file():
            raise FileNotFoundError("Persisted PDF is unavailable.")

        content_units, ocr_pages, table_units, image_units, masking_stats = (
            process_document(
                document_path=document_path,
                document_id=document_id,
                apply_masking=bool(job.mask_content),
                on_stage=stage_callback,
            )
        )
        if not content_units:
            raise NoRetrievableContentError(
                "No retrievable content was extracted from this PDF."
            )

        text_units = [
            unit
            for unit in content_units
            if unit.get("content_type") in {"text", "table"}
            and unit.get("content", "").strip()
        ]
        graph_result: dict | None = None
        graph_error: str | None = None
        if graph_service.is_configured() and text_units:
            text_for_graph = "\n".join(
                unit.get("content", "") for unit in text_units
            )
            graph_pages = [
                {
                    "page_number": unit.get("page_number"),
                    "source_unit_id": unit.get("id"),
                    "content": unit.get("content", ""),
                }
                for unit in text_units
            ]
            graph_result = ingest_document_graph(
                document_id=document_id,
                title=job.filename,
                content=text_for_graph,
                pages=graph_pages,
                on_stage=stage_callback,
            )
        elif not graph_service.is_configured():
            graph_error = "Neo4j is not configured; graph ingestion skipped."
            stage_callback(
                "knowledge_graph",
                "skipped",
                0.0,
                "neo4j_not_configured",
            )

        masked_text_stored = False
        if masking_stats["applied"]:
            artifact_started = time.perf_counter()
            try:
                masked_text_stored = (
                    write_masked_text_artifact(document_id, content_units)
                    is not None
                )
            except OSError as error:
                stage_callback(
                    "masked_text_artifact",
                    "failed",
                    (time.perf_counter() - artifact_started) * 1000,
                    type(error).__name__,
                )
                raise
            else:
                stage_callback(
                    "masked_text_artifact",
                    "completed",
                    (time.perf_counter() - artifact_started) * 1000,
                    None,
                )
            if document_path.exists():
                document_path.unlink()

        result = {
            "content_units": len(content_units),
            "ocr_pages": ocr_pages,
            "tables_extracted": len(table_units),
            "images_extracted": len(image_units),
            "masking_applied": masking_stats["applied"],
            "masking_safety_passed": masking_stats["safety_passed"],
            "masked_entities": masking_stats["masked_entities"],
            "newly_masked_entities": masking_stats.get(
                "masked_entities", 0
            ),
            "existing_placeholders": masking_stats.get(
                "existing_placeholders", 0
            ),
            "existing_placeholder_occurrences": masking_stats.get(
                "existing_placeholder_occurrences", 0
            ),
            "masked_text_stored": masked_text_stored,
            "knowledge_graph": {
                "ingested": graph_result is not None,
                "entities": (
                    graph_result["entities_ingested"]
                    if graph_result
                    else None
                ),
                "relationships": (
                    graph_result["entity_relations_ingested"]
                    if graph_result
                    else None
                ),
                "error": graph_error,
            },
        }
        stage_callback(
            "completion",
            "completed",
            (time.perf_counter() - ingestion_started) * 1000,
            None,
        )
        ingestion_jobs.update_job(
            job_id,
            status="completed",
            stage="completed",
            result=result,
            error_code=None,
            error_message=None,
        )
        logger.info(
            "document_ingestion_completed job_id=%s document_id=%s "
            "duration_ms=%.1f content_units=%s graph_ingested=%s",
            job_id,
            document_id,
            (time.perf_counter() - ingestion_started) * 1000,
            len(content_units),
            graph_result is not None,
        )
    except Exception as error:
        error_code, error_message = _safe_ingestion_failure(
            current_stage,
            error,
        )
        if isinstance(
            error,
            (PDFProcessingError, MaskingRejectedError),
        ):
            try:
                document_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                logger.warning(
                    "Could not remove rejected upload job_id=%s "
                    "error_code=%s",
                    job_id,
                    type(cleanup_error).__name__,
                )
            try:
                delete_stored_document_files(document_id)
            except Exception as cleanup_error:
                logger.warning(
                    "Could not remove rejected document artifacts "
                    "document_id=%s error_code=%s",
                    document_id,
                    type(cleanup_error).__name__,
                )
        try:
            ingestion_jobs.update_job(
                job_id,
                status="failed",
                stage=current_stage,
                error_code=error_code,
                error_message=error_message,
            )
        except Exception as status_error:
            logger.error(
                "Could not persist ingestion failure job_id=%s "
                "error_code=%s",
                job_id,
                type(status_error).__name__,
            )
            return
        logger.error(
            "document_ingestion_failed job_id=%s document_id=%s "
            "stage=%s error_code=%s exception_type=%s duration_ms=%.1f",
            job_id,
            document_id,
            current_stage,
            error_code,
            type(error).__name__,
            (time.perf_counter() - ingestion_started) * 1000,
        )


def _schedule_ingestion_job(job_id: str) -> bool:
    with _SCHEDULED_INGESTION_LOCK:
        if job_id in _SCHEDULED_INGESTION_JOBS:
            return False
        _SCHEDULED_INGESTION_JOBS.add(job_id)

    def run_and_release() -> None:
        try:
            _execute_ingestion_job(job_id)
        finally:
            with _SCHEDULED_INGESTION_LOCK:
                _SCHEDULED_INGESTION_JOBS.discard(job_id)

    try:
        _INGESTION_EXECUTOR.submit(run_and_release)
    except Exception:
        with _SCHEDULED_INGESTION_LOCK:
            _SCHEDULED_INGESTION_JOBS.discard(job_id)
        raise
    return True


@app.post("/documents/upload", status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    # Optional multipart field; when "true", extracted text (including
    # OCR output) is masked and re-validated BEFORE embedding/indexing.
    mask_content: str = Form(default="false"),
    idempotency_key: str | None = Header(
        default=None,
        alias="Idempotency-Key",
        max_length=200,
    ),
):
    started = time.perf_counter()
    try:
        if (
            not file.filename
            or Path(file.filename).suffix.lower() != ".pdf"
        ):
            raise HTTPException(
                status_code=400,
                detail="Please upload a PDF file.",
            )
        apply_masking = _parse_mask_flag(mask_content)
        file_bytes = await _read_upload_bytes(file)
        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        # A renamed non-PDF must not reach the PDF parsers.
        if not file_bytes.startswith(PDF_SIGNATURE):
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is not a valid PDF.",
            )

        validation_ms = (time.perf_counter() - started) * 1000
        content_hash = hashlib.sha256(file_bytes).hexdigest()
        key_source = (
            f"client:{idempotency_key.strip()}:content:{content_hash}:"
            f"mask={int(apply_masking)}"
            if idempotency_key and idempotency_key.strip()
            else f"request:{uuid.uuid4()}"
        )
        safe_idempotency_key = hashlib.sha256(
            key_source.encode("utf-8")
        ).hexdigest()
        document_id = str(uuid.uuid4())
        document_path = DOCUMENT_DIR / f"{document_id}.pdf"
        try:
            job, created = ingestion_jobs.create_or_get_job(
                idempotency_key=safe_idempotency_key,
                filename=Path(file.filename or "upload.pdf").name,
                file_path=str(document_path),
                mask_content=apply_masking,
            )
        except Exception as error:
            logger.error(
                "Ingestion job persistence failed error_code=%s",
                type(error).__name__,
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "Durable ingestion job storage is unavailable. "
                    "Apply the database migrations and retry."
                ),
                headers={"X-Error-Code": "ingestion_job_store_unavailable"},
            ) from error

        if created:
            persistence_started = time.perf_counter()
            try:
                document_path.write_bytes(file_bytes)
                _record_ingestion_stage(
                    job.id,
                    job.document_id,
                    "upload_validation",
                    "completed",
                    validation_ms,
                    None,
                )
                _record_ingestion_stage(
                    job.id,
                    job.document_id,
                    "upload_persistence",
                    "completed",
                    (time.perf_counter() - persistence_started) * 1000,
                    None,
                )
            except Exception as error:
                ingestion_jobs.update_job(
                    job.id,
                    status="failed",
                    stage="upload_persistence",
                    error_code="upload_persistence_failed",
                    error_message="The uploaded PDF could not be stored.",
                )
                if document_path.exists():
                    document_path.unlink()
                raise HTTPException(
                    status_code=500,
                    detail="The uploaded PDF could not be stored.",
                ) from error
            try:
                _schedule_ingestion_job(job.id)
            except RuntimeError as error:
                ingestion_jobs.update_job(
                    job.id,
                    status="failed",
                    stage="queued",
                    error_code="ingestion_worker_unavailable",
                    error_message=(
                        "The ingestion worker is unavailable. Retry this job."
                    ),
                )
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "The ingestion worker is unavailable. "
                        "Retry the accepted job."
                    ),
                ) from error

        payload = ingestion_jobs.job_response(job)
        payload["status_url"] = f"/documents/jobs/{job.id}"
        payload["duplicate_request"] = not created
        return payload
    finally:
        await file.close()


@app.get("/documents/jobs/{job_id}")
def read_ingestion_job(job_id: str):
    try:
        normalized_id = str(uuid.UUID(job_id))
        job = ingestion_jobs.get_job(normalized_id)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail="Invalid ingestion job ID.",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Ingestion job status is unavailable.",
        ) from error
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found.")
    return ingestion_jobs.job_response(job)


@app.post("/documents/jobs/{job_id}/retry", status_code=202)
def retry_ingestion_job(job_id: str):
    try:
        normalized_id = str(uuid.UUID(job_id))
        job = ingestion_jobs.get_job(normalized_id)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail="Invalid ingestion job ID.",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Ingestion job status is unavailable.",
        ) from error
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found.")
    if job.status != "failed":
        raise HTTPException(
            status_code=409,
            detail="Only failed ingestion jobs can be retried.",
        )
    if not Path(job.file_path).is_file():
        raise HTTPException(
            status_code=409,
            detail="The persisted PDF is unavailable; upload it again.",
        )
    ingestion_jobs.update_job(
        job.id,
        status="queued",
        stage="queued",
        error_code=None,
        error_message=None,
    )
    try:
        _schedule_ingestion_job(job.id)
    except RuntimeError as error:
        ingestion_jobs.update_job(
            job.id,
            status="failed",
            stage="queued",
            error_code="ingestion_worker_unavailable",
            error_message="The ingestion worker is unavailable.",
        )
        raise HTTPException(
            status_code=503,
            detail="The ingestion worker is unavailable. Retry this job.",
        ) from error
    refreshed = ingestion_jobs.get_job(job.id)
    return ingestion_jobs.job_response(refreshed)


# --------------------------------------------------
# Semantic document search
# --------------------------------------------------

@app.post("/documents/search")
def search_documents(request: DocumentSearchRequest):
    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty.",
        )

    try:
        query_embedding = generate_text_embedding(query)

        results = search_similar_content(
            query_embedding=query_embedding,
            top_k=request.top_k,
            document_id=request.document_id,
        )

        return {
            "query": query,
            "results": results,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        # Configuration problems: the message names the missing
        # environment variable and contains no credentials.
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except Exception:
        logger.exception("Document search failed")
        raise HTTPException(
            status_code=503,
            detail="Document search failed unexpectedly.",
        ) from None


# --------------------------------------------------
# Context Engine question answering
# --------------------------------------------------

@app.post("/documents/ask")
def ask_document(request: RAGRequest, db=Depends(get_db_optional)):
    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    # When the request explicitly targets a conversation, the same
    # access checks as POST /chat apply: the conversation must exist
    # and the caller must own it. This also enables conversation
    # persistence in the frontend without bypassing ownership.
    conversation = None

    requested_conversation_id = _normalize_conversation_id(
        request.conversation_id
    )

    if requested_conversation_id is not None:
        conversation = _load_accessible_conversation(
            db,
            requested_conversation_id,
            request.session_id,
        )

    # If the caller names a document, retrieval is scoped to it. If that
    # returns nothing, the Context Engine falls back to the stored full
    # text so a short resume or a scanned PDF still grounds the answer.
    target_document_id = _validated_document_id(request.document_id)

    try:
        # A named conversation adds conversation, memory and graph
        # evidence to the document evidence, exactly like POST /chat.
        if conversation is not None:
            context = assemble_context(
                query=query,
                top_k=request.top_k,
                document_id=target_document_id,
                conversation_id=conversation.id,
                user_id=request.user_id,
            )

            document_summary = (
                target_document_id is not None
                and is_summary_request(query)
            )
            if document_summary:
                answer = _summarize_document_context(
                    context,
                    target_document_id,
                )
                return {
                    "query": query,
                    "answer": answer,
                    "sources": {
                        "conversation": None,
                        "memory": [],
                        "rag": context.get("rag", []),
                        "knowledge_graph": [],
                    },
                    "source_errors": context.get("source_errors", []),
                    "images_sent_to_llm": 0,
                }

            if target_document_id is not None and not context.get("rag"):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "The selected document has no retrievable "
                        "content. Choose a different document or upload "
                        "a PDF that contains selectable or extracted "
                        "text."
                    ),
                )

            formatted_context = format_context(context)

            # Trusted instructions go to the system role; the assembled
            # evidence (retrieved chunks, memories, graph, conversation)
            # is untrusted and stays in the user content. When the
            # request names a conversation, that conversation's own
            # custom prompt is composed exactly like POST /chat.
            system_text, _custom_prompt_applied = build_instructions(
                custom_prompt=conversation.system_prompt or ""
            )

            # Resolve image references and avoid sending duplicate image
            # files to the vision model. Keep the source records
            # unchanged in the response so the caller can inspect the
            # original retrieval results.
            image_paths = []
            seen_image_paths = set()

            for result in context.get("rag", []):
                if result.get("content_type") != "image":
                    continue

                image_path = resolve_image_path(result)
                if not image_path:
                    continue

                # Normalize paths so equivalent references count as one.
                normalized_path = str(Path(image_path).resolve())
                if normalized_path in seen_image_paths:
                    continue

                seen_image_paths.add(normalized_path)
                image_paths.append(normalized_path)

            # Use multimodal inference only when image files exist.
            if image_paths:
                answer = generate_multimodal_response(
                    message=formatted_context,
                    image_paths=image_paths,
                    system=system_text,
                )
            else:
                answer = generate_response(
                    prompt=formatted_context,
                    system=system_text,
                )

            return {
                "query": query,
                "answer": answer,
                "sources": {
                    "conversation": context.get("conversation"),
                    "memory": context.get("memory", []),
                    "rag": context.get("rag", []),
                    "knowledge_graph": context.get(
                        "knowledge_graph", []
                    ),
                },
                "source_errors": context.get("source_errors", []),
                "images_sent_to_llm": len(image_paths),
            }

        # No conversation: document-grounded questions only. Assemble
        # through the same authoritative path so a resume upload can be
        # summarised from its extracted text.
        context = assemble_context(
            query=query,
            top_k=request.top_k,
            document_id=target_document_id,
            conversation_id=None,
            user_id=request.user_id,
        )

        document_summary = (
            target_document_id is not None
            and is_summary_request(query)
        )
        if document_summary:
            answer = _summarize_document_context(
                context,
                target_document_id,
            )
            return {
                "query": query,
                "answer": answer,
                "sources": {
                    "conversation": None,
                    "memory": [],
                    "rag": context.get("rag", []),
                    "knowledge_graph": [],
                },
                "source_errors": context.get("source_errors", []),
                "images_sent_to_llm": 0,
            }

        # With no selected document and no retrievable evidence there is
        # nothing to ground an answer in. Report that instead of sending
        # a question-only prompt that invites a fabricated summary.
        if not target_document_id and not context.get("rag"):
            raise HTTPException(
                status_code=400,
                detail=(
                    "No document was selected or uploaded. Upload a PDF "
                    "first, then ask to summarise it, or re-run the request "
                    "with a document_id that contains the resume."
                ),
            )

        # A selected document with no retrievable text is also a client
        # error: the model must not invent document content.
        if target_document_id and not context.get("rag"):
            raise HTTPException(
                status_code=400,
                detail=(
                    "The selected document has no retrievable content. "
                    "Choose a different document or upload a PDF that "
                    "contains selectable or extracted text."
                ),
            )

        formatted_context = format_context(context)

        # Trusted instructions go to the system role; the assembled
        # evidence is untrusted and stays in the user content.
        system_text, _custom_prompt_applied = build_instructions()

        answer = generate_response(
            prompt=formatted_context,
            system=system_text,
        )

        return {
            "query": query,
            "answer": answer,
            "sources": {
                "conversation": context.get("conversation"),
                "memory": context.get("memory", []),
                "rag": context.get("rag", []),
                "knowledge_graph": context.get(
                    "knowledge_graph", []
                ),
            },
            "source_errors": context.get("source_errors", []),
            "images_sent_to_llm": 0,
        }

    except HTTPException:
        raise

    except LLMError as error:
        raise _llm_http_exception(error) from error

    except Exception:
        # Log technical details on the server; do not expose internal
        # exception messages, paths, or connection details to API clients.
        logger.exception("Question answering failed")
        raise HTTPException(
            status_code=500,
            detail="Question answering failed unexpectedly.",
        ) from None


# --------------------------------------------------
# Context assembly preview
# --------------------------------------------------
# Read-only inspection of the same authoritative assembly path used by
# POST /documents/ask. It calls assemble_context and format_context and
# sends nothing to the language model itself, so no answer is generated
# and no prompt completion is billed.
#
# Caveat: when conversation_id is supplied, the conversation branch of
# assemble_context refreshes the stored conversation summary, which can
# call the configured LLM once (exactly as POST /documents/ask does).
# No chat answer is generated; /context/preview never calls
# generate_response itself.
#
# Session ownership matches POST /chat and POST /documents/ask: an
# unknown or not-owned conversation returns 404.

class ContextPreviewRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: str | None = None
    conversation_id: str | None = Field(default=None, max_length=36)
    user_id: str | None = Field(default=None, max_length=100)

    # Session-ownership claim, mirroring ChatRequest and RAGRequest:
    # when a conversation_id is supplied, the same access rules as
    # POST /chat and POST /documents/ask apply to it.
    session_id: str | None = Field(default=None, max_length=64)


@app.post("/context/preview")
def preview_context(
    request: ContextPreviewRequest,
    db=Depends(get_db_optional),
):
    """Report the context that would be assembled for a question.

    Read-only with respect to chat: no answer is generated. When a
    conversation is supplied, it must exist and the caller must own it
    under the same session-claim rules as POST /chat and
    POST /documents/ask.
    """

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    # Resolve the selected conversation with the same access checks as
    # the conversational endpoints. Unknown and not-owned conversations
    # both return 404, so a conversation id cannot be probed.
    conversation = None

    requested_conversation_id = _normalize_conversation_id(
        request.conversation_id
    )

    if requested_conversation_id is not None:
        conversation = _load_accessible_conversation(
            db,
            requested_conversation_id,
            request.session_id,
        )

    started_at = time.perf_counter()

    try:
        context = assemble_context(
            query=query,
            top_k=request.top_k,
            document_id=request.document_id,
            conversation_id=(
                conversation.id if conversation is not None else None
            ),
            user_id=request.user_id,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception:
        logger.exception("Context assembly failed")
        raise HTTPException(
            status_code=500,
            detail="Context assembly failed unexpectedly.",
        ) from None

    # Count retrieved image units that resolve to a stored file. Only
    # the count is returned: local file paths stay on the server.
    image_references = 0

    for result in context.get("rag", []):
        if result.get("content_type") != "image":
            continue

        if resolve_image_path(result):
            image_references += 1

    assembly = build_context_report(context)
    diagnostics = build_context_diagnostics(
        context,
        conversation_requested=conversation is not None,
        memory_requested=bool(request.user_id),
        duration_ms=int((time.perf_counter() - started_at) * 1000),
        assembly=assembly,
    )

    return {
        "query": query,
        "sources": {
            "conversation": context.get("conversation"),
            "memory": context.get("memory", []),
            "rag": context.get("rag", []),
            "knowledge_graph": context.get("knowledge_graph", []),
        },
        "source_errors": context.get("source_errors", []),
        # Image units that would be attached to a vision request.
        "image_references": image_references,
        # Prompt state in effect for this question (the selected
        # conversation's custom prompt, if any), so the Context
        # Inspector can show which trusted instructions were composed
        # into the prompt.
        "system_prompt": {
            "using_default": not (
                (conversation.system_prompt or "").strip()
                if conversation is not None
                else ""
            ),
            "default_prompt_id": "default_system_prompt",
        },
        "model": describe_active_provider(),
        # The assembled context text (before the built-in security
        # instructions are prepended), i.e. exactly the evidence an
        # answer would be based on.
        "formatted_context": assembly["formatted_context"],
        "diagnostics": diagnostics,
    }



# Define the input expected by the masking endpoint.
class MaskingTextRequest(BaseModel):
    # Limit input size to avoid excessively large requests.
    text: str = Field(min_length=1, max_length=50000)


# Expose a dedicated endpoint for sensitive text masking.
@app.post("/masking/text")
def mask_text_endpoint(request: MaskingTextRequest):
    # Reject whitespace-only input.
    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty.",
        )

    try:
        # Run detection, masking, and secondary validation.
        pipeline_output = mask_and_validate(
            text=request.text,
            document_id="text-input",
        )

        # Retrieve the internal result and safety decision.
        result = pipeline_output["result"]
        safety = pipeline_output["safety"]

        # Return masked content and metadata only.
        # Never return the original text or raw entity values.
        # If the secondary validation did not pass, the candidate output
        # is withheld: it may still contain a value the detectors missed,
        # so no client can mistake it for approved text.
        safety_passed = safety["passed"] is True

        return {
            "masked_text": result.masked_text if safety_passed else None,
            "safety_passed": safety_passed,
            "detected_count": len(result.entities),
            "detected_categories": sorted({
                entity.entity_type
                for entity in result.entities
            }),
            "safety_reasons": safety["reasons"],
        }

    except ValueError as error:
        # Convert invalid input into a client error.
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        # Fail closed: do not return original text on failure.
        raise HTTPException(
            status_code=500,
            detail="Masking failed. No result was approved.",
        ) from error


# --------------------------------------------------
# Database session dependency
# --------------------------------------------------

def get_db():
    """Yield a database session, or fail with a clear service error.

    A request that raises mid-transaction is rolled back before the
    session is returned to the pool, so a failed write cannot leak an
    open transaction into the next request.
    """

    # Imported here so a missing DATABASE_URL only affects the routes
    # that actually need storage.
    from app.context_engine.database import SessionLocal

    try:
        db = SessionLocal()

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    try:
        yield db

    except Exception:
        try:
            db.rollback()
        except SQLAlchemyError:
            logger.warning(
                "Database rollback failed after a request error.",
                exc_info=True,
            )
        raise

    finally:
        db.close()


def _database_error() -> HTTPException:
    """Convert a storage failure into a clear service error.

    The underlying driver message is deliberately not returned: it can
    contain connection details and is not useful to the client.
    """

    return HTTPException(
        status_code=503,
        detail="The conversation database is unavailable.",
    )


# --------------------------------------------------
# Conversation-specific system prompts
# --------------------------------------------------
# Every conversation carries its own optional custom system prompt,
# persisted in PostgreSQL (conversations.system_prompt). There is no
# global active prompt: selecting another conversation loads its own
# prompt, and resetting one conversation never affects another. The
# built-in security instructions are not editable and are always
# composed on top of the custom prompt by build_instructions, so a
# custom prompt can never disable them.

class ConversationSystemPromptUpdateRequest(BaseModel):
    # Empty string resets the conversation to the default instructions.
    custom_prompt: str = Field(
        default="",
        max_length=MAX_CUSTOM_PROMPT_LENGTH,
    )


@app.get("/conversations/{conversation_id}/system-prompt")
def read_conversation_system_prompt(
    conversation_id: str,
    session_id: str | None = Query(default=None, max_length=64),
    db=Depends(get_db),
):
    """Report one conversation's custom system prompt (and status)."""

    conversation = _load_accessible_conversation(
        db,
        conversation_id,
        session_id,
    )

    try:
        prompt = get_conversation_system_prompt(
            db,
            conversation.id,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except SQLAlchemyError as error:
        raise _database_error() from error

    return {
        **prompt,
        "security_instructions": SECURITY_INSTRUCTIONS,
        "max_custom_prompt_length": MAX_CUSTOM_PROMPT_LENGTH,
    }


@app.put("/conversations/{conversation_id}/system-prompt")
def update_conversation_system_prompt(
    conversation_id: str,
    request: ConversationSystemPromptUpdateRequest,
    session_id: str | None = Query(default=None, max_length=64),
    db=Depends(get_db),
):
    """Persist one conversation's custom system prompt.

    An empty prompt resets the conversation to the default system
    instructions. Only this conversation is affected.
    """

    conversation = _load_accessible_conversation(
        db,
        conversation_id,
        session_id,
    )

    try:
        saved = save_conversation_system_prompt(
            db=db,
            conversation_id=conversation.id,
            custom_prompt=request.custom_prompt,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except SQLAlchemyError as error:
        raise _database_error() from error

    return {
        **saved,
        "security_instructions": SECURITY_INSTRUCTIONS,
        "max_custom_prompt_length": MAX_CUSTOM_PROMPT_LENGTH,
    }


@app.delete("/conversations/{conversation_id}/system-prompt")
def delete_conversation_system_prompt(
    conversation_id: str,
    session_id: str | None = Query(default=None, max_length=64),
    db=Depends(get_db),
):
    """Reset one conversation to the default system instructions.

    The custom prompt is deleted for this conversation only; its
    messages and summary are untouched.
    """

    conversation = _load_accessible_conversation(
        db,
        conversation_id,
        session_id,
    )

    try:
        reset = reset_conversation_system_prompt(
            db=db,
            conversation_id=conversation.id,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except SQLAlchemyError as error:
        raise _database_error() from error

    return {
        **reset,
        "security_instructions": SECURITY_INSTRUCTIONS,
        "max_custom_prompt_length": MAX_CUSTOM_PROMPT_LENGTH,
    }


# --------------------------------------------------
# Indexed documents
# --------------------------------------------------

@app.get("/documents")
def read_documents():
    """List indexed documents with their storage status."""

    try:
        listing = list_indexed_documents()

    except (RuntimeError, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except Exception:
        logger.exception("Document listing failed")
        raise HTTPException(
            status_code=503,
            detail="Document listing failed unexpectedly.",
        ) from None

    documents = []

    for document in listing["documents"]:
        try:
            storage = document_storage_status(
                document["document_id"]
            )

        except ValueError:
            # An id stored in the index that this application could not
            # have generated is reported without local file details.
            storage = {
                "pdf_stored": False,
                "images_stored": False,
                "masked_text_stored": False,
            }

        documents.append({**document, **storage})

    return {
        "documents": documents,
        "scanned_vectors": listing["scanned_vectors"],
        "scan_truncated": listing["scan_truncated"],
    }


@app.get("/documents/{document_id}")
def read_document(document_id: str):
    """Describe one indexed document's content units.

    Unit metadata is returned without the stored text, because indexed
    document content has not been masked.
    """

    try:
        validate_document_id(document_id)

        details = get_document_content_units(document_id)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except Exception:
        logger.exception("Document lookup failed for %s", document_id)
        raise HTTPException(
            status_code=503,
            detail="Document lookup failed unexpectedly.",
        ) from None

    if details["content_units"] == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id!r} was not found.",
        )

    try:
        storage = document_storage_status(document_id)

    except ValueError:
        storage = {
            "pdf_stored": False,
            "images_stored": False,
            "masked_text_stored": False,
        }

    return {**details, **storage}


@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    """Delete one document's vectors and its locally stored files.

    The index is cleaned first: if vector deletion fails the local files
    are kept, so the document remains discoverable instead of becoming
    an orphaned index entry.
    """

    try:
        validate_document_id(document_id)

        index_result = delete_document_from_index(document_id)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except Exception:
        logger.exception("Document deletion failed for %s", document_id)
        raise HTTPException(
            status_code=503,
            detail="Document deletion failed unexpectedly.",
        ) from None

    if not index_result["index_delete_requested"]:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id!r} was not found.",
        )

    try:
        removed_files = delete_stored_document_files(document_id)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    return {
        "document_id": document_id,
        "index_delete_requested": True,
        "matched_content_units_on_first_page": index_result[
            "matched_content_units_on_first_page"
        ],
        "removed_files": removed_files,
    }


# --------------------------------------------------
# Conversations
# --------------------------------------------------

class ConversationCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)

    # Optional session-ownership claim (trusted, like the memory API's
    # user_id). Empty/absent keeps the conversation shared.
    session_id: str | None = Field(default=None, max_length=64)


@app.get("/conversations")
def read_conversations(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session_id: str | None = Query(default=None, max_length=64),
    db=Depends(get_db),
):
    """List conversations, newest first, with message counts.

    With ``session_id`` the listing is scoped to that session (shared
    conversations plus its own). Without it, every conversation is
    listed, preserving the pre-existing contract.
    """

    try:
        conversations = list_conversations(
            db=db,
            limit=limit,
            offset=offset,
            session_id=session_id,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except SQLAlchemyError as error:
        raise _database_error() from error

    return {"conversations": conversations}


@app.post("/conversations", status_code=201)
def create_new_conversation(
    request: ConversationCreateRequest,
    db=Depends(get_db),
):
    """Create a conversation and return its identifier."""

    title = request.title.strip() if request.title else None

    try:
        conversation = create_conversation(
            db,
            title=title or None,
            session_id=request.session_id,
        )

    except SQLAlchemyError as error:
        logger.exception("Conversation creation failed")
        raise _database_error() from error

    return {
        "id": conversation.id,
        "title": conversation.title,
        "session_id": conversation.session_id,
        "created_at": (
            conversation.created_at.isoformat()
            if conversation.created_at
            else None
        ),
    }


@app.get("/conversations/{conversation_id}/messages")
def read_conversation_messages(
    conversation_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    # "recent" (default) keeps the pre-existing contract: the newest N
    # messages, newest first. "chronological" returns the earliest
    # messages first, the natural reading order of a transcript.
    order: str = Query(default="recent", pattern="^(recent|chronological)$"),
    db=Depends(get_db),
):
    """Return one conversation's messages."""

    try:
        conversation = get_conversation(db, conversation_id)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except SQLAlchemyError as error:
        raise _database_error() from error

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation {conversation_id!r} was not found.",
        )

    try:
        if order == "chronological":
            messages = list_conversation_messages(
                db=db,
                conversation_id=conversation_id,
                limit=limit,
                chronological=True,
            )
        else:
            messages = get_last_n_messages(
                db=db,
                conversation_id=conversation_id,
                n=limit,
            )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except SQLAlchemyError as error:
        raise _database_error() from error

    return {
        "conversation_id": conversation_id,
        "title": conversation.title,
        "order": order,
        "messages": messages,
    }


@app.delete("/conversations/{conversation_id}")
def remove_conversation(
    conversation_id: str,
    session_id: str | None = Query(default=None, max_length=64),
    db=Depends(get_db),
):
    """Delete one conversation with its messages and stored summary.

    The stored summary and every message are removed in the same
    transaction as the conversation, so no orphaned rows remain. An
    unknown or not-owned conversation returns 404 without revealing
    whether it exists.
    """

    conversation = _load_accessible_conversation(
        db,
        conversation_id,
        session_id,
    )

    try:
        result = delete_conversation(db, conversation.id)

    except SQLAlchemyError as error:
        raise _database_error() from error

    return {
        "deleted": result["deleted"],
        "conversation_id": conversation.id,
        "messages_deleted": result["messages_deleted"],
    }


@app.delete("/conversations/{conversation_id}/messages")
def remove_conversation_messages(
    conversation_id: str,
    session_id: str | None = Query(default=None, max_length=64),
    db=Depends(get_db),
):
    """Clear one conversation's messages and summary, keeping it.

    Used to implement a safe "reset conversation" that preserves the
    conversation identifier.
    """

    conversation = _load_accessible_conversation(
        db,
        conversation_id,
        session_id,
    )

    try:
        result = clear_conversation(db, conversation.id)

    except SQLAlchemyError as error:
        raise _database_error() from error

    return {
        "cleared": result["cleared"],
        "conversation_id": conversation.id,
        "messages_deleted": result["messages_deleted"],
        "summaries_deleted": result["summaries_deleted"],
    }


# --------------------------------------------------
# Long-term memory
# --------------------------------------------------

class MemoryCreateRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)
    memory_type: str = Field(min_length=1, max_length=20)
    content: str = Field(min_length=1, max_length=10000)
    importance: int = Field(default=3, ge=1, le=5)
    conversation_id: str | None = Field(
        default=None,
        max_length=36,
    )


@app.get("/memory")
def read_memories(
    user_id: str = Query(min_length=1, max_length=100),
    limit: int = Query(default=100, ge=1, le=500),
    db=Depends(get_db),
):
    """List one user's stored memories."""

    try:
        memories = list_memories(
            db=db,
            user_id=user_id,
            limit=limit,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except SQLAlchemyError as error:
        raise _database_error() from error

    return {"user_id": user_id, "memories": memories}


@app.post("/memory", status_code=201)
def create_new_memory(
    request: MemoryCreateRequest,
    db=Depends(get_db),
):
    """Store a new memory using the existing memory service."""

    try:
        memory = create_memory(
            db=db,
            user_id=request.user_id,
            memory_type=request.memory_type,
            content=request.content,
            importance=request.importance,
            conversation_id=request.conversation_id,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except SQLAlchemyError as error:
        raise _database_error() from error

    return {
        "id": memory.id,
        "user_id": memory.user_id,
        "memory_type": memory.memory_type,
        "content": memory.content,
        "importance": memory.importance,
        "conversation_id": memory.conversation_id,
        "created_at": (
            memory.created_at.isoformat()
            if memory.created_at
            else None
        ),
    }


@app.delete("/memory/{memory_id}")
def delete_memory_endpoint(
    memory_id: int,
    user_id: str = Query(min_length=1, max_length=100),
    db=Depends(get_db),
):
    """Delete one memory record belonging to a user."""

    try:
        deleted = delete_memory(
            db=db,
            user_id=user_id,
            memory_id=memory_id,
        )

    except SQLAlchemyError as error:
        raise _database_error() from error

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Memory {memory_id} was not found for "
                "this user."
            ),
        )

    try:
        remaining_memory_count = count_memories(db, user_id)
    except SQLAlchemyError as error:
        raise _database_error() from error

    return {
        "deleted": True,
        "memory_id": memory_id,
        "dry_run": False,
        "other_memories_preserved": True,
        "remaining_memory_count": remaining_memory_count,
    }


# --------------------------------------------------
# Knowledge graph
# --------------------------------------------------


@app.get("/knowledge-graph/status")
def read_knowledge_graph_status():
    """Report graph reachability and aggregate counts without exposing config."""

    if not graph_service.is_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "Neo4j is not configured. Set NEO4J_URI and "
                "NEO4J_PASSWORD in the root .env file."
            ),
        )

    try:
        verify_graph_connection()
        counts = graph_counts()
    except RuntimeError as error:
        detail = str(error)
        if "NEO4J_URI" not in detail:
            detail = "Knowledge graph is unreachable."
        raise HTTPException(status_code=503, detail=detail) from error
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Knowledge graph is unreachable.",
        ) from None

    return {"available": True, "counts": counts}


@app.get("/knowledge-graph/entities")
def read_knowledge_graph_entities(
    search: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    entity_type: str | None = Query(default=None, max_length=40),
):
    """Read graph entities with optional search, type, and paging filters."""

    clean_search = search.strip() if search else ""
    clean_entity_type = entity_type.strip() if entity_type else None

    try:
        if clean_search:
            if offset:
                entities = find_entities(
                    clean_search,
                    limit=limit,
                    entity_type=clean_entity_type,
                    offset=offset,
                )
            else:
                entities = find_entities(
                    clean_search,
                    limit=limit,
                    entity_type=clean_entity_type,
                )
        else:
            entities = list_entities(
                limit=limit,
                offset=offset,
                entity_type=clean_entity_type,
            )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        # Neo4j is not configured or unreachable.
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except Exception:
        logger.exception("Knowledge graph entity lookup failed")
        raise HTTPException(
            status_code=503,
            detail="Knowledge graph lookup failed unexpectedly.",
        ) from None

    return {
        "entities": entities,
        "search": clean_search or None,
    }


@app.get("/knowledge-graph/entities/{entity_id}/neighborhood")
def read_entity_neighborhood(
    entity_id: str,
    limit: int = Query(default=25, ge=1, le=100),
):
    """Read an entity and its bounded incoming/outgoing graph edges."""

    if not entity_id.strip():
        raise HTTPException(
            status_code=400,
            detail="entity_id cannot be empty.",
        )

    try:
        neighborhood = get_entity_neighborhood(entity_id, limit=limit)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error
    except Exception:
        logger.exception("Knowledge graph neighborhood lookup failed")
        raise HTTPException(
            status_code=503,
            detail="Knowledge graph neighborhood lookup failed.",
        ) from None

    return {
        "entity_id": entity_id,
        "entity": neighborhood["entity"],
        "edges": neighborhood["edges"],
    }


@app.get("/knowledge-graph/entities/{entity_id}/documents")
def read_entity_documents(
    entity_id: str,
    limit: int = Query(default=50, ge=1, le=100),
):
    """Read documents linked to an entity through MENTIONS edges."""

    if not entity_id.strip():
        raise HTTPException(
            status_code=400,
            detail="entity_id cannot be empty.",
        )

    try:
        documents = find_entity_documents(entity_id, limit=limit)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error
    except Exception:
        logger.exception("Knowledge graph entity-document lookup failed")
        raise HTTPException(
            status_code=503,
            detail="Knowledge graph entity-document lookup failed.",
        ) from None

    return {"entity_id": entity_id, "documents": documents}


@app.get(
    "/knowledge-graph/entities/{entity_id}/relationships"
)
def read_entity_relationships(entity_id: str):
    """Return one entity's incoming and outgoing relationships.

    An empty list is returned both for an entity without relationships
    and for an unknown entity id, because the existing graph service
    reports both the same way.
    """

    if not entity_id.strip():
        raise HTTPException(
            status_code=400,
            detail="entity_id cannot be empty.",
        )

    try:
        relationships = get_entity_relationships(entity_id)

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except Exception:
        logger.exception("Knowledge graph relationship lookup failed")
        raise HTTPException(
            status_code=503,
            detail=(
                "Knowledge graph relationship lookup failed "
                "unexpectedly."
            ),
        ) from None

    return {
        "entity_id": entity_id,
        "relationships": relationships,
    }


@app.get("/knowledge-graph/documents")
def read_graph_documents(
    limit: int = Query(default=25, ge=1, le=100),
):
    """List documents that were ingested into the graph."""

    try:
        documents = list_documents(limit=limit)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except Exception:
        logger.exception("Knowledge graph document listing failed")
        raise HTTPException(
            status_code=503,
            detail=(
                "Knowledge graph document listing failed unexpectedly."
            ),
        ) from None

    return {"documents": documents}


# --------------------------------------------------
# Knowledge-graph legacy cleanup (safe by default)
# --------------------------------------------------
# The GET endpoint is strictly read-only: it produces the dry-run
# report without modifying Neo4j. The POST endpoint defaults to the
# same dry run; destructive changes require an explicit
# {"confirm": true} body and only ever touch application-owned records.


class GraphCleanupRequest(BaseModel):
    confirm: bool = False


@app.get("/knowledge-graph/cleanup-report")
def read_graph_cleanup_report():
    """Read-only cleanup report (dry run; nothing is modified)."""

    if not graph_service.is_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "Neo4j is not configured. Set NEO4J_URI and "
                "NEO4J_PASSWORD in the root .env file."
            ),
        )

    try:
        snapshot = graph_cleanup.fetch_snapshot()

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except Exception:
        logger.exception("Graph cleanup report failed")
        raise HTTPException(
            status_code=503,
            detail="The cleanup report could not be generated.",
        ) from None

    return graph_cleanup.build_report(snapshot)


@app.post("/knowledge-graph/cleanup")
def run_graph_cleanup(request: GraphCleanupRequest):
    """Run the legacy cleanup.

    Default (confirm=false) is a dry run that returns the report and
    the planned actions without modifying the graph. With
    confirm=true the plan is executed against application-owned
    records only, and the remaining-actions counts prove idempotency.
    """

    if not graph_service.is_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "Neo4j is not configured. Set NEO4J_URI and "
                "NEO4J_PASSWORD in the root .env file."
            ),
        )

    try:
        result = graph_cleanup.run_cleanup(confirm=request.confirm)

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except Exception:
        logger.exception("Graph cleanup failed")
        raise HTTPException(
            status_code=503,
            detail="The cleanup could not be executed.",
        ) from None

    if request.confirm:
        logger.warning(
            "Graph cleanup executed (destructive): "
            "merges=%d removals=%d errors=%d",
            len(result["applied"]["merges"]),
            len(result["applied"]["removals"]),
            len(result["applied"]["errors"]),
        )

    return result


@app.get("/knowledge-graph/documents/{document_id}/graph")
def read_graph_document_graph(document_id: str):
    """Return one document's entities and the relationships between them."""

    try:
        graph = find_document_graph(document_id)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except Exception:
        logger.exception("Knowledge graph document graph lookup failed")
        raise HTTPException(
            status_code=503,
            detail=(
                "Knowledge graph document graph lookup failed "
                "unexpectedly."
            ),
        ) from None

    return graph


@app.get("/knowledge-graph/documents/{document_id}/entities")
def read_graph_document_entities(document_id: str):
    """List the graph entities a single document mentions."""

    try:
        entities = find_document_entities(document_id)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except Exception:
        logger.exception("Knowledge graph document entities lookup failed")
        raise HTTPException(
            status_code=503,
            detail=(
                "Knowledge graph document entities lookup failed "
                "unexpectedly."
            ),
        ) from None

    return {
        "document_id": document_id,
        "entities": entities,
    }
