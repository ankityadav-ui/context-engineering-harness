
# Import FastAPI's request validation tools.
import json

from pydantic import BaseModel, Field

# Import the masking and safety pipeline.
from app.masking.safety_pipeline import mask_and_validate

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import os

from fastapi import (
    Depends,
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
)
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
from app.context_engine import graph_service
from app.rag import embedding_service, ocr, pinecone_service
from app.context_engine.engine import (
    assemble_context,
    format_context,
)
from app.context_engine.conversation_service import (
    create_conversation,
    get_conversation,
    get_last_n_messages,
    list_conversations,
)
from app.context_engine.graph_service import (
    find_document_entities,
    find_entities,
    get_entity_relationships,
    ingest_document_graph,
    list_documents,
    list_entities,
)
from app.context_engine.memory_service import (
    create_memory,
    delete_memory,
    list_memories,
)
from app.system_prompt import (
    MAX_CUSTOM_PROMPT_LENGTH,
    SECURITY_INSTRUCTIONS,
    build_instructions,
    get_active_skill,
    get_default_prompt_text,
    get_skill_text,
    get_system_prompt,
    list_skills,
    save_active_skill,
    save_system_prompt,
)


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
                "label": f"Language model (Ollama, {llm_active['backend']})",
                "configured": llm_is_configured(),
                "model": llm_active["model"],
                "required_env": (
                    ["OLLAMA_TEXT_MODEL", "Ollama_API_KEY"]
                    if llm_active["backend"] == "cloud"
                    else ["OLLAMA_TEXT_MODEL"]
                ),
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

@app.post("/chat")
def chat(request: ChatRequest):
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    # Trusted instructions (stored custom prompt, active skill, and the
    # built-in security rules) travel as a separate system message; the
    # user's message stays user content.
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
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error


# --------------------------------------------------
# Streaming chat (Server-Sent Events)
# --------------------------------------------------
# A separate endpoint keeps POST /chat's JSON contract untouched. The
# request body is identical, and the instructions are composed by the
# same build_instructions path as POST /chat, so streaming never
# bypasses the system-prompt composition or security instructions.
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
def chat_stream(request: ChatRequest):
    """Stream an assistant reply as server-sent events.

    Chunks are emitted as the provider generates them. Provider
    failures surface as a terminal `error` event (never leaking the
    prompt or credentials). When the client disconnects, Starlette
    closes this generator, which closes the provider's upstream
    stream and releases the connection.
    """

    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    # Same instruction-composition path as POST /chat: trusted
    # instructions are sent as the system message, and the user's
    # message is sent as user content.
    system_text, custom_prompt_applied = build_instructions()

    # Captured at request time so the completion event reports the
    # provider that actually served this stream.
    active = describe_active_provider()

    def event_stream():
        emitted: list[str] = []
        deltas = stream_response(prompt=message, system=system_text)

        try:
            for delta in deltas:
                emitted.append(delta)
                yield _sse_event("chunk", {"delta": delta})

            answer = "".join(emitted)

            yield _sse_event(
                "done",
                {
                    "response": answer,
                    "provider": active["provider"],
                    "model": active["model"],
                    "system_prompt_active": custom_prompt_applied,
                },
            )

        except LLMError as error:
            # The failure message comes from our own LLM layer: it
            # describes configuration/provider problems without the
            # prompt or any credentials.
            yield _sse_event(
                "error",
                {
                    "detail": str(error),
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
        # Cleanup is best-effort; the original failure is reported.
        pass


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    # Optional multipart field; when "true", extracted text (including
    # OCR output) is masked and re-validated BEFORE embedding/indexing.
    mask_content: str = Form(default="false"),
):
    if (
        not file.filename
        or Path(file.filename).suffix.lower() != ".pdf"
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file.",
        )

    apply_masking = _parse_mask_flag(mask_content)

    document_id = str(uuid.uuid4())
    document_path = DOCUMENT_DIR / f"{document_id}.pdf"

    try:
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

        document_path.write_bytes(file_bytes)

        # This function performs text extraction, OCR,
        # table/image extraction, embedding, and indexing.
        # When apply_masking is set, extracted text is masked and
        # re-validated before embedding; image extraction is skipped.
        (
            content_units,
            ocr_pages,
            table_units,
            image_units,
            masking_stats,
        ) = process_document(
            document_path=document_path,
            document_id=document_id,
            apply_masking=apply_masking,
        )

        if not content_units:
            raise HTTPException(
                status_code=422,
                detail="No retrievable content was extracted from this PDF.",
            )

        # Knowledge-graph ingestion is best-effort: a Neo4j outage must
        # not fail the upload, but it is reported honestly in the
        # response (and logged) instead of being silently ignored.
        graph_result: dict | None = None
        graph_error: str | None = None

        if graph_service.is_configured():
            try:
                text_for_graph = "\n".join(
                    unit.get("content", "")
                    for unit in content_units
                    if unit.get("content_type") in {"text", "table"}
                    and unit.get("content", "").strip()
                )

                if text_for_graph.strip():
                    graph_result = ingest_document_graph(
                        document_id=document_id,
                        title=file.filename,
                        content=text_for_graph,
                    )

            except Exception as error:
                graph_error = f"{type(error).__name__}: {error}"
                logger.warning(
                    "Graph ingestion failed for document %s: %s",
                    document_id,
                    graph_error,
                )
        else:
            graph_error = (
                "Neo4j is not configured; graph ingestion skipped."
            )

        # Masked documents are never kept as raw files on disk: only
        # the sanitized content units were embedded and indexed.
        if masking_stats["applied"] and document_path.exists():
            document_path.unlink()

        return {
            "message": (
                "Document processed successfully. Masked content was "
                "indexed and the raw PDF was not stored."
                if masking_stats["applied"]
                else "Document processed successfully."
            ),
            "document_id": document_id,
            "filename": file.filename,
            "content_units": len(content_units),
            "ocr_pages": ocr_pages,
            "tables_extracted": len(table_units),
            "images_extracted": len(image_units),
            "masking_applied": masking_stats["applied"],
            "masking_safety_passed": masking_stats["safety_passed"],
            "masked_entities": masking_stats["masked_entities"],
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

    except HTTPException:
        raise

    except MaskingRejectedError as error:
        # Fail closed: nothing was embedded or indexed. Remove every
        # artifact so no raw sensitive content remains on disk.
        _remove_upload_artifacts(document_id, document_path)

        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception:
        # Remove the uploaded PDF (and any extracted images) if
        # processing fails. The internal error is logged server-side and
        # never returned: it can contain parser details, file paths, or
        # connection information.
        _remove_upload_artifacts(document_id, document_path)

        logger.exception("Document processing failed for %s", document_id)

        raise HTTPException(
            status_code=500,
            detail=(
                "Document processing failed unexpectedly. "
                "No content was indexed."
            ),
        ) from None

    finally:
        await file.close()


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
def ask_document(request: RAGRequest):
    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        # Assemble all context through one authoritative path.
        context = assemble_context(
            query=query,
            top_k=request.top_k,
            document_id=request.document_id,
            conversation_id=request.conversation_id,
            user_id=request.user_id,
        )

        formatted_context = format_context(context)

        # Trusted instructions go to the system role; the assembled
        # evidence (retrieved chunks, memories, graph, conversation) is
        # untrusted and stays in the user content.
        system_text, _custom_prompt_applied = build_instructions()

        # Resolve image references and avoid sending duplicate image files
        # to the vision model. Keep the source records unchanged in the
        # response so the caller can inspect the original retrieval results.
        image_paths = []
        seen_image_paths = set()

        for result in context.get("rag", []):
            if result.get("content_type") != "image":
                continue

            image_path = resolve_image_path(result)
            if not image_path:
                continue

            # Normalize paths so equivalent references count as one image.
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

    except LLMError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

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

class ContextPreviewRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: str | None = None
    conversation_id: str | None = Field(default=None, max_length=36)
    user_id: str | None = Field(default=None, max_length=100)


@app.post("/context/preview")
def preview_context(request: ContextPreviewRequest):
    """Report the context that would be assembled for a question."""

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    try:
        context = assemble_context(
            query=query,
            top_k=request.top_k,
            document_id=request.document_id,
            conversation_id=request.conversation_id,
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
        # Prompt/skill identifiers in effect for this question, so the
        # Context Inspector can show which trusted instructions were
        # composed into the prompt.
        "system_prompt": {
            "using_default": not bool(
                get_system_prompt()["custom_prompt"].strip()
            ),
            "active_skill": get_active_skill(),
            "default_prompt_id": "default_system_prompt",
        },
        "model": describe_active_provider(),
        # The assembled context text (before the built-in security
        # instructions are prepended), i.e. exactly the evidence an
        # answer would be based on.
        "formatted_context": format_context(context),
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
        return {
            "masked_text": result.masked_text,
            "safety_passed": safety["passed"],
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
# System-prompt settings
# --------------------------------------------------
# The custom prompt is persisted server-side and prepended to every
# LLM request. The built-in security instructions are not editable and
# are always applied on top of the custom prompt.

class SystemPromptUpdateRequest(BaseModel):
    # Empty string resets to the default (no custom prompt).
    custom_prompt: str = Field(
        default="",
        max_length=MAX_CUSTOM_PROMPT_LENGTH,
    )


@app.get("/settings/system-prompt")
def read_system_prompt_settings():
    """Report the stored custom system prompt (and its status)."""

    return {
        **get_system_prompt(),
        "security_instructions": SECURITY_INSTRUCTIONS,
        "max_custom_prompt_length": MAX_CUSTOM_PROMPT_LENGTH,
    }


@app.put("/settings/system-prompt")
async def update_system_prompt_settings(
    request: SystemPromptUpdateRequest,
):
    """Persist the custom system prompt (empty string resets it)."""

    try:
        saved = save_system_prompt(request.custom_prompt)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except OSError as error:
        raise HTTPException(
            status_code=503,
            detail="The system prompt settings could not be stored.",
        ) from error

    return {
        **saved,
        "security_instructions": SECURITY_INSTRUCTIONS,
        "max_custom_prompt_length": MAX_CUSTOM_PROMPT_LENGTH,
    }


# --------------------------------------------------
# Skills (trusted instruction files)
# --------------------------------------------------

@app.get("/settings/skills")
def list_available_skills():
    """List trusted skill files and the currently active skill."""

    return {
        "skills": list_skills(),
        "active_skill": get_active_skill(),
        "default_prompt_id": "default_system_prompt",
    }


class SkillSelectionRequest(BaseModel):
    skill_id: str = Field(default="", max_length=100)


@app.post("/settings/skills/select")
def select_active_skill(request: SkillSelectionRequest):
    """Select (or clear) the active skill instruction file."""

    try:
        if request.skill_id.strip():
            text = get_skill_text(request.skill_id.strip())
        else:
            text = ""

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    try:
        saved = save_active_skill(request.skill_id.strip())

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except OSError as error:
        raise HTTPException(
            status_code=503,
            detail="The skill selection could not be stored.",
        ) from error

    return {
        **saved,
        "skill_preview_chars": len(text),
    }


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
        storage = {"pdf_stored": False, "images_stored": False}

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


@app.get("/conversations")
def read_conversations(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db=Depends(get_db),
):
    """List conversations, newest first, with message counts."""

    try:
        conversations = list_conversations(
            db=db,
            limit=limit,
            offset=offset,
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
        )

    except SQLAlchemyError as error:
        raise _database_error() from error

    return {
        "id": conversation.id,
        "title": conversation.title,
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
    db=Depends(get_db),
):
    """Return the most recent messages of one conversation."""

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
        "messages": messages,
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

    return {"deleted": True, "memory_id": memory_id}


# --------------------------------------------------
# Knowledge graph
# --------------------------------------------------

@app.get("/knowledge-graph/entities")
def read_knowledge_graph_entities(
    search: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=25, ge=1, le=100),
):
    """List graph entities, optionally filtered by name."""

    clean_search = search.strip() if search else ""

    try:
        if clean_search:
            entities = find_entities(clean_search)
        else:
            entities = list_entities(limit=limit)

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
