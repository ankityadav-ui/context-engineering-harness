"""Thin HTTP client for the FastAPI backend.

Streamlit never talks to PostgreSQL, Pinecone, Neo4j, or any LLM
directly: every feature goes through this client, so the backend stays
the single authority for chat, RAG, memory, graph, and masking logic.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator, Optional

import requests
from dotenv import load_dotenv

# Load the root .env once so BACKEND_API_URL can be configured there.
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DEFAULT_BACKEND_URL = os.environ.get(
    "BACKEND_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")


class APIError(Exception):
    """Raised when the backend returns an error response."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _base_url(api_url: str | None) -> str:
    url = api_url or DEFAULT_BACKEND_URL
    return url.rstrip("/")


def _handle(response: requests.Response) -> dict[str, Any]:
    """Convert a non-2xx backend response into a clear error."""

    if response.ok:
        return response.json()

    try:
        payload = response.json()
        message = payload.get("detail", str(payload))

    except ValueError:
        message = (
            f"Backend error {response.status_code}: "
            f"{response.text[:200]}"
        )

    raise APIError(str(message), status_code=response.status_code)


# --------------------------------------------------


def delete_conversation(conversation_id: str, session_id: str | None = None, api_url: str | None = None) -> dict:
    """Delete one conversation with its messages and stored summary.

    The conversation, its messages, and its conversation-specific summary
    are removed in the same transaction. An unknown or not-owned
    conversation returns 404 without revealing whether it exists.
    """
    url = f"{_base_url(api_url)}/conversations/{conversation_id}"
    params: dict[str, str] = {}
    if session_id:
        params["session_id"] = session_id
    response = requests.delete(url, params=params, timeout=30)
    return _handle(response)

# Health, capabilities, and model info
# --------------------------------------------------

def health(api_url: str | None = None) -> dict:
    response = requests.get(
        f"{_base_url(api_url)}/health", timeout=5
    )
    return _handle(response)


def capabilities(api_url: str | None = None) -> dict:
    response = requests.get(
        f"{_base_url(api_url)}/capabilities", timeout=10
    )
    return _handle(response)


def model_info(api_url: str | None = None) -> dict:
    """Active model description from GET /capabilities."""

    response = capabilities(api_url)

    for service in response.get("services", []):
        if service.get("key") == "llm":
            return {
                "model": service.get("model", "unknown"),
                "configured": service.get("configured", False),
            }

    return {"model": "unknown", "configured": False}

# --------------------------------------------------
# Chat
# --------------------------------------------------

def chat(
    message: str,
    document_id: str | None = None,
    api_url: str | None = None,
) -> dict:
    """Send one chat message (JSON response).

    ``document_id`` scopes retrieval and the full-text fallback to one
    indexed document, so "summarise this" resolves the uploaded file.
    """

    payload: dict[str, Any] = {"message": message}

    if document_id:
        payload["document_id"] = document_id

    response = requests.post(
        f"{_base_url(api_url)}/chat",
        json=payload,
        timeout=120,
    )

    return _handle(response)


def chat_stream(
    message: str,
    document_id: str | None = None,
    api_url: str | None = None,
) -> Iterator[dict]:
    """Stream a chat reply as parsed server-sent events.

    Yields {"type": "chunk", "delta": ...}, {"type": "done", ...} and
    {"type": "error", ...} frames exactly as the backend emits them.
    ``document_id`` has the same document-scoping meaning as :func:`chat`.
    """

    payload: dict[str, Any] = {"message": message}

    if document_id:
        payload["document_id"] = document_id

    stream = requests.post(
        f"{_base_url(api_url)}/chat/stream",
        json=payload,
        stream=True,
        timeout=(5, 300),
    )

    stream.raise_for_status()

    event: Optional[str] = None
    data_lines: list[str] = []

    try:
        for raw_line in stream.iter_lines(decode_unicode=True):
            line = raw_line or ""

            if not line.strip():
                # Blank line ends one event frame.
                if event and data_lines:
                    parsed = _parse_event(event, data_lines)
                    if parsed is not None:
                        yield parsed
                    if parsed and parsed.get("type") in {"done", "error"}:
                        return
                event = None
                data_lines = []
                continue

            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())
    finally:
        stream.close()


def _parse_event(event: str, data_lines: list[str]) -> dict | None:
    raw = "".join(data_lines)

    try:
        payload = json.loads(raw)

    except ValueError:
        return {"type": "error", "detail": f"Unreadable event: {raw!r}"}

    payload["type"] = event
    return payload


# --------------------------------------------------
# Documents and RAG
# --------------------------------------------------

def upload_document_file(
    file_obj,
    filename: str,
    mask_content: bool,
    api_url: str | None = None,
    *,
    idempotency_key: str | None = None,
) -> dict:
    """Upload one PDF from an open file-like object.

    The backend accepts PDFs only (multipart form field ``file``) with
    an optional ``mask_content`` flag; processing, OCR, embedding, and
    indexing continue in a durable backend job after an HTTP 202 response.
    """

    response = requests.post(
        f"{_base_url(api_url)}/documents/upload",
        files={"file": (filename, file_obj, "application/pdf")},
        data={"mask_content": "true" if mask_content else "false"},
        headers=(
            {"Idempotency-Key": idempotency_key}
            if idempotency_key
            else {}
        ),
        timeout=60,
    )

    return _handle(response)


def upload_document(
    file_path: str,
    filename: str,
    mask_content: bool,
    api_url: str | None = None,
    *,
    idempotency_key: str | None = None,
) -> dict:
    with open(file_path, "rb") as handle:
        return upload_document_file(
            handle,
            filename,
            mask_content,
            api_url=api_url,
            idempotency_key=idempotency_key,
        )


def get_ingestion_job(
    job_id: str,
    api_url: str | None = None,
) -> dict:
    response = requests.get(
        f"{_base_url(api_url)}/documents/jobs/{job_id}",
        timeout=10,
    )
    return _handle(response)


def retry_ingestion_job(
    job_id: str,
    api_url: str | None = None,
) -> dict:
    response = requests.post(
        f"{_base_url(api_url)}/documents/jobs/{job_id}/retry",
        timeout=15,
    )
    return _handle(response)


def list_documents(api_url: str | None = None) -> dict:
    response = requests.get(
        f"{_base_url(api_url)}/documents", timeout=60
    )
    return _handle(response)


def get_document(document_id: str, api_url: str | None = None) -> dict:
    response = requests.get(
        f"{_base_url(api_url)}/documents/{document_id}", timeout=60
    )
    return _handle(response)


def delete_document(document_id: str, api_url: str | None = None) -> dict:
    response = requests.delete(
        f"{_base_url(api_url)}/documents/{document_id}", timeout=60
    )
    return _handle(response)


def search_documents(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
    api_url: str | None = None,
) -> dict:
    response = requests.post(
        f"{_base_url(api_url)}/documents/search",
        json={
            "query": query,
            "top_k": top_k,
            "document_id": document_id or None,
        },
        timeout=120,
    )
    return _handle(response)


def ask_question(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    api_url: str | None = None,
) -> dict:
    """Ask a question through the backend RAG pipeline.

    ``conversation_id`` (with its ``session_id`` ownership claim)
    targets one conversation's history and saved system prompt, using
    the same access rules as chat.
    """

    response = requests.post(
        f"{_base_url(api_url)}/documents/ask",
        json={
            "query": query,
            "top_k": top_k,
            "document_id": document_id or None,
            "conversation_id": conversation_id or None,
            "user_id": user_id or None,
            "session_id": session_id or None,
        },
        timeout=300,
    )
    return _handle(response)


# --------------------------------------------------
# Context preview (Context Inspector)
# --------------------------------------------------

def context_preview(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    api_url: str | None = None,
) -> dict:
    """Preview the context the backend would assemble for a question.

    This endpoint never generates a chat answer. ``conversation_id``
    uses the same session-ownership rules as ``/chat``: a conversation
    created with a session claim requires that same ``session_id``.
    """

    response = requests.post(
        f"{_base_url(api_url)}/context/preview",
        json={
            "query": query,
            "top_k": top_k,
            "document_id": document_id or None,
            "conversation_id": conversation_id or None,
            "user_id": user_id or None,
            "session_id": session_id or None,
        },
        timeout=300,
    )
    return _handle(response)


# --------------------------------------------------
# Masking
# --------------------------------------------------

def mask_text(text: str, api_url: str | None = None) -> dict:
    response = requests.post(
        f"{_base_url(api_url)}/masking/text",
        json={"text": text},
        timeout=300,
    )
    return _handle(response)


# --------------------------------------------------
# Conversations and memory
# --------------------------------------------------

def create_conversation(title: str | None = None, api_url: str | None = None) -> dict:
    response = requests.post(
        f"{_base_url(api_url)}/conversations",
        json={"title": title},
        timeout=30,
    )
    return _handle(response)


def list_conversations(api_url: str | None = None) -> dict:
    response = requests.get(
        f"{_base_url(api_url)}/conversations", timeout=30
    )
    return _handle(response)


def get_messages(
    conversation_id: str,
    api_url: str | None = None,
) -> dict:
    response = requests.get(
        f"{_base_url(api_url)}/conversations/{conversation_id}/messages",
        timeout=30,
    )
    return _handle(response)


def list_memories(user_id: str, api_url: str | None = None) -> dict:
    response = requests.get(
        f"{_base_url(api_url)}/memory",
        params={"user_id": user_id},
        timeout=30,
    )
    return _handle(response)


def create_memory(
    user_id: str,
    memory_type: str,
    content: str,
    importance: int = 3,
    api_url: str | None = None,
) -> dict:
    response = requests.post(
        f"{_base_url(api_url)}/memory",
        json={
            "user_id": user_id,
            "memory_type": memory_type,
            "content": content,
            "importance": importance,
        },
        timeout=30,
    )
    return _handle(response)


def delete_memory(memory_id: int, user_id: str, api_url: str | None = None) -> dict:
    response = requests.delete(
        f"{_base_url(api_url)}/memory/{memory_id}",
        params={"user_id": user_id},
        timeout=30,
    )
    return _handle(response)


# --------------------------------------------------
# Knowledge graph cleanup
# --------------------------------------------------
# The knowledge graph is an internal backend service (Neo4j): entity
# browsing was removed from the UI, and the only graph call the
# frontend still makes is the read-only cleanup report shown on the
# Diagnostics page.

def cleanup_report(api_url: str | None = None) -> dict:
    """Read-only cleanup report (dry run; nothing is modified).

    Calls the backend /knowledge-graph/cleanup-report endpoint which returns
    a report of orphaned/uncertain records without performing any deletions.
    Use the ``mode`` field to confirm the report is read-only.
    """
    response = requests.get(
        f"{_base_url(api_url)}/knowledge-graph/cleanup-report",
        timeout=60,
    )
    return _handle(response)


# --------------------------------------------------
# Conversation-specific system prompts
# --------------------------------------------------

def get_system_prompt(
    conversation_id: str,
    api_url: str | None = None,
) -> dict:
    """Fetch one conversation's saved system prompt."""

    response = requests.get(
        f"{_base_url(api_url)}/conversations/{conversation_id}"
        "/system-prompt",
        timeout=30,
    )
    return _handle(response)


def save_system_prompt(
    conversation_id: str,
    custom_prompt: str,
    api_url: str | None = None,
) -> dict:
    """Persist one conversation's custom system prompt."""

    response = requests.put(
        f"{_base_url(api_url)}/conversations/{conversation_id}"
        "/system-prompt",
        json={"custom_prompt": custom_prompt},
        timeout=30,
    )
    return _handle(response)


def reset_system_prompt(
    conversation_id: str,
    api_url: str | None = None,
) -> dict:
    """Reset one conversation to the default system instructions."""

    response = requests.delete(
        f"{_base_url(api_url)}/conversations/{conversation_id}"
        "/system-prompt",
        timeout=30,
    )
    return _handle(response)
