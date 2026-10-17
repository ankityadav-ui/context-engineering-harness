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
    api_url: str | None = None,
) -> dict:
    """Send one chat message (JSON response)."""

    response = requests.post(
        f"{_base_url(api_url)}/chat",
        json={"message": message},
        timeout=120,
    )

    return _handle(response)


def chat_stream(
    message: str,
    api_url: str | None = None,
) -> Iterator[dict]:
    """Stream a chat reply as parsed server-sent events.

    Yields {"type": "chunk", "delta": ...}, {"type": "done", ...} and
    {"type": "error", ...} frames exactly as the backend emits them.
    """

    stream = requests.post(
        f"{_base_url(api_url)}/chat/stream",
        json={"message": message},
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

def upload_document(
    file_path: str,
    filename: str,
    mask_content: bool,
    api_url: str | None = None,
) -> dict:
    with open(file_path, "rb") as handle:
        response = requests.post(
            f"{_base_url(api_url)}/documents/upload",
            files={"file": (filename, handle, "application/pdf")},
            data={"mask_content": "true" if mask_content else "false"},
            timeout=900,
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
    api_url: str | None = None,
) -> dict:
    response = requests.post(
        f"{_base_url(api_url)}/documents/ask",
        json={
            "query": query,
            "top_k": top_k,
            "document_id": document_id or None,
            "conversation_id": conversation_id or None,
            "user_id": user_id or None,
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
    api_url: str | None = None,
) -> dict:
    response = requests.post(
        f"{_base_url(api_url)}/context/preview",
        json={
            "query": query,
            "top_k": top_k,
            "document_id": document_id or None,
            "conversation_id": conversation_id or None,
            "user_id": user_id or None,
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
# Knowledge graph
# --------------------------------------------------

def graph_entities(
    search: str | None = None,
    limit: int = 25,
    api_url: str | None = None,
) -> dict:
    response = requests.get(
        f"{_base_url(api_url)}/knowledge-graph/entities",
        params={"search": search or None, "limit": limit},
        timeout=60,
    )
    return _handle(response)


def entity_relationships(entity_id: str, api_url: str | None = None) -> dict:
    response = requests.get(
        f"{_base_url(api_url)}/knowledge-graph/entities/{entity_id}/relationships",
        timeout=60,
    )
    return _handle(response)


def graph_documents(api_url: str | None = None) -> dict:
    response = requests.get(
        f"{_base_url(api_url)}/knowledge-graph/documents", timeout=60
    )
    return _handle(response)


def graph_document_entities(document_id: str, api_url: str | None = None) -> dict:
    response = requests.get(
        f"{_base_url(api_url)}/knowledge-graph/documents/{document_id}/entities",
        timeout=60,
    )
    return _handle(response)


# --------------------------------------------------
# System prompt and skills
# --------------------------------------------------

def get_system_prompt(api_url: str | None = None) -> dict:
    response = requests.get(
        f"{_base_url(api_url)}/settings/system-prompt", timeout=30
    )
    return _handle(response)


def save_system_prompt(custom_prompt: str, api_url: str | None = None) -> dict:
    response = requests.put(
        f"{_base_url(api_url)}/settings/system-prompt",
        json={"custom_prompt": custom_prompt},
        timeout=30,
    )
    return _handle(response)


def list_skills(api_url: str | None = None) -> dict:
    response = requests.get(
        f"{_base_url(api_url)}/settings/skills", timeout=30
    )
    return _handle(response)


def select_skill(skill_id: str, api_url: str | None = None) -> dict:
    response = requests.post(
        f"{_base_url(api_url)}/settings/skills/select",
        json={"skill_id": skill_id},
        timeout=30,
    )
    return _handle(response)
