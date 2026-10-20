"""Diagnostics for the authoritative context assembly pipeline.

The page answers "what did this request actually gather?" from the
request-scoped ``diagnostics`` block returned by ``POST /context/preview``.
That block is produced by the same ``assemble_context`` call chat uses, so
opening the page or running it never issues a second retrieval pass and
never asks the model for an answer.

Opening the page performs only the pre-existing health/capabilities
checks. A preview is fetched only when the user presses *Run diagnostics*.
"""

from __future__ import annotations

from typing import Any

import requests

import api_client

# Source rows in the order they contribute to the assembled context.
SOURCE_ROWS = (
    ("short_term", "Short-term conversation"),
    ("conversation_summary", "Conversation summary"),
    ("memory", "Long-term memory"),
    ("rag", "Document retrieval (RAG)"),
    ("knowledge_graph", "Knowledge graph"),
)

SECTION_LABELS = {
    "question": "Current question",
    "conversation_summary": "Conversation summary",
    "recent_conversation": "Recent conversation",
    "memory": "Long-term memories",
    "knowledge_graph": "Knowledge-graph relationships",
    "document_evidence": "Document evidence",
}

# Every status is derived from the backend's per-source record; nothing is
# inferred from the mere presence of a retrieval list.
STATUS_LABELS = {
    "ok": "returned content",
    "empty": "completed with no results",
    "not_requested": "not requested",
    "error": "failed",
}

STATUS_ICONS = {
    "ok": "✅",
    "empty": "⚪",
    "not_requested": "➖",
    "error": "❌",
}

ERROR_SOURCE_LABELS = {
    "rag": "Document retrieval",
    "full_text": "Full-text fallback",
    "conversation": "Conversation history",
    "memory": "Long-term memory",
    "knowledge_graph": "Knowledge graph",
}


# --------------------------------------------------
# Fetch helpers (monkeypatched in tests)
# --------------------------------------------------

def fetch_preview(
    query: str,
    *,
    top_k: int = 5,
    document_id: str | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    api_url: str | None = None,
) -> dict[str, Any]:
    """Call the existing preview endpoint once and return its JSON."""

    preview = api_client.context_preview(
        query=query,
        top_k=top_k,
        document_id=document_id,
        conversation_id=conversation_id,
        user_id=user_id,
        session_id=session_id,
        api_url=api_url,
    )

    if not isinstance(preview, dict):
        raise ValueError("The backend returned an unreadable preview.")

    return preview


def fetch_health(api_url: str | None = None) -> dict[str, Any]:
    return api_client.health(api_url=api_url)


def fetch_capabilities(api_url: str | None = None) -> dict[str, Any]:
    return api_client.capabilities(api_url=api_url)


# --------------------------------------------------
# Pure helpers
# --------------------------------------------------

def diagnostics_of(preview: dict[str, Any]) -> dict[str, Any] | None:
    value = preview.get("diagnostics")

    return value if isinstance(value, dict) else None


def source_status(preview: dict[str, Any], source: str) -> str | None:
    """Return the backend-reported status for one source, if present."""

    diagnostics = diagnostics_of(preview)

    if diagnostics is None:
        return None

    sources = diagnostics.get("sources")

    if not isinstance(sources, dict):
        return None

    entry = sources.get(source)

    if not isinstance(entry, dict):
        return None

    status = str(entry.get("status") or "").strip()

    return status or None


def source_entry(preview: dict[str, Any], source: str) -> dict[str, Any]:
    diagnostics = diagnostics_of(preview) or {}
    sources = diagnostics.get("sources") or {}

    entry = sources.get(source) if isinstance(sources, dict) else None

    return entry if isinstance(entry, dict) else {}


def section_counts(preview: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics = diagnostics_of(preview) or {}
    assembly = diagnostics.get("assembly") or {}

    sections = assembly.get("sections") if isinstance(assembly, dict) else None

    if not isinstance(sections, list):
        return []

    return [
        {
            "name": str(item.get("name") or ""),
            "chars": int(item.get("chars") or 0),
        }
        for item in sections
        if isinstance(item, dict)
    ]


def _source_detail(
    preview: dict[str, Any], source: str, status: str | None
) -> str:
    """One metadata-only line explaining a source's outcome."""

    entry = source_entry(preview, source)
    sources = preview.get("sources") or {}
    value = sources.get(source) if isinstance(sources, dict) else None

    if status == "not_requested":
        explanations = {
            "short_term": "No conversation was selected, so no message window was read.",
            "conversation_summary": (
                "No conversation was selected, or it has no older messages to summarize."
            ),
            "memory": "No memory owner (user id) was supplied.",
        }
        return explanations.get(source, "This source was not requested.")

    if status == "error":
        return "This source failed; other sources still contributed."

    if source == "short_term":
        messages = entry.get("messages")
        if messages is None:
            messages = len((value or {}).get("recent_messages") or []) if isinstance(value, dict) else 0
        return f"{messages} recent message(s) in the window."

    if source == "conversation_summary":
        chars = entry.get("chars")
        older = entry.get("older_messages")
        if chars is None:
            summary = (value or {}).get("summary") if isinstance(value, dict) else ""
            chars = len(summary or "")
        parts = [f"{int(chars or 0)} summary character(s)"]
        if older is not None:
            parts.append(f"{int(older)} older message(s)")
        return " · ".join(parts) + "."

    if source == "memory":
        items = entry.get("items")
        if items is None:
            items = len(value or []) if isinstance(value, list) else 0
        return f"{items} relevant memor{'y' if items == 1 else 'ies'} retrieved."

    if source == "rag":
        chunks = entry.get("chunks")
        documents = entry.get("documents")
        if chunks is None:
            chunks = len(value or []) if isinstance(value, list) else 0
        if documents is None:
            documents = len(
                {
                    item.get("document_id")
                    for item in (value or [])
                    if isinstance(item, dict) and item.get("document_id")
                }
            )
        detail = f"{chunks} chunk(s) from {documents} document(s)."

        if status == "empty":
            detail = (
                "No indexed document content matched this query. "
                "Chat still answers from the other sources."
            )

        fallback = entry.get("fallback_source")

        if fallback:
            detail += (
                " Served by the full-text fallback, not vector retrieval."
            )

        if entry.get("fallback_failed"):
            detail += " The full-text fallback also failed."

        return detail

    if source == "knowledge_graph":
        relationships = entry.get("relationships")
        entities = entry.get("entities")
        if relationships is None:
            relationships = len(value or []) if isinstance(value, list) else 0
        detail = f"{relationships} relationship(s) from {int(entities or 0)} entit(y/ies)."
        if status == "empty":
            detail += " No entities in the graph matched this question."
        return detail

    return ""


def _runs_without_evidence(preview: dict[str, Any]) -> bool:
    assembly = (diagnostics_of(preview) or {}).get("assembly") or {}

    return bool(isinstance(assembly, dict) and assembly.get("empty"))


# --------------------------------------------------
# Rendering
# --------------------------------------------------

def _render_health(st, api_url: str, health, capabilities) -> None:
    """Preserve the original health/capabilities section, harmlessly."""

    st.subheader("Service health")

    try:
        st.json(health(api_url))
    except Exception as exc:  # health probes must never break the page
        st.error(f"Could not load API health: {exc}")

    try:
        data = capabilities(api_url)
    except Exception as exc:
        st.warning(f"Could not load backend capabilities: {exc}")
        return

    services = data.get("services", []) if isinstance(data, dict) else []

    if not services:
        st.json(data)
        return

    for service in services:
        if not isinstance(service, dict):
            continue

        label = service.get("label", service.get("key", "Service"))

        if service.get("configured", False):
            st.success(f"{label}: configured")
        else:
            st.warning(f"{label}: not configured")


def _render_overview(
    st,
    preview: dict[str, Any],
    *,
    conversation_title: str | None,
    conversation_requested: bool,
) -> None:
    diagnostics = diagnostics_of(preview)
    duration = diagnostics.get("duration_ms") if diagnostics else None

    st.subheader("Request overview")

    if conversation_requested:
        label = (conversation_title or "").strip() or "selected conversation"
        st.caption(f"Conversation: {label}")
    else:
        st.caption("Conversation: none selected")

    st.caption("Request status: succeeded")
    st.caption(
        "Context assembly: "
        + (
            "succeeded with evidence"
            if not _runs_without_evidence(preview)
            else "succeeded without evidence (question only)"
        )
    )

    if isinstance(duration, int):
        st.caption(f"Assembly duration: {duration} ms")
    else:
        st.caption(
            "Assembly duration: not reported by this backend version."
        )


def _render_sources(st, preview: dict[str, Any]) -> None:
    st.subheader("Context source status")

    diagnostics = diagnostics_of(preview)

    if diagnostics is None:
        st.warning(
            "This backend did not return structured diagnostics. Source "
            "statuses below are derived from the returned preview."
        )

    for key, label in SOURCE_ROWS:
        status = source_status(preview, key)

        if status is None:
            # Backward-compatible fallback for a preview without the
            # diagnostics block: classify from the existing contract.
            status = _legacy_status(preview, key)

        icon = STATUS_ICONS.get(status, "❔")
        status_label = STATUS_LABELS.get(status, status or "unknown")

        st.markdown(f"{icon} **{label}** — {status_label}")

        detail = _source_detail(preview, key, status)

        if detail:
            st.caption(detail)


def _legacy_status(preview: dict[str, Any], source: str) -> str:
    """Best-effort classification when ``diagnostics`` is absent."""

    sources = preview.get("sources") or {}
    errors = preview.get("source_errors") or []

    failed = {
        str(item.get("source") or "")
        for item in errors
        if isinstance(item, dict)
    }

    if source in failed or (
        source == "conversation" and "conversation" in failed
    ):
        return "error"

    if source == "conversation_summary":
        conversation = sources.get("conversation") if isinstance(sources, dict) else None
        if not isinstance(conversation, dict):
            return "not_requested"
        return "ok" if (conversation.get("summary") or "").strip() else "not_requested"

    value = sources.get(source) if isinstance(sources, dict) else None

    if value is None and source == "conversation":
        return "not_requested"

    if isinstance(value, dict):
        return "ok" if value else "empty"

    if isinstance(value, list):
        return "ok" if value else "empty"

    return "not_requested"


def _render_retrievals(st, preview: dict[str, Any]) -> None:
    st.subheader("Retrieval details")

    diagnostics = diagnostics_of(preview)
    sources = preview.get("sources") or {}

    # Conversation counts come from the preview itself; diagnostics add
    # the status split and the timing.
    conversation = sources.get("conversation") if isinstance(sources, dict) else None

    if isinstance(conversation, dict):
        st.markdown(
            "**Conversation** — "
            f"{conversation.get('recent_count', 0)} recent · "
            f"{conversation.get('older_count', 0)} older · "
            f"{conversation.get('total_messages', 0)} total message(s)"
        )

    memories = sources.get("memory") if isinstance(sources, dict) else None

    if isinstance(memories, list):
        st.markdown(f"**Long-term memory** — {len(memories)} match(es)")

    rag = sources.get("rag") if isinstance(sources, dict) else None

    if isinstance(rag, list):
        document_ids = sorted(
            {
                str(item.get("document_id"))
                for item in rag
                if isinstance(item, dict) and item.get("document_id")
            }
        )
        content_types = sorted(
            {
                str(item.get("content_type") or "text")
                for item in rag
                if isinstance(item, dict)
            }
        )

        st.markdown(
            f"**Documents** — {len(rag)} chunk(s) · "
            f"{len(document_ids)} document(s) · "
            f"types: {', '.join(content_types) if content_types else 'none'}"
        )

        metadata: list[dict[str, Any]] = []

        for index, item in enumerate(rag, start=1):
            if not isinstance(item, dict):
                continue

            metadata.append(
                {
                    "chunk": index,
                    "document_id": item.get("document_id") or "—",
                    "content_type": item.get("content_type") or "text",
                    "page": item.get("page_number")
                    if item.get("page_number") is not None
                    else "—",
                    "score": item.get("score")
                    if item.get("score") is not None
                    else "—",
                }
            )

        if metadata:
            # Metadata only: chunk text is never rendered here.
            st.markdown("**Retrieved chunk metadata**")
            for row in metadata:
                st.caption(
                    f"chunk {row['chunk']} · document {row['document_id']} · "
                    f"{row['content_type']} · page {row['page']} · "
                    f"score {row['score']}"
                )

        fallback = source_entry(preview, "rag").get("fallback_source")

        if fallback:
            st.info(
                "Vector retrieval returned nothing, so the selected "
                "document's full text was used"
                + (" (masked artifact)." if fallback == "masked_full_text" else ".")
            )

    graph = sources.get("knowledge_graph") if isinstance(sources, dict) else None

    if isinstance(graph, list):
        entities = len(
            {
                item.get("matched_entity")
                for item in graph
                if isinstance(item, dict) and item.get("matched_entity")
            }
        )
        st.markdown(
            f"**Knowledge graph** — {len(graph)} relationship(s) · "
            f"{entities} matching entit(y/ies)"
        )

        if not graph:
            st.caption(
                "No entity in the graph matched this question. This is an "
                "empty result, not a failure."
            )

    if diagnostics is None:
        st.caption(
            "Per-source timings are unavailable: this backend returned "
            "the legacy preview contract."
        )


def _render_assembly(st, preview: dict[str, Any]) -> None:
    st.subheader("Context assembly")

    diagnostics = diagnostics_of(preview) or {}
    assembly = diagnostics.get("assembly") or {}

    if not isinstance(assembly, dict):
        assembly = {}

    sections = section_counts(preview)
    chars = assembly.get("chars")
    formatted = preview.get("formatted_context")
    formatted_text = formatted if isinstance(formatted, str) else ""

    if not isinstance(chars, int) and formatted_text:
        # Legacy previews carry no assembly block; the rendered text is
        # still a reliable character count.
        chars = len(formatted_text)

    st.caption("Assembly status: succeeded")
    st.caption(
        f"Final context: {int(chars) if isinstance(chars, int) else 'unknown'} "
        "character(s)"
    )

    if sections:
        st.markdown("**Sections contributed**")

        for section in sections:
            label = SECTION_LABELS.get(section["name"], section["name"])

            st.caption(f"{label} — {section['chars']} character(s)")
    elif diagnostics_of(preview) is None:
        st.caption(
            "Section sizes are unavailable: this backend returned the "
            "legacy preview contract."
        )
    else:
        st.caption(
            "No evidence sections were assembled. The context contained "
            "only the current question."
        )

    if assembly.get("truncated") is True:
        st.warning(
            "The assembled context hit the backend's character ceiling and "
            "was truncated; a truncation notice was included in the prompt."
        )

    if _runs_without_evidence(preview):
        st.info(
            "No context source contributed evidence. Chat does not block "
            "an unscoped question in this situation: it answers from the "
            "current question alone. With a selected document scope, chat "
            "refuses rather than fabricate a summary."
        )

    with st.expander("🧾 Raw assembled context (may contain document text)"):
        st.caption(
            "Shown exactly as assembled for the model. Masked uploads keep "
            "their placeholders; documents ingested without masking may "
            "contain sensitive text, so open this only when authorized."
        )

        if formatted_text.strip():
            st.code(formatted_text, language="text")
        else:
            st.info(
                "The backend assembled no context text for this query. "
                "This is an empty result, not a context payload."
            )


def _render_errors(st, preview: dict[str, Any]) -> None:
    st.subheader("Errors")

    diagnostics = diagnostics_of(preview) or {}
    errors = diagnostics.get("errors")

    if isinstance(errors, list) and errors:
        for item in errors:
            if not isinstance(item, dict):
                continue

            source = str(item.get("source") or "")
            label = ERROR_SOURCE_LABELS.get(source, source or "Source")
            message = str(item.get("message") or "This source failed.")
            category = str(item.get("category") or "internal")

            st.error(f"{label}: {message} (category: {category})")

        st.caption(
            "Details stay in the server logs; only a fixed, sanitized "
            "message and category are shown here."
        )
        return

    # Legacy previews expose source names only; raw exception text is
    # deliberately not rendered on this page.
    failed_sources = sorted(
        {
            str(item.get("source") or "")
            for item in (preview.get("source_errors") or [])
            if isinstance(item, dict)
        }
    )

    if failed_sources:
        labels = ", ".join(
            ERROR_SOURCE_LABELS.get(name, name) for name in failed_sources
        )
        st.warning(
            f"The backend reported failures for: {labels}. This backend "
            "did not return sanitized diagnostics; check the server logs."
        )
        return

    st.success("No source errors were reported for this run.")


def render_page(
    st,
    *,
    api_url: str,
    conversation_id: str | None,
    state,
    session_id: str | None = None,
    conversation_title: str | None = None,
    document_id: str | None = None,
    document_name: str | None = None,
    fetch=None,
    health=None,
    capabilities=None,
) -> None:
    """Render the Diagnostics page.

    The preview is fetched only inside the button branch, so opening or
    rerunning the page causes no context retrieval and no LLM answer.
    """

    st.title("Diagnostics")
    st.caption(
        "Trace one context assembly run: which sources were queried, what "
        "they contributed, and what was finally sent to the model. "
        "Read-only — no answer is generated."
    )
    st.caption(
        "Running diagnostics performs one real context assembly. If the "
        "selected conversation has older messages, the backend may refresh "
        "its stored summary, which can call the model once."
    )

    _render_health(
        st,
        api_url,
        health or fetch_health,
        capabilities or fetch_capabilities,
    )

    st.divider()

    if conversation_id:
        label = (conversation_title or "").strip() or conversation_id
        st.caption(f"Selected conversation: {label}")

        use_conversation = st.checkbox(
            "Include the selected conversation's history and summary",
            value=True,
            key="diagnostics_use_conversation",
        )
    else:
        use_conversation = False
        st.info(
            "No conversation is selected. Conversation history and its "
            "summary will not be part of this run."
        )

    if document_id:
        st.caption(
            "Document scope: "
            f"{(document_name or '').strip() or document_id}"
        )

    query = st.text_input(
        "Question to diagnose",
        key="diagnostics_query",
        placeholder="Enter a question...",
    )

    memory_user_id = st.text_input(
        "Long-term memory owner (user id)",
        key="diagnostics_user_id",
        placeholder="Leave empty to skip memory retrieval",
    )

    top_k = st.selectbox(
        "Document results (top_k)",
        (3, 5, 10, 20),
        index=1,
        key="diagnostics_top_k",
    )

    if st.button(
        "Run diagnostics",
        type="primary",
        key="diagnostics_run",
    ):
        cleaned_query = (query or "").strip()

        if not cleaned_query:
            st.warning("Enter a question first.")
        else:
            target_conversation = (
                conversation_id if use_conversation else None
            )
            target_user = (memory_user_id or "").strip() or None
            loader = fetch or fetch_preview

            try:
                preview = loader(
                    cleaned_query,
                    top_k=int(top_k),
                    document_id=document_id,
                    conversation_id=target_conversation,
                    user_id=target_user,
                    session_id=session_id,
                    api_url=api_url,
                )

            except (
                requests.RequestException,
                api_client.APIError,
                ValueError,
            ) as exc:
                # A stale run must never look like the current result.
                state["diagnostics_result"] = None
                st.error(f"Could not run diagnostics: {exc}")

            else:
                state["diagnostics_result"] = {
                    "preview": preview,
                    "query": cleaned_query,
                    "conversation_id": target_conversation,
                    "user_id": target_user,
                    "document_id": document_id,
                }

    stored = state.get("diagnostics_result")

    if not isinstance(stored, dict):
        st.caption(
            "No diagnostics run yet. Enter a question and press "
            "**Run diagnostics**."
        )
        return

    preview = stored.get("preview")

    if not isinstance(preview, dict):
        return

    if stored.get("conversation_id") != (
        conversation_id if use_conversation else None
    ):
        st.caption(
            "These diagnostics were collected for a different conversation "
            "selection. Run again to refresh them."
        )

    st.divider()
    _render_overview(
        st,
        preview,
        conversation_title=conversation_title,
        conversation_requested=bool(stored.get("conversation_id")),
    )
    _render_sources(st, preview)
    _render_retrievals(st, preview)
    _render_assembly(st, preview)
    _render_errors(st, preview)
