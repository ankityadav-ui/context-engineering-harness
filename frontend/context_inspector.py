"""Context Inspector backed by the authoritative ``/context/preview`` API.

Opening the page performs no backend call. A preview is fetched only
when the user presses *Inspect context*, and the result is stored in the
Streamlit session so reruns keep displaying the same evidence without
re-querying the backend or generating an answer.
"""

from __future__ import annotations

from typing import Any

import requests

import api_client

# Source keys returned by POST /context/preview under "sources".
SOURCE_LABELS = {
    "conversation": "Conversation",
    "memory": "Long-term memory",
    "rag": "Document retrieval",
    "knowledge_graph": "Knowledge graph",
}

_SOURCE_ORDER = ("conversation", "memory", "rag", "knowledge_graph")


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
    """Call the existing preview endpoint and return its JSON response."""

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


def source_errors(preview: dict[str, Any]) -> dict[str, str]:
    """Map each failed source name to the error the backend reported."""

    errors: dict[str, str] = {}

    for item in preview.get("source_errors") or []:
        if not isinstance(item, dict):
            continue

        source = str(item.get("source") or "").strip()

        if source:
            errors[source] = str(item.get("error") or "").strip()

    return errors


def source_state(
    preview: dict[str, Any],
    source: str,
    *,
    requested: bool = True,
) -> str:
    """Classify one source as error, ok, empty, or not requested.

    A legitimate empty result (the service answered with nothing) is
    deliberately distinct from a retrieval/service error.
    """

    if source in source_errors(preview):
        return "error"

    if not requested:
        return "not_requested"

    sources = preview.get("sources")

    if not isinstance(sources, dict):
        return "empty"

    value = sources.get(source)

    if source == "conversation":
        return "ok" if isinstance(value, dict) and value else "empty"

    if isinstance(value, list):
        return "ok" if value else "empty"

    return "empty"


def _clip(text: object, limit: int) -> str:
    """Return readable text with an explicit truncation marker."""

    value = str(text or "")

    if len(value) <= limit:
        return value

    return value[:limit].rstrip() + "\n\n[truncated for display]"


def _role_label(message: dict[str, Any]) -> str:
    role = str(message.get("role") or "unknown").strip().upper()

    return role or "UNKNOWN"


def _conversation_panel(st, preview, state: str, errors: dict) -> None:
    with st.expander("🧠 Conversation context (summary + recent messages)"):
        if state == "error":
            st.error(
                "Conversation context retrieval failed: "
                + (errors.get("conversation") or "unknown error")
            )
            return

        if state == "not_requested":
            st.info(
                "No conversation was selected, so conversation history "
                "and its summary were not requested."
            )
            return

        conversation = (preview.get("sources") or {}).get("conversation")

        if state == "empty" or not isinstance(conversation, dict):
            st.caption(
                "The conversation exists but returned no message window."
            )
            return

        total = conversation.get("total_messages", 0)
        older = conversation.get("older_count", 0)
        recent = conversation.get("recent_count", 0)

        st.caption(
            f"{total} message(s) total · {recent} recent · "
            f"{older} older summarized"
        )

        summary = conversation.get("summary")

        if isinstance(summary, str) and summary.strip():
            st.markdown("**Summary of older messages**")
            st.markdown(_clip(summary, 4000))
        elif older:
            st.caption(
                "Older messages exist, but the backend returned no summary."
            )
        else:
            st.caption("No older messages need summarizing.")

        recent_messages = conversation.get("recent_messages") or []

        if not recent_messages:
            st.caption(
                "No recent messages are in the context window for this "
                "conversation."
            )
            return

        for message in recent_messages:
            if not isinstance(message, dict):
                continue

            st.markdown(f"**{_role_label(message)}**")
            st.markdown(_clip(message.get("content"), 4000))


def _memory_panel(st, preview, state: str, errors: dict) -> None:
    with st.expander("🗂️ Long-term memory (LTM)"):
        if state == "error":
            st.error(
                "Long-term memory retrieval failed: "
                + (errors.get("memory") or "unknown error")
            )
            return

        if state == "not_requested":
            st.info(
                "No memory owner (user id) was supplied, so long-term "
                "memory was not retrieved."
            )
            return

        memories = (preview.get("sources") or {}).get("memory") or []

        if state == "empty" or not memories:
            st.caption("No long-term memories matched this query.")
            return

        for item in memories:
            if not isinstance(item, dict):
                continue

            memory_type = str(item.get("memory_type") or "memory")
            importance = item.get("importance", "?")
            score = item.get("score")

            score_label = (
                f" · score {score:.2f}"
                if isinstance(score, (int, float))
                else ""
            )

            st.markdown(
                f"**{memory_type}** · importance {importance}/5"
                f"{score_label}"
            )
            st.markdown(_clip(item.get("content"), 2000))

            matched = item.get("matched_keywords") or []

            if matched:
                st.caption(
                    "Matched keywords: "
                    + ", ".join(str(word) for word in matched)
                )

            st.divider()


def _rag_panel(st, preview, state: str, errors: dict) -> None:
    results = (preview.get("sources") or {}).get("rag") or []
    label = f"📚 Document retrieval ({len(results)} result(s))"

    with st.expander(label):
        if state == "error":
            st.error(
                "Document retrieval failed: "
                + (errors.get("rag") or "unknown error")
            )
            return

        if state == "empty" or not results:
            st.caption("No indexed document content matched this query.")
            return

        for index, item in enumerate(results, start=1):
            if not isinstance(item, dict):
                continue

            document_id = item.get("document_id") or "unknown"
            content_type = item.get("content_type") or "text"
            page_number = item.get("page_number")
            score = item.get("score")

            details = [f"document `{document_id}`"]

            if page_number is not None:
                details.append(f"page {page_number}")

            details.append(str(content_type))

            if isinstance(score, (int, float)):
                details.append(f"score {score:.2f}")

            st.markdown(f"**[{index}] " + " · ".join(details) + "**")

            reference = item.get("content_ref") or item.get("image_id")

            if reference:
                st.caption(f"Reference: {reference}")

            content = str(item.get("content") or "").strip()

            if content:
                st.text(_clip(content, 1500))
            else:
                st.caption("Image unit reference with no text content.")

            st.divider()


def _graph_panel(st, preview, state: str, errors: dict) -> None:
    relationships = (
        (preview.get("sources") or {}).get("knowledge_graph") or []
    )
    label = (
        f"🕸️ Knowledge graph ({len(relationships)} relationship(s))"
    )

    with st.expander(label):
        if state == "error":
            st.error(
                "Knowledge-graph retrieval failed: "
                + (errors.get("knowledge_graph") or "unknown error")
            )
            return

        if state == "empty" or not relationships:
            st.caption(
                "No knowledge-graph relationships matched this question."
            )
            return

        for item in relationships:
            if not isinstance(item, dict):
                continue

            source = item.get("source") or "Unknown"
            relationship = item.get("relationship") or "RELATED_TO"
            target = item.get("target") or "Unknown"
            direction = item.get("direction")
            matched = item.get("matched_entity")

            line = f"{source} --{relationship}--> {target}"

            if direction:
                line += f" ({direction})"

            if matched:
                line = f"**{matched}** · {line}"

            st.markdown(f"- {line}")


def render_preview(
    st,
    stored: dict[str, Any],
    *,
    conversation_requested: bool,
    memory_requested: bool,
) -> None:
    """Render one stored preview response using real backend data."""

    preview = stored.get("preview") or {}
    query = stored.get("query") or ""
    errors = source_errors(preview)
    sources = preview.get("sources") or {}

    st.subheader("Assembled context")
    st.caption(f'Preview for: "{query}"')

    total_rag = len(sources.get("rag") or [])
    total_memory = len(sources.get("memory") or [])
    total_graph = len(sources.get("knowledge_graph") or [])

    st.caption(
        f"{total_rag} document result(s) · {total_memory} memory "
        f"result(s) · {total_graph} graph relationship(s) · "
        f"{len(errors)} source error(s)"
    )

    system_prompt = preview.get("system_prompt")

    if isinstance(system_prompt, dict):
        if system_prompt.get("using_default") is False:
            st.caption(
                "Conversation custom instructions are in effect. "
                "Built-in security instructions remain enforced."
            )
        else:
            st.caption(
                "Default system instructions apply. Built-in security "
                "instructions remain enforced."
            )

    if errors:
        st.warning(
            "The backend reported retrieval errors for: "
            + ", ".join(sorted(errors))
            + ". Those panels show the failure instead of treating it "
            "as an empty result."
        )

    _conversation_panel(
        st,
        preview,
        source_state(
            preview,
            "conversation",
            requested=conversation_requested,
        ),
        errors,
    )
    _memory_panel(
        st,
        preview,
        source_state(preview, "memory", requested=memory_requested),
        errors,
    )
    _rag_panel(st, preview, source_state(preview, "rag"), errors)
    _graph_panel(
        st,
        preview,
        source_state(preview, "knowledge_graph"),
        errors,
    )

    formatted = preview.get("formatted_context")
    formatted_text = formatted if isinstance(formatted, str) else ""
    image_references = preview.get("image_references", 0)

    with st.expander(
        "🧾 Final formatted context (evidence supplied to the model)"
    ):
        st.caption(
            "Assembled evidence text, shown exactly as indexed. Documents "
            "ingested without masking may still contain sensitive text, "
            "so open this only when you are authorized to view it."
        )

        if formatted_text.strip():
            st.code(formatted_text, language="text")
        else:
            st.info(
                "The backend assembled no evidence for this query. This "
                "is an empty result, not a context payload."
            )

        st.caption(
            f"{len(formatted_text)} characters · "
            f"{image_references} image reference(s) resolved"
        )


def render_page(
    st,
    *,
    api_url: str,
    conversation_id: str | None,
    state,
    session_id: str | None = None,
    conversation_title: str | None = None,
    fetch=None,
) -> None:
    """Render the Context Inspector page.

    The preview is fetched only inside the button branch, so merely
    opening or rerunning the page never contacts the backend.
    """

    st.title("Context Inspector")
    st.caption(
        "Inspect the evidence the backend would assemble for a question. "
        "No chat answer is generated."
    )
    st.caption(
        "Inspection never adds chat messages. If the selected conversation "
        "has older messages, the backend may refresh its stored summary "
        "during inspection, which can call the model once."
    )

    if conversation_id:
        label = (conversation_title or "").strip() or conversation_id

        st.caption(f"Selected conversation: {label}")

        use_conversation = st.checkbox(
            "Include the selected conversation's history and summary",
            value=True,
            key="context_inspector_use_conversation",
            help=(
                "Uses the same conversation and session-ownership rules "
                "as chat and document Q&A."
            ),
        )
    else:
        use_conversation = False

        st.info(
            "No conversation is selected. Conversation history and its "
            "summary will be omitted from this preview."
        )

    query = st.text_input(
        "Question to inspect",
        key="context_inspector_query",
        placeholder="Enter a question...",
    )

    memory_user_id = st.text_input(
        "Long-term memory owner (user id)",
        key="context_inspector_user_id",
        placeholder="Leave empty to skip memory retrieval",
        help=(
            "Only memories belonging to this user id are retrieved. "
            "This is the same trusted user_id claim as the memory API."
        ),
    )

    if st.button(
        "Inspect context",
        type="primary",
        key="context_inspector_run",
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
                # Never leave an older preview looking like the result
                # of this failed request.
                state["context_inspector_result"] = None
                st.error(f"Could not load the context preview: {exc}")

            else:
                state["context_inspector_result"] = {
                    "preview": preview,
                    "query": cleaned_query,
                    "conversation_id": target_conversation,
                    "user_id": target_user,
                }

    stored = state.get("context_inspector_result")

    if not isinstance(stored, dict):
        return

    if not isinstance(stored.get("preview"), dict):
        return

    current_conversation = conversation_id if use_conversation else None

    if stored.get("conversation_id") != current_conversation:
        st.caption(
            "This preview was assembled for a different conversation "
            "selection. Inspect again to refresh it."
        )

    render_preview(
        st,
        stored,
        conversation_requested=bool(stored.get("conversation_id")),
        memory_requested=bool(stored.get("user_id")),
    )
