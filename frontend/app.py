
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

from context_inspector import render_page as render_context_inspector
from masking_workflow import MaskingError, prepare_chat_message

# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

API_URL = os.getenv(
    "BACKEND_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

st.set_page_config(
    page_title="Context Engineering Harness",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# Theme and bottom composer styling
# =========================================================

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
    }

    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: #111318 !important;
        color: #ECECF1 !important;
    }

    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {
        background: #111318 !important;
    }

    [data-testid="stDecoration"] {
        display: none !important;
    }

    [data-testid="stSidebar"] {
        background: #191B22 !important;
        border-right: 1px solid #2B2E37;
    }

    [data-testid="stSidebar"] * {
        color: #ECECF1;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 9rem;
    }

    h1, h2, h3, p, label {
        color: #ECECF1;
    }

    .brand {
        font-size: 1.18rem;
        font-weight: 750;
        letter-spacing: -0.035em;
        padding: 0.35rem 0 1.2rem;
    }

    .eyebrow {
        font-size: 0.73rem;
        font-weight: 650;
        color: #A1A1AA;
        text-transform: uppercase;
        letter-spacing: 0.11em;
        margin: 1rem 0 0.55rem;
    }

    .muted {
        color: #A1A1AA !important;
        font-size: 0.88rem;
    }

    .welcome-card {
        background: #1B1E26;
        border: 1px solid #2C303A;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1.4rem 0;
    }

    .welcome-card h3 {
        margin: 0 0 0.55rem;
        font-size: 1.3rem;
    }

    .welcome-card p {
        margin: 0;
        color: #A1A1AA;
        line-height: 1.7;
    }

    div[data-testid="stChatMessage"] {
        background: #191C23;
        border: 1px solid #292D36;
        border-radius: 14px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
    }

    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div {
        background: #1B1E26 !important;
        border: 1px solid #3A3E49 !important;
        border-radius: 14px !important;
        color: #ECECF1 !important;
    }

    [data-testid="stChatInput"] textarea {
        color: #ECECF1 !important;
    }

    [data-testid="stChatInput"] button {
        color: #ECECF1 !important;
    }

    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] section {
        background: #1B1E26 !important;
        border-color: #3A3E49 !important;
    }

    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input {
        background: #1B1E26 !important;
        color: #ECECF1 !important;
        border-color: #3A3E49 !important;
    }

    [data-testid="stExpander"] {
        background: #1B1E26;
        border: 1px solid #2C303A;
        border-radius: 12px;
    }

    div.stButton > button {
        border-radius: 9px;
        min-height: 2.55rem;
    }

    div.stButton > button[kind="secondary"] {
        background: #20232B;
        border: 1px solid #30343E;
        color: #E4E4E7;
    }

    div.stButton > button[kind="primary"] {
        background: #5146E5;
        border: 1px solid #5146E5;
        color: white;
    }

    /* Position the masking control near the fixed chat composer.
       This selector uses Streamlit's keyed container. */
    [class*="st-key-composer_masking"] {
        position: fixed;
        bottom: 1.15rem;
        left: 50%;
        transform: translateX(-50%);
        width: min(650px, 38vw);
        z-index: 1000;
        background: #1B1E26;
        border: 1px solid #3A3E49;
        border-radius: 10px;
        padding: 0.25rem 0.7rem;
    }

    [class*="st-key-composer_masking"] label {
        font-size: 0.85rem;
    }

    @media (max-width: 900px) {
        [class*="st-key-composer_masking"] {
            left: 10px;
            transform: none;
            width: 42%;
            bottom: 1.1rem;
        }
    }

    hr {
        border-color: #2C303A !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Session state
# =========================================================

DEFAULTS = {
    "messages": [],
    "active_page": "Chat",
    "masking_enabled": False,
    "system_prompt": "",
    # Conversation whose saved prompt is loaded in the editor (None:
    # nothing loaded yet).
    "system_prompt_loaded_for": None,
    "system_prompt_max_chars": 4000,
    "system_prompt_load_error": None,
    "system_prompt_saved_notice": "",
    "system_prompt_action_error": "",
    "conversation_id": None,
    "conversation_list": [],
    "conversation_list_error": None,
    "document_list": [],
    "document_list_error": None,
    # SHA-256 fingerprints of PDFs uploaded this session, so a Streamlit
    # rerun can never push the same file twice.
    "uploaded_document_fingerprints": set(),
    # Last successful upload result, shown once after a rerun.
    "documents_last_upload": None,
    # Document the chat is currently grounded in. Set on every successful
    # upload (chat or Documents page) so "summarise this" resolves the
    # document the user just added instead of searching the whole index.
    "active_document_id": None,
    "active_document_name": None,
    "active_document_masked": False,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# API helpers
# =========================================================

def api_get(path: str, timeout: int = 8) -> dict:
    response = requests.get(f"{API_URL}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict, timeout: int = 120) -> dict:
    response = requests.post(
        f"{API_URL}{path}",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def api_put(path: str, payload: dict, timeout: int = 15) -> dict:
    """Update backend settings through FastAPI."""
    response = requests.put(
        f"{API_URL}{path}",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def api_delete(path: str, timeout: int = 15) -> dict:
    """Delete a backend resource through FastAPI."""
    response = requests.delete(f"{API_URL}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def list_conversations() -> list[dict]:
    """Fetch persisted conversations from FastAPI."""
    data = api_get("/conversations?limit=50&offset=0")
    conversations = data.get("conversations", [])
    return conversations if isinstance(conversations, list) else []


def create_conversation(title: str | None = None) -> dict:
    """Create a durable conversation in PostgreSQL through FastAPI."""
    payload = {"title": title} if title else {}
    return api_post("/conversations", payload, timeout=15)


def load_conversation_messages(conversation_id: str) -> list[dict]:
    """Load a conversation transcript in chronological order."""
    data = api_get(
        f"/conversations/{conversation_id}/messages?limit=200&order=chronological",
        timeout=15,
    )
    raw_messages = data.get("messages", [])
    messages: list[dict] = []
    if isinstance(raw_messages, list):
        for item in raw_messages:
            if not isinstance(item, dict):
                continue
            role = item.get("role", "assistant")
            content = item.get("content", "")
            if role in {"user", "assistant", "system"} and isinstance(content, str):
                messages.append({"role": role, "content": content})
    return messages


def refresh_conversation_list() -> None:
    """Refresh the sidebar cache without breaking the chat on API errors."""
    try:
        st.session_state.conversation_list = list_conversations()
        st.session_state.conversation_list_error = None
    except (requests.RequestException, ValueError) as exc:
        st.session_state.conversation_list = []
        st.session_state.conversation_list_error = str(exc)


def http_error_detail(exc: requests.HTTPError) -> str:
    """Extract the backend's actionable error detail, when present."""
    try:
        return str(exc.response.json().get("detail", ""))
    except (ValueError, AttributeError):
        return ""


def list_documents() -> dict:
    """Fetch the indexed-document listing from FastAPI."""
    return api_get("/documents", timeout=60)


def refresh_document_list() -> None:
    """Refresh the document cache without breaking the page on errors."""
    try:
        data = list_documents()
        documents = data.get("documents", [])
        st.session_state.document_list = (
            documents if isinstance(documents, list) else []
        )
        st.session_state.document_list_error = None
    except (requests.RequestException, ValueError) as exc:
        st.session_state.document_list = []
        st.session_state.document_list_error = str(exc)


def upload_pdf_document(
    file_bytes: bytes,
    filename: str,
    mask_content: bool,
) -> dict:
    """Upload one PDF to the backend ingestion endpoint.

    Multipart POST /documents/upload; the response is only returned
    after the backend finished extraction, OCR, embedding, and
    indexing, so callers can trust the reported status.
    """
    buffer = io.BytesIO(file_bytes)
    response = requests.post(
        f"{API_URL}/documents/upload",
        files={"file": (filename, buffer, "application/pdf")},
        data={"mask_content": "true" if mask_content else "false"},
        timeout=900,
    )
    response.raise_for_status()
    return response.json()


def ask_documents(payload: dict) -> dict:
    """Ask a question through the backend document Q&A pipeline."""
    return api_post("/documents/ask", payload, timeout=300)


def backend_status() -> tuple[bool, str]:
    try:
        data = api_get("/health", timeout=4)
        if data.get("status") == "ok":
            return True, "Connected"
        return False, "Unexpected health response"
    except requests.RequestException:
        return False, "Unavailable"


def sync_system_prompt_editor() -> None:
    """Load the selected conversation's saved prompt into the editor.

    Every conversation has its own custom system prompt stored in the
    backend. When the selected conversation changes, the editor is
    refilled from that conversation's saved prompt (or cleared when it
    has none), so switching and returning restores each prompt.
    """

    conversation_id = st.session_state.get("conversation_id")

    if st.session_state.system_prompt_loaded_for == conversation_id:
        return

    # No conversation selected: an empty editor on the default
    # instructions; saving stays disabled until one is selected.
    if not conversation_id:
        st.session_state.system_prompt = ""
        st.session_state.system_prompt_loaded_for = None
        st.session_state.system_prompt_load_error = None
        return

    try:
        settings = api_get(
            f"/conversations/{conversation_id}/system-prompt",
            timeout=10,
        )

        saved_prompt = settings.get("custom_prompt", "")

        if isinstance(saved_prompt, str):
            st.session_state.system_prompt = saved_prompt

        max_chars = settings.get("max_custom_prompt_length")

        if isinstance(max_chars, int) and max_chars > 0:
            st.session_state.system_prompt_max_chars = max_chars

        st.session_state.system_prompt_loaded_for = conversation_id
        st.session_state.system_prompt_load_error = None

    except (requests.RequestException, ValueError) as exc:
        st.session_state.system_prompt_load_error = str(exc)
        # Not marked loaded: the next rerun retries the fetch.


def get_answer(payload: dict) -> str:
    for key in ("answer", "response", "reply", "message"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value

    result = payload.get("result")
    if isinstance(result, str) and result.strip():
        return result

    return str(payload)


def send_chat_message(message: str) -> None:
    """Mask when enabled, then send exactly the approved text.

    Masking is fail-closed: if the backend cannot confirm the masking
    safety check, nothing is sent to chat. The original text is kept
    only in this browser session for display; the backend stores the
    masked text it receives.
    """

    try:
        prepared = prepare_chat_message(
            message,
            st.session_state.masking_enabled,
            api_url=API_URL,
        )

    except MaskingError as exc:
        # The message is shown for context, but it is not sent and its
        # masking status is explicitly reported as not applied.
        st.session_state.messages.append({
            "role": "user",
            "content": message,
            "masking": {"applied": False, "error": str(exc)},
        })
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Masking failed, so the message was not sent. {exc}",
        })
        return

    masking_report = None

    if prepared.masking is not None:
        masking_report = {
            "applied": True,
            "detected_count": prepared.masking.detected_count,
            "categories": prepared.masking.detected_categories,
        }

    st.session_state.messages.append({
        "role": "user",
        "content": message,
        "masking": masking_report,
    })

    try:
        payload = {"message": prepared.content}
        conversation_id = st.session_state.get("conversation_id")
        if conversation_id:
            payload["conversation_id"] = conversation_id

        # Scope the request to the document the user most recently
        # uploaded, so the backend can use its full-text fallback when
        # vector search returns nothing for a short resume.
        document_id = st.session_state.get("active_document_id")
        if document_id:
            payload["document_id"] = document_id

        result = api_post(
            "/chat",
            payload,
            timeout=180,
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": get_answer(result),
        })

    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except (ValueError, AttributeError):
            pass

        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Backend error: {detail or exc}",
        })

    except requests.RequestException as exc:
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "Couldn't reach FastAPI. Check that Uvicorn is running. "
                f"Details: {exc}"
            ),
        })


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:
    st.markdown(
        '<div class="brand">🧠 Context Engineering</div>',
        unsafe_allow_html=True,
    )

    if st.button("＋  New chat", use_container_width=True, type="primary"):
        try:
            created = create_conversation("New chat")
            st.session_state.conversation_id = created.get("id")
            st.session_state.messages = []
            # A new conversation starts unscoped; upload a PDF to scope it.
            st.session_state.active_document_id = None
            st.session_state.active_document_name = None
            st.session_state.active_document_masked = False
            st.session_state.active_page = "Chat"
            refresh_conversation_list()
            st.rerun()
        except (requests.RequestException, ValueError) as exc:
            st.error(f"Could not create a conversation: {exc}")

    st.markdown(
        '<div class="eyebrow">Workspace</div>',
        unsafe_allow_html=True,
    )

    pages = [
        ("Chat", "💬"),
        ("Documents", "📄"),
        ("Context Inspector", "🧩"),
        ("Knowledge Graph", "🕸️"),
        ("Diagnostics", "⚙️"),
    ]

    for page_name, icon in pages:
        if st.button(
            f"{icon}  {page_name}",
            key=f"nav_{page_name}",
            use_container_width=True,
            type=(
                "primary"
                if st.session_state.active_page == page_name
                else "secondary"
            ),
        ):
            st.session_state.active_page = page_name
            st.rerun()

    st.divider()

    # The system-prompt editor sits above the conversation list: every
    # conversation has its own custom prompt, saved per conversation in
    # the backend.
    with st.expander("System instructions", expanded=False):
        # Refill the editor from the selected conversation's saved
        # prompt before creating the keyed text_area, so Streamlit
        # receives the right value on every rerun.
        sync_system_prompt_editor()

        selected_conversation = st.session_state.get("conversation_id")

        if st.session_state.system_prompt_load_error:
            st.warning(
                "Couldn't load this conversation's saved system "
                "instructions. Check that FastAPI is running; this panel "
                "retries on the next rerun."
            )

        st.text_area(
            "System prompt",
            key="system_prompt",
            height=180,
            max_chars=st.session_state.system_prompt_max_chars,
            disabled=not selected_conversation,
            placeholder=(
                "Describe the assistant's behavior..."
                if selected_conversation
                else "Create or select a conversation first..."
            ),
            help=(
                "This custom prompt is saved for this conversation only. "
                "Built-in security instructions remain enforced."
            ),
        )

        def save_system_prompt_callback() -> None:
            """Save the editor value against the selected conversation."""
            conversation_id = st.session_state.get("conversation_id")

            if not conversation_id:
                st.session_state.system_prompt_saved_notice = ""
                st.session_state.system_prompt_action_error = (
                    "Create or select a conversation before saving "
                    "system instructions."
                )
                return

            try:
                api_put(
                    f"/conversations/{conversation_id}/system-prompt",
                    {"custom_prompt": st.session_state.get("system_prompt", "")},
                    timeout=15,
                )
                st.session_state.system_prompt_saved_notice = "System instructions saved."
                st.session_state.system_prompt_action_error = ""
                st.session_state.system_prompt_load_error = None
                st.session_state.system_prompt_loaded_for = conversation_id
            except requests.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.response.json().get("detail", "")
                except (ValueError, AttributeError):
                    pass
                st.session_state.system_prompt_saved_notice = ""
                st.session_state.system_prompt_action_error = (
                    f"Couldn't save system instructions: {detail or exc}"
                )
            except (requests.RequestException, ValueError) as exc:
                st.session_state.system_prompt_saved_notice = ""
                st.session_state.system_prompt_action_error = (
                    f"Couldn't save system instructions: {exc}"
                )

        def reset_system_prompt_callback() -> None:
            """Reset this conversation to the default system instructions."""
            conversation_id = st.session_state.get("conversation_id")

            if not conversation_id:
                st.session_state.system_prompt_saved_notice = ""
                st.session_state.system_prompt_action_error = (
                    "Create or select a conversation before resetting "
                    "system instructions."
                )
                return

            try:
                api_delete(
                    f"/conversations/{conversation_id}/system-prompt",
                    timeout=15,
                )
                # The custom prompt text disappears from the editor.
                st.session_state.system_prompt = ""
                st.session_state.system_prompt_saved_notice = (
                    "Custom prompt reset to default."
                )
                st.session_state.system_prompt_action_error = ""
                st.session_state.system_prompt_load_error = None
                st.session_state.system_prompt_loaded_for = conversation_id
            except (requests.RequestException, ValueError) as exc:
                st.session_state.system_prompt_saved_notice = ""
                st.session_state.system_prompt_action_error = (
                    f"Couldn't reset system instructions: {exc}"
                )

        save_col, reset_col = st.columns(2)
        with save_col:
            st.button(
                "Save instructions",
                key="save_system_prompt",
                type="primary",
                use_container_width=True,
                disabled=not selected_conversation,
                on_click=save_system_prompt_callback,
            )

        with reset_col:
            st.button(
                "Reset prompt",
                key="reset_system_prompt",
                use_container_width=True,
                disabled=not selected_conversation,
                on_click=reset_system_prompt_callback,
            )

        if st.session_state.get("system_prompt_saved_notice"):
            st.success(st.session_state.system_prompt_saved_notice)
            st.session_state.system_prompt_saved_notice = ""
        if st.session_state.get("system_prompt_action_error"):
            st.error(st.session_state.system_prompt_action_error)
            st.session_state.system_prompt_action_error = ""

        if not selected_conversation:
            st.caption(
                "Create or select a conversation to edit its system "
                "instructions."
            )

        st.caption(
            "Saved per conversation in PostgreSQL. Leave the field empty "
            "to use the default instructions for this conversation. "
            "Built-in security instructions cannot be removed by this "
            "editor."
        )

    st.divider()

    st.markdown(
        '<div class="eyebrow">Conversations</div>',
        unsafe_allow_html=True,
    )

    # Load history from the backend on each Streamlit rerun.
    refresh_conversation_list()
    if st.session_state.conversation_list_error:
        st.caption("Conversation history is temporarily unavailable.")
    elif not st.session_state.conversation_list:
        st.caption("No saved conversations yet.")
    else:
        for conversation in st.session_state.conversation_list:
            conversation_id = conversation.get("id")
            title = (conversation.get("title") or "New chat").strip()
            if len(title) > 32:
                title = title[:29] + "..."
            is_active = conversation_id == st.session_state.conversation_id
            message_count = conversation.get("message_count", 0)
            label = f"{'● ' if is_active else ''}{title}  ·  {message_count}"
            if st.button(
                label,
                key=f"conversation_{conversation_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
                help=f"Open conversation {conversation_id}",
            ):
                try:
                    st.session_state.messages = load_conversation_messages(conversation_id)
                    st.session_state.conversation_id = conversation_id
                    # Do not carry one conversation's document scope into
                    # another conversation.
                    st.session_state.active_document_id = None
                    st.session_state.active_document_name = None
                    st.session_state.active_document_masked = False
                    st.session_state.active_page = "Chat"
                    st.rerun()
                except (requests.RequestException, ValueError) as exc:
                    st.error(f"Could not load conversation: {exc}")

    st.divider()

    healthy, _ = backend_status()

    if healthy:
        st.success("API connected")
    else:
        st.error("API unavailable")

    st.caption(f"FastAPI: {API_URL}")


# =========================================================
# Main content
# =========================================================

page = st.session_state.active_page

if page == "Chat":
    st.title("Chat")
    st.markdown(
        '<p class="muted">Your unified workspace for conversations, '
        'documents, memory and knowledge retrieval.</p>',
        unsafe_allow_html=True,
    )

    if st.session_state.get("active_document_id"):
        scope_col, clear_col = st.columns([5, 2])

        with scope_col:
            scope_name = (
                st.session_state.get("active_document_name")
                or st.session_state["active_document_id"]
            )
            masked_note = (
                " · masked"
                if st.session_state.get("active_document_masked")
                else ""
            )
            st.caption(
                f"📄 Questions in this chat are grounded in "
                f"**{scope_name}**{masked_note}. "
                "Ask “summarise this” to summarise it."
            )

        with clear_col:
            if st.button(
                "Ask across all documents",
                key="clear_active_document",
                use_container_width=True,
            ):
                st.session_state.active_document_id = None
                st.session_state.active_document_name = None
                st.session_state.active_document_masked = False
                st.rerun()

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="welcome-card">
                <h3>How can I help?</h3>
                <p>
                    Ask a question or explore your knowledge base through
                    your context engine.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🧠 Context-aware**")
            st.caption("Use the backend context assembly pipeline.")

        with col2:
            st.markdown("**📚 Document Q&A**")
            st.caption("Upload PDFs and ask questions on the Documents page.")

        with col3:
            st.markdown("**🛡️ Privacy controls**")
            st.caption("Mask text before sending it to chat.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            masking = message.get("masking")

            if isinstance(masking, dict):
                if masking.get("applied"):
                    count = masking.get("detected_count", 0)
                    categories = masking.get("categories") or []
                    noun = "entity" if count == 1 else "entities"
                    details = f"{count} {noun} masked"

                    if categories:
                        details += " (" + ", ".join(categories) + ")"

                    st.caption(
                        "🛡️ Masked by the backend before sending · "
                        f"{details} · safety check passed"
                    )
                elif masking.get("error"):
                    st.caption(f"🛡️ Not sent: {masking['error']}")

    # Masking toggle is positioned beside the bottom chat composer.
    with st.container(key="composer_masking"):
        st.toggle(
            "🛡️ Enable PII masking",
            key="masking_enabled",
            help=(
                "When enabled, every outgoing chat message is masked "
                "and safety-checked by the backend before it is sent. "
                "PDFs attached in chat are ingested with masking enabled."
            ),
        )

    # Native Streamlit chat input supports attachments in the composer.
    # The backend ingestion pipeline accepts PDFs only, so no other
    # file type is offered here.
    prompt = st.chat_input(
        "Message your context engine...",
        accept_file=True,
        file_type=["pdf"],
    )

    if prompt:
        if isinstance(prompt, str):
            message_text = prompt.strip()
            attached_files = []
        else:
            message_text = (getattr(prompt, "text", "") or "").strip()
            attached_files = getattr(prompt, "files", []) or []

        if attached_files:
            # Attachments are ingested through the same backend pipeline
            # as the Documents page: extraction, OCR, embedding, and
            # indexing all happen server-side before we report success.
            for attached in attached_files:
                attached_name = getattr(attached, "name", "attachment")

                if not attached_name.lower().endswith(".pdf"):
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": (
                            f"Skipped {attached_name}: the backend "
                            "ingestion pipeline only accepts PDF files. "
                            "Convert the file to PDF and attach it again."
                        ),
                    })
                    continue

                attached_bytes = attached.getvalue()
                fingerprint = hashlib.sha256(attached_bytes).hexdigest()

                if (
                    fingerprint
                    in st.session_state.uploaded_document_fingerprints
                ):
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": (
                            f"{attached_name} was already uploaded in "
                            "this session, so it was not indexed again."
                        ),
                    })
                    continue

                try:
                    with st.spinner(
                        f"Uploading and indexing {attached_name}..."
                    ):
                        result = upload_pdf_document(
                            attached_bytes,
                            attached_name,
                            mask_content=st.session_state.masking_enabled,
                        )

                    st.session_state.uploaded_document_fingerprints.add(
                        fingerprint
                    )
                    # Ground the rest of this chat in the new document.
                    st.session_state.active_document_id = result.get(
                        "document_id"
                    )
                    st.session_state.active_document_name = attached_name
                    st.session_state.active_document_masked = bool(
                        result.get("masking_applied")
                    )
                    refresh_document_list()

                    masking_note = ""

                    if result.get("masking_applied"):
                        masking_note = (
                            f", masking confirmed by the backend: "
                            f"{result.get('masked_entities', 0)} "
                            f"entities masked, safety check passed: "
                            f"{result.get('masking_safety_passed')}"
                        )
                    elif st.session_state.masking_enabled:
                        masking_note = (
                            ", but masking was requested and was NOT "
                            "confirmed by the backend"
                        )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": (
                            f"Uploaded and indexed {attached_name} "
                            f"(document id `{result.get('document_id')}`, "
                            f"{result.get('content_units', 0)} content "
                            f"units{masking_note}). Ask questions about it "
                            "on the Documents page."
                        ),
                    })

                except requests.HTTPError as exc:
                    detail = http_error_detail(exc)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": (
                            f"Upload failed for {attached_name}: "
                            f"{detail or exc}"
                        ),
                    })

                except requests.RequestException as exc:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": (
                            f"Couldn't upload {attached_name}: FastAPI "
                            f"is unreachable. Details: {exc}"
                        ),
                    })

        if message_text:
            if not st.session_state.get("conversation_id"):
                try:
                    created = create_conversation(message_text[:60] or "New chat")
                    st.session_state.conversation_id = created.get("id")
                    refresh_conversation_list()
                except (requests.RequestException, ValueError) as exc:
                    st.error(f"Could not start a persistent conversation: {exc}")
                    st.stop()

            # send_chat_message appends the user turn after masking, so
            # a failed safety check never leaves a sent-looking message.
            with st.spinner("Waiting for the backend response..."):
                send_chat_message(message_text)

        st.rerun()


elif page == "Documents":
    st.title("Documents")
    st.caption("Document ingestion and retrieval workspace.")

    # -------------------------------------------------- Upload and index
    st.subheader("Upload a PDF")

    uploaded_file = st.file_uploader(
        "PDF file",
        type=["pdf"],
        key="documents_uploader",
        help=(
            "The backend extracts text, runs OCR on scanned pages, "
            "embeds the content, and indexes it in the vector store."
        ),
    )

    mask_before_indexing = st.toggle(
        "🛡️ Mask sensitive content before indexing",
        key="documents_mask",
        help=(
            "When enabled, extracted text is masked and safety-"
            "validated before embedding; image extraction is skipped "
            "and the raw PDF is not stored."
        ),
    )

    upload_clicked = st.button(
        "Upload and index",
        type="primary",
        key="documents_upload_button",
        disabled=uploaded_file is None,
    )

    if upload_clicked and uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        fingerprint = hashlib.sha256(file_bytes).hexdigest()

        if fingerprint in st.session_state.uploaded_document_fingerprints:
            # Rerun/double-click protection: identical bytes are never
            # pushed through ingestion twice in one session.
            st.info(
                f"{uploaded_file.name} was already uploaded in this "
                "session with identical content, so it was not indexed "
                "again."
            )
        else:
            try:
                with st.spinner(
                    "Uploading and indexing... large or scanned PDFs "
                    "can take a while."
                ):
                    result = upload_pdf_document(
                        file_bytes,
                        uploaded_file.name,
                        mask_content=mask_before_indexing,
                    )

            except requests.HTTPError as exc:
                detail = http_error_detail(exc)
                st.error(f"Upload failed: {detail or exc}")

            except requests.RequestException as exc:
                st.error(f"Couldn't reach FastAPI: {exc}")

            else:
                st.session_state.uploaded_document_fingerprints.add(
                    fingerprint
                )
                # The next chat question about "this document" should
                # target what was just uploaded, wherever it came from.
                st.session_state.active_document_id = result.get(
                    "document_id"
                )
                st.session_state.active_document_name = uploaded_file.name
                st.session_state.active_document_masked = bool(
                    result.get("masking_applied")
                )
                # Record which masking mode this result belongs to, so
                # the notice can never imply masking that was not done.
                st.session_state.documents_last_upload = {
                    **result,
                    "_mask_requested": mask_before_indexing,
                }
                # Refresh the listing before the success notice shows on
                # the next run, so the new document is visible at once.
                refresh_document_list()
                st.rerun()

    if st.session_state.documents_last_upload:
        result = st.session_state.documents_last_upload

        # The backend only returns this response after extraction,
        # embedding, and indexing actually succeeded.
        st.success(
            result.get("message", "Document processed successfully.")
        )
        st.caption(
            f"Document id: {result.get('document_id')} · "
            f"{result.get('content_units', 0)} content units · "
            f"{result.get('ocr_pages', 0)} OCR pages · "
            f"{result.get('tables_extracted', 0)} tables · "
            f"{result.get('images_extracted', 0)} images"
        )

        if result.get("masking_applied"):
            st.caption(
                "Masking confirmed by the backend: "
                f"{result.get('masked_entities', 0)} entities masked · "
                f"safety check passed: "
                f"{result.get('masking_safety_passed')}"
            )
        elif result.get("_mask_requested"):
            st.warning(
                "Masking was requested but the backend did not confirm "
                "it was applied. Do not assume this document is masked."
            )

        graph = result.get("knowledge_graph") or {}

        if graph.get("error"):
            st.warning(f"Knowledge graph: {graph['error']}")

        # Shown once, then cleared so reruns do not repeat it.
        st.session_state.documents_last_upload = None

    st.divider()

    # -------------------------------------------------- Indexed documents
    st.subheader("Indexed documents")

    # Refreshed on every rerun (same pattern as the conversation list)
    # so a successful upload or external change is reflected at once.
    refresh_document_list()

    if st.session_state.document_list_error:
        st.warning(
            "The document listing is temporarily unavailable. Details: "
            f"{st.session_state.document_list_error}"
        )

    elif not st.session_state.document_list:
        st.caption("No documents are indexed yet. Upload a PDF above.")

    else:
        for document in st.session_state.document_list:
            sources = document.get("sources") or []
            label = (
                sources[0]
                if sources
                else str(document.get("document_id", "document"))
            )
            pages = document.get("pages") or []
            page_label = (
                f"{len(pages)} pages" if pages else "page count unknown"
            )

            with st.expander(
                f"📄 {label} · {document.get('content_units', 0)} units"
            ):
                st.caption(
                    f"Document id: {document.get('document_id')}"
                )
                st.caption(
                    f"Content types: "
                    f"{', '.join(document.get('content_types') or []) or 'text'}"
                    f" · {page_label}"
                    f" · stored PDF: "
                    f"{'yes' if document.get('pdf_stored') else 'no'}"
                )

    st.divider()

    # -------------------------------------------------- Document Q&A
    st.subheader("Ask a question")

    question = st.text_input(
        "Question",
        placeholder="Ask something about your documents...",
        key="documents_ask_question",
    )

    scope_options = ["All documents"]
    scope_ids: list[str | None] = [None]

    for document in st.session_state.document_list:
        sources = document.get("sources") or []
        label = (
            sources[0]
            if sources
            else str(document.get("document_id", "document"))
        )
        scope_options.append(
            f"{label} ({document.get('document_id')})"
        )
        scope_ids.append(document.get("document_id"))

    selected_scope = st.selectbox(
        "Scope",
        scope_options,
        key="documents_ask_scope",
        help="Restrict retrieval to one indexed document, or search all.",
    )

    use_conversation = False

    if st.session_state.conversation_id:
        use_conversation = st.checkbox(
            "Use the selected conversation's history and saved "
            "instructions",
            value=True,
            key="documents_ask_use_conversation",
            help=(
                "Sends this question with the selected conversation, so "
                "its saved system prompt and history apply (same access "
                "rules as chat)."
            ),
        )

    st.caption(
        "The masking toggle above applies to PDF ingestion. Question text "
        "sent by Ask is not masked by that toggle."
    )

    if st.button("Ask", type="primary", key="documents_ask_button"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            payload = {
                "query": question.strip(),
                "top_k": 5,
                "document_id": scope_ids[scope_options.index(selected_scope)],
            }

            if use_conversation and st.session_state.conversation_id:
                payload["conversation_id"] = st.session_state.conversation_id

            try:
                with st.spinner(
                    "Retrieving context and asking the model..."
                ):
                    result = ask_documents(payload)

            except requests.HTTPError as exc:
                # LLM/config failures surface here (e.g. 503) and are
                # distinct from "no documents matched" below.
                detail = http_error_detail(exc)
                st.error(f"Backend error: {detail or exc}")

            except requests.RequestException as exc:
                st.error(
                    "Couldn't reach FastAPI. Check that Uvicorn is "
                    f"running. Details: {exc}"
                )

            else:
                answer = result.get("answer", "")

                if answer:
                    st.markdown("**Answer**")
                    st.markdown(answer)
                else:
                    st.error(
                        "The backend returned no answer. This usually "
                        "means the model service failed; try again or "
                        "check the Diagnostics page."
                    )

                rag_sources = (result.get("sources") or {}).get("rag") or []

                if rag_sources:
                    st.markdown("**Sources**")

                    for index, item in enumerate(rag_sources, start=1):
                        page_number = item.get("page_number")
                        page_label = (
                            f", page {page_number}"
                            if page_number is not None
                            else ""
                        )
                        score = item.get("score")
                        score_label = (
                            f" · score {score:.2f}"
                            if isinstance(score, (int, float))
                            else ""
                        )
                        content = str(item.get("content", "")).strip()
                        snippet = (
                            content[:300] + "..."
                            if len(content) > 300
                            else content
                        )

                        st.caption(
                            f"[{index}] document "
                            f"{item.get('document_id', 'unknown')}"
                            f"{page_label} · "
                            f"{item.get('content_type', 'text')}"
                            f"{score_label}"
                        )
                        st.markdown(snippet)
                else:
                    st.info(
                        "No matching document content was found for "
                        "this question. The answer above, if any, is "
                        "based on the model and any other available "
                        "context."
                    )

                source_errors = result.get("source_errors") or []

                if source_errors:
                    st.warning(
                        "Some context sources reported errors; the "
                        "answer may be incomplete."
                    )
                    st.json(source_errors)


elif page == "Context Inspector":
    selected_conversation_id = st.session_state.get("conversation_id")
    selected_conversation_title = next(
        (
            conversation.get("title")
            for conversation in st.session_state.conversation_list
            if conversation.get("id") == selected_conversation_id
        ),
        None,
    )

    render_context_inspector(
        st,
        api_url=API_URL,
        conversation_id=selected_conversation_id,
        conversation_title=selected_conversation_title,
        # This Streamlit app does not claim a session, so its
        # conversations are shared (legacy behavior) and no claim is
        # sent. If one is configured, it is forwarded unchanged.
        session_id=st.session_state.get("session_id"),
        state=st.session_state,
    )


elif page == "Knowledge Graph":
    st.title("Knowledge Graph")
    st.caption("Explore entities and their relationships.")
    st.info("Graph entity and relationship browsing will be connected next.")


elif page == "Diagnostics":
    st.title("Diagnostics")

    try:
        st.subheader("API health")
        st.json(api_get("/health"))

        capabilities = api_get("/capabilities")
        st.subheader("Backend capabilities")

        services = capabilities.get("services", [])

        if services:
            for service in services:
                label = service.get("label", service.get("key", "Service"))
                if service.get("configured", False):
                    st.success(f"{label}: configured")
                else:
                    st.warning(f"{label}: not configured")
        else:
            st.json(capabilities)

    except requests.RequestException as exc:
        st.error(f"Could not load diagnostics: {exc}")
