"""Chat page: conversation history + streaming replies from FastAPI."""

from __future__ import annotations

import uuid

import streamlit as st

import api_client
from ui_theme import chip, state_error

DEFAULT_USER_ID = "local-user"


def render() -> None:  # noqa: C901
    if st.session_state is not None:
        st.session_state.setdefault("ui_messages", [])
        st.session_state.setdefault("conversation_id", None)

    left, right = st.columns([1, 1])

    left.markdown("#### Chat")
    right.markdown(
        '<div class="ceh-muted">Answers stream from the real backend; '
        "no example responses are stored client-side.</div>",
        unsafe_allow_html=True,
    )

    # Conversation controls (persist messages server-side so the
    # context engine can summarize older turns).
    control_cols = st.columns(2)

    if control_cols[0].button("New conversation", use_container_width=True):
        try:
            created = api_client.create_conversation(
                title="Streamlit chat",
            )
            st.session_state.conversation_id = created["id"]
            st.session_state.ui_messages = []
            st.info(f"Started conversation {created['id'][:8]}…")
        except Exception as error:
            state_error(error, fallback="Could not create conversation")

    if st.session_state.conversation_id:
        control_cols[1].markdown(
            f'<span class="ceh-chip">Conversation: '
            f"{str(st.session_state.conversation_id)[:8]}…</span>",
            unsafe_allow_html=True,
        )

    # Message history (persisted in session; the backend keeps the
    # durable record when a conversation is started).
    for message in st.session_state.ui_messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "🛡️"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask something…")

    if not prompt:
        return

    # Remember the user's message immediately.
    st.session_state.ui_messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    if not st.session_state.conversation_id:
        try:
            created = api_client.create_conversation(
                title=prompt[:60],
            )
            st.session_state.conversation_id = created["id"]
        except Exception as error:
            state_error(
                error,
                fallback=(
                    "Conversation storage unavailable; the reply is "
                    "being streamed without context assembly"
                ),
            )

    # Stream the assistant answer.
    with st.chat_message("assistant", avatar="🛡️"):
        placeholder = st.empty()
        status = st.empty()
        status.caption("Waiting for the backend…")

        collected: list[str] = []
        failure: str | None = None

        try:
            for frame in api_client.chat_stream(prompt):
                if frame["type"] == "chunk":
                    collected.append(frame.get("delta", ""))
                    placeholder.markdown("".join(collected) + "▌")
                    status.caption("Streaming…")
                elif frame["type"] == "done":
                    placeholder.markdown(frame.get("response", "").join(["", ""]) or "".join(collected))
                    status.caption("Done.")
                elif frame["type"] == "error":
                    failure = frame.get("detail", "Unknown streaming error")
                    break
        except Exception as error:
            failure = str(error)

        if not collected and failure:
            placeholder.error(f"Chat failed: {failure}")
        elif failure:
            placeholder.markdown("".join(collected))
            st.warning(f"Stream ended early: {failure}")

    st.session_state.ui_messages.append(
        {"role": "assistant", "content": "".join(collected)}
    )
