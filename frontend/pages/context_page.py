"""Context Inspector page.

Displays what POST /context/preview actually assembles for a question:
recent messages, the older-messages summary, relevant long-term
memories, retrieved document chunks with sources, Neo4j entities and
relationships, and the selected prompt/skill identifiers. Only real
data returned by the backend is displayed.
"""

from __future__ import annotations

import streamlit as st

import api_client
from ui_theme import chip, state_error


def render() -> None:
    st.markdown("#### Context Inspector")

    left, right = st.columns(2)

    with left:
        query = st.text_input(
            "Question",
            placeholder="Show the exact context the model would receive…",
        )
        top_k = st.slider("Top K", 1, 20, 5, key="inspector-k")

        use_conversation = st.toggle(
            "Use an existing conversation",
            value=False,
        )

        conversation_id = None

        if use_conversation:
            try:
                conversations = api_client.list_conversations().get(
                    "conversations", []
                )

                if conversations:
                    options = [
                        f"{c['id'][:8]}… · {c.get('title') or 'untitled'}"
                        for c in conversations
                    ]
                    choice = st.selectbox("Conversation", options)
                    conversation_id = conversations[
                        options.index(choice)
                    ]["id"]
                else:
                    st.caption("No conversations exist yet.")
            except Exception as error:
                state_error(error, fallback="Conversations unavailable")

        user_id = st.text_input(
            "User id (long-term memory)",
            value="local-user",
        )

    with right:
        st.caption(
            "The preview calls the same assemble_context path as "
            "/documents/ask. When a conversation is attached the stored "
            "summary may be refreshed (one LLM call)."
        )

        if st.button(
            "Assemble context preview",
            use_container_width=True,
            disabled=not query.strip(),
        ):
            try:
                with st.spinner("Assembling context…"):
                    preview = api_client.context_preview(
                        query=query.strip(),
                        top_k=top_k,
                        conversation_id=conversation_id,
                        user_id=user_id.strip() or None,
                    )

                st.session_state["context_preview"] = preview

            except Exception as error:
                state_error(
                    error,
                    fallback="Context preview failed",
                )

    preview = st.session_state.get("context_preview")

    if not preview:
        st.info("Run a preview to inspect the assembled context.")
        _render_prompt_settings()
        return

    sources = preview.get("sources", {})

    st.markdown("---")

    # Prompt/skill identifiers + model info.
    system_prompt = preview.get("system_prompt", {})
    model = preview.get("model", {})

    chip(
        "Prompt: default"
        if system_prompt.get("using_default")
        else "Prompt: custom",
    )

    skill = system_prompt.get("active_skill")

    if skill:
        chip(f"Skill: {skill}")
    else:
        st.caption("No skill file selected (default prompt only).")

    chip(f"Model: {model.get('model', 'unknown')}")

    # Conversation summary and recent messages.
    conversation = sources.get("conversation")

    st.markdown("**Conversation**")

    if conversation:
        summary = conversation.get("summary") or ""

        if summary:
            st.markdown("*Summary of older messages*")
            st.markdown(summary)
        else:
            st.caption("No older messages to summarize.")

        recent = conversation.get("recent_messages") or []

        if recent:
            st.markdown(f"*Recent messages ({len(recent)})*")

            for message in recent:
                st.caption(
                    f"{message.get('role', '?')}: {message.get('content', '')}"
                )

        st.caption(
            f"Total messages: {conversation.get('total_messages', 0)} · "
            f"older: {conversation.get('older_count', 0)} · "
            f"recent: {conversation.get('recent_count', 0)}"
        )
    else:
        st.caption("No conversation attached.")

    # Long-term memories.
    st.markdown("**Long-term memories**")
    memories = sources.get("memory") or []

    if memories:
        for memory in memories:
            st.caption(
                f"[{memory.get('memory_type')}] {memory.get('content')} "
                f"(importance {memory.get('importance')})"
            )
    else:
        st.caption("No relevant memories retrieved.")

    # Document chunks.
    st.markdown("**Retrieved document chunks**")
    rag = sources.get("rag") or []

    if rag:
        for result in rag:
            st.caption(
                f"{result.get('document_id', '')} · "
                f"page {result.get('page_number', '?')} · "
                f"{result.get('content_type', '')}"
            )
            st.write(result.get("content", ""))
    else:
        st.caption("No document chunks matched.")

    # Knowledge graph.
    st.markdown("**Knowledge graph relationships**")
    graph = sources.get("knowledge_graph") or []

    if graph:
        for relationship in graph:
            st.caption(
                f"{relationship.get('matched_entity', '')}: "
                f"{relationship.get('source', '?')} "
                f"-[{relationship.get('relationship', '?')}]→ "
                f"{relationship.get('target', '?')}"
            )
    else:
        st.caption("No graph relationships matched.")

    for source_error in preview.get("source_errors") or []:
        st.caption(
            f"Source unavailable ({source_error.get('source')}): "
            f"{source_error.get('error')}"
        )

    st.divider()
    st.markdown("**Assembled context text (evidence returned to the model)**")
    st.code(preview.get("formatted_context") or "", language="markdown")

    _render_prompt_settings()


def _render_prompt_settings() -> None:
    """Show the persisted prompt/skill management controls."""

    st.markdown("---")
    st.markdown("**System prompt & skills**")

    try:
        prompt_settings = api_client.get_system_prompt()
        skills = api_client.list_skills()
    except Exception as error:
        state_error(error, fallback="Prompt settings unavailable")
        return

    active_skill = skills.get("active_skill")
    skill_options = [s["id"] for s in skills.get("skills", [])]

    choice = st.selectbox(
        "Active skill (trusted instruction file)",
        ["(none)", *skill_options],
        index=(
            (["(none)", *skill_options].index(active_skill))
            if active_skill in skill_options
            else 0
        ),
    )

    if st.button("Save skill selection", use_container_width=True):
        try:
            api_client.select_skill(
                "" if choice == "(none)" else choice
            )
            st.success("Skill selection saved.")
        except Exception as error:
            state_error(error, fallback="Selection not saved")

    st.caption(prompt_settings.get("security_instructions", "")[:400])
