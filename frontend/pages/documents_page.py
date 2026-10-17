"""Documents & RAG page: upload, index, search, and question answering."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

import api_client
from ui_theme import chip, state_error


def render() -> None:
    st.markdown("#### Documents & RAG")

    tab_upload, tab_search, tab_ask = st.tabs(
        ["Upload & index", "Semantic search", "Ask the documents"]
    )

    # --------------------------------------------------
    # Upload & index
    # --------------------------------------------------
    with tab_upload:
        uploaded = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            key="rag-upload",
        )

        mask_before_index = st.checkbox(
            "Mask sensitive content before indexing",
            value=False,
            help=(
                "When enabled, detected entities/patterns are replaced "
                "with placeholders before embedding. NER + regex checks "
                "cannot guarantee complete de-identification."
            ),
        )

        if uploaded is not None and st.button(
            "Extract, embed, and index",
            use_container_width=True,
        ):
            with st.status("Processing document…", expanded=False) as status_row:
                st.caption("Extraction, OCR, tables, embeddings, indexing…")

                temp_dir = tempfile.mkdtemp()
                temp_path = Path(temp_dir) / uploaded.name
                temp_path.write_bytes(uploaded.getvalue())

                try:
                    result = api_client.upload_document(
                        temp_path,
                        uploaded.name,
                        mask_before_index,
                    )
                    status_row.update(
                        label="Document indexed",
                        state="complete",
                        expanded=True,
                    )

                    st.toast("Indexing finished")
                    st.markdown("---")
                    st.caption("Ingestion report")
                    st.json(result, expanded=True)

                except api_client.APIError as error:
                    status_row.update(label="Failed", state="error")
                    state_error(error, fallback="Upload rejected")

                except Exception as error:
                    status_row.update(label="Failed", state="error")
                    state_error(error, fallback="Upload failed")

    # --------------------------------------------------
    # Semantic search
    # --------------------------------------------------
    with tab_search:
        try:
            listing = api_client.list_documents()
            documents = listing.get("documents", [])
        except Exception as error:
            documents = []
            state_error(error, fallback="Document listing failed")

        if not documents:
            st.info("No indexed documents available yet.")
        else:
            ids = [doc["document_id"] for doc in documents]
            selected = st.selectbox(
                "Scope (optional)",
                ["all documents", *ids],
            )

            query = st.text_input("Search query")
            top_k = st.slider("Top K", 1, 20, 5)

            if st.button("Search", use_container_width=True) and query.strip():
                try:
                    with st.spinner("Searching the vector index…"):
                        results = api_client.search_documents(
                            query,
                            top_k=top_k,
                            document_id=(
                                selected
                                if selected != "all documents"
                                else None
                            ),
                        )

                    matches = results.get("results", [])

                    if not matches:
                        st.warning("No matching content found.")
                    else:
                        st.success(f"{len(matches)} result(s)")

                        for match in matches:
                            meta = [
                                match.get("document_id", ""),
                                f"page {match.get('page_number', '?')}",
                                match.get("content_type", ""),
                            ]
                            st.markdown(
                                f"**{' · '.join(str(m) for m in meta if m)}**"
                            )
                            st.write(match.get("content", ""))
                            st.divider()

                except Exception as error:
                    state_error(error, fallback="Search failed")

    # --------------------------------------------------
    # Ask the documents (context engine path)
    # --------------------------------------------------
    with tab_ask:
        query = st.text_input("Question")
        top_k = st.slider("Context size (Top K)", 1, 20, 5, key="ask-k")

        if st.button(
            "Answer with sources",
            use_container_width=True,
            disabled=not query.strip(),
        ):
            try:
                with st.spinner("Assembling context and asking the model…"):
                    answer = api_client.ask_question(
                        query=query.strip(),
                        top_k=top_k,
                        user_id="local-user",
                    )

                st.markdown(answer.get("answer", ""))

                sources = answer.get("sources", {})
                rag_results = sources.get("rag", [])

                if rag_results:
                    st.markdown("**Sources**")
                    for result in rag_results:
                        st.caption(
                            f"{result.get('document_id', '')} · "
                            f"page {result.get('page_number', '?')} · "
                            f"{result.get('content_type', '')}"
                        )

                if answer.get("images_sent_to_llm"):
                    st.caption(
                        f"{answer['images_sent_to_llm']} extracted image "
                        "reference(s) were passed to the vision model."
                    )

                for source_error in answer.get("source_errors", []):
                    st.caption(
                        f"Source unavailable ({source_error.get('source')}): "
                        f"{source_error.get('error')}"
                    )

            except Exception as error:
                state_error(error, fallback="Question answering failed")
