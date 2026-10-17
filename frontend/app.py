"""Streamlit shell for the Masked RAG Chatbot.

Every page talks only to the FastAPI backend through ``api_client``;
Streamlit holds no app logic of its own (no LLM, RAG, memory, graph, or
masking logic is duplicated here).

Run with:

    streamlit run frontend/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports whether launched from the repo root or from frontend/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

import api_client
from ui_theme import apply_theme, chip, state_error

st.set_page_config(
    page_title="Masked RAG Chatbot",
    page_icon="🛡️",
    layout="wide",
)

apply_theme()

# --------------------------------------------------
# Sidebar: backend status and navigation
# --------------------------------------------------

with st.sidebar:
    st.markdown("### Masked RAG Chatbot")

    with st.expander("Backend status", expanded=True):
        try:
            _health = api_client.health()
            chip("FastAPI online", "success")
        except Exception as error:
            chip("FastAPI unreachable", "error")
            st.caption(
                "Start the backend first: "
                "`uvicorn app.main:app --port 8000`"
            )
            st.caption(f"Details: {error}")

        try:
            _caps = api_client.capabilities()
            _model = api_client.model_info()

            chip(f"Model: {_model['model']}")

            for service in _caps.get("services", []):
                name = service.get("label", service.get("key"))
                status = (
                    "• configured"
                    if service.get("configured")
                    else "• not configured"
                )
                st.caption(f"{name}: {status}")

        except Exception as error:
            st.caption(f"Capabilities unavailable: {error}")

    st.divider()

    page = st.radio(
        "Navigate",
        [
            "Chat",
            "Documents & RAG",
            "Document Masking",
            "Context Inspector",
        ],
        label_visibility="collapsed",
    )

    st.caption("Backend URL comes from BACKEND_API_URL in .env.")


# --------------------------------------------------
# Page routing
# --------------------------------------------------

if page == "Chat":
    from pages.chat_page import render

elif page == "Documents & RAG":
    from pages.documents_page import render

elif page == "Document Masking":
    from pages.masking_page import render

else:
    from pages.context_page import render

try:
    render()
except Exception as error:
    state_error(error, fallback="Page failed")
