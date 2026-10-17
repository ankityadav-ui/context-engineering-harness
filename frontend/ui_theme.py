"""Shared UI helpers: design tokens and CSS for the Streamlit app.

The visual style is a clean, card-based layout that mirrors the
"Masked RAG Chatbot Webpage" design intent. The exact Stitch screen
could not be retrieved (no authenticated access from this environment),
so no fabricated brand assets are used anywhere below.
"""

from __future__ import annotations

import streamlit as st

ACCENT = "#4338CA"          # indigo
ACCENT_SOFT = "#EEF2FF"     # indigo 50
SURFACE = "#FFFFFF"
BORDER = "#E5E7EB"
TEXT = "#111827"
TEXT_MUTED = "#6B7280"
SUCCESS = "#059669"
DANGER = "#DC2626"
WARNING = "#D97706"


def apply_theme() -> None:
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

            html, body, [class*="css"] {{
                font-family: 'Inter', 'Segoe UI', sans-serif;
                color: {TEXT};
            }}

            .stApp {{
                background: {ACCENT_SOFT};
            }}

            section[data-testid="stSidebar"] > div {{
                background: {SURFACE};
            }}

            .stApp header {{
                background: transparent;
            }}

            div[data-testid="stMarkdownContainer"] p {{
                color: {TEXT};
            }}

            .ceh-card {{
                background: {SURFACE};
                border: 1px solid {BORDER};
                border-radius: 16px;
                padding: 20px;
                margin-bottom: 12px;
            }}

            .ceh-chip {{
                display: inline-block;
                padding: 4px 12px;
                border-radius: 999px;
                background: {ACCENT_SOFT};
                color: {ACCENT};
                font-size: 0.8rem;
                font-weight: 600;
                margin-right: 6px;
            }}

            .ceh-chip-error {{
                display: inline-block;
                padding: 4px 12px;
                border-radius: 999px;
                background: #FEE2E2;
                color: {DANGER};
                font-size: 0.8rem;
                font-weight: 600;
            }}

            .ceh-chip-success {{
                display: inline-block;
                padding: 4px 12px;
                border-radius: 999px;
                background: #D1FAE5;
                color: {SUCCESS};
                font-size: 0.8rem;
                font-weight: 600;
            }}

            .ceh-muted {{
                color: {TEXT_MUTED};
                font-size: 0.85rem;
            }}

            .stButton > button {{
                border-radius: 10px;
                border: 1px solid {BORDER};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def card(content: str, *, unsafe: bool = False) -> None:
    """Render the inner HTML of a card-based container."""

    st.markdown(
        f'<div class="ceh-card">{content}</div>',
        unsafe_allow_html=unsafe or True,
    )


def chip(text: str, kind: str = "accent") -> None:
    style = {
        "accent": "ceh-chip",
        "success": "ceh-chip-success",
        "error": "ceh-chip-error",
    }.get(kind, "ceh-chip")

    st.markdown(
        f'<span class="{style}">{text}</span>',
        unsafe_allow_html=True,
    )


def state_error(error: Exception, *, fallback: str = "Request failed") -> None:
    """Show a clean, honest error row (no fabricated fallback data)."""

    st.error(f"{fallback}: {error}")
