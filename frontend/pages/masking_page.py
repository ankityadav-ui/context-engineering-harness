"""Document Masking page.

Two verified backend behaviors live here:
1. Standalone text masking through POST /masking/text.
2. Masked ingestion: upload a specific document with masking ON; the
   extracted text (including OCR output) is masked and re-validated
   BEFORE embedding, the raw PDF is not stored, and the upload fails
   closed if the masked output does not pass safety validation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

import api_client
from ui_theme import chip, state_error


def render() -> None:
    st.markdown("#### Document Masking")

    st.caption(
        "NER (spaCy) and regex rules reduce sensitive data but cannot "
        "guarantee complete de-identification. Image pixels are never "
        "masked, and previously stored/indexed content is unchanged."
    )

    tab_text, tab_document = st.tabs(["Mask text", "Mask a document"])

    # --------------------------------------------------
    # Standalone text masking
    # --------------------------------------------------
    with tab_text:
        text = st.text_area(
            "Text to mask",
            placeholder="Paste the text that should be sanitized…",
            max_chars=50000,
        )

        if st.button("Mask text", use_container_width=True) and text.strip():
            try:
                with st.spinner("Detecting and masking entities…"):
                    result = api_client.mask_text(text)

                if result.get("safety_passed") is False:
                    chip("Safety check FAILED", "error")
                else:
                    chip("Safety check passed", "success")

                st.markdown(result.get("masked_text", ""))

                st.caption(
                    f"{result.get('detected_count', 0)} entity/entities "
                    f"in categories: "
                    f"{', '.join(result.get('detected_categories', []))}"
                )

                if result.get("safety_reasons"):
                    st.caption(
                        "Safety notes: "
                        + "; ".join(result["safety_reasons"])
                    )

            except Exception as error:
                state_error(error, fallback="Masking failed")

    # --------------------------------------------------
    # Masked document ingestion (mask ON/OFF control)
    # --------------------------------------------------
    with tab_document:
        st.caption(
            "This control changes actual backend processing: masked "
            "content is what gets embedded and indexed; unmasked "
            "uploads follow the normal workflow."
        )

        uploaded = st.file_uploader(
            "Upload a PDF to mask",
            type=["pdf"],
            key="masking-upload",
        )

        mask_on = st.toggle(
            "Mask the selected document before indexing",
            value=True,
        )

        if uploaded is not None and st.button(
            "Process document with masking "
            + ("ON" if mask_on else "OFF"),
            use_container_width=True,
        ):
            with st.spinner("Processing document…"):
                temp_dir = tempfile.mkdtemp()
                temp_path = Path(temp_dir) / uploaded.name

                try:
                    temp_path.write_bytes(uploaded.getvalue())

                    result = api_client.upload_document(
                        temp_path,
                        uploaded.name,
                        mask_on,
                    )

                    document_id = result.get("document_id", "")

                    st.markdown("**Result**")

                    st.caption(
                        f"Document {document_id[:8]}… · "
                        f"{result.get('document_id')}"
                    )

                    if mask_on:
                        if result.get("masking_safety_passed") is False:
                            chip("Masked output safety validation FAILED", "error")
                        else:
                            chip("Masked before indexing", "success")
                            chip(
                                f"{result.get('masked_entities', 0)} "
                                "sensitive value(s) replaced"
                            )
                        st.caption(
                            "The raw PDF was not stored; only sanitized "
                            "content was embedded and indexed. If the "
                            "original PDF was never kept, masked output "
                            "cannot be shown without the raw file."
                        )
                    else:
                        chip("Masking OFF — normal workflow", "accent")

                    st.caption(
                        f"content_units={result.get('content_units')} · "
                        f"ocr_pages={result.get('ocr_pages')} · "
                        f"tables={result.get('tables_extracted')} · "
                        f"images={result.get('images_extracted')}"
                    )

                    st.caption(
                        "Semantic masking output preview is not stored "
                        "server-side; rerun masking on the text when a "
                        "full preview is required."
                    )

                except api_client.APIError as error:
                    state_error(error, fallback="Masked upload rejected")

                except Exception as error:
                    state_error(error, fallback="Masked upload failed")
