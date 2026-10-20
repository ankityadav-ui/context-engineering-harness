"""Tests for the Streamlit Diagnostics page.

The page must show only what the backend reported, must not run a preview
when merely opened, and must never surface raw exception text.
"""

import sys
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"

if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

import api_client  # noqa: E402
import diagnostics  # noqa: E402


class _FakeExpander:
    def __init__(self, st, label):
        self.st = st
        self.label = label

    def __enter__(self):
        return self.st

    def __exit__(self, *_exc):
        return False


class FakeStreamlit:
    def __init__(
        self,
        *,
        button_pressed=False,
        query="",
        memory_user_id="",
        checkbox_value=True,
        selectbox_index=1,
    ):
        self.button_pressed = button_pressed
        self.query = query
        self.memory_user_id = memory_user_id
        self.checkbox_value = checkbox_value
        self.selectbox_index = selectbox_index
        self.calls = []

    def _record(self, name, *args, **kwargs):
        self.calls.append({"name": name, "args": args, "kwargs": kwargs})

    def title(self, text, **kwargs):
        self._record("title", text, **kwargs)

    def subheader(self, text, **kwargs):
        self._record("subheader", text, **kwargs)

    def caption(self, text, **kwargs):
        self._record("caption", text, **kwargs)

    def markdown(self, text, **kwargs):
        self._record("markdown", text, **kwargs)

    def code(self, text, **kwargs):
        self._record("code", text, **kwargs)

    def json(self, data, **kwargs):
        self._record("json", data, **kwargs)

    def success(self, text, **kwargs):
        self._record("success", text, **kwargs)

    def warning(self, text, **kwargs):
        self._record("warning", text, **kwargs)

    def error(self, text, **kwargs):
        self._record("error", text, **kwargs)

    def info(self, text, **kwargs):
        self._record("info", text, **kwargs)

    def divider(self):
        self._record("divider")

    def text_input(self, label, **kwargs):
        self._record("text_input", label, **kwargs)

        if label.startswith("Question"):
            return self.query

        return self.memory_user_id

    def checkbox(self, label, **kwargs):
        self._record("checkbox", label, **kwargs)
        return self.checkbox_value

    def selectbox(self, label, options, index=0, **kwargs):
        self._record("selectbox", label, options, index=index, **kwargs)
        return options[index]

    def button(self, label, **kwargs):
        self._record("button", label, **kwargs)
        return self.button_pressed

    def expander(self, label, **kwargs):
        self._record("expander", label, **kwargs)
        return _FakeExpander(self, label)

    def all_text(self):
        return "\n".join(
            str(call["args"][0])
            for call in self.calls
            if call["args"] and isinstance(call["args"][0], str)
        )

    def named(self, name):
        return [call for call in self.calls if call["name"] == name]


def health_ok(_api_url=None):
    return {"status": "ok"}


def capabilities_ok(_api_url=None):
    return {
        "services": [
            {"key": "database", "label": "PostgreSQL", "configured": True}
        ]
    }


def full_preview():
    return {
        "query": "What is the policy?",
        "sources": {
            "conversation": {
                "total_messages": 8,
                "older_count": 2,
                "recent_count": 6,
                "summary": "Older messages summarized.",
                "recent_messages": [
                    {"role": "user", "content": "Remember the codename."}
                ],
            },
            "memory": [
                {
                    "memory_type": "preference",
                    "content": "Prefers concise answers.",
                }
            ],
            "rag": [
                {
                    "content_type": "text",
                    "content": "The return window is 30 days.",
                    "document_id": "doc-1",
                    "page_number": 2,
                    "score": 0.87,
                },
                {
                    "content_type": "text",
                    "content": "Returns need a receipt.",
                    "document_id": "doc-1",
                    "page_number": 3,
                    "score": 0.72,
                },
            ],
            "knowledge_graph": [
                {
                    "matched_entity": "Acme",
                    "source": "Acme",
                    "relationship": "RELATES_TO",
                    "target": "Policy",
                    "direction": "outgoing",
                }
            ],
        },
        "source_errors": [],
        "image_references": 0,
        "system_prompt": {"using_default": True},
        "formatted_context": (
            "CURRENT QUESTION:\nWhat is the policy?\n\n"
            "RELEVANT DOCUMENT EVIDENCE:\nThe return window is 30 days."
        ),
        "diagnostics": {
            "status": "ok",
            "duration_ms": 42,
            "conversation_requested": True,
            "memory_requested": True,
            "document_scoped": False,
            "query_chars": 21,
            "errors": [],
            "sources": {
                "short_term": {
                    "status": "ok",
                    "requested": True,
                    "messages": 6,
                    "elapsed_ms": 4,
                },
                "conversation_summary": {
                    "status": "ok",
                    "requested": True,
                    "chars": 26,
                    "older_messages": 2,
                    "elapsed_ms": 4,
                },
                "memory": {
                    "status": "ok",
                    "requested": True,
                    "items": 1,
                    "elapsed_ms": 2,
                },
                "rag": {
                    "status": "ok",
                    "requested": True,
                    "chunks": 2,
                    "documents": 1,
                    "document_ids": ["doc-1"],
                    "content_types": ["text"],
                    "elapsed_ms": 20,
                },
                "knowledge_graph": {
                    "status": "ok",
                    "requested": True,
                    "relationships": 1,
                    "entities": 1,
                    "elapsed_ms": 8,
                },
            },
            "assembly": {
                "empty": False,
                "truncated": False,
                "chars": 500,
                "sections": [
                    {"name": "conversation_summary", "chars": 26},
                    {"name": "recent_conversation", "chars": 60},
                    {"name": "memory", "chars": 40},
                    {"name": "document_evidence", "chars": 120},
                    {"name": "knowledge_graph", "chars": 45},
                ],
            },
        },
    }


def empty_preview():
    preview = full_preview()
    preview["sources"]["memory"] = []
    preview["sources"]["rag"] = []
    preview["sources"]["knowledge_graph"] = []
    preview["sources"]["conversation"] = None
    preview["formatted_context"] = "CURRENT QUESTION:\nWhat is the policy?"
    preview["diagnostics"] = {
        "status": "ok",
        "duration_ms": 5,
        "conversation_requested": False,
        "memory_requested": False,
        "document_scoped": False,
        "query_chars": 21,
        "errors": [
            {
                "source": "knowledge_graph",
                "category": "unavailable",
                "message": "The knowledge graph could not be queried.",
            }
        ],
        "sources": {
            "short_term": {"status": "not_requested", "requested": False},
            "conversation_summary": {
                "status": "not_requested",
                "requested": False,
            },
            "memory": {"status": "not_requested", "requested": False},
            "rag": {"status": "empty", "requested": True, "chunks": 0},
            "knowledge_graph": {"status": "error", "requested": True},
        },
        "assembly": {
            "empty": True,
            "truncated": False,
            "chars": 38,
            "sections": [],
        },
    }
    return preview


def render(st, *, fetch=None, state=None, health=health_ok, capabilities=capabilities_ok):
    diagnostics.render_page(
        st,
        api_url="http://api",
        conversation_id="conversation-1",
        conversation_title="Policy chat",
        session_id="session-a",
        state={} if state is None else state,
        fetch=fetch or (lambda *_args, **_kwargs: full_preview()),
        health=health,
        capabilities=capabilities,
    )


def test_opening_the_page_does_not_fetch_a_preview():
    st = FakeStreamlit(button_pressed=False)
    state = {}

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("Opening Diagnostics must not retrieve context.")

    render(st, fetch=fail_if_called, state=state)

    assert "No diagnostics run yet" in st.all_text()
    assert "diagnostics_result" not in state
    # The pre-existing health check still renders.
    assert "PostgreSQL: configured" in st.all_text()


def test_run_renders_overview_statuses_counts_and_assembly():
    st = FakeStreamlit(button_pressed=True, query="What is the policy?")
    state = {}

    render(st, state=state)

    text = st.all_text()

    assert state["diagnostics_result"]["query"] == "What is the policy?"

    # Overview
    assert "Conversation: Policy chat" in text
    assert "Request status: succeeded" in text
    assert "Assembly duration: 42 ms" in text
    assert "Context assembly: succeeded with evidence" in text

    # Source statuses and counts
    assert "✅ **Short-term conversation** — returned content" in text
    assert "6 recent message(s) in the window." in text
    assert "✅ **Conversation summary** — returned content" in text
    assert "26 summary character(s)" in text
    assert "✅ **Long-term memory** — returned content" in text
    assert "1 relevant memory retrieved." in text
    assert "✅ **Document retrieval (RAG)** — returned content" in text
    assert "2 chunk(s) from 1 document(s)." in text
    assert "✅ **Knowledge graph** — returned content" in text
    assert "1 relationship(s) from 1 entit(y/ies)." in text

    # Retrieval metadata, never chunk text.
    assert "document doc-1" in text
    assert "score 0.87" in text
    assert "The return window is 30 days." in text  # only in the expander
    assert st.named("expander")[0]["args"][0].startswith(
        "🧾 Raw assembled context"
    )

    # Assembly
    assert "Final context: 500 character(s)" in text
    assert "Document evidence — 120 character(s)" in text

    assert "No source errors were reported for this run." in text


def test_empty_not_requested_and_failed_sources_are_explained():
    st = FakeStreamlit(button_pressed=True, query="What is the policy?")
    state = {}

    render(st, state=state, fetch=lambda *_a, **_k: empty_preview())

    text = st.all_text()

    assert "➖ **Short-term conversation** — not requested" in text
    assert "➖ **Conversation summary** — not requested" in text
    assert "➖ **Long-term memory** — not requested" in text
    assert "⚪ **Document retrieval (RAG)** — completed with no results" in text
    assert "❌ **Knowledge graph** — failed" in text
    assert "This source failed; other sources still contributed." in text
    assert "The knowledge graph could not be queried." in text
    assert "(category: unavailable)" in text
    assert "No context source contributed evidence." in text
    assert "No evidence sections were assembled." in text


def test_failed_run_clears_stale_results_and_reports_the_error():
    st = FakeStreamlit(button_pressed=True, query="New question")
    state = {
        "diagnostics_result": {
            "preview": full_preview(),
            "query": "Old question",
            "conversation_id": "conversation-1",
            "user_id": None,
            "document_id": None,
        }
    }

    def failing_fetch(*_args, **_kwargs):
        raise api_client.APIError("Backend unavailable", status_code=503)

    render(st, fetch=failing_fetch, state=state)

    assert state["diagnostics_result"] is None
    assert "Could not run diagnostics: Backend unavailable" in st.all_text()
    assert "What is the policy?" not in st.all_text()


def test_legacy_preview_is_labeled_and_never_echoes_raw_errors():
    legacy = full_preview()
    legacy.pop("diagnostics")
    legacy["source_errors"] = [
        {"source": "rag", "error": "password=hunter2 connection refused"}
    ]

    st = FakeStreamlit(button_pressed=True, query="What is the policy?")
    state = {}

    render(st, state=state, fetch=lambda *_a, **_k: legacy)

    text = st.all_text()

    assert "did not return structured diagnostics" in text
    assert "❌ **Document retrieval (RAG)** — failed" in text
    assert "The backend reported failures for: Document retrieval" in text
    assert "hunter2" not in text
    assert "password" not in text.lower()


def test_health_failure_does_not_block_a_diagnostics_run():
    def broken_health(_api_url=None):
        raise RuntimeError("health probe failed")

    def broken_capabilities(_api_url=None):
        raise RuntimeError("capabilities probe failed")

    st = FakeStreamlit(button_pressed=True, query="What is the policy?")

    render(
        st,
        state={},
        health=broken_health,
        capabilities=broken_capabilities,
    )

    text = st.all_text()

    assert "Could not load API health: health probe failed" in text
    assert "Could not load backend capabilities: capabilities probe failed" in text
    # The run still happened afterwards.
    assert "Request status: succeeded" in text


def test_empty_question_warns_without_calling_the_backend():
    st = FakeStreamlit(button_pressed=True, query="   ")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("A blank question must not be sent.")

    render(st, fetch=fail_if_called, state={})

    assert "Enter a question first." in st.all_text()
    assert "No diagnostics run yet" in st.all_text()
