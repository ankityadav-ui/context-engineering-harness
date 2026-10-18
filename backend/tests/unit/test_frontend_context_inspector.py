"""Tests for the Streamlit Context Inspector.

A recording stand-in for Streamlit verifies that opening the page makes
no backend call, that an explicit inspect action uses the real preview
contract, and that empty results, source errors, and returned data are
displayed differently.
"""

import sys
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"

if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

import api_client  # noqa: E402
import context_inspector  # noqa: E402


class _FakeExpander:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class FakeStreamlit:
    """Records the calls the inspector makes."""

    def __init__(
        self,
        *,
        button_pressed=False,
        query="",
        memory_user_id="",
        checkbox_value=True,
    ):
        self.button_pressed = button_pressed
        self.query = query
        self.memory_user_id = memory_user_id
        self.checkbox_value = checkbox_value
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

    def text(self, text, **kwargs):
        self._record("text", text, **kwargs)

    def code(self, text, **kwargs):
        self._record("code", text, **kwargs)

    def info(self, text, **kwargs):
        self._record("info", text, **kwargs)

    def warning(self, text, **kwargs):
        self._record("warning", text, **kwargs)

    def error(self, text, **kwargs):
        self._record("error", text, **kwargs)

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

    def button(self, label, **kwargs):
        self._record("button", label, **kwargs)
        return self.button_pressed

    def expander(self, label, **kwargs):
        self._record("expander", label, **kwargs)
        return _FakeExpander(label)

    def all_text(self):
        return "\n".join(
            str(call["args"][0])
            for call in self.calls
            if call["args"] and isinstance(call["args"][0], str)
        )

    def named(self, name):
        return [
            call
            for call in self.calls
            if call["name"] == name
        ]


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
                    {
                        "role": "user",
                        "content": "Remember the Falcon codename.",
                    },
                    {
                        "role": "assistant",
                        "content": "Noted.",
                    },
                ],
            },
            "memory": [
                {
                    "memory_type": "preference",
                    "content": "Prefers concise answers.",
                    "importance": 5,
                    "score": 7.0,
                    "matched_keywords": ["concise"],
                }
            ],
            "rag": [
                {
                    "content_type": "text",
                    "content": "The return window is 30 days.",
                    "document_id": "doc-1",
                    "page_number": 2,
                    "score": 0.87,
                }
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
        "system_prompt": {"using_default": False},
        "formatted_context": (
            "CURRENT QUESTION:\nWhat is the policy?\n\n"
            "RELEVANT DOCUMENT EVIDENCE:\n[1] TYPE=text; "
            "document_id=doc-1, page=2\nThe return window is 30 days."
        ),
    }


def empty_preview(**overrides):
    preview = {
        "query": "Anything?",
        "sources": {
            "conversation": None,
            "memory": [],
            "rag": [],
            "knowledge_graph": [],
        },
        "source_errors": [],
        "image_references": 0,
        "system_prompt": {"using_default": True},
        "formatted_context": "CURRENT QUESTION:\nAnything?",
    }
    preview.update(overrides)
    return preview


def test_opening_the_inspector_does_not_call_the_backend():
    st = FakeStreamlit(button_pressed=False)
    state = {}

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError(
            "Opening the inspector must not fetch or generate anything."
        )

    context_inspector.render_page(
        st,
        api_url="http://api",
        conversation_id=None,
        state=state,
        fetch=fail_if_called,
    )

    assert "context_inspector_result" not in state
    assert "Assembled context" not in st.all_text()


def test_inspect_action_uses_real_preview_contract_and_displays_sources():
    st = FakeStreamlit(
        button_pressed=True,
        query="  What is the policy?  ",
        memory_user_id="  user-1  ",
    )
    state = {}
    calls = []

    def fake_fetch(query, **kwargs):
        calls.append({"query": query, **kwargs})
        return full_preview()

    context_inspector.render_page(
        st,
        api_url="http://api",
        conversation_id="conversation-1",
        conversation_title="Policy chat",
        session_id="session-a",
        state=state,
        fetch=fake_fetch,
    )

    # Query is trimmed; the selected conversation and its session claim
    # are sent to the real endpoint.
    assert calls == [
        {
            "query": "What is the policy?",
            "conversation_id": "conversation-1",
            "user_id": "user-1",
            "session_id": "session-a",
            "api_url": "http://api",
        }
    ]

    text = st.all_text()

    assert "Selected conversation: Policy chat" in text
    assert "Older messages summarized." in text
    assert "Remember the Falcon codename." in text
    assert "Prefers concise answers." in text
    assert "document `doc-1`" in text
    assert "page 2" in text
    assert "score 0.87" in text
    assert "Acme --RELATES_TO--> Policy (outgoing)" in text
    assert "Conversation custom instructions are in effect" in text

    # The formatted context is available only inside its own expander.
    code_calls = st.named("code")

    assert len(code_calls) == 1
    assert "CURRENT QUESTION:" in code_calls[0]["args"][0]
    assert "The return window is 30 days." in code_calls[0]["args"][0]


def test_empty_results_are_not_presented_as_retrieved_context():
    st = FakeStreamlit(button_pressed=True, query="Anything?")
    state = {}

    context_inspector.render_page(
        st,
        api_url="http://api",
        conversation_id=None,
        state=state,
        fetch=lambda *_args, **_kwargs: empty_preview(),
    )

    text = st.all_text()

    assert "No long-term memories matched this query." not in text
    assert "No memory owner (user id) was supplied" in text
    assert "No conversation was selected" in text
    assert "No knowledge-graph relationships matched this question." in text
    # "No context provided" must never masquerade as retrieved context.
    assert "No context provided" not in text


def test_empty_formatted_context_is_labelled_as_empty():
    st = FakeStreamlit(button_pressed=True, query="Anything?")
    state = {}

    context_inspector.render_page(
        st,
        api_url="http://api",
        conversation_id=None,
        state=state,
        fetch=lambda *_args, **_kwargs: empty_preview(
            formatted_context=""
        ),
    )

    assert st.named("code") == []
    assert "This is an empty result, not a context payload." in st.all_text()


def test_source_error_is_shown_instead_of_an_empty_rag_result():
    st = FakeStreamlit(button_pressed=True, query="Anything?")
    state = {}

    preview = empty_preview(
        source_errors=[
            {"source": "rag", "error": "vector store unavailable"}
        ]
    )

    context_inspector.render_page(
        st,
        api_url="http://api",
        conversation_id=None,
        state=state,
        fetch=lambda *_args, **_kwargs: preview,
    )

    text = st.all_text()

    assert "Document retrieval failed: vector store unavailable" in text
    assert "No indexed document content matched this query." not in text
    assert "The backend reported retrieval errors for: rag" in text


def test_failed_preview_clears_stale_results():
    st = FakeStreamlit(button_pressed=True, query="New question")
    state = {
        "context_inspector_result": {
            "preview": full_preview(),
            "query": "Old question",
            "conversation_id": None,
            "user_id": None,
        }
    }

    def failing_fetch(*_args, **_kwargs):
        raise api_client.APIError("Backend unavailable", status_code=503)

    context_inspector.render_page(
        st,
        api_url="http://api",
        conversation_id=None,
        state=state,
        fetch=failing_fetch,
    )

    assert state["context_inspector_result"] is None
    assert "Could not load the context preview: Backend unavailable" in (
        st.all_text()
    )
    assert "Old question" not in st.all_text()


def test_source_state_distinguishes_error_empty_and_not_requested():
    preview = empty_preview(
        source_errors=[{"source": "memory", "error": "database down"}]
    )

    assert (
        context_inspector.source_state(preview, "memory") == "error"
    )
    assert context_inspector.source_state(preview, "rag") == "empty"
    assert (
        context_inspector.source_state(
            preview,
            "conversation",
            requested=False,
        )
        == "not_requested"
    )
