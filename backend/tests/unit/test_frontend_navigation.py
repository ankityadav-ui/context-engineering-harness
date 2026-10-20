"""Frontend navigation tests for the Streamlit app.

The knowledge graph is an internal backend service now, so the UI must
expose exactly Chat, Documents, Context Inspector, and Diagnostics —
with no Knowledge Graph page and no Graph Explorer.

These tests execute the real ``frontend/app.py`` with a recording
stand-in for Streamlit and no backend reachable, so the assertions are
about the actual navigation code rather than a copy of it.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import requests

FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"

if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))


SUPPORTED_PAGES = ["Chat", "Documents", "Context Inspector", "Diagnostics"]

REMOVED_PAGES = ["Knowledge Graph", "Graph Explorer"]


class _Context:
    """Stand-in for ``with st.sidebar:`` and similar containers."""

    def __init__(self, st):
        self.st = st

    def __enter__(self):
        return self.st

    def __exit__(self, *_exc):
        return False


class _SessionState(dict):
    """Session state supporting both item and attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


class FakeStreamlit:
    """Records every Streamlit call the app makes while rendering."""

    def __init__(self, *, active_page=None, pressed_keys=()):
        self.session_state = _SessionState()
        if active_page is not None:
            self.session_state["active_page"] = active_page
        self.pressed_keys = set(pressed_keys)
        self.calls = []

    def _record(self, name, *args, **kwargs):
        self.calls.append({"name": name, "args": args, "kwargs": kwargs})

    # --- containers ----------------------------------------------------

    @property
    def sidebar(self):
        return _Context(self)

    def container(self, **kwargs):
        return _Context(self)

    def expander(self, label, **kwargs):
        return _Context(self)

    def columns(self, spec, **kwargs):
        if isinstance(spec, int):
            count = spec
        else:
            count = len(spec)
        return [_Context(self) for _ in range(count)]

    def spinner(self, *args, **kwargs):
        return _Context(self)

    def fragment(self, **_kwargs):
        return lambda function: function

    def chat_message(self, *args, **kwargs):
        return _Context(self)

    # --- display -------------------------------------------------------

    def set_page_config(self, **kwargs):
        self._record("set_page_config", **kwargs)

    def title(self, text, **kwargs):
        self._record("title", text, **kwargs)

    def subheader(self, text, **kwargs):
        self._record("subheader", text, **kwargs)

    def header(self, text, **kwargs):
        self._record("header", text, **kwargs)

    def markdown(self, text, **kwargs):
        self._record("markdown", text, **kwargs)

    def caption(self, text, **kwargs):
        self._record("caption", text, **kwargs)

    def text(self, text, **kwargs):
        self._record("text", text, **kwargs)

    def code(self, text, **kwargs):
        self._record("code", text, **kwargs)

    def json(self, value, **kwargs):
        self._record("json", value, **kwargs)

    def write(self, value, **kwargs):
        self._record("write", value, **kwargs)

    def success(self, text, **kwargs):
        self._record("success", text, **kwargs)

    def info(self, text, **kwargs):
        self._record("info", text, **kwargs)

    def warning(self, text, **kwargs):
        self._record("warning", text, **kwargs)

    def error(self, text, **kwargs):
        self._record("error", text, **kwargs)

    def progress(self, value, **kwargs):
        self._record("progress", value, **kwargs)

    def divider(self, **kwargs):
        self._record("divider", **kwargs)

    def rerun(self, **kwargs):
        self._record("rerun", **kwargs)

    # --- widgets -------------------------------------------------------

    def button(self, label, **kwargs):
        self._record("button", label, **kwargs)
        key = kwargs.get("key")
        return label in self.pressed_keys or key in self.pressed_keys

    def toggle(self, label, **kwargs):
        self._record("toggle", label, **kwargs)
        return False

    def checkbox(self, label, **kwargs):
        self._record("checkbox", label, **kwargs)
        return False

    def selectbox(self, label, options, **kwargs):
        self._record("selectbox", label, options, **kwargs)
        return options[0] if options else None

    def text_input(self, label, **kwargs):
        self._record("text_input", label, **kwargs)
        return ""

    def text_area(self, label, **kwargs):
        self._record("text_area", label, **kwargs)
        return ""

    def slider(self, label, *args, **kwargs):
        self._record("slider", label, *args, **kwargs)
        return kwargs.get("value", args[2] if len(args) > 2 else None)

    def file_uploader(self, label, **kwargs):
        self._record("file_uploader", label, **kwargs)
        return None

    def chat_input(self, label, **kwargs):
        self._record("chat_input", label, **kwargs)
        return None

    # --- inspection helpers ---------------------------------------------

    def records(self, name):
        return [call for call in self.calls if call["name"] == name]

    def texts(self, name):
        return [str(call["args"][0]) for call in self.records(name)]

    def all_text(self):
        texts = []
        for call in self.calls:
            texts.extend(str(arg) for arg in call.get("args", ()))
            texts.extend(str(value) for value in call.get("kwargs", {}).values())
        return "\n".join(texts)

    def nav_labels(self):
        labels = []
        for call in self.records("button"):
            key = str(call["kwargs"].get("key", ""))
            if key.startswith("nav_"):
                labels.append(str(call["args"][0]))
        return labels


def _run_app(monkeypatch, *, active_page=None, pressed_keys=()):
    """Execute frontend/app.py once with a fake Streamlit and no backend."""

    st = FakeStreamlit(active_page=active_page, pressed_keys=pressed_keys)

    monkeypatch.setitem(sys.modules, "streamlit", st)

    # The tests never contact a live backend: every HTTP call fails
    # immediately, which the app catches per its degradation paths.
    def _offline(*_args, **_kwargs):
        raise requests.ConnectionError("backend unreachable in tests")

    for method in ("get", "post", "put", "delete"):
        monkeypatch.setattr(requests, method, _offline)

    spec = importlib.util.spec_from_file_location(
        "frontend_app_under_test",
        FRONTEND_DIR / "app.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return st


def test_navigation_exposes_exactly_the_supported_pages(monkeypatch):
    st = _run_app(monkeypatch)

    labels = st.nav_labels()

    for page_name in SUPPORTED_PAGES:
        assert any(page_name in label for label in labels), (
            f"{page_name} is missing from the navigation"
        )

    for removed in REMOVED_PAGES:
        assert all(removed not in label for label in labels), (
            f"{removed} is still exposed in the navigation"
        )


def test_no_graph_explorer_content_is_rendered_anywhere(monkeypatch):
    st = _run_app(monkeypatch)

    rendered = st.all_text()

    assert "Knowledge Graph" not in rendered
    assert "Graph Explorer" not in rendered
    assert "graph_explorer" not in rendered


def test_chat_page_still_renders(monkeypatch):
    st = _run_app(monkeypatch, active_page="Chat")

    assert "Chat" in st.texts("title")


def test_documents_page_still_renders(monkeypatch):
    st = _run_app(monkeypatch, active_page="Documents")

    assert "Documents" in st.texts("title")
    assert "Upload a PDF" in st.texts("subheader")
    uploader = st.records("file_uploader")[0]
    assert uploader["kwargs"]["max_upload_size"] == 25


def test_chat_attachment_size_matches_backend_limit(monkeypatch):
    st = _run_app(monkeypatch, active_page="Chat")

    chat_input = st.records("chat_input")[0]
    assert chat_input["kwargs"]["max_upload_size"] == 25


def test_context_inspector_page_still_renders(monkeypatch):
    st = _run_app(monkeypatch, active_page="Context Inspector")

    assert "Context Inspector" in st.texts("title")


def test_diagnostics_page_still_renders(monkeypatch):
    st = _run_app(monkeypatch, active_page="Diagnostics")

    # The page runs its API-health section before the (offline) backend
    # call fails, proving the page itself is reachable and intact.
    assert "API health" in st.texts("subheader")


def test_stale_knowledge_graph_session_falls_back_to_chat(monkeypatch):
    """A session saved before the page was removed must not blank out."""

    st = _run_app(monkeypatch, active_page="Knowledge Graph")

    assert st.session_state["active_page"] == "Chat"
    assert "Chat" in st.texts("title")
    assert "Knowledge Graph" not in st.all_text()


def test_pressing_a_nav_button_switches_page(monkeypatch):
    st = _run_app(monkeypatch, pressed_keys={"nav_Documents"})

    assert st.session_state["active_page"] == "Documents"
    assert "Documents" in st.texts("title")


def test_graph_explorer_module_and_page_are_gone():
    assert not (FRONTEND_DIR / "graph_explorer.py").exists()

    source = (FRONTEND_DIR / "app.py").read_text(encoding="utf-8")

    assert "graph_explorer" not in source
    assert '("Knowledge Graph"' not in source
    assert 'page == "Knowledge Graph"' not in source
