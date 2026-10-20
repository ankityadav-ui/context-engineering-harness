"""Minimal Streamlit AppTest shim: runs a Streamlit script with Fake st
so the page can be exercised without importing Streamlit 1.60's heavy
AppTest. The real Streamlit UI calls are replaced here.
"""

from __future__ import annotations


class _FakeColumn:
    def __init__(self, st):
        self.st = st

    def __enter__(self):
        return self.st

    def __exit__(self, *_exc):
        return False


class FakeStreamlit:
    """Records Streamlit calls and replays widget values by key."""

    def __init__(self, selectbox_values=None, number_input_values=None):
        self.calls = []
        self.selectbox_values = dict(selectbox_values or {})
        self.number_input_values = dict(number_input_values or {})
        self.text_input_values = {}
        self.button_values = {}
        self.session_state = {}

    def _record(self, name, *args, **kwargs):
        self.calls.append({"name": name, "args": args, "kwargs": kwargs})

    # --- basic display -----------------------------------------------
    def title(self, text, **kwargs):
        self._record("title", text, kwargs)

    def subheader(self, text, **kwargs):
        self._record("subheader", text, kwargs)

    def caption(self, text, **kwargs):
        self._record("caption", text, kwargs)

    def markdown(self, text, **kwargs):
        self._record("markdown", text, kwargs)

    def info(self, text, **kwargs):
        self._record("info", text, kwargs)

    def error(self, text, **kwargs):
        self._record("error", text, kwargs)

    def warning(self, text, **kwargs):
        self._record("warning", text, kwargs)

    def success(self, text, **kwargs):
        self._record("success", text, kwargs)

    def graphviz_chart(self, dot, **kwargs):
        self._record("graphviz_chart", dot, kwargs)

    # --- widgets -----------------------------------------------------
    def text_input(self, label, **kwargs):
        self._record("text_input", label, kwargs)
        return self.text_input_values.get(label, "")

    def selectbox(self, label, options, index=0, key=None, **kwargs):
        self._record("selectbox", label, options, index, key, kwargs)
        if key in self.selectbox_values:
            return self.selectbox_values[key]
        return options[index] if options else None

    def number_input(self, label, min_value=None, max_value=None, value=0, step=None, key=None, **kwargs):
        self._record("number_input", label, min_value, max_value, value, step, key, kwargs)
        return self.number_input_values.get(key, value)

    def button(self, label, **kwargs):
        self._record("button", label, kwargs)
        key = kwargs.get("key")
        return key in self.button_values or label in self.button_values

    def columns(self, count, **kwargs):
        self._record("columns", count, kwargs)
        return [_FakeColumn(self) for _ in range(count)]

    # --- misc -----------------------------------------------------
    def progress(self, value, **kwargs):
        self._record("progress", value, kwargs)

    def empty(self):
        self._record("empty")
        return self

    def meter(self, label, value, **kwargs):
        self._record("meter", label, value, kwargs)

    def balloon(self):
        self._record("balloon")

    def egg(self):
        self._record("egg")

    def snow(self):
        self._record("snow")

    def color_picker(self, label, value="", **kwargs):
        self._record("color_picker", label, value, kwargs)
        return value

    def slider(self, label, min_value=0, max_value=100, value=50, **kwargs):
        self._record("slider", label, min_value, max_value, value, kwargs)
        return value

    def multiselect(self, label, options, default=None, **kwargs):
        self._record("multiselect", label, options, default, kwargs)
        return default or []

    def radio(self, label, options, default=0, **kwargs):
        self._record("radio", label, options, default, kwargs)
        return options[default]

    def checkboxes(self, label, options, default=False, **kwargs):
        self._record("checkboxes", label, options, default, kwargs)
        return {opt: bool(default) for opt in options}

    def file_uploader(self, label, type=None, accept=None, **kwargs):
        self._record("file_uploader", label, type, accept, kwargs)
        return None

    def download_button(self, label, data="", file_name="download.txt", mime="text/plain", **kwargs):
        self._record("download_button", label, data, file_name, mime, kwargs)
        return False

    def link_button(self, label, url, **kwargs):
        self._record("link_button", label, url, kwargs)
        return False

    def write(self, *args, **kwargs):
        self._record("write", args, kwargs)

    def markdown_(self, *args, **kwargs):
        self._record("markdown_", args, kwargs)

    def error_(self, *args, **kwargs):
        self._record("error_", args, kwargs)

    def json(self, *args, **kwargs):
        self._record("json", args, kwargs)

    def dataframe(self, data, **kwargs):
        self._record("dataframe", data, kwargs)

    def table(self, data, **kwargs):
        self._record("table", data, kwargs)

    def anonymize_handler(self):
        self._record("anonymize_handler")

    # --- session-state helpers ------------------------------------
    def session_state_get(self, key, default=None):
        return self.session_state.get(key, default)

    def session_state_set(self, key, value):
        self.session_state[key] = value

    def session_state_pop(self, key, default=None):
        return self.session_state.pop(key, default)
