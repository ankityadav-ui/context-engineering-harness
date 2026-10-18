"""Tests for POST /chat/stream (server-sent event streaming).

The active Ollama client is replaced with an in-memory double, so no
Ollama server, no cloud endpoint, and no paid API is ever contacted.

Covered here:
- multiple chunks arriving progressively over the wire,
- the final assembled response and the terminal `done` event,
- provider failure mid-stream (`error` event, no `done`),
- invalid/empty provider chunks being skipped,
- client cancellation closing the upstream stream,
- provider/model reporting in the completion event,
- the custom system prompt and the built-in security instructions
  reaching the provider as a real `system` message,
- no regression in the non-streaming POST /chat contract.
"""

import json

import pytest

import app.llm as llm_module
from app.system_prompt import SECURITY_INSTRUCTIONS


# ---------------------------------------------------------------- fakes


class FakeStreamProvider:
    """In-memory Ollama chat client double.

    Mirrors the real SDK contract: ``chat()`` returns one object whose
    ``message.content`` holds the text, and with ``stream=True`` returns
    an iterable of those objects. Every call records the full ``messages``
    list so tests can inspect the system and user roles separately.
    """

    def __init__(self, chunks, model="fake-model", fail_after=None):
        self.chunks = list(chunks)
        self.model = model
        self.fail_after = fail_after
        # One entry per chat() call: the messages list that was sent.
        self.seen_messages: list[list[dict]] = []
        self.closed = False

    @staticmethod
    def _response(content):
        response = type("ChatResponse", (), {})()
        response.message = type("Message", (), {})()
        response.message.content = content
        return response

    def generate_text(self, message):
        return "".join(
            chunk for chunk in self.chunks if isinstance(chunk, str)
        )

    def _stream(self):
        try:
            for index, chunk in enumerate(self.chunks):
                if self.fail_after is not None and index == self.fail_after:
                    raise RuntimeError("upstream exploded")

                yield self._response(chunk)
        finally:
            # Mirrors the real SDK: runs on completion, failure, and
            # client cancellation (GeneratorExit).
            self.closed = True

    def chat(self, **kwargs):
        messages = kwargs.get("messages", [])
        self.seen_messages.append(messages)

        if kwargs.get("stream"):
            return self._stream()

        return self._response(
            self.generate_text(messages[-1].get("content", ""))
        )

    # -- assertions helpers -------------------------------------------

    @property
    def last_messages(self):
        return self.seen_messages[-1]

    @property
    def last_system_text(self):
        return "".join(
            message.get("content", "")
            for message in self.last_messages
            if message.get("role") == "system"
        )

    @property
    def last_user_text(self):
        return "".join(
            message.get("content", "")
            for message in self.last_messages
            if message.get("role") == "user"
        )


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    """Keep tests independent from the local Ollama environment."""

    monkeypatch.delenv("OLLAMA_VISION_MODEL", raising=False)


@pytest.fixture()
def fake_provider(monkeypatch):
    """Install a fake streaming provider on the shared llm module."""

    def install(chunks, **kwargs):
        provider = FakeStreamProvider(chunks, **kwargs)
        monkeypatch.setattr(llm_module, "_get_client", lambda: provider)
        llm_module.reset_client()
        return provider

    return install


# --------------------------------------------------------------- helpers


def read_sse_events(response):
    """Parse SSE frames into a list of (event, data) tuples."""
    events = []
    event_name = None
    data_lines = []

    for raw_line in response.iter_lines():
        line = raw_line.rstrip("\r")

        if line == "":
            if data_lines or event_name:
                events.append(
                    (event_name, json.loads("".join(data_lines) or "{}"))
                )
            event_name = None
            data_lines = []
            continue

        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())

    if data_lines or event_name:
        events.append((event_name, json.loads("".join(data_lines) or "{}")))

    return events


def stream_events(client, message, **kwargs):
    # Extra keyword arguments (e.g. conversation_id) are sent as part
    # of the JSON request body.
    with client.stream(
        "POST",
        "/chat/stream",
        json={"message": message, **kwargs},
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        return read_sse_events(response)


def split_events(events, name):
    return [data for event, data in events if event == name]


# ------------------------------------------------------------- the tests


def test_stream_delivers_multiple_progressive_chunks(api_client, fake_provider):
    provider = fake_provider(["Hel", "lo ", "world"])

    events = stream_events(api_client, "Hi there")

    chunks = split_events(events, "chunk")

    assert [chunk["delta"] for chunk in chunks] == ["Hel", "lo ", "world"]
    # Chunks are separate frames: they cannot arrive as one payload.
    assert len(chunks) == 3
    assert len(provider.seen_messages) >= 1
    assert "Hi there" in provider.last_user_text


def test_stream_done_reports_assembled_response_and_provider(
    api_client, fake_provider
):
    provider = fake_provider(["A", "B", "C"])

    events = stream_events(api_client, "Assemble me")

    done = split_events(events, "done")
    errors = split_events(events, "error")

    assert len(done) + len(errors) >= 1
    assert len(done) == 1

    # The assembled response equals the concatenation of the deltas, and
    # the event reports the Ollama backend that served this stream.
    assert done[0]["response"] == "ABC"
    assert done[0]["provider"] == "ollama"
    assert done[0]["model"] == llm_module.get_text_model()
    assert done[0]["system_prompt_active"] is False

    # Exactly one terminal event: no error event follows a done event.
    assert split_events(events, "error") == []
    assert len(provider.seen_messages) >= 1
    assert "Assemble me" in provider.last_user_text


def test_stream_terminates_with_exactly_one_terminal_event(
    api_client, fake_provider
):
    provider = fake_provider(["one"])

    events = stream_events(api_client, "terminal check")

    terminal = [
        event
        for event, _ in events
        if event in {"done", "error"}
    ]

    assert len(terminal) == 1
    assert terminal[0] in {"done", "error"}
    assert len(provider.seen_messages) >= 1
    assert "terminal check" in provider.last_user_text


def test_stream_provider_failure_mid_stream_sends_error_event(
    api_client, fake_provider
):
    provider = fake_provider(
        ["first chunk", "second chunk", "third chunk"],
        fail_after=1,
    )

    events = stream_events(api_client, "This will break")

    chunks = split_events(events, "chunk")
    errors = split_events(events, "error")

    # The first chunk reached the client before the failure.
    assert [chunk["delta"] for chunk in chunks] == ["first chunk"]
    assert len(errors) == 1

    # The error is the module's own translated, secret-free message: the
    # upstream exception text is not echoed to the client.
    assert "failed" in errors[0]["detail"]
    assert "upstream exploded" not in errors[0]["detail"]

    # The partial text is reported so the UI can keep it.
    assert errors[0]["partial"] == "first chunk"
    # No completion event after a failure.
    assert split_events(events, "done") == []
    # The provider's cleanup ran despite the failure.
    assert provider.closed is True

    # The provider saw the user message that triggered the stream.
    assert len(provider.seen_messages) >= 1
    assert "This will break" in provider.last_user_text


def test_stream_skips_invalid_provider_chunks(api_client, fake_provider):
    provider = fake_provider(["valid-1", "", None, 123, "valid-2"])

    events = stream_events(api_client, "invalid chunks")

    chunks = split_events(events, "chunk")

    assert [chunk["delta"] for chunk in chunks] == [
        "valid-1",
        "valid-2",
    ]
    assert split_events(events, "done")[0]["response"] == "valid-1valid-2"
    assert len(provider.seen_messages) >= 1
    assert "invalid chunks" in provider.last_user_text


def test_stream_empty_provider_output_sends_error(api_client, fake_provider):
    provider = fake_provider([None, "", 42])

    events = stream_events(api_client, "nothing at all")

    assert split_events(events, "done") == []
    errors = split_events(events, "error")

    assert len(errors) == 1
    assert "empty response" in errors[0]["detail"]
    assert len(provider.seen_messages) >= 1
    assert "nothing at all" in provider.last_user_text


def test_stream_provider_without_streaming_support_reports_error(
    api_client, monkeypatch
):
    class NonStreamingProvider:
        model = "no-stream"

        def chat(self, **kwargs):
            if kwargs.get("stream"):
                raise TypeError("streaming not supported")

            response = type("ChatResponse", (), {})()
            response.message = type("Message", (), {})()
            response.message.content = "should not be used"
            return response

        def list(self):
            return [type("Model", (), {"name": "models/gemma4:31b"})()]

        def show(self, model=None):
            return type("Show", (), {"capabilities": []})()

        def close(self):
            pass

    monkeypatch.setattr(
        llm_module, "_get_client", lambda: NonStreamingProvider()
    )

    events = stream_events(api_client, "Hi")

    assert split_events(events, "chunk") == []
    errors = split_events(events, "error")

    assert len(errors) == 1
    assert "does not support streaming" in errors[0]["detail"]


def test_stream_applies_saved_system_prompt_to_provider(
    api_client, fake_provider, isolated_assembly
):
    provider = fake_provider(["bonjour"])

    # The custom prompt belongs to one conversation and is saved
    # against it in the database.
    created = api_client.post(
        "/conversations", json={"title": "French"}
    ).json()

    saved = api_client.put(
        f"/conversations/{created['id']}/system-prompt",
        json={"custom_prompt": "Answer in French."},
    )
    assert saved.status_code == 200

    events = stream_events(
        api_client,
        "Hello?",
        conversation_id=created["id"],
    )

    # The custom instructions reached the provider in the system role,
    # together with the built-in security instructions, while the user
    # message stayed user content.
    assert "Answer in French." in provider.last_system_text
    assert SECURITY_INSTRUCTIONS in provider.last_system_text
    assert "Hello?" in provider.last_user_text
    assert "Answer in French." not in provider.last_user_text

    done = split_events(events, "done")
    assert done[0]["system_prompt_active"] is True


def test_stream_without_custom_prompt_keeps_builtin_security(
    api_client, fake_provider
):
    provider = fake_provider(["plain"])

    events = stream_events(api_client, "Raw message only")

    done = split_events(events, "done")
    assert len(done) == 1
    assert done[0]["response"] == "plain"
    assert done[0]["system_prompt_active"] is False

    # The built-in security instructions are always present, and the
    # user's message never travels in the system role.
    assert SECURITY_INSTRUCTIONS in provider.last_system_text
    assert "Raw message only" in provider.last_user_text
    assert "Raw message only" not in provider.last_system_text


def test_stream_rejects_whitespace_only_message(api_client):
    response = api_client.post("/chat/stream", json={"message": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "Message cannot be empty."


def test_stream_rejects_missing_message(api_client):
    response = api_client.post("/chat/stream", json={})

    # The same ChatRequest validation as POST /chat.
    assert response.status_code == 422


def test_closing_the_stream_generator_closes_the_provider_stream(
    fake_provider,
):
    provider = fake_provider(["one", "two", "three"])

    stream = llm_module.stream_response("cancel me")

    assert next(stream) == "one"
    assert provider.closed is False

    # Closing the iterator is what the endpoint does when the HTTP
    # client disconnects.
    try:
        stream.close()
    except Exception:
        pass

    assert provider.closed is True


def test_client_disconnect_closes_the_upstream_stream(
    api_client, fake_provider
):
    provider = fake_provider(["chunk-1", "chunk-2", "chunk-3"])

    with api_client.stream(
        "POST",
        "/chat/stream",
        json={"message": "stop early"},
    ) as response:
        for line in response.iter_lines():
            if line.startswith("data:"):
                break
        # Leaving the block aborts the response without reading the
        # rest of the stream.

    # The endpoint closes its generator, which closes the provider's
    # stream. Give the ASGI layer a moment to unwind.
    import time

    for _ in range(50):
        if provider.closed:
            break
        time.sleep(0.02)

    assert provider.closed is True


def test_non_streaming_chat_contract_is_unchanged(
    api_client, fake_provider
):
    provider = fake_provider(["plain answer"])

    response = api_client.post("/chat", json={"message": "Hello"})

    assert response.status_code == 200
    body = response.json()

    # The original JSON contract: same keys, JSON content type.
    assert set(body.keys()) == {"response", "system_prompt_active"}
    assert body["response"] == "plain answer"
    assert "text/event-stream" not in response.headers["content-type"]
    assert len(provider.seen_messages) >= 1
    assert "Hello" in provider.last_user_text
    assert SECURITY_INSTRUCTIONS in provider.last_system_text
