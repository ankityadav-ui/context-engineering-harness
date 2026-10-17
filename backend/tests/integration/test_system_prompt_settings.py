"""Tests for persisted system-prompt settings and real prompt composition.

The settings file is redirected into a temporary directory and the LLM
provider call is replaced with a stub, so no file outside tmp_path is
written and no paid API is contacted.
"""

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
import app.llm as llm_module
from app.main import app
from app.system_prompt import (
    SECURITY_INSTRUCTIONS,
    build_ask_prompt,
    build_chat_prompt,
)


@pytest.fixture()
def settings_file(tmp_path, monkeypatch):
    """Point the settings store at a temporary file."""
    path = tmp_path / "system_prompt.json"
    monkeypatch.setenv("CEH_SETTINGS_FILE", str(path))
    return path


# ---------------------------------------------------------- settings routes


def test_settings_default_reports_no_custom_prompt(settings_file):
    with TestClient(app) as client:
        response = client.get("/settings/system-prompt")

    assert response.status_code == 200

    body = response.json()

    assert body["custom_prompt"] == ""
    assert body["using_default"] is True
    # The built-in security instructions are reported read-only.
    assert body["security_instructions"] == SECURITY_INSTRUCTIONS


def test_settings_put_persists_across_requests(settings_file):
    with TestClient(app) as client:
        saved = client.put(
            "/settings/system-prompt",
            json={"custom_prompt": "Answer like a pirate."},
        )

        assert saved.status_code == 200
        assert saved.json()["custom_prompt"] == "Answer like a pirate."
        assert saved.json()["using_default"] is False

        # A later request (same process) reads the persisted value.
        loaded = client.get("/settings/system-prompt")

    assert loaded.json()["custom_prompt"] == "Answer like a pirate."


def test_settings_put_empty_string_resets_to_default(settings_file):
    with TestClient(app) as client:
        client.put(
            "/settings/system-prompt",
            json={"custom_prompt": "Custom instruction."},
        )

        reset = client.put(
            "/settings/system-prompt",
            json={"custom_prompt": "   "},
        )

        loaded = client.get("/settings/system-prompt")

    assert reset.status_code == 200
    assert reset.json()["using_default"] is True
    assert loaded.json()["custom_prompt"] == ""


def test_settings_put_rejects_overlong_prompt(settings_file):
    with TestClient(app) as client:
        response = client.put(
            "/settings/system-prompt",
            json={"custom_prompt": "x" * 4001},
        )

    assert response.status_code == 422


# ------------------------------------------------------ prompt composition


def test_chat_prompt_defaults_to_builtin_security_instructions(settings_file):
    prompt, custom_applied = build_chat_prompt("What is 2+2?")

    assert custom_applied is False
    assert prompt.startswith(SECURITY_INSTRUCTIONS)
    assert "User message:\nWhat is 2+2?" in prompt


def test_chat_prompt_includes_custom_and_security_instructions(
    settings_file,
):
    with TestClient(app) as client:
        client.put(
            "/settings/system-prompt",
            json={"custom_prompt": "Answer in French."},
        )

    prompt, custom_applied = build_chat_prompt("Hello?")

    assert custom_applied is True
    assert prompt.startswith("Answer in French.")
    # Security instructions survive a custom prompt.
    assert SECURITY_INSTRUCTIONS in prompt
    assert "Hello?" in prompt


def test_ask_prompt_always_keeps_security_instructions(settings_file):
    prompt, custom_applied = build_ask_prompt("[context here]")

    assert custom_applied is False
    assert SECURITY_INSTRUCTIONS in prompt
    assert "[context here]" in prompt

    with TestClient(app) as client:
        client.put(
            "/settings/system-prompt",
            json={"custom_prompt": "Be brief."},
        )

    prompt, custom_applied = build_ask_prompt("[context here]")

    assert custom_applied is True
    assert prompt.startswith("Be brief.")
    assert SECURITY_INSTRUCTIONS in prompt


# ------------------------------------------------------------- /chat wiring


def test_chat_route_applies_persisted_prompt_to_llm_request(
    settings_file,
    monkeypatch,
):
    captured = {"prompts": []}

    class FakeClient:
        def __init__(self):
            self.seen_messages = []

        def chat(self, **kwargs):
            captured["prompts"].extend(
                message.get("content", "")
                for message in kwargs.get("messages", [])
            )
            response = type("ChatResponse", (), {})()
            response.message = type("Message", (), {})()
            response.message.content = "stubbed answer"
            return response

        def list(self):
            return [type("Model", (), {"name": "models/gemma4:31b"})()]

        def show(self, model=None):
            return type("Show", (), {"capabilities": ["vision"]})()

        def close(self):
            pass

    fake_client = FakeClient()

    def fake_get_client():
        return fake_client

    monkeypatch.setattr(llm_module, "generate_response", lambda prompt, system=None: "stubbed answer")
    monkeypatch.setattr(llm_module, "_get_client", fake_get_client)
    monkeypatch.setattr(llm_module, "reset_client", lambda: None)

    with TestClient(app) as client:
        client.put(
            "/settings/system-prompt",
            json={"custom_prompt": "Answer in French."},
        )

        response = client.post("/chat", json={"message": "Hello?"})

    assert response.status_code == 200
    assert response.json()["response"] == "stubbed answer"
    assert response.json()["system_prompt_active"] is True

    # The prompt actually handed to the provider carries the custom
    # instruction, the built-in security text, and the user message.
    joined = "\n".join(captured["prompts"])
    assert joined.startswith("Answer in French.")
    assert SECURITY_INSTRUCTIONS in joined
    assert "Hello?" in joined


def test_chat_route_keeps_builtin_security_without_custom_prompt(
    settings_file,
    monkeypatch,
):
    captured = {"prompts": []}

    class FakeClient:
        def __init__(self):
            self.seen_messages = []

        def chat(self, **kwargs):
            captured["prompts"].extend(
                message.get("content", "")
                for message in kwargs.get("messages", [])
            )
            response = type("ChatResponse", (), {})()
            response.message = type("Message", (), {})()
            response.message.content = "stubbed answer"
            return response

        def list(self):
            return [type("Model", (), {"name": "models/gemma4:31b"})()]

        def show(self, model=None):
            return type("Show", (), {"capabilities": ["vision"]})()

        def close(self):
            pass

    fake_client = FakeClient()

    def fake_get_client():
        return fake_client

    monkeypatch.setattr(llm_module, "generate_response", lambda prompt, system=None: "stubbed answer")
    monkeypatch.setattr(llm_module, "_get_client", fake_get_client)
    monkeypatch.setattr(llm_module, "reset_client", lambda: None)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "Hello?"})

    assert response.status_code == 200
    assert response.json()["system_prompt_active"] is False

    joined = "\n".join(captured["prompts"])
    assert joined.startswith(SECURITY_INSTRUCTIONS)
    assert "Hello?" in joined
