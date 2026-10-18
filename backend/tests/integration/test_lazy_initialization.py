"""Initialization tests.

These verify two things:
1. External-service clients raise clear errors only when a feature is
   actually used, instead of breaking module import.
2. The API (including /health) can start with the unrelated external
   services unconfigured.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# tests/integration/test_lazy_initialization.py -> backend/
BACKEND_DIR = Path(__file__).resolve().parents[2]


def test_pinecone_index_requires_configuration_on_use(monkeypatch):
    from app.rag import pinecone_service

    monkeypatch.setattr(pinecone_service, "_client", None)
    monkeypatch.setattr(pinecone_service, "_index", None)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="PINECONE_API_KEY"):
        pinecone_service.get_index()

    with pytest.raises(RuntimeError, match="PINECONE_API_KEY"):
        pinecone_service.describe_index_model()


def test_gemini_client_requires_configuration_on_use(monkeypatch):
    from app.rag import embedding_service

    monkeypatch.setattr(embedding_service, "_client", None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        embedding_service.get_client()


def test_database_engine_requires_configuration_on_use(monkeypatch):
    from app.context_engine import database

    monkeypatch.setattr(database, "_engine", None)
    monkeypatch.setattr(database, "_session_factory", None)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        database.get_engine()


def test_importing_function_modules_does_not_connect():
    """Importing the service modules must succeed without any config."""

    # Fresh import in a subprocess with the .env values cleared, so
    # load_dotenv() cannot repopulate them (empty values are respected).
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND_DIR)
    env["LLM_PROVIDER"] = "ollama"
    for key in (
        "PINECONE_API_KEY",
        "GEMINI_API_KEY",
        "DATABASE_URL",
        "NEO4J_URI",
        "NEO4J_PASSWORD",
        "NEO4J_USERNAME",
        "NEO4J_DATABASE",
        "OPENROUTER_API_KEY",
    ):
        env[key] = ""

    code = (
        "import app.main\n"
        "from fastapi.testclient import TestClient\n"
        "with TestClient(app.main.app) as client:\n"
        "    response = client.get('/health')\n"
        "assert response.status_code == 200, response.text\n"
        "assert response.json() == {'status': 'ok'}, response.json()\n"
        "print('IMPORT_AND_HEALTH_OK')\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(BACKEND_DIR),
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )
    assert "IMPORT_AND_HEALTH_OK" in result.stdout
