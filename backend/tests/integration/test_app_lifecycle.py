"""Application lifespan: startup validation and shutdown cleanup.

The LLM startup check is replaced with a canned result, and every cleanup
target is recorded, so no external service is contacted.
"""

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app


@pytest.fixture()
def recorded_cleanup(monkeypatch):
    """Record shutdown cleanup calls instead of closing real resources."""

    calls = []

    monkeypatch.setattr(
        main_module,
        "validate_llm_config",
        lambda: {
            "configured": True,
            "model": "test-model",
            "backend": "local",
            "model_available": True,
            "vision_supported": None,
            "warning": None,
        },
    )
    monkeypatch.setattr(
        main_module.graph_service,
        "close_graph_connection",
        lambda: calls.append("graph"),
    )
    monkeypatch.setattr(
        main_module.context_database,
        "dispose_engine",
        lambda: calls.append("database"),
    )
    monkeypatch.setattr(
        main_module,
        "reset_llm_client",
        lambda: calls.append("llm"),
    )

    return calls


def test_shutdown_releases_every_external_resource(recorded_cleanup):
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    # Neo4j, PostgreSQL, and the Ollama client are all released.
    assert recorded_cleanup == ["graph", "database", "llm"]


def test_one_failing_cleanup_does_not_block_the_others(
    recorded_cleanup,
    monkeypatch,
):
    def failing_close():
        recorded_cleanup.append("graph")
        raise RuntimeError("Neo4j close failed")

    monkeypatch.setattr(
        main_module.graph_service,
        "close_graph_connection",
        failing_close,
    )

    # Shutdown must not raise because one close failed.
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert recorded_cleanup == ["graph", "database", "llm"]


def test_startup_warning_is_reported_without_blocking_startup(
    monkeypatch,
):
    monkeypatch.setattr(
        main_module,
        "validate_llm_config",
        lambda: {
            "configured": True,
            "model": "test-model",
            "backend": "local",
            "model_available": None,
            "vision_supported": None,
            "warning": "Model availability could not be checked: ConnectError",
        },
    )

    with TestClient(app) as client:
        response = client.get("/health")

    # A transient probe failure is a warning, not a startup failure.
    assert response.status_code == 200
