"""Shared fixtures for the automated test suite.

Everything here uses in-memory storage only. No test contacts
PostgreSQL, Pinecone, Neo4j, Gemini, or an LLM.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.context_engine import database
from app.context_engine.database import Base

# Import the models so every table is registered on Base.metadata.
import app.context_engine.models  # noqa: F401

from app.main import app, get_db, get_db_optional


@pytest.fixture(autouse=True)
def mock_gemini_configuration(monkeypatch):
    """Keep the test suite independent of local credentials and .env."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GEMINI_TEXT_MODEL", "gemini-test-model")
    monkeypatch.setenv("INGESTION_JOB_RECOVERY_ENABLED", "false")


@pytest.fixture(autouse=True)
def isolate_ingestion_job_store(monkeypatch, sqlite_session_factory):
    """Keep durable job tests in the shared in-memory test database."""
    from app.context_engine import ingestion_jobs
    from app import main

    monkeypatch.setattr(ingestion_jobs, "SessionLocal", sqlite_session_factory)

    class InlineExecutor:
        def submit(self, callback):
            callback()

    monkeypatch.setattr(main, "_INGESTION_EXECUTOR", InlineExecutor())


@pytest.fixture()
def sqlite_session_factory():
    """A session factory bound to one shared in-memory SQLite database."""

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    Base.metadata.create_all(bind=engine)

    factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )

    try:
        yield factory
    finally:
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture()
def api_client(sqlite_session_factory, monkeypatch):
    """A TestClient whose storage dependency is an in-memory database."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GEMINI_TEXT_MODEL", "gemini-test-model")

    def override_get_db():
        db = sqlite_session_factory()

        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    # The chat routes persist only when a conversation_id is supplied;
    # both dependencies must share the same in-memory database.
    app.dependency_overrides[get_db_optional] = override_get_db

    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


@pytest.fixture()
def isolated_assembly(monkeypatch, sqlite_session_factory):
    """Keep assemble_context off live services, still exercising the
    real conversation/summary code paths against SQLite."""

    monkeypatch.setattr(
        database,
        "SessionLocal",
        sqlite_session_factory,
    )

    # Pinecone, Neo4j, and memory lookups stay off live services; the
    # conversation/summary branches keep running for real.
    import app.context_engine.engine as engine

    monkeypatch.setattr(engine, "retrieve_context", lambda **_k: [])
    monkeypatch.setattr(engine, "_get_graph_context", lambda _q: [])

    # The summarizer calls the LLM; keep it deterministic.
    import app.context_engine.summarizer as summarizer

    monkeypatch.setattr(
        summarizer,
        "generate_response",
        lambda *_args, **_kwargs: "Older messages summarized.",
    )

    return monkeypatch
