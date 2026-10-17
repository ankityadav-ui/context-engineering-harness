"""Shared fixtures for the automated test suite.

Everything here uses in-memory storage only. No test contacts
PostgreSQL, Pinecone, Neo4j, Gemini, or an LLM.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.context_engine.database import Base

# Import the models so every table is registered on Base.metadata.
import app.context_engine.models  # noqa: F401

from app.main import app, get_db


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
def api_client(sqlite_session_factory):
    """A TestClient whose storage dependency is an in-memory database."""

    def override_get_db():
        db = sqlite_session_factory()

        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
