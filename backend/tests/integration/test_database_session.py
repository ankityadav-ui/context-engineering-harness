"""Database session and shutdown reliability.

No PostgreSQL connection is opened: the session factory and engine are
replaced with in-memory stand-ins.
"""

import pytest

from sqlalchemy.exc import SQLAlchemyError

import app.context_engine.database as database
from app.main import get_db


class FakeSession:
    """Minimal session double recording its lifecycle calls."""

    def __init__(self):
        self.rolled_back = False
        self.closed = False

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def _install_session(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(database, "SessionLocal", lambda: session)
    return session


def test_session_is_closed_when_the_request_succeeds(monkeypatch):
    session = _install_session(monkeypatch)

    dependency = get_db()
    yielded = next(dependency)

    assert yielded is session
    assert session.rolled_back is False

    with pytest.raises(StopIteration):
        next(dependency)

    assert session.closed is True


def test_session_is_rolled_back_and_closed_when_the_request_fails(
    monkeypatch,
):
    session = _install_session(monkeypatch)

    dependency = get_db()
    next(dependency)

    with pytest.raises(RuntimeError):
        dependency.throw(RuntimeError("route failed"))

    # A failed request must not leak an open transaction.
    assert session.rolled_back is True
    assert session.closed is True


def test_a_failing_rollback_does_not_mask_the_original_error(
    monkeypatch,
):
    class BrokenSession(FakeSession):
        def rollback(self):
            raise SQLAlchemyError("rollback failed")

    session = BrokenSession()
    monkeypatch.setattr(database, "SessionLocal", lambda: session)

    dependency = get_db()
    next(dependency)

    with pytest.raises(RuntimeError, match="route failed"):
        dependency.throw(RuntimeError("route failed"))

    assert session.closed is True


def test_unconfigured_database_returns_a_service_error(monkeypatch):
    def missing_url():
        raise RuntimeError("DATABASE_URL is not configured.")

    monkeypatch.setattr(database, "SessionLocal", missing_url)

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as error:
        next(get_db())

    assert error.value.status_code == 503
    assert "DATABASE_URL" in error.value.detail


def test_dispose_engine_is_safe_without_a_configured_database(monkeypatch):
    disposed = []

    class FakeEngine:
        def dispose(self):
            disposed.append(True)

    monkeypatch.setattr(database, "_engine", FakeEngine())
    monkeypatch.setattr(database, "_session_factory", object())

    database.dispose_engine()

    assert disposed == [True]
    assert database._engine is None
    assert database._session_factory is None

    # Calling it again with nothing configured must not raise.
    database.dispose_engine()
