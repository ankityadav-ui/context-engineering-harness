import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


# Find the project root, where the shared .env file lives.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Load the existing project's environment variables.
load_dotenv(PROJECT_ROOT / ".env")

# Base class for all SQLAlchemy database models.
# This does not require a configured connection URL.
Base = declarative_base()


# Hold the engine and session factory until storage is actually needed.
# Creating them at import time would stop unrelated API routes from
# starting when DATABASE_URL is not configured.
_engine = None
_session_factory = None


def get_database_url() -> str:
    """Return the configured database URL or raise a clear error."""

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not configured. Set it in the root "
            ".env file to use conversation and memory storage."
        )

    return database_url


def is_configured() -> bool:
    """Report whether DATABASE_URL is set, without connecting."""

    return bool(os.getenv("DATABASE_URL"))


def get_engine():
    """Return the SQLAlchemy engine, creating it on first use."""

    global _engine

    if _engine is None:
        _engine = create_engine(
            get_database_url(),
            pool_pre_ping=True,
        )

    return _engine


def get_session_factory():
    """Return the session factory, creating it on first use."""

    global _session_factory

    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            autocommit=False,
        )

    return _session_factory


def SessionLocal():
    """Create a new database session bound to the configured engine."""

    return get_session_factory()()


def dispose_engine() -> None:
    """Dispose the connection pool and forget the cached objects.

    Called on application shutdown so pooled PostgreSQL connections are
    returned instead of being dropped at process exit. Safe to call when
    no database was ever configured.
    """

    global _engine, _session_factory

    if _engine is not None:
        _engine.dispose()

    _engine = None
    _session_factory = None


def __getattr__(name):
    # Preserve the previous module-level name for callers that import
    # `engine` directly, while still connecting lazily.
    if name == "engine":
        return get_engine()

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
