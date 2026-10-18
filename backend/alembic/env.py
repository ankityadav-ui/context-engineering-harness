"""Alembic environment for the Context Engineering Harness backend.

Design constraints:

- The database URL always comes from the application's existing
  environment configuration (``DATABASE_URL`` in the repository-root
  .env). No URL, credential, or secret is stored in alembic.ini.
- ``config.attributes["search_path"]`` may be set by the live
  integration tests to run every migration inside a disposable,
  clearly-named PostgreSQL schema instead of the application schema.
  Nothing outside that schema is touched when it is set.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

from app.context_engine import database as app_database

# Import every model so Base.metadata describes the full schema.
import app.context_engine.models  # noqa: F401

config = context.config

# Echo SQL only when explicitly requested; never log parameter values.
config.attributes.setdefault("sql_logger", None)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = app_database.Base.metadata


def _connectable():
    """Return an engine for the configured database.

    When the caller (the live integration tests) provided a
    ``search_path`` attribute, the engine scopes every session to that
    schema, so migrations and writes cannot reach the application's own
    tables.
    """

    search_path = config.attributes.get("search_path")

    if search_path:
        return create_engine(
            app_database.get_database_url(),
            connect_args={
                "options": f"-c search_path={search_path}"
            },
        )

    # The application's lazily-created engine (DATABASE_URL).
    return app_database.get_engine()


def run_migrations_online() -> None:
    connectable = _connectable()

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            # The baseline revision is intentionally inspector-guarded
            # (create-if-missing) so it can be applied to both a clean
            # database and the pre-Alembic development schema.
            render_as_batch=connection.dialect.name == "sqlite",
        )

        with connection.begin():
            context.run_migrations()


run_migrations_online()
