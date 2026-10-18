"""Add conversations.session_id for the session-ownership model.

The application has no authentication system. Following the existing
pattern used by the memory API (a caller-supplied ``user_id`` claim),
conversations can now carry an optional caller-supplied ``session_id``.
A conversation without a session_id stays shared/legacy-visible; a
conversation with one is only listed, continued, cleared, or deleted by
callers presenting the same session_id.

Revision ID: 0002_session_id
Revises: 0001_baseline
Create Date: 2026-10-17

"""
from alembic import op
import sqlalchemy as sa

revision = "0002_session_id"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # Inspector-guarded so the revision is safe to apply on top of the
    # pre-Alembic development schema and is idempotent when re-run
    # against a database whose tables were created by init_db.py after
    # this column already existed.
    if "conversations" in inspector.get_table_names():
        columns = {
            column["name"]
            for column in inspector.get_columns("conversations")
        }

        if "session_id" not in columns:
            op.add_column(
                "conversations",
                sa.Column(
                    "session_id",
                    sa.String(length=64),
                    nullable=True,
                ),
            )

    existing_indexes = {
        index["name"]
        for index in inspector.get_indexes("conversations")
    }

    if "ix_conversations_session_id" not in existing_indexes:
        op.create_index(
            "ix_conversations_session_id",
            "conversations",
            ["session_id"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    existing_indexes = {
        index["name"]
        for index in inspector.get_indexes("conversations")
    }

    if "ix_conversations_session_id" in existing_indexes:
        op.drop_index(
            "ix_conversations_session_id",
            table_name="conversations",
        )

    if "conversations" in inspector.get_table_names():
        columns = {
            column["name"]
            for column in inspector.get_columns("conversations")
        }

        if "session_id" in columns:
            op.drop_column("conversations", "session_id")
