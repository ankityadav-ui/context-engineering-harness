"""Baseline: the pre-existing application tables (conversations,
messages, conversation_summaries, memories).

This revision is inspector-guarded: every table and index is created
only when it does not already exist. That makes one migration chain
apply cleanly both to a fresh database and to the pre-Alembic
development schema that was created by init_db.py / metadata.create_all.

Tables are NOT dropped or altered here; changing an existing table's
shape belongs in a dedicated later revision (see 0002_session_id).

Revision ID: 0001_baseline
Revises:
Create Date: 2026-10-17

"""
from alembic import op
import sqlalchemy as sa

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def _create_missing_tables_and_indexes() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())

    # ------------------------------------------------------------
    # conversations
    # ------------------------------------------------------------
    if "conversations" not in existing:
        op.create_table(
            "conversations",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("title", sa.String(length=200), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
            ),
        )

    conversation_indexes = {
        index["name"]
        for index in inspector.get_indexes("conversations")
    }

    if "ix_conversations_session_id" not in conversation_indexes:
        # Added by 0002 for new databases; created here only when the
        # column exists (fresh databases created from this baseline get
        # the column in 0002 before this index is ever needed).
        columns = {
            column["name"]
            for column in inspector.get_columns("conversations")
        }

        if "session_id" in columns:
            op.create_index(
                "ix_conversations_session_id",
                "conversations",
                ["session_id"],
            )

    # ------------------------------------------------------------
    # messages
    # ------------------------------------------------------------
    if "messages" not in existing:
        op.create_table(
            "messages",
            sa.Column(
                "id",
                sa.Integer(),
                autoincrement=True,
                nullable=False,
                primary_key=True,
            ),
            sa.Column(
                "conversation_id",
                sa.String(length=36),
                nullable=False,
            ),
            sa.Column("role", sa.String(length=20), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
            ),
            sa.ForeignKeyConstraint(
                ["conversation_id"],
                ["conversations.id"],
                ondelete="CASCADE",
            ),
        )

    message_indexes = {
        index["name"]
        for index in inspector.get_indexes("messages")
    }

    if "ix_messages_conversation_id" not in message_indexes:
        op.create_index(
            "ix_messages_conversation_id",
            "messages",
            ["conversation_id"],
        )

    # ------------------------------------------------------------
    # conversation_summaries
    # ------------------------------------------------------------
    if "conversation_summaries" not in existing:
        op.create_table(
            "conversation_summaries",
            sa.Column(
                "id",
                sa.Integer(),
                autoincrement=True,
                nullable=False,
                primary_key=True,
            ),
            sa.Column(
                "conversation_id",
                sa.String(length=36),
                nullable=False,
            ),
            sa.Column("summary_text", sa.Text(), nullable=False),
            sa.Column(
                "covered_through_message_id",
                sa.Integer(),
                nullable=False,
            ),
            sa.Column("last_query", sa.Text(), nullable=False),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
            ),
            sa.ForeignKeyConstraint(
                ["conversation_id"],
                ["conversations.id"],
                ondelete="CASCADE",
            ),
            sa.UniqueConstraint(
                "conversation_id",
                name="uq_summary_conversation_id",
            ),
        )

    summary_indexes = {
        index["name"]
        for index in inspector.get_indexes("conversation_summaries")
    }

    if (
        "ix_conversation_summaries_conversation_id"
        not in summary_indexes
    ):
        op.create_index(
            "ix_conversation_summaries_conversation_id",
            "conversation_summaries",
            ["conversation_id"],
        )

    # ------------------------------------------------------------
    # memories
    # ------------------------------------------------------------
    if "memories" not in existing:
        op.create_table(
            "memories",
            sa.Column(
                "id",
                sa.Integer(),
                autoincrement=True,
                nullable=False,
                primary_key=True,
            ),
            sa.Column(
                "user_id",
                sa.String(length=100),
                nullable=False,
            ),
            sa.Column(
                "memory_type",
                sa.String(length=20),
                nullable=False,
            ),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column(
                "importance",
                sa.Integer(),
                nullable=False,
            ),
            sa.Column(
                "conversation_id",
                sa.String(length=36),
                nullable=True,
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
            ),
            sa.ForeignKeyConstraint(
                ["conversation_id"],
                ["conversations.id"],
                ondelete="SET NULL",
            ),
        )

    memory_indexes = {
        index["name"]
        for index in inspector.get_indexes("memories")
    }

    for index_name, column in (
        ("ix_memories_user_id", "user_id"),
        ("ix_memories_memory_type", "memory_type"),
        ("ix_memories_conversation_id", "conversation_id"),
    ):
        if index_name not in memory_indexes:
            op.create_index(index_name, "memories", [column])


def upgrade() -> None:
    _create_missing_tables_and_indexes()


def downgrade() -> None:
    # The baseline represents schema that predates migrations; it is
    # never dropped here. Reverting to an empty database is a manual,
    # reviewed operation.
    pass
