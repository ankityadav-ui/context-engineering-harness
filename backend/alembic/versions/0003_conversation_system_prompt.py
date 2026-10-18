"""Add conversations.system_prompt for per-conversation custom prompts.

Each conversation can now carry its own custom system prompt. The
column is nullable: NULL means "use the default system instructions",
which is the state of every pre-existing conversation row, so all
existing conversations and messages are preserved untouched.

Following the 0002 pattern, the change is inspector-guarded so the
revision is safe to apply on top of schemas created by init_db.py /
metadata.create_all (where the column may already exist) and is
idempotent when re-run.

Revision ID: 0003_conversation_system_prompt
Revises: 0002_session_id
Create Date: 2026-10-18

"""
from alembic import op
import sqlalchemy as sa

revision = "0003_conversation_system_prompt"
down_revision = "0002_session_id"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "conversations" in inspector.get_table_names():
        columns = {
            column["name"]
            for column in inspector.get_columns("conversations")
        }

        if "system_prompt" not in columns:
            op.add_column(
                "conversations",
                sa.Column(
                    "system_prompt",
                    sa.Text(),
                    nullable=True,
                ),
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "conversations" in inspector.get_table_names():
        columns = {
            column["name"]
            for column in inspector.get_columns("conversations")
        }

        if "system_prompt" in columns:
            op.drop_column("conversations", "system_prompt")
