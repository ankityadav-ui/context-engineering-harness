"""add chat_mode to chat_sessions

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6g7
Create Date: 2026-08-30 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c3d4e5f6a7b8'
down_revision: Union[str, Sequence[str], None] = 'b2c3d4e5f6g7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add chat_mode column with default 'document' for existing rows
    op.add_column(
        'chat_sessions',
        sa.Column(
            'chat_mode',
            sa.String(length=20),
            nullable=False,
            server_default='document',
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('chat_sessions', 'chat_mode')
