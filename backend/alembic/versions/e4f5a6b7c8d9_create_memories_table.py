"""create memories table

Revision ID: e4f5a6b7c8d9
Revises: d2c7ae4f70db
Create Date: 2026-08-30 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e4f5a6b7c8d9'
down_revision: Union[str, None] = 'c3d4e5f6a7b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'memories',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('user_id', sa.String(255), nullable=False, index=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('memory_type', sa.String(50), nullable=False, server_default='fact'),
        sa.Column('case_id', sa.Integer(), sa.ForeignKey('cases.id'), nullable=True, index=True),
        sa.Column('importance', sa.Float(), nullable=False, server_default='0.5'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table('memories')
