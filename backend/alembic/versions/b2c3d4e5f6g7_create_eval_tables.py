"""create eval queries and results tables

Revision ID: b2c3d4e5f6g7
Revises: a1b2c3d4e5f6
Create Date: 2026-08-30 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6g7'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    # Create eval_queries table
    op.create_table(
        'eval_queries',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('case_id', sa.Integer(), nullable=False),
        sa.Column('query', sa.Text(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('expected_document_ids', sa.Text(), nullable=False),
        sa.Column('expected_chunk_indices', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['case_id'], ['cases.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_eval_queries_id'),
        'eval_queries',
        ['id'],
        unique=False,
    )
    op.create_index(
        op.f('ix_eval_queries_case_id'),
        'eval_queries',
        ['case_id'],
        unique=False,
    )

    # Create eval_results table
    op.create_table(
        'eval_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('eval_query_id', sa.Integer(), nullable=False),
        sa.Column('top_k', sa.Integer(), nullable=False),
        sa.Column('retrieved_doc_ids', sa.Text(), nullable=False),
        sa.Column('retrieved_distances', sa.Text(), nullable=False),
        sa.Column('hit_at_k', sa.Boolean(), nullable=False),
        sa.Column('precision_at_k', sa.Float(), nullable=False),
        sa.Column('recall_at_k', sa.Float(), nullable=False),
        sa.Column('reciprocal_rank', sa.Float(), nullable=False),
        sa.Column('passed', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ['eval_query_id'],
            ['eval_queries.id'],
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_eval_results_id'),
        'eval_results',
        ['id'],
        unique=False,
    )
    op.create_index(
        op.f('ix_eval_results_eval_query_id'),
        'eval_results',
        ['eval_query_id'],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_index(
        op.f('ix_eval_results_eval_query_id'),
        table_name='eval_results',
    )
    op.drop_index(
        op.f('ix_eval_results_id'),
        table_name='eval_results',
    )
    op.drop_table('eval_results')

    op.drop_index(
        op.f('ix_eval_queries_case_id'),
        table_name='eval_queries',
    )
    op.drop_index(
        op.f('ix_eval_queries_id'),
        table_name='eval_queries',
    )
    op.drop_table('eval_queries')
