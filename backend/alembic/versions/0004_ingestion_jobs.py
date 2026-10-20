"""Add durable PDF ingestion job records.

Revision ID: 0004_ingestion_jobs
Revises: 0003_conversation_system_prompt
Create Date: 2026-10-20
"""
from alembic import op
import sqlalchemy as sa


revision = "0004_ingestion_jobs"
down_revision = "0003_conversation_system_prompt"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "ingestion_jobs" in inspector.get_table_names():
        return

    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.Column("idempotency_key", sa.String(length=64), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("mask_content", sa.Boolean(), nullable=False, default=False),
        sa.Column("status", sa.String(length=20), nullable=False, default="queued"),
        sa.Column("stage", sa.String(length=40), nullable=False, default="queued"),
        sa.Column("stage_timings", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error_code", sa.String(length=80), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False, default=0),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id",
            name="uq_ingestion_jobs_document_id",
        ),
        sa.UniqueConstraint(
            "idempotency_key",
            name="uq_ingestion_job_idempotency_key",
        ),
    )
    op.create_index(
        "ix_ingestion_jobs_document_id",
        "ingestion_jobs",
        ["document_id"],
        unique=False,
    )
    op.create_index(
        "ix_ingestion_jobs_status",
        "ingestion_jobs",
        ["status"],
        unique=False,
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "ingestion_jobs" not in inspector.get_table_names():
        return
    op.drop_index("ix_ingestion_jobs_status", table_name="ingestion_jobs")
    op.drop_index("ix_ingestion_jobs_document_id", table_name="ingestion_jobs")
    op.drop_table("ingestion_jobs")
