"""Persistence helpers for restart-recoverable PDF ingestion jobs."""

import uuid
from datetime import datetime, timezone

from sqlalchemy.exc import IntegrityError

from app.context_engine.database import SessionLocal
from app.context_engine.models import IngestionJob


def create_or_get_job(
    *,
    idempotency_key: str,
    filename: str,
    file_path: str,
    mask_content: bool,
) -> tuple[IngestionJob, bool]:
    """Create one durable job or return the existing idempotent request."""

    with SessionLocal() as db:
        existing = (
            db.query(IngestionJob)
            .filter_by(idempotency_key=idempotency_key)
            .one_or_none()
        )
        if existing is not None:
            return existing, False

        job = IngestionJob(
            id=str(uuid.uuid4()),
            document_id=str(uuid.uuid4()),
            idempotency_key=idempotency_key,
            filename=filename[:255],
            file_path=file_path,
            mask_content=mask_content,
            status="queued",
            stage="queued",
            stage_timings={},
        )
        db.add(job)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            existing = (
                db.query(IngestionJob)
                .filter_by(idempotency_key=idempotency_key)
                .one_or_none()
            )
            if existing is None:
                raise
            return existing, False
        db.refresh(job)
        return job, True


def get_job(job_id: str) -> IngestionJob | None:
    with SessionLocal() as db:
        return db.query(IngestionJob).filter_by(id=job_id).one_or_none()


def update_job(job_id: str, **values) -> bool:
    with SessionLocal() as db:
        job = db.query(IngestionJob).filter_by(id=job_id).one_or_none()
        if job is None:
            return False
        for key, value in values.items():
            setattr(job, key, value)
        job.updated_at = datetime.now(timezone.utc)
        db.commit()
        return True


def update_stage_timing(
    job_id: str,
    *,
    stage: str,
    status: str,
    duration_ms: float,
    error_code: str | None = None,
) -> bool:
    with SessionLocal() as db:
        job = db.query(IngestionJob).filter_by(id=job_id).one_or_none()
        if job is None:
            return False
        timings = dict(job.stage_timings or {})
        timings[stage] = {
            "status": status,
            "duration_ms": round(duration_ms, 1),
            "error_code": error_code,
        }
        job.stage_timings = timings
        job.stage = stage
        job.updated_at = datetime.now(timezone.utc)
        db.commit()
        return True


def recover_interrupted_jobs() -> list[str]:
    """Reset interrupted jobs to queued and return IDs ready to resume."""

    with SessionLocal() as db:
        jobs = (
            db.query(IngestionJob)
            .filter(IngestionJob.status.in_(("queued", "processing")))
            .order_by(IngestionJob.created_at)
            .all()
        )
        job_ids = []
        for job in jobs:
            job.status = "queued"
            job.stage = "queued"
            job.error_code = None
            job.error_message = None
            job.updated_at = datetime.now(timezone.utc)
            job_ids.append(job.id)
        db.commit()
        return job_ids


def job_response(job: IngestionJob) -> dict:
    """Serialize safe status fields; never return local paths or source text."""

    return {
        "job_id": job.id,
        "document_id": job.document_id,
        "filename": job.filename,
        "status": job.status,
        "stage": job.stage,
        "stage_timings": job.stage_timings or {},
        "attempts": job.attempts,
        "result": job.result,
        "error": (
            {
                "code": job.error_code,
                "message": job.error_message,
            }
            if job.error_code
            else None
        ),
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
    }
