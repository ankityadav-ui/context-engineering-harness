"""API regressions for durable, idempotent PDF ingestion jobs."""

import app.main as main_module


def _successful_document(**_kwargs):
    return (
        [
            {
                "id": "unit-1",
                "document_id": "ignored-by-worker",
                "content_type": "text",
                "page_number": 1,
                "content": "A sufficiently long example source passage.",
            }
        ],
        [],
        [],
        [],
        {
            "applied": False,
            "safety_passed": None,
            "masked_entities": 0,
        },
    )


def _post_upload(client, *, headers=None):
    return client.post(
        "/documents/upload",
        files={"file": ("sample.pdf", b"%PDF-1.4 test", "application/pdf")},
        headers=headers,
    )


def test_upload_accepts_a_durable_job_and_exposes_safe_status(
    api_client,
    monkeypatch,
):
    monkeypatch.setattr(main_module, "process_document", _successful_document)
    monkeypatch.setattr(
        main_module.graph_service, "is_configured", lambda: False
    )

    response = _post_upload(api_client)

    assert response.status_code == 202
    accepted = response.json()
    assert accepted["status"] == "queued"
    assert accepted["job_id"]
    assert accepted["document_id"]
    assert "file_path" not in accepted

    status = api_client.get(
        f"/documents/jobs/{accepted['job_id']}"
    )
    assert status.status_code == 200
    job = status.json()
    assert job["status"] == "completed"
    assert job["result"]["content_units"] == 1
    assert job["stage_timings"]["completion"]["status"] == "completed"
    assert "file_path" not in job


def test_explicit_idempotency_deduplicates_but_new_upload_reprocesses(
    api_client,
    monkeypatch,
):
    calls = []

    def record_process(**kwargs):
        calls.append(kwargs["document_id"])
        return _successful_document(**kwargs)

    monkeypatch.setattr(main_module, "process_document", record_process)
    monkeypatch.setattr(
        main_module.graph_service, "is_configured", lambda: False
    )

    first = _post_upload(
        api_client,
        headers={"Idempotency-Key": "same-request"},
    )
    duplicate = _post_upload(
        api_client,
        headers={"Idempotency-Key": "same-request"},
    )
    reprocess = _post_upload(api_client)

    first_body = first.json()
    duplicate_body = duplicate.json()
    reprocess_body = reprocess.json()
    assert first.status_code == duplicate.status_code == reprocess.status_code == 202
    assert first_body["document_id"] == duplicate_body["document_id"]
    assert duplicate_body["duplicate_request"] is True
    assert reprocess_body["document_id"] != first_body["document_id"]
    assert len(calls) == 2


def test_failed_job_retries_with_the_same_document_id(
    api_client,
    monkeypatch,
):
    attempts = 0

    def fail_once_then_succeed(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("private upstream details")
        return _successful_document(**kwargs)

    monkeypatch.setattr(
        main_module,
        "process_document",
        fail_once_then_succeed,
    )
    monkeypatch.setattr(
        main_module.graph_service, "is_configured", lambda: False
    )

    accepted = _post_upload(api_client).json()
    failed = api_client.get(
        f"/documents/jobs/{accepted['job_id']}"
    ).json()
    assert failed["status"] == "failed"
    assert failed["error"]["message"] != "private upstream details"

    retry = api_client.post(
        f"/documents/jobs/{accepted['job_id']}/retry"
    )
    assert retry.status_code == 202
    completed = api_client.get(
        f"/documents/jobs/{accepted['job_id']}"
    ).json()
    assert completed["status"] == "completed"
    assert completed["document_id"] == accepted["document_id"]
    assert completed["attempts"] == 2
