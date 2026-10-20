"""Contract tests for the frontend Context Inspector API client."""

import sys
from io import BytesIO
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"

if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

import api_client  # noqa: E402


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.ok = True
        self.status_code = 200
        self.text = str(payload)

    def json(self):
        return self._payload


def test_context_preview_posts_the_full_ownership_contract(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse({"sources": {}})

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    result = api_client.context_preview(
        query="What is the policy?",
        top_k=3,
        document_id="doc-1",
        conversation_id="conversation-1",
        user_id="user-1",
        session_id="session-a",
        api_url="http://backend:8000/",
    )

    assert captured["url"] == "http://backend:8000/context/preview"
    assert captured["json"] == {
        "query": "What is the policy?",
        "top_k": 3,
        "document_id": "doc-1",
        "conversation_id": "conversation-1",
        "user_id": "user-1",
        "session_id": "session-a",
    }
    assert captured["timeout"] == 300
    assert result == {"sources": {}}


def test_context_preview_sends_null_for_absent_optional_fields(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["json"] = json
        return FakeResponse({"sources": {}})

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    api_client.context_preview("Question", api_url="http://backend:8000")

    assert captured["json"] == {
        "query": "Question",
        "top_k": 5,
        "document_id": None,
        "conversation_id": None,
        "user_id": None,
        "session_id": None,
    }


def test_upload_uses_job_acceptance_and_idempotency_header(monkeypatch):
    captured = {}

    def fake_post(url, files=None, data=None, headers=None, timeout=None):
        captured.update(
            {
                "url": url,
                "files": files,
                "data": data,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return FakeResponse({"job_id": "job-1", "status": "queued"})

    monkeypatch.setattr(api_client.requests, "post", fake_post)
    result = api_client.upload_document_file(
        BytesIO(b"%PDF-test"),
        "sample.pdf",
        True,
        api_url="http://backend:8000",
        idempotency_key="request-1",
    )

    assert captured["url"] == "http://backend:8000/documents/upload"
    assert captured["data"] == {"mask_content": "true"}
    assert captured["headers"] == {"Idempotency-Key": "request-1"}
    assert captured["timeout"] == 60
    assert result["status"] == "queued"


def test_job_status_and_retry_client_routes(monkeypatch):
    calls = []

    def fake_get(url, timeout=None):
        calls.append(("get", url, timeout))
        return FakeResponse({"job_id": "job-1", "status": "processing"})

    def fake_post(url, timeout=None):
        calls.append(("post", url, timeout))
        return FakeResponse({"job_id": "job-1", "status": "queued"})

    monkeypatch.setattr(api_client.requests, "get", fake_get)
    monkeypatch.setattr(api_client.requests, "post", fake_post)

    api_client.get_ingestion_job("job-1", api_url="http://backend:8000")
    api_client.retry_ingestion_job("job-1", api_url="http://backend:8000")

    assert calls == [
        (
            "get",
            "http://backend:8000/documents/jobs/job-1",
            10,
        ),
        (
            "post",
            "http://backend:8000/documents/jobs/job-1/retry",
            15,
        ),
    ]
