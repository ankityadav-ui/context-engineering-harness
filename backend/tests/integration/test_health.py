from fastapi.testclient import TestClient

from app.main import app


def test_health_reports_api_is_running():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_does_not_claim_external_service_status():
    with TestClient(app) as client:
        body = client.get("/health").json()

    # The endpoint reports only the API process itself; it must not
    # assert that Pinecone, Gemini, Neo4j, or PostgreSQL are healthy.
    assert set(body.keys()) == {"status"}
