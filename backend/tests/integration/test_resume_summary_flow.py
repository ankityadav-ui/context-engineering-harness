"""End-to-end regression tests for the resume summarisation flow.

The reported failure was not an LLM problem: a resume could upload
successfully and ``summarise this`` still produced a "garbled text"
answer. These tests drive the production request path:

    PDF -> extraction -> masking -> chunking -> indexing
        -> retrieval (stubbed) -> Context Engine -> final LLM prompt

Only the paid remote stages (Gemini embeddings, Pinecone upsert, graph
ingestion) are replaced. Storage, extraction, masking, chunking, the
Context Engine, FastAPI routing, and SQLite conversation persistence are
the real implementations.
"""

import pymupdf
import pytest

import app.context_engine.engine as engine_module
import app.main as main_module
from app.context_engine import database
from app.rag import document_service, embedding_service

# Fictional resume content. Nothing here belongs to a real person.
# The header name is deliberately one that ``en_core_web_sm`` does not
# recognise, which is the real-world resume case (for example several
# South-Asian names are labelled as nothing at all).
RESUME_LINES = [
    "Vikram Rathore",
    "vikram.rathore@example.com | +91 9876543210",
    "Education",
    "Example University, B.Tech in Computer Science, 2020 - 2024",
    "Skills",
    "Python, SQL, FastAPI",
    "Project",
    "Document search application using RAG.",
]


def build_resume_pdf(path, lines=None):
    """Write a one-page resume PDF with selectable text."""

    document = pymupdf.open()
    page = document.new_page()

    y = 72
    for line in lines or RESUME_LINES:
        page.insert_text((72, y), line)
        y += 16

    document.save(path)
    document.close()


@pytest.fixture()
def resume_pipeline(tmp_path, monkeypatch):
    """Redirect storage and stub the remote embedding/index stages."""

    document_dir = tmp_path / "storage" / "documents"
    image_dir = tmp_path / "storage" / "images"
    document_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)

    monkeypatch.setattr(main_module, "DOCUMENT_DIR", document_dir)
    monkeypatch.setattr(
        document_service, "DOCUMENT_STORAGE_DIR", document_dir
    )
    monkeypatch.setattr(document_service, "IMAGE_STORAGE_DIR", image_dir)
    monkeypatch.setattr(document_service, "BASE_DIR", tmp_path)

    # Image references are resolved relative to the backend directory.
    monkeypatch.setattr(embedding_service, "BACKEND_DIR", tmp_path)

    # Neo4j is never written to from a test run.
    monkeypatch.setattr(
        main_module.graph_service, "is_configured", lambda: False
    )

    record = {"embedded": [], "upserted": 0}

    def fake_embed(content_units):
        record["embedded"] = [
            unit["content"] for unit in content_units
        ]
        return [
            {**unit, "embedding": [0.0, 0.0]}
            for unit in content_units
        ]

    def fake_upsert(embedded_units):
        record["upserted"] = len(embedded_units)
        return len(embedded_units)

    monkeypatch.setattr(document_service, "embed_content_units", fake_embed)
    monkeypatch.setattr(document_service, "upsert_content_units", fake_upsert)

    record["document_dir"] = document_dir

    return record


@pytest.fixture()
def resume_app(
    api_client,
    resume_pipeline,
    isolated_assembly,
    sqlite_session_factory,
):
    """Wire the real API routes to SQLite plus the temporary storage.

    ``isolated_assembly`` already keeps Pinecone/Neo4j and summary
    generation deterministic; it returns the shared ``monkeypatch`` used
    here to capture the final prompt.
    """

    # Conversation summary refresh reads through this factory.
    isolated_assembly.setattr(
        database, "SessionLocal", sqlite_session_factory
    )

    # The reported failure used an empty/relevant-miss vector result, so
    # the full-text fallback is what must ground ``summarise this``.
    isolated_assembly.setattr(
        engine_module, "retrieve_context", lambda **_kwargs: []
    )

    return {
        "client": api_client,
        "monkeypatch": isolated_assembly,
        "record": resume_pipeline,
        "document_dir": resume_pipeline["document_dir"],
    }


def upload(client, pdf_path, mask_content="false"):
    return client.post(
        "/documents/upload",
        files={
            "file": (
                "resume.pdf",
                pdf_path.read_bytes(),
                "application/pdf",
            )
        },
        data={"mask_content": mask_content},
    )


def ask_chat(resume_app, message, document_id, conversation_id=None):
    """Capture the exact prompt the backend sends to the model."""

    prompts = []

    def fake_generate_response(prompt, system=None):
        prompts.append(prompt)
        return "The model answered from the supplied evidence."

    resume_app["monkeypatch"].setattr(
        main_module, "generate_response", fake_generate_response
    )

    payload = {"message": message, "document_id": document_id}

    if conversation_id is not None:
        payload["conversation_id"] = conversation_id

    response = resume_app["client"].post("/chat", json=payload)

    return response, prompts


def test_chat_summarise_this_uses_the_uploaded_resume(tmp_path, resume_app):
    """Upload -> /chat('summarise this') must carry the resume text."""

    pdf_path = tmp_path / "resume.pdf"
    build_resume_pdf(pdf_path)

    upload_response = upload(resume_app["client"], pdf_path)
    assert upload_response.status_code == 200, upload_response.text

    document_id = upload_response.json()["document_id"]
    conversation = resume_app["client"].post(
        "/conversations", json={"title": "Resume"}
    ).json()

    response, prompts = ask_chat(
        resume_app,
        "summarise this",
        document_id=document_id,
        conversation_id=conversation["id"],
    )

    assert response.status_code == 200, response.text
    body = response.json()

    assert "The model answered" in body["response"]
    assert prompts, "the model must receive an assembled context"

    prompt = prompts[-1]

    # The chosen document is the one the user just uploaded, and the
    # full-text fallback reached the model even with an empty index.
    assert document_id in prompt
    assert "Vikram Rathore" in prompt
    assert "Example University" in prompt
    assert "B.Tech in Computer Science" in prompt
    assert "Python, SQL, FastAPI" in prompt
    assert "Document search application using RAG." in prompt


def test_masked_resume_keeps_structure_and_never_leaks_raw_pii(
    tmp_path, resume_app
):
    """Masked indexing must stay safe and still produce a usable context.

    The raw PDF is deleted after masked indexing, so the fallback must
    read the sanitized text artifact instead of re-extracting the PDF.
    Section headings must survive masking (the line-merging NER bug turned
    them into placeholder soup).
    """

    pdf_path = tmp_path / "resume.pdf"
    build_resume_pdf(pdf_path)

    upload_response = upload(
        resume_app["client"], pdf_path, mask_content="true"
    )
    assert upload_response.status_code == 200, upload_response.text

    upload_body = upload_response.json()

    assert upload_body["masking_applied"] is True
    assert upload_body["masking_safety_passed"] is True
    assert upload_body["masked_entities"] >= 1
    assert upload_body["masked_text_stored"] is True

    # The raw PDF is gone; only sanitized text remains on disk.
    assert list(resume_app["document_dir"].glob("*.pdf")) == []
    masked_artifacts = list(
        resume_app["document_dir"].glob("*.masked.txt")
    )
    assert len(masked_artifacts) == 1

    document_id = upload_body["document_id"]

    response, prompts = ask_chat(
        resume_app, "summarise this", document_id=document_id
    )

    assert response.status_code == 200, response.text

    prompt = prompts[-1]

    # Structure survives, so the summary is not a wall of placeholders.
    assert "Education" in prompt
    assert "Skills" in prompt
    assert "Project" in prompt
    assert "Python" in prompt
    assert "FastAPI" in prompt

    # Intended masked equivalents are present.
    assert "[PERSON_1]" in prompt
    assert "[EMAIL_1]" in prompt
    assert "[ORG_" in prompt

    # No raw sensitive value reaches the model. The header name is the
    # case spaCy misses on real resumes, so this is the regression guard.
    assert "Vikram Rathore" not in prompt
    assert "vikram.rathore@example.com" not in prompt
    assert "9876543210" not in prompt

    # The sanitized artifact on disk contains no raw PII either.
    artifact_text = masked_artifacts[0].read_text(encoding="utf-8")
    assert "Vikram Rathore" not in artifact_text
    assert "vikram.rathore@example.com" not in artifact_text


def test_upload_succeeds_with_a_realistic_pinecone_client_shape(
    tmp_path, monkeypatch, api_client
):
    """Regression for the production 500 on every real upload.

    The Pinecone SDK's data-plane ``Index`` handle has **no**
    ``describe_index``; that method lives on the control-plane
    ``Pinecone`` client. The dimension check used to call it on the
    index handle, so ``upsert_content_units`` raised ``AttributeError``
    and every upload failed before indexing anything.
    """

    from app.rag import pinecone_service

    document_dir = tmp_path / "storage" / "documents"
    image_dir = tmp_path / "storage" / "images"
    document_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)

    monkeypatch.setattr(main_module, "DOCUMENT_DIR", document_dir)
    monkeypatch.setattr(
        document_service, "DOCUMENT_STORAGE_DIR", document_dir
    )
    monkeypatch.setattr(document_service, "IMAGE_STORAGE_DIR", image_dir)
    monkeypatch.setattr(document_service, "BASE_DIR", tmp_path)
    monkeypatch.setattr(
        main_module.graph_service, "is_configured", lambda: False
    )

    # Embeddings are stubbed; the real upsert (and its dimension check)
    # runs against the fake client/index below.
    monkeypatch.setattr(
        document_service,
        "embed_content_units",
        lambda units: [{**unit, "embedding": [0.0] * 1024} for unit in units],
    )

    class _Field:
        def __init__(self, dimension):
            self.read_parameters = {"dimension": dimension}

    class _Schema:
        def __init__(self, dimension):
            self.fields = {"text": _Field(dimension)}

    class _IndexModel:
        dimension = None
        schema = _Schema(1024)

    class _DataPlaneIndex:
        """What Pinecone.Index() returns: upsert/query only."""

        def __init__(self):
            self.upserted = []

        def upsert(self, vectors, namespace):
            self.upserted.extend(vectors)

    class _ControlPlaneClient:
        """What Pinecone(api_key=...) returns: describe_index lives here."""

        def __init__(self):
            self.described = []

        def describe_index(self, name):
            self.described.append(name)
            return _IndexModel()

    data_plane = _DataPlaneIndex()
    control = _ControlPlaneClient()

    assert not hasattr(data_plane, "describe_index")

    pinecone_service.reset_index_cache()
    monkeypatch.setattr(pinecone_service, "get_index", lambda: data_plane)
    monkeypatch.setattr(pinecone_service, "get_client", lambda: control)

    pdf_path = tmp_path / "resume.pdf"
    build_resume_pdf(pdf_path)

    response = upload(api_client, pdf_path)

    assert response.status_code == 200, response.text
    assert response.json()["content_units"] >= 1

    # The dimension came from the control-plane client and the vector was
    # written through the data-plane handle.
    assert control.described == [pinecone_service.PINECONE_INDEX_NAME]
    assert len(data_plane.upserted) == 1


def test_deleting_a_masked_document_removes_its_artifact(
    tmp_path, resume_app
):
    """Deleting the document must also drop the sanitized full text."""

    from app.rag.document_service import delete_stored_document_files

    pdf_path = tmp_path / "resume.pdf"
    build_resume_pdf(pdf_path)

    upload_response = upload(
        resume_app["client"], pdf_path, mask_content="true"
    )
    assert upload_response.status_code == 200, upload_response.text

    document_id = upload_response.json()["document_id"]
    artifacts = list(resume_app["document_dir"].glob("*.masked.txt"))
    assert len(artifacts) == 1

    removed = delete_stored_document_files(document_id)

    expected = str(artifacts[0].relative_to(tmp_path)).replace("\\", "/")
    assert expected in [item.replace("\\", "/") for item in removed]
    assert list(resume_app["document_dir"].glob("*.masked.txt")) == []


def test_chat_with_no_retrievable_document_reports_clearly(resume_app):
    """A missing/unreadable document must not invite a fabricated summary."""

    prompts = []

    def fake_generate_response(prompt, system=None):
        prompts.append(prompt)
        return "This answer should never be generated."

    resume_app["monkeypatch"].setattr(
        main_module, "generate_response", fake_generate_response
    )

    response = resume_app["client"].post(
        "/chat",
        json={
            "message": "summarise this",
            "document_id": "00000000-0000-4000-8000-000000000000",
        },
    )

    assert response.status_code == 400
    assert "no retrievable content" in response.json()["detail"].lower()
    assert prompts == []


def test_chat_without_a_document_keeps_the_plain_contract(resume_app):
    """Unscoped chat still works and never invents document evidence."""

    prompts = []

    def fake_generate_response(prompt, system=None):
        prompts.append(prompt)
        return "A plain answer."

    resume_app["monkeypatch"].setattr(
        main_module, "generate_response", fake_generate_response
    )

    response = resume_app["client"].post(
        "/chat", json={"message": "Hello there"}
    )

    assert response.status_code == 200
    assert set(response.json()) == {"response", "system_prompt_active"}
    assert prompts == ["Hello there"]
