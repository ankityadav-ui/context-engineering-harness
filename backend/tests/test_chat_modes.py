"""
Chat Mode Routing Tests

Verifies that:
1. chat_mode="normal" → generate_normal_answer() is called
2. chat_mode="document" → generate_rag_answer() is called
3. Unknown/invalid chat_mode → falls back to generate_normal_answer()
4. Normal chat works without any documents
5. Document chat with no context returns NO_CONTEXT_RESPONSE
6. Session chat_mode persists and is used as source of truth

Uses TestClient + MockLLM so no external services needed.
"""

import os
import sys
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Base, ChatSession, ChatMessage, Case


# ============================================================
# SETUP: Override database before importing app
# ============================================================

test_engine = create_engine(
    "sqlite:///:memory:",
    echo=False,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
Base.metadata.create_all(test_engine)
TestSessionLocal = sessionmaker(bind=test_engine)

import app.database as database_mod
import app.main as main_mod

database_mod.engine = test_engine


def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


main_mod.app.dependency_overrides[main_mod.get_db] = override_get_db

from fastapi.testclient import TestClient


# ============================================================
# MOCK LLM HELPER
# ============================================================

class FakeLLM:
    """A fake LLM that returns controlled answers."""

    def __init__(self, default_answer="Mock answer"):
        self.default_answer = default_answer
        self.last_messages = None

    def generate(self, request):
        from app.llm.base import LLMResponse

        messages = request.messages
        self.last_messages = messages

        system = messages[0]["content"] if messages else ""
        user = messages[-1]["content"] if messages else ""

        if "general-purpose assistant" in system:
            return LLMResponse(text="I am a helpful general-purpose assistant.")
        if "helpful assistant for a Context Engineering" in system:
            return LLMResponse(text="Based on the documents, here is the answer.")
        return LLMResponse(text=self.default_answer)


fake_llm = FakeLLM()


# ============================================================
# FIXTURES
# ============================================================

def setup_test_case():
    db = TestSessionLocal()
    case = Case(name="Mode Test Case", description="For chat mode tests")
    db.add(case)
    db.commit()
    db.refresh(case)
    case_id = case.id
    db.close()
    return case_id


def cleanup_chat_data(session_ids, case_ids):
    db = TestSessionLocal()
    for sid in session_ids:
        db.query(ChatMessage).filter(ChatMessage.session_id == sid).delete()
        db.query(ChatSession).filter(ChatSession.id == sid).delete()
    for cid in case_ids:
        db.query(Case).filter(Case.id == cid).delete()
    db.commit()
    db.close()


# ============================================================
# TEST 1: Normal chat uses generate_normal_answer()
# ============================================================

def test_normal_chat_routes_to_normal_answer():
    """Verify chat_mode='normal' calls generate_normal_answer, not generate_rag_answer."""
    print("\nTEST: Normal chat routes to generate_normal_answer")
    client = TestClient(main_mod.app)
    case_id = setup_test_case()

    resp = client.post(f"/cases/{case_id}/chats", json={
        "title": "Normal Routing Test",
        "chat_mode": "normal",
    })
    assert resp.status_code == 200
    session_id = resp.json()["id"]
    assert resp.json()["chat_mode"] == "normal"

    with patch("app.main.generate_normal_answer") as mock_normal, \
         patch("app.main.generate_rag_answer") as mock_rag:
        mock_normal.return_value = {
            "query": "Hello",
            "answer": "Normal answer",
            "chunks": [],
            "chunk_count": 0,
            "graph_results": [],
            "metadata": {},
        }

        resp = client.post(f"/chats/{session_id}/messages", json={
            "content": "Hello, who are you?",
        })

        assert resp.status_code == 200
        assert resp.json()["content"] == "Normal answer"
        mock_normal.assert_called_once()
        mock_rag.assert_not_called()
        print("  PASSED: generate_normal_answer was called, generate_rag_answer was NOT")

    cleanup_chat_data([session_id], [case_id])


# ============================================================
# TEST 2: Document chat uses generate_rag_answer()
# ============================================================

def test_document_chat_routes_to_rag_answer():
    """Verify chat_mode='document' calls generate_rag_answer, not generate_normal_answer."""
    print("\nTEST: Document chat routes to generate_rag_answer")
    client = TestClient(main_mod.app)
    case_id = setup_test_case()

    resp = client.post(f"/cases/{case_id}/chats", json={
        "title": "Document Routing Test",
        "chat_mode": "document",
    })
    assert resp.status_code == 200
    session_id = resp.json()["id"]
    assert resp.json()["chat_mode"] == "document"

    with patch("app.main.generate_normal_answer") as mock_normal, \
         patch("app.main.generate_rag_answer") as mock_rag:
        mock_rag.return_value = {
            "query": "What is X?",
            "case_id": case_id,
            "answer": "RAG answer",
            "chunks": [],
            "chunk_count": 0,
            "graph_results": [],
            "metadata": {},
        }

        resp = client.post(f"/chats/{session_id}/messages", json={
            "content": "What is X?",
        })

        assert resp.status_code == 200
        assert resp.json()["content"] == "RAG answer"
        mock_rag.assert_called_once()
        mock_normal.assert_not_called()
        print("  PASSED: generate_rag_answer was called, generate_normal_answer was NOT")

    cleanup_chat_data([session_id], [case_id])


# ============================================================
# TEST 3: Unknown mode falls back to normal
# ============================================================

def test_unknown_mode_falls_back_to_normal():
    """Verify unknown chat_mode falls back to generate_normal_answer."""
    print("\nTEST: Unknown mode falls back to generate_normal_answer")
    client = TestClient(main_mod.app)
    case_id = setup_test_case()

    # Manually create a session with an invalid mode
    db = TestSessionLocal()
    session = ChatSession(
        case_id=case_id,
        title="Invalid Mode Session",
        chat_mode="invalid_mode",
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    session_id = session.id
    db.close()

    with patch("app.main.generate_normal_answer") as mock_normal, \
         patch("app.main.generate_rag_answer") as mock_rag:
        mock_normal.return_value = {
            "query": "Test",
            "answer": "Fallback answer",
            "chunks": [],
            "chunk_count": 0,
            "graph_results": [],
            "metadata": {},
        }

        resp = client.post(f"/chats/{session_id}/messages", json={
            "content": "Test unknown mode",
        })

        assert resp.status_code == 200
        assert resp.json()["content"] == "Fallback answer"
        mock_normal.assert_called_once()
        mock_rag.assert_not_called()
        print("  PASSED: Unknown mode fell back to generate_normal_answer")

    cleanup_chat_data([session_id], [case_id])


# ============================================================
# TEST 4: Normal chat without documents succeeds
# ============================================================

def test_normal_chat_without_documents():
    """Normal chat should work even with no documents uploaded."""
    print("\nTEST: Normal chat without documents succeeds")
    client = TestClient(main_mod.app)
    case_id = setup_test_case()

    resp = client.post(f"/cases/{case_id}/chats", json={
        "title": "Normal No Docs",
        "chat_mode": "normal",
    })
    session_id = resp.json()["id"]

    with patch("app.chat.create_llm", return_value=fake_llm):
        resp = client.post(f"/chats/{session_id}/messages", json={
            "content": "Hello, who are you?",
        })

    assert resp.status_code == 200
    answer = resp.json()["content"]
    assert answer, "Answer should not be empty"
    assert "could not find enough relevant information" not in answer.lower(), (
        f"Normal chat should NOT return no-context response: {answer}"
    )
    print(f"  Answer: {answer}")
    print("  PASSED: Normal chat works without documents")

    cleanup_chat_data([session_id], [case_id])


# ============================================================
# TEST 5: Document chat with no context returns safe response
# ============================================================

def test_document_chat_no_context_returns_safe_response():
    """Document chat with no uploaded documents should return NO_CONTEXT_RESPONSE."""
    print("\nTEST: Document chat with no context returns NO_CONTEXT_RESPONSE")
    client = TestClient(main_mod.app)
    case_id = setup_test_case()

    resp = client.post(f"/cases/{case_id}/chats", json={
        "title": "Doc No Context",
        "chat_mode": "document",
    })
    session_id = resp.json()["id"]

    # No documents uploaded, no memories
    with patch("app.chat.create_llm", return_value=fake_llm):
        resp = client.post(f"/chats/{session_id}/messages", json={
            "content": "What is in the documents?",
        })

    assert resp.status_code == 200
    answer = resp.json()["content"]
    assert "could not find enough relevant information" in answer.lower(), (
        f"Document chat with no context should return NO_CONTEXT_RESPONSE: {answer}"
    )
    print(f"  Answer: {answer}")
    print("  PASSED: Correctly returned no-context response")

    cleanup_chat_data([session_id], [case_id])


# ============================================================
# TEST 6: Session mode is persisted correctly
# ============================================================

def test_session_mode_persisted_correctly():
    """Verify chat_mode is stored and returned correctly."""
    print("\nTEST: Session mode persisted correctly")
    client = TestClient(main_mod.app)
    case_id = setup_test_case()

    # Create normal session
    resp = client.post(f"/cases/{case_id}/chats", json={
        "title": "Persist Normal",
        "chat_mode": "normal",
    })
    assert resp.status_code == 200
    normal_id = resp.json()["id"]
    assert resp.json()["chat_mode"] == "normal"

    # Create document session
    resp = client.post(f"/cases/{case_id}/chats", json={
        "title": "Persist Document",
        "chat_mode": "document",
    })
    assert resp.status_code == 200
    doc_id = resp.json()["id"]
    assert resp.json()["chat_mode"] == "document"

    # Verify via list endpoint
    resp = client.get(f"/cases/{case_id}/chats")
    assert resp.status_code == 200
    sessions = resp.json()
    modes = {s["id"]: s["chat_mode"] for s in sessions}
    assert modes[normal_id] == "normal"
    assert modes[doc_id] == "document"

    # Verify via history endpoint
    resp = client.get(f"/chats/{normal_id}")
    assert resp.status_code == 200
    assert resp.json()["chat_mode"] == "normal"

    resp = client.get(f"/chats/{doc_id}")
    assert resp.status_code == 200
    assert resp.json()["chat_mode"] == "document"

    print("  PASSED: Both modes persisted and returned correctly")
    cleanup_chat_data([normal_id, doc_id], [case_id])


# ============================================================
# TEST 7: Switching sessions preserves mode
# ============================================================

def test_session_switch_preserves_mode():
    """Verify switching between sessions correctly updates the active mode."""
    print("\nTEST: Session switch preserves mode")
    client = TestClient(main_mod.app)
    case_id = setup_test_case()

    # Create two sessions
    resp = client.post(f"/cases/{case_id}/chats", json={
        "title": "Normal Session",
        "chat_mode": "normal",
    })
    normal_id = resp.json()["id"]

    resp = client.post(f"/cases/{case_id}/chats", json={
        "title": "Document Session",
        "chat_mode": "document",
    })
    doc_id = resp.json()["id"]

    # Verify each session's mode via history endpoint
    resp = client.get(f"/chats/{normal_id}")
    assert resp.json()["chat_mode"] == "normal"

    resp = client.get(f"/chats/{doc_id}")
    assert resp.json()["chat_mode"] == "document"

    # Send message to normal session - should use normal path
    with patch("app.main.generate_normal_answer") as mock_normal, \
         patch("app.main.generate_rag_answer") as mock_rag:
        mock_normal.return_value = {
            "query": "Test", "answer": "Normal", "chunks": [],
            "chunk_count": 0, "graph_results": [], "metadata": {},
        }
        resp = client.post(f"/chats/{normal_id}/messages", json={
            "content": "Hello",
        })
        assert resp.status_code == 200
        mock_normal.assert_called_once()
        mock_rag.assert_not_called()

    # Send message to document session - should use RAG path
    with patch("app.main.generate_normal_answer") as mock_normal, \
         patch("app.main.generate_rag_answer") as mock_rag:
        mock_rag.return_value = {
            "query": "Test", "case_id": case_id, "answer": "RAG",
            "chunks": [], "chunk_count": 0, "graph_results": [], "metadata": {},
        }
        resp = client.post(f"/chats/{doc_id}/messages", json={
            "content": "What about docs?",
        })
        assert resp.status_code == 200
        mock_rag.assert_called_once()
        mock_normal.assert_not_called()

    print("  PASSED: Mode routing correct after session switching")
    cleanup_chat_data([normal_id, doc_id], [case_id])


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":
    tests = [
        test_normal_chat_routes_to_normal_answer,
        test_document_chat_routes_to_rag_answer,
        test_unknown_mode_falls_back_to_normal,
        test_normal_chat_without_documents,
        test_document_chat_no_context_returns_safe_response,
        test_session_mode_persisted_correctly,
        test_session_switch_preserves_mode,
    ]

    passed = 0
    failed = 0
    errors = []

    print("=" * 60)
    print("CHAT MODE ROUTING TEST SUITE")
    print("=" * 60)

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            print(f"  FAILED: {e}")

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    if errors:
        print("\nFAILURES:")
        for name, err in errors:
            print(f"  {name}: {err}")

    sys.exit(1 if failed else 0)
