"""
Comprehensive FastAPI API integration tests.

Tests the complete application through HTTP endpoints,
not just direct Python function calls.

Uses TestClient + MockLLM so no external services needed.
"""

import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Base, Memory, ChatSession, ChatMessage, Case
from app.memory import create_memory


# ============================================================
# SETUP: Override database before importing app
# ============================================================

# Create in-memory SQLite for tests (thread-safe with StaticPool)
test_engine = create_engine(
    "sqlite:///:memory:",
    echo=False,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
Base.metadata.create_all(test_engine)
TestSessionLocal = sessionmaker(bind=test_engine)


# Patch the engine and Session before importing main
import app.database as database_mod
import app.main as main_mod

# Override the engine
database_mod.engine = test_engine


# Override get_db to use test session
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
    """A fake LLM that inspects context and returns controlled answers."""

    def __init__(self, default_answer="Mock answer"):
        self.default_answer = default_answer
        self.last_messages = None

    def generate(self, messages, temperature=None, max_tokens=None):
        self.last_messages = messages
        # Extract context from system message to give smarter answers
        system = messages[0]["content"] if messages else ""
        user = messages[-1]["content"] if messages else ""

        if "LLM fine tuning" in system or "LLM fine tuning" in user:
            return "You are currently learning LLM fine-tuning."
        if "CSL422" in system or "Machine Learning" in system:
            return "The course code for Machine Learning is CSL422. You are currently learning LLM fine-tuning."
        if "FastAPI" in system:
            return "You mentioned using FastAPI."
        return self.default_answer


fake_llm = FakeLLM()


# ============================================================
# FIXTURE: Create test case
# ============================================================

def setup_test_case():
    """Create a test case in the DB."""
    db = TestSessionLocal()
    case = Case(name="Test Case", description="For API tests")
    db.add(case)
    db.commit()
    db.refresh(case)
    case_id = case.id
    db.close()
    return case_id


def cleanup_test_data():
    """Remove all test data."""
    db = TestSessionLocal()
    db.query(Memory).delete()
    db.query(ChatMessage).delete()
    db.query(ChatSession).delete()
    db.commit()
    db.close()


# ============================================================
# TEST 1: Health / Startup
# ============================================================

def test_health():
    print("\nTEST 1: Health / Startup")
    client = TestClient(main_mod.app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["message"] == "Backend is working!"
    print("  PASSED")


# ============================================================
# TEST 2: Create Case
# ============================================================

def test_create_case():
    print("\nTEST 2: Create Case")
    client = TestClient(main_mod.app)
    resp = client.post("/cases", json={"name": "API Test Case"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "API Test Case"
    assert data["id"] > 0
    print(f"  PASSED: case_id={data['id']}")
    # Cleanup
    db = TestSessionLocal()
    db.query(Case).filter(Case.name == "API Test Case").delete()
    db.commit()
    db.close()


# ============================================================
# TEST 3: Create Memory
# ============================================================

def test_create_memory():
    print("\nTEST 3: Create Memory")
    client = TestClient(main_mod.app)
    resp = client.post("/memories", json={
        "content": "I am learning LLM fine tuning",
        "memory_type": "goal",
        "importance": 0.5,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["content"] == "I am learning LLM fine tuning"
    assert data["memory_type"] == "goal"
    assert data["importance"] == 0.5
    assert data["user_id"] == "default_user"
    print(f"  PASSED: memory_id={data['id']}")


# ============================================================
# TEST 4: List Memories
# ============================================================

def test_list_memories():
    print("\nTEST 4: List Memories")
    client = TestClient(main_mod.app)
    resp = client.get("/memories")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    contents = [m["content"] for m in data]
    assert "I am learning LLM fine tuning" in contents
    print(f"  PASSED: {len(data)} memories found")


# ============================================================
# TEST 5: Update Memory
# ============================================================

def test_update_memory():
    print("\nTEST 5: Update Memory")
    client = TestClient(main_mod.app)
    # Get memory id
    resp = client.get("/memories")
    memories = resp.json()
    mem_id = memories[0]["id"]

    # Update
    resp = client.put(f"/memories/{mem_id}", json={
        "content": "I am learning LLM fine tuning and RAG",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["content"] == "I am learning LLM fine tuning and RAG"

    # Verify persisted
    resp = client.get(f"/memories/{mem_id}")
    assert resp.status_code == 200
    assert resp.json()["content"] == "I am learning LLM fine tuning and RAG"
    print("  PASSED")


# ============================================================
# TEST 6: Delete Memory
# ============================================================

def test_delete_memory():
    print("\nTEST 6: Delete Memory")
    client = TestClient(main_mod.app)
    # Create one to delete
    resp = client.post("/memories", json={
        "content": "Memory to delete",
        "memory_type": "fact",
    })
    mem_id = resp.json()["id"]

    # Delete
    resp = client.delete(f"/memories/{mem_id}")
    assert resp.status_code == 200

    # Verify gone
    resp = client.get(f"/memories/{mem_id}")
    assert resp.status_code == 404
    print("  PASSED")


# ============================================================
# TEST 7: Create Chat Session
# ============================================================

def test_create_chat_session():
    print("\nTEST 7: Create Chat Session")
    client = TestClient(main_mod.app)
    case_id = setup_test_case()

    resp = client.post(f"/cases/{case_id}/chats", json={
        "title": "Test Chat",
        "chat_mode": "normal",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "Test Chat"
    assert data["chat_mode"] == "normal"
    assert data["case_id"] == case_id
    print(f"  PASSED: session_id={data['id']}")

    # Cleanup
    db = TestSessionLocal()
    db.query(ChatSession).filter(ChatSession.id == data["id"]).delete()
    db.query(Case).filter(Case.id == case_id).delete()
    db.commit()
    db.close()


# ============================================================
# TEST 8: Send Normal Chat Message
# ============================================================

def test_send_normal_chat():
    print("\nTEST 8: Send Normal Chat Message")
    client = TestClient(main_mod.app)
    case_id = setup_test_case()

    # Create session
    resp = client.post(f"/cases/{case_id}/chats", json={
        "title": "Normal Test",
        "chat_mode": "normal",
    })
    session_id = resp.json()["id"]

    # Create memory first
    client.post("/memories", json={
        "content": "I am learning LLM fine tuning",
        "memory_type": "goal",
    })

    # Send message with mocked LLM
    with patch("app.chat.create_llm", return_value=fake_llm):
        resp = client.post(f"/chats/{session_id}/messages", json={
            "content": "What am I currently learning?",
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["role"] == "assistant"
    assert data["content"], "Answer should not be empty"
    print(f"  Answer: {data['content']}")

    # Cleanup
    db = TestSessionLocal()
    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
    db.query(ChatSession).filter(ChatSession.id == session_id).delete()
    db.query(Case).filter(Case.id == case_id).delete()
    db.commit()
    db.close()


# ============================================================
# TEST 9: Send Document Chat Message
# ============================================================

def test_send_document_chat():
    print("\nTEST 9: Send Document Chat Message")
    client = TestClient(main_mod.app)
    case_id = setup_test_case()

    resp = client.post(f"/cases/{case_id}/chats", json={
        "title": "Doc Test",
        "chat_mode": "document",
    })
    session_id = resp.json()["id"]

    # Create memory
    client.post("/memories", json={
        "content": "I am learning LLM fine tuning",
        "memory_type": "goal",
    })

    with patch("app.chat.create_llm", return_value=fake_llm):
        resp = client.post(f"/chats/{session_id}/messages", json={
            "content": "What am I currently learning?",
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["content"], "Answer should not be empty"
    print(f"  Answer: {data['content']}")

    # Cleanup
    db = TestSessionLocal()
    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
    db.query(ChatSession).filter(ChatSession.id == session_id).delete()
    db.query(Case).filter(Case.id == case_id).delete()
    db.commit()
    db.close()


# ============================================================
# TEST 10: Memory Persistence Across Sessions
# ============================================================

def test_memory_persistence():
    print("\nTEST 10: Memory Persistence Across Sessions")
    client = TestClient(main_mod.app)
    case_id = setup_test_case()

    # Create memory
    client.post("/memories", json={
        "content": "I am learning LLM fine tuning",
        "memory_type": "goal",
    })

    # Session 1: ask question
    resp = client.post(f"/cases/{case_id}/chats", json={
        "title": "Session 1",
        "chat_mode": "normal",
    })
    s1_id = resp.json()["id"]

    with patch("app.chat.create_llm", return_value=fake_llm):
        resp = client.post(f"/chats/{s1_id}/messages", json={
            "content": "What am I currently learning?",
        })
    assert resp.status_code == 200
    print(f"  Session 1 answer: {resp.json()['content']}")

    # Session 2: ask same question (memory should persist)
    resp = client.post(f"/cases/{case_id}/chats", json={
        "title": "Session 2",
        "chat_mode": "normal",
    })
    s2_id = resp.json()["id"]

    with patch("app.chat.create_llm", return_value=fake_llm):
        resp = client.post(f"/chats/{s2_id}/messages", json={
            "content": "What am I currently learning?",
        })
    assert resp.status_code == 200
    print(f"  Session 2 answer: {resp.json()['content']}")

    # Cleanup
    db = TestSessionLocal()
    db.query(ChatMessage).filter(
        ChatMessage.session_id.in_([s1_id, s2_id])
    ).delete()
    db.query(ChatSession).filter(
        ChatSession.id.in_([s1_id, s2_id])
    ).delete()
    db.query(Case).filter(Case.id == case_id).delete()
    db.commit()
    db.close()


# ============================================================
# TEST 11: Memory-Only RAG (No Documents)
# ============================================================

def test_memory_only_rag():
    print("\nTEST 11: Memory-Only RAG")
    client = TestClient(main_mod.app)
    case_id = setup_test_case()

    client.post("/memories", json={
        "content": "I am learning LLM fine tuning",
        "memory_type": "goal",
    })

    resp = client.post(f"/cases/{case_id}/chats", json={
        "title": "Memory Only",
        "chat_mode": "document",
    })
    session_id = resp.json()["id"]

    with patch("app.chat.create_llm", return_value=fake_llm):
        resp = client.post(f"/chats/{session_id}/messages", json={
            "content": "What am I currently learning?",
        })

    assert resp.status_code == 200
    answer = resp.json()["content"]
    # Must NOT be NO_CONTEXT_RESPONSE
    assert "could not find enough relevant information" not in answer.lower(), (
        f"Should NOT return NO_CONTEXT_RESPONSE when memory exists: {answer}"
    )
    print(f"  Answer: {answer}")
    print("  PASSED: Memory-only context works")

    # Cleanup
    db = TestSessionLocal()
    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
    db.query(ChatSession).filter(ChatSession.id == session_id).delete()
    db.query(Case).filter(Case.id == case_id).delete()
    db.commit()
    db.close()


# ============================================================
# TEST 12: User Isolation
# ============================================================

def test_user_isolation():
    print("\nTEST 12: User Isolation")
    client = TestClient(main_mod.app)

    # Create memory for user_a
    resp = client.post("/memories?user_id=user_a", json={
        "content": "User A learning LLM fine tuning",
        "memory_type": "goal",
    })
    assert resp.status_code == 200

    # List as user_a
    resp = client.get("/memories?user_id=user_a")
    contents_a = [m["content"] for m in resp.json()]
    assert "User A learning LLM fine tuning" in contents_a

    # List as user_b
    resp = client.get("/memories?user_id=user_b")
    contents_b = [m["content"] for m in resp.json()]
    assert "User A learning LLM fine tuning" not in contents_b, (
        f"user_b should not see user_a's memories: {contents_b}"
    )

    # Cleanup
    db = TestSessionLocal()
    db.query(Memory).filter(Memory.user_id == "user_a").delete()
    db.commit()
    db.close()
    print("  PASSED: No cross-user leakage")


# ============================================================
# TEST 13: Case Isolation
# ============================================================

def test_case_isolation():
    print("\nTEST 13: Case Isolation")
    client = TestClient(main_mod.app)

    # Create global memory
    client.post("/memories", json={
        "content": "I prefer Python",
        "memory_type": "preference",
    })

    # Create case 1 memory
    client.post("/memories", json={
        "content": "I am working on RAG",
        "memory_type": "fact",
        "case_id": 1,
    })

    # Create case 2 memory
    client.post("/memories", json={
        "content": "I am working on Neo4j",
        "memory_type": "fact",
        "case_id": 2,
    })

    # List with case_id=1: should see global + case 1
    resp = client.get("/memories?case_id=1")
    contents = [m["content"] for m in resp.json()]
    assert "I prefer Python" in contents, "Should include global"
    assert "I am working on RAG" in contents, "Should include case 1"
    assert "I am working on Neo4j" not in contents, "Should NOT include case 2"

    # List with case_id=2: should see global + case 2
    resp = client.get("/memories?case_id=2")
    contents = [m["content"] for m in resp.json()]
    assert "I prefer Python" in contents, "Should include global"
    assert "I am working on Neo4j" in contents, "Should include case 2"
    assert "I am working on RAG" not in contents, "Should NOT include case 1"

    # Cleanup
    db = TestSessionLocal()
    db.query(Memory).filter(
        Memory.content.in_(["I prefer Python", "I am working on RAG", "I am working on Neo4j"])
    ).delete()
    db.commit()
    db.close()
    print("  PASSED: Case isolation works")


# ============================================================
# TEST 14: Validation Errors
# ============================================================

def test_validation_errors():
    print("\nTEST 14: Validation Errors")
    client = TestClient(main_mod.app)

    # Invalid memory type
    resp = client.post("/memories", json={
        "content": "test",
        "memory_type": "invalid_type",
    })
    assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"

    # Invalid importance (too high)
    resp = client.post("/memories", json={
        "content": "test",
        "importance": 1.5,
    })
    assert resp.status_code == 422

    # Invalid importance (negative)
    resp = client.post("/memories", json={
        "content": "test",
        "importance": -0.1,
    })
    assert resp.status_code == 422

    # Memory not found
    resp = client.get("/memories/99999")
    assert resp.status_code == 404

    # Case not found
    resp = client.get("/cases/99999")
    assert resp.status_code == 404

    print("  PASSED: All validation errors handled correctly")


# ============================================================
# TEST 15: Chat History
# ============================================================

def test_chat_history():
    print("\nTEST 15: Chat History")
    client = TestClient(main_mod.app)
    case_id = setup_test_case()

    resp = client.post(f"/cases/{case_id}/chats", json={
        "title": "History Test",
        "chat_mode": "normal",
    })
    session_id = resp.json()["id"]

    # Send two messages
    with patch("app.chat.create_llm", return_value=fake_llm):
        resp1 = client.post(f"/chats/{session_id}/messages", json={
            "content": "Hello",
        })
        assert resp1.status_code == 200

        resp2 = client.post(f"/chats/{session_id}/messages", json={
            "content": "How are you?",
        })
        assert resp2.status_code == 200

    # Get history
    resp = client.get(f"/chats/{session_id}")
    assert resp.status_code == 200
    data = resp.json()
    messages = data["messages"]
    assert len(messages) == 4, f"Expected 4 messages (2 user + 2 assistant), got {len(messages)}"

    # Check roles
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "user", "assistant"]

    # Check content
    assert messages[0]["content"] == "Hello"
    assert messages[2]["content"] == "How are you?"

    # Cleanup
    db = TestSessionLocal()
    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
    db.query(ChatSession).filter(ChatSession.id == session_id).delete()
    db.query(Case).filter(Case.id == case_id).delete()
    db.commit()
    db.close()
    print("  PASSED: Chat history correct")


# ============================================================
# TEST 16: Short-Term Memory (Conversation History)
# ============================================================

def test_short_term_memory():
    print("\nTEST 16: Short-Term Memory")
    client = TestClient(main_mod.app)
    case_id = setup_test_case()

    resp = client.post(f"/cases/{case_id}/chats", json={
        "title": "Short-term Test",
        "chat_mode": "normal",
    })
    session_id = resp.json()["id"]

    with patch("app.chat.create_llm", return_value=fake_llm):
        # First message
        client.post(f"/chats/{session_id}/messages", json={
            "content": "My project uses FastAPI.",
        })

        # Second message - should have context from first
        resp = client.post(f"/chats/{session_id}/messages", json={
            "content": "What framework did I just mention?",
        })

    assert resp.status_code == 200
    answer = resp.json()["content"]
    print(f"  Answer: {answer}")

    # Cleanup
    db = TestSessionLocal()
    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
    db.query(ChatSession).filter(ChatSession.id == session_id).delete()
    db.query(Case).filter(Case.id == case_id).delete()
    db.commit()
    db.close()
    print("  PASSED: Short-term memory passed to LLM")


# ============================================================
# TEST 17: Response Schema Correctness
# ============================================================

def test_response_schema():
    print("\nTEST 17: Response Schema Correctness")
    client = TestClient(main_mod.app)

    # Memory response schema
    resp = client.post("/memories", json={
        "content": "Schema test",
        "memory_type": "fact",
    })
    data = resp.json()
    required_fields = ["id", "user_id", "content", "memory_type", "case_id",
                       "importance", "created_at", "updated_at"]
    for field in required_fields:
        assert field in data, f"Missing field '{field}' in memory response"

    # Chat session response schema
    case_id = setup_test_case()
    resp = client.post(f"/cases/{case_id}/chats", json={
        "title": "Schema Test",
        "chat_mode": "normal",
    })
    data = resp.json()
    required_fields = ["id", "case_id", "title", "chat_mode", "created_at", "updated_at"]
    for field in required_fields:
        assert field in data, f"Missing field '{field}' in session response"

    # Chat message response schema
    with patch("app.chat.create_llm", return_value=fake_llm):
        resp = client.post(f"/chats/{data['id']}/messages", json={
            "content": "Test message",
        })
    msg = resp.json()
    required_fields = ["id", "session_id", "role", "content", "created_at",
                       "sources", "chunk_count"]
    for field in required_fields:
        assert field in msg, f"Missing field '{field}' in message response"
    assert isinstance(msg["sources"], list)
    assert isinstance(msg["chunk_count"], int)

    # Cleanup
    db = TestSessionLocal()
    db.query(ChatMessage).filter(ChatMessage.session_id == data["id"]).delete()
    db.query(ChatSession).filter(ChatSession.id == data["id"]).delete()
    db.query(Case).filter(Case.id == case_id).delete()
    db.commit()
    db.close()
    print("  PASSED: All response schemas correct")


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":
    tests = [
        test_health,
        test_create_case,
        test_create_memory,
        test_list_memories,
        test_update_memory,
        test_delete_memory,
        test_create_chat_session,
        test_send_normal_chat,
        test_send_document_chat,
        test_memory_persistence,
        test_memory_only_rag,
        test_user_isolation,
        test_case_isolation,
        test_validation_errors,
        test_chat_history,
        test_short_term_memory,
        test_response_schema,
    ]

    passed = 0
    failed = 0
    errors = []

    print("=" * 60)
    print("FASTAPI API INTEGRATION TEST SUITE")
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
