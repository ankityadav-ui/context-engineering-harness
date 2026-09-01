"""
Comprehensive tests for the long-term memory system.

Tests cover:
  1. Create memory
  2. Retrieve by keyword
  3. Retrieve by natural language question
  4. build_memory_context
  5. Normal chat (skipped - needs LLM)
  6. RAG + memory (skipped - needs LLM + graph)
  7. User isolation
  8. Case isolation
  9. Update memory
  10. Delete memory
  11. Empty database safety
  12. FastAPI import check (skipped - slow)
  13. Schema validation
  14. list_memories with memory_type filter
  15. store_extracted_memories
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, Memory
from app.memory import (
    create_memory,
    list_memories,
    get_memory,
    update_memory,
    delete_memory,
    retrieve_relevant_memories,
    build_memory_context,
    store_extracted_memories,
)


# ============================================================
# SETUP: In-memory SQLite database
# ============================================================

engine = create_engine("sqlite:///:memory:", echo=False)
Base.metadata.create_all(engine)
TestSession = sessionmaker(bind=engine)


def get_test_db():
    session = TestSession()
    try:
        yield session
    finally:
        session.close()


def fresh_db():
    """Return a clean session for each test."""
    session = TestSession()
    # Clean all data
    session.query(Memory).delete()
    session.commit()
    return session


# ============================================================
# TEST 1: Create memory
# ============================================================

def test_1_create_memory():
    print("\nTEST 1: Create memory")
    db = fresh_db()

    memory = create_memory(
        content="I am learning LLM fine tuning",
        memory_type="goal",
        user_id="default_user",
        case_id=None,
        importance=0.5,
        db=db,
    )

    assert memory is not None, "Memory should not be None"
    assert memory.id > 0, f"Memory should have an ID, got {memory.id}"
    assert memory.content == "I am learning LLM fine tuning"
    assert memory.memory_type == "goal"
    assert memory.user_id == "default_user"
    assert memory.case_id is None
    assert memory.importance == 0.5
    assert memory.created_at is not None
    assert memory.updated_at is not None

    print(f"  PASSED: Created memory id={memory.id}")
    db.close()


# ============================================================
# TEST 2: Retrieve by keyword "learning"
# ============================================================

def test_2_retrieve_by_keyword():
    print("\nTEST 2: Retrieve by keyword 'learning'")
    db = fresh_db()

    mem = create_memory(
        content="I am learning LLM fine tuning",
        memory_type="goal",
        user_id="default_user",
        importance=0.5,
        db=db,
    )

    results = retrieve_relevant_memories(
        query="learning",
        user_id="default_user",
        db=db,
    )

    assert len(results) >= 1, f"Expected at least 1 result, got {len(results)}"
    contents = [r["content"] for r in results]
    assert "I am learning LLM fine tuning" in contents, (
        f"Expected 'I am learning LLM fine tuning' in {contents}"
    )

    print(f"  PASSED: Found {len(results)} matching memories")
    db.close()


# ============================================================
# TEST 3: Retrieve by natural language question
# ============================================================

def test_3_retrieve_natural_language():
    print("\nTEST 3: Retrieve by 'What am I currently learning?'")
    db = fresh_db()

    mem = create_memory(
        content="I am learning LLM fine tuning",
        memory_type="goal",
        user_id="default_user",
        importance=0.5,
        db=db,
    )

    # Test multiple natural language queries
    queries = [
        "What am I currently learning?",
        "What am I learning?",
        "What am I studying?",
        "Which topic am I learning?",
        "Tell me what I'm currently learning",
        "learning",
    ]

    for query in queries:
        results = retrieve_relevant_memories(
            query=query,
            user_id="default_user",
            db=db,
        )
        contents = [r["content"] for r in results]
        assert "I am learning LLM fine tuning" in contents, (
            f"Failed for query '{query}': expected memory not found in {contents}"
        )

    print(f"  PASSED: All {len(queries)} natural language queries returned the memory")
    db.close()


# ============================================================
# TEST 4: build_memory_context
# ============================================================

def test_4_build_memory_context():
    print("\nTEST 4: build_memory_context")
    db = fresh_db()

    mem = create_memory(
        content="I am learning LLM fine tuning",
        memory_type="goal",
        user_id="default_user",
        importance=0.5,
        db=db,
    )

    context = build_memory_context(
        query="What am I currently learning?",
        user_id="default_user",
        db=db,
    )

    assert context, "Context should not be empty"
    assert "USER MEMORIES:" in context, f"Expected 'USER MEMORIES:' in context, got: {context}"
    assert "[goal]" in context, f"Expected '[goal]' in context, got: {context}"
    assert "I am learning LLM fine tuning" in context, (
        f"Expected memory content in context, got: {context}"
    )

    print(f"  PASSED: Context = {repr(context)}")
    db.close()


# ============================================================
# TEST 5: Normal chat (needs LLM - test memory retrieval only)
# ============================================================

def test_5_normal_chat_memory_retrieval():
    print("\nTEST 5: Normal chat memory retrieval (LLM not available, testing retrieval)")
    db = fresh_db()

    mem = create_memory(
        content="I am learning LLM fine tuning",
        memory_type="goal",
        user_id="default_user",
        importance=0.5,
        db=db,
    )

    # Verify that build_memory_context returns the right context
    # which would be injected into the LLM prompt
    context = build_memory_context(
        query="What am I currently learning?",
        user_id="default_user",
        db=db,
    )

    assert "I am learning LLM fine tuning" in context
    print("  PASSED: Memory retrieval for normal chat works")
    db.close()


# ============================================================
# TEST 7: User isolation
# ============================================================

def test_7_user_isolation():
    print("\nTEST 7: User isolation")
    db = fresh_db()

    # Create memory for user_a
    mem_a = create_memory(
        content="User A private memory",
        memory_type="fact",
        user_id="user_a",
        importance=0.5,
        db=db,
    )

    # Retrieve as user_b
    results = retrieve_relevant_memories(
        query="private memory",
        user_id="user_b",
        db=db,
    )

    assert len(results) == 0, (
        f"User B should not see User A's memories, got {results}"
    )

    # Verify user_a can see their own memory
    results_a = retrieve_relevant_memories(
        query="private memory",
        user_id="user_a",
        db=db,
    )

    assert len(results_a) == 1, (
        f"User A should see their own memory, got {len(results_a)}"
    )

    # Verify get_memory enforces user ownership
    mem = get_memory(
        memory_id=mem_a.id,
        user_id="user_b",
        db=db,
    )
    assert mem is None, "User B should not access User A's memory"

    # Verify update_memory enforces user ownership
    updated = update_memory(
        memory_id=mem_a.id,
        content="hacked",
        user_id="user_b",
        db=db,
    )
    assert updated is None, "User B should not update User A's memory"

    # Verify delete_memory enforces user ownership
    deleted = delete_memory(
        memory_id=mem_a.id,
        user_id="user_b",
        db=db,
    )
    assert deleted is False, "User B should not delete User A's memory"

    print("  PASSED: All user isolation checks passed")
    db.close()


# ============================================================
# TEST 8: Case isolation
# ============================================================

def test_8_case_isolation():
    print("\nTEST 8: Case isolation")
    db = fresh_db()

    # Create global memory (case_id=None)
    global_mem = create_memory(
        content="Global memory",
        memory_type="fact",
        user_id="default_user",
        case_id=None,
        importance=0.5,
        db=db,
    )

    # Create case 1 memory
    case1_mem = create_memory(
        content="Case 1 specific memory",
        memory_type="fact",
        user_id="default_user",
        case_id=1,
        importance=0.5,
        db=db,
    )

    # Create case 2 memory
    case2_mem = create_memory(
        content="Case 2 specific memory",
        memory_type="fact",
        user_id="default_user",
        case_id=2,
        importance=0.5,
        db=db,
    )

    # Retrieve with case_id=1: should get global + case 1, NOT case 2
    results = retrieve_relevant_memories(
        query="memory",
        user_id="default_user",
        case_id=1,
        db=db,
    )

    contents = [r["content"] for r in results]
    assert "Global memory" in contents, "Should include global memory"
    assert "Case 1 specific memory" in contents, "Should include case 1 memory"
    assert "Case 2 specific memory" not in contents, (
        f"Should NOT include case 2 memory, got {contents}"
    )

    # Retrieve without case_id: should get only global
    results_global = retrieve_relevant_memories(
        query="memory",
        user_id="default_user",
        case_id=None,
        db=db,
    )

    contents_global = [r["content"] for r in results_global]
    assert "Global memory" in contents_global, "Should include global memory"
    assert "Case 1 specific memory" not in contents_global, (
        f"Should NOT include case 1 memory when no case_id, got {contents_global}"
    )

    # Test list_memories with case_id
    listed = list_memories(
        user_id="default_user",
        case_id=1,
        db=db,
    )
    listed_contents = [m.content for m in listed]
    assert "Global memory" in listed_contents, "list_memories should include global"
    assert "Case 1 specific memory" in listed_contents, "list_memories should include case 1"
    assert "Case 2 specific memory" not in listed_contents, (
        f"list_memories should NOT include case 2, got {listed_contents}"
    )

    print("  PASSED: All case isolation checks passed")
    db.close()


# ============================================================
# TEST 9: Update memory
# ============================================================

def test_9_update_memory():
    print("\nTEST 9: Update memory")
    db = fresh_db()

    mem = create_memory(
        content="I am learning LLM fine tuning",
        memory_type="goal",
        user_id="default_user",
        importance=0.5,
        db=db,
    )

    updated = update_memory(
        memory_id=mem.id,
        content="I am learning LLM fine tuning and RAG",
        user_id="default_user",
        db=db,
    )

    assert updated is not None, "Update should return the memory"
    assert updated.content == "I am learning LLM fine tuning and RAG", (
        f"Content should be updated, got: {updated.content}"
    )

    # Verify in database
    fetched = get_memory(
        memory_id=mem.id,
        user_id="default_user",
        db=db,
    )
    assert fetched.content == "I am learning LLM fine tuning and RAG", (
        f"Database should reflect update, got: {fetched.content}"
    )

    print("  PASSED: Memory updated successfully")
    db.close()


# ============================================================
# TEST 10: Delete memory
# ============================================================

def test_10_delete_memory():
    print("\nTEST 10: Delete memory")
    db = fresh_db()

    mem = create_memory(
        content="Memory to delete",
        memory_type="fact",
        user_id="default_user",
        importance=0.5,
        db=db,
    )

    mem_id = mem.id

    deleted = delete_memory(
        memory_id=mem_id,
        user_id="default_user",
        db=db,
    )

    assert deleted is True, "Delete should return True"

    # Verify it's gone
    fetched = get_memory(
        memory_id=mem_id,
        user_id="default_user",
        db=db,
    )
    assert fetched is None, "Deleted memory should not be found"

    print("  PASSED: Memory deleted successfully")
    db.close()


# ============================================================
# TEST 11: Empty database safety
# ============================================================

def test_11_empty_database_safety():
    print("\nTEST 11: Empty database safety")
    db = fresh_db()

    # All functions should handle empty database gracefully

    results = retrieve_relevant_memories(
        query="anything",
        user_id="default_user",
        db=db,
    )
    assert results == [], f"Should return empty list, got {results}"

    context = build_memory_context(
        query="anything",
        user_id="default_user",
        db=db,
    )
    assert context == "", f"Should return empty string, got {repr(context)}"

    mem = get_memory(
        memory_id=999,
        user_id="default_user",
        db=db,
    )
    assert mem is None, "Should return None for non-existent memory"

    updated = update_memory(
        memory_id=999,
        content="hacked",
        user_id="default_user",
        db=db,
    )
    assert updated is None, "Should return None for non-existent memory"

    deleted = delete_memory(
        memory_id=999,
        user_id="default_user",
        db=db,
    )
    assert deleted is False, "Should return False for non-existent memory"

    listed = list_memories(
        user_id="default_user",
        db=db,
    )
    assert listed == [], f"Should return empty list, got {listed}"

    # Test with None db
    results_none = retrieve_relevant_memories(
        query="anything",
        db=None,
    )
    assert results_none == []

    context_none = build_memory_context(
        query="anything",
        db=None,
    )
    assert context_none == ""

    mem_none = get_memory(memory_id=1, db=None)
    assert mem_none is None

    updated_none = update_memory(memory_id=1, content="x", db=None)
    assert updated_none is None

    deleted_none = delete_memory(memory_id=1, db=None)
    assert deleted_none is False

    listed_none = list_memories(db=None)
    assert listed_none == []

    print("  PASSED: All empty database / None db cases handled safely")
    db.close()


# ============================================================
# TEST 12: Schema validation
# ============================================================

def test_12_schema_validation():
    print("\nTEST 12: Schema validation")
    from app.schemas import MemoryCreate, MemoryUpdate, MemoryResponse
    from pydantic import ValidationError

    # Valid memory creation
    valid = MemoryCreate(
        content="Test",
        memory_type="goal",
        importance=0.5,
    )
    assert valid.memory_type == "goal"
    assert valid.importance == 0.5

    # All allowed types
    for mt in ["fact", "preference", "context", "note", "goal"]:
        m = MemoryCreate(content="x", memory_type=mt)
        assert m.memory_type == mt

    # Invalid memory type
    try:
        MemoryCreate(content="x", memory_type="invalid_type")
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass

    # Invalid importance - too high
    try:
        MemoryCreate(content="x", importance=1.5)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass

    # Invalid importance - too low
    try:
        MemoryCreate(content="x", importance=-0.1)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass

    # Valid importance boundaries
    MemoryCreate(content="x", importance=0.0)
    MemoryCreate(content="x", importance=1.0)

    # MemoryUpdate with invalid type
    try:
        MemoryUpdate(memory_type="bad")
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass

    # MemoryUpdate with None (valid)
    update = MemoryUpdate()
    assert update.content is None
    assert update.memory_type is None
    assert update.importance is None

    # MemoryUpdate with valid type
    update_goal = MemoryUpdate(memory_type="goal")
    assert update_goal.memory_type == "goal"

    print("  PASSED: All schema validation checks passed")


# ============================================================
# TEST 13: list_memories with memory_type filter
# ============================================================

def test_13_list_memories_memory_type_filter():
    print("\nTEST 13: list_memories with memory_type filter")
    db = fresh_db()

    create_memory(content="Fact 1", memory_type="fact", user_id="u1", importance=0.5, db=db)
    create_memory(content="Fact 2", memory_type="fact", user_id="u1", importance=0.5, db=db)
    create_memory(content="Goal 1", memory_type="goal", user_id="u1", importance=0.6, db=db)
    create_memory(content="Pref 1", memory_type="preference", user_id="u1", importance=0.4, db=db)

    # List all (no filter)
    all_mems = list_memories(user_id="u1", db=db)
    assert len(all_mems) == 4, f"Expected 4, got {len(all_mems)}"

    # Filter by goal
    goals = list_memories(user_id="u1", memory_type="goal", db=db)
    assert len(goals) == 1, f"Expected 1 goal, got {len(goals)}"
    assert goals[0].memory_type == "goal"

    # Filter by fact
    facts = list_memories(user_id="u1", memory_type="fact", db=db)
    assert len(facts) == 2, f"Expected 2 facts, got {len(facts)}"

    # Filter by nonexistent type
    notes = list_memories(user_id="u1", memory_type="note", db=db)
    assert len(notes) == 0, f"Expected 0 notes, got {len(notes)}"

    print("  PASSED: list_memories memory_type filter works correctly")
    db.close()


# ============================================================
# TEST 14: store_extracted_memories
# ============================================================

def test_14_store_extracted_memories():
    print("\nTEST 14: store_extracted_memories")
    db = fresh_db()

    # Test goal extraction
    memories = store_extracted_memories(
        user_message="I want to master deep learning by December",
        assistant_response="That sounds great!",
        user_id="default_user",
        db=db,
    )
    assert len(memories) >= 1, f"Expected at least 1 memory, got {len(memories)}"
    assert memories[0].memory_type == "goal"

    # Test learning extraction
    memories2 = store_extracted_memories(
        user_message="I am learning PyTorch for neural networks",
        assistant_response="Great choice!",
        user_id="default_user",
        db=db,
    )
    assert len(memories2) >= 1

    # Test short message (should not extract)
    memories3 = store_extracted_memories(
        user_message="hi",
        assistant_response="Hello!",
        user_id="default_user",
        db=db,
    )
    assert len(memories3) == 0, "Short messages should not be extracted"

    # Test duplicate prevention
    memories4 = store_extracted_memories(
        user_message="I want to master deep learning by December",
        assistant_response="That sounds great!",
        user_id="default_user",
        db=db,
    )
    assert len(memories4) == 0, "Duplicate should not be created"

    # Test with None db
    memories5 = store_extracted_memories(
        user_message="I want something",
        assistant_response="ok",
        user_id="default_user",
        db=None,
    )
    assert len(memories5) == 0

    print("  PASSED: store_extracted_memories works correctly")
    db.close()


# ============================================================
# TEST 15: Importance doesn't block relevant memories
# ============================================================

def test_15_importance_doesnt_block():
    print("\nTEST 15: Importance=0.5 with matching keyword still returns")
    db = fresh_db()

    mem = create_memory(
        content="I am learning LLM fine tuning",
        memory_type="goal",
        user_id="default_user",
        importance=0.5,
        db=db,
    )

    results = retrieve_relevant_memories(
        query="learning",
        user_id="default_user",
        db=db,
    )

    assert len(results) >= 1, (
        f"Memory with importance=0.5 and matching keyword should be retrievable, "
        f"got {len(results)} results"
    )

    print("  PASSED: Low importance does not block relevant memories")
    db.close()


# ============================================================
# TEST 16: Context budget respect
# ============================================================

def test_16_context_budget():
    print("\nTEST 16: build_memory_context respects MEMORY_MAX_CONTEXT_CHARS")
    db = fresh_db()

    # Create a very long memory
    long_content = "X " * 5000  # ~10000 chars
    create_memory(
        content=long_content,
        memory_type="note",
        user_id="default_user",
        importance=0.5,
        db=db,
    )

    context = build_memory_context(
        query="X",
        user_id="default_user",
        db=db,
    )

    # MEMORY_MAX_CONTEXT_CHARS is 2000 by default
    assert len(context) <= 2000 + 50, (
        f"Context should respect budget (2000 chars), got {len(context)}"
    )

    print(f"  PASSED: Context length = {len(context)} (budget: 2000)")
    db.close()


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":
    tests = [
        test_1_create_memory,
        test_2_retrieve_by_keyword,
        test_3_retrieve_natural_language,
        test_4_build_memory_context,
        test_5_normal_chat_memory_retrieval,
        test_7_user_isolation,
        test_8_case_isolation,
        test_9_update_memory,
        test_10_delete_memory,
        test_11_empty_database_safety,
        test_12_schema_validation,
        test_13_list_memories_memory_type_filter,
        test_14_store_extracted_memories,
        test_15_importance_doesnt_block,
        test_16_context_budget,
    ]

    passed = 0
    failed = 0
    errors = []

    print("=" * 60)
    print("MEMORY SYSTEM TEST SUITE")
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
