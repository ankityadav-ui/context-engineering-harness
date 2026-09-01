from datetime import datetime

from pydantic import BaseModel, field_validator


# ============================================================
# CASES
# ============================================================


class CaseCreate(BaseModel):
    name: str
    description: str | None = None


class CaseResponse(BaseModel):
    id: int
    name: str
    description: str | None
    created_at: datetime

    class Config:
        from_attributes = True


# ============================================================
# DOCUMENTS
# ============================================================


# ============================================================
# CHAT
# ============================================================


class ChatMessageCreate(BaseModel):
    content: str


class ChatMessageResponse(BaseModel):
    id: int
    session_id: int
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


class ChatSessionCreate(BaseModel):
    title: str = "New Chat"
    chat_mode: str = "document"


class ChatSessionResponse(BaseModel):
    id: int
    case_id: int
    title: str
    chat_mode: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ChatHistoryResponse(BaseModel):
    id: int
    case_id: int
    title: str
    chat_mode: str
    created_at: datetime
    updated_at: datetime
    messages: list[ChatMessageResponse]


class SourceChunk(BaseModel):
    document_id: int | None = None
    filename: str = "unknown"
    chunk_index: int | None = None
    text: str | None = None
    distance: float | None = None


class ChatMessageWithSourcesResponse(BaseModel):
    id: int
    session_id: int
    role: str
    content: str
    created_at: datetime
    sources: list[SourceChunk]
    chunk_count: int


# ============================================================
# EVALUATION
# ============================================================


class EvalQueryCreate(BaseModel):
    query: str
    description: str | None = None
    expected_document_ids: list[int] = []
    expected_chunk_indices: list[int] | None = None


class EvalQueryResponse(BaseModel):
    id: int
    case_id: int
    query: str
    description: str | None
    expected_document_ids: list[int]
    expected_chunk_indices: list[int] | None
    created_at: datetime


class EvalResultResponse(BaseModel):
    id: int
    eval_query_id: int
    top_k: int
    retrieved_doc_ids: list[int]
    retrieved_distances: list[float]
    hit_at_k: bool
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    passed: bool
    created_at: datetime


class EvalQueryDetail(BaseModel):
    id: int
    case_id: int
    query: str
    description: str | None
    expected_document_ids: list[int]
    expected_chunk_indices: list[int] | None
    created_at: datetime
    last_result: EvalResultResponse | None = None


class EvalAggregate(BaseModel):
    total_queries: int = 0
    queries_with_results: int = 0
    total_passed: int = 0
    total_failed: int = 0
    hit_at_k: float = 0.0
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    avg_distance: float | None = None


class EvalRunResponse(BaseModel):
    case_id: int
    top_k: int
    aggregate: EvalAggregate
    query_results: list[dict]


# ============================================================
# MEMORY
# ============================================================


class MemoryCreate(BaseModel):
    content: str
    memory_type: str = "fact"
    case_id: int | None = None
    importance: float = 0.5

    @field_validator("memory_type")
    @classmethod
    def validate_memory_type(cls, v):
        allowed = {
            "fact",
            "preference",
            "context",
            "note",
            "goal",
        }

        if v not in allowed:
            raise ValueError(
                f"memory_type must be one of: {', '.join(sorted(allowed))}"
            )

        return v

    @field_validator("importance")
    @classmethod
    def validate_importance(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                "importance must be between 0.0 and 1.0"
            )

        return v


class MemoryUpdate(BaseModel):
    content: str | None = None
    memory_type: str | None = None
    importance: float | None = None

    @field_validator("memory_type")
    @classmethod
    def validate_memory_type(cls, v):
        if v is None:
            return v

        allowed = {
            "fact",
            "preference",
            "context",
            "note",
            "goal",
        }

        if v not in allowed:
            raise ValueError(
                f"memory_type must be one of: {', '.join(sorted(allowed))}"
            )

        return v

    @field_validator("importance")
    @classmethod
    def validate_importance(cls, v):
        if v is None:
            return v

        if not 0.0 <= v <= 1.0:
            raise ValueError(
                "importance must be between 0.0 and 1.0"
            )

        return v


class MemoryResponse(BaseModel):
    id: int
    user_id: str
    content: str
    memory_type: str
    case_id: int | None
    importance: float
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
