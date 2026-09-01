from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


class Base(DeclarativeBase):
    pass


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True
    )

    case_id: Mapped[int] = mapped_column(
        ForeignKey("cases.id"),
        nullable=False,
        index=True
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default="New Chat"
    )

    chat_mode: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="document"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )

    case: Mapped["Case"] = relationship(
        back_populates="chat_sessions"
    )

    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at"
    )


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    documents: Mapped[list["Document"]] = relationship(
        back_populates="case",
        cascade="all, delete-orphan"
    )

    chat_sessions: Mapped[list["ChatSession"]] = relationship(
        back_populates="case",
        cascade="all, delete-orphan"
    )

    eval_queries: Mapped[list["EvalQuery"]] = relationship(
        back_populates="case",
        cascade="all, delete-orphan"
    )

    memories: Mapped[list["Memory"]] = relationship(
        back_populates="case"
    )


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True
    )

    case_id: Mapped[int] = mapped_column(
        ForeignKey("cases.id"),
        nullable=False,
        index=True
    )

    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    file_path: Mapped[str] = mapped_column(
        String(1000),
        nullable=False
    )

    text_content: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    case: Mapped["Case"] = relationship(
        back_populates="documents"
    )

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True
    )

    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id"),
        nullable=False,
        index=True
    )

    chunk_index: Mapped[int] = mapped_column(
        nullable=False
    )

    text: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    document: Mapped["Document"] = relationship(
        back_populates="chunks"
    )


# ============================================================
# EVALUATION
# ============================================================


class EvalQuery(Base):
    __tablename__ = "eval_queries"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True
    )

    case_id: Mapped[int] = mapped_column(
        ForeignKey("cases.id"),
        nullable=False,
        index=True
    )

    query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    expected_document_ids: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="[]",
    )

    expected_chunk_indices: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        default=None,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    case: Mapped["Case"] = relationship(
        back_populates="eval_queries"
    )

    results: Mapped[list["EvalResult"]] = relationship(
        back_populates="eval_query",
        cascade="all, delete-orphan",
    )


class EvalResult(Base):
    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True
    )

    eval_query_id: Mapped[int] = mapped_column(
        ForeignKey("eval_queries.id"),
        nullable=False,
        index=True
    )

    top_k: Mapped[int] = mapped_column(
        nullable=False,
        default=5,
    )

    retrieved_doc_ids: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="[]",
    )

    retrieved_distances: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="[]",
    )

    hit_at_k: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
    )

    precision_at_k: Mapped[float] = mapped_column(
        nullable=False,
        default=0.0,
    )

    recall_at_k: Mapped[float] = mapped_column(
        nullable=False,
        default=0.0,
    )

    reciprocal_rank: Mapped[float] = mapped_column(
        nullable=False,
        default=0.0,
    )

    passed: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    eval_query: Mapped["EvalQuery"] = relationship(
        back_populates="results"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True
    )

    session_id: Mapped[int] = mapped_column(
        ForeignKey("chat_sessions.id"),
        nullable=False,
        index=True
    )

    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    session: Mapped["ChatSession"] = relationship(
        back_populates="messages"
    )


# ============================================================
# LONG-TERM MEMORY
# ============================================================


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True
    )

    user_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        default="default_user",
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    memory_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="fact",
    )

    case_id: Mapped[int | None] = mapped_column(
        ForeignKey("cases.id"),
        nullable=True,
        index=True,
    )

    importance: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.5,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    case: Mapped["Case"] = relationship(
        back_populates="memories"
    )