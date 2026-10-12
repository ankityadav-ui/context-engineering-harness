
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from app.context_engine.database import Base

from sqlalchemy import Boolean

from sqlalchemy import UniqueConstraint


class Conversation(Base):
    # Map this class to the conversations table.
    __tablename__ = "conversations"

    # Unique identifier for each conversation.
    id = Column(String(36), primary_key=True)

    # Optional human-readable conversation title.
    title = Column(String(200), nullable=True)

    # Record when the conversation was created.
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Connect one conversation to all its messages.
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.id",
    )


class Message(Base):
    # Map this class to the messages table.
    __tablename__ = "messages"

    # Auto-incrementing message identifier.
    id = Column(Integer, primary_key=True, index=True)

    # Associate this message with its conversation.
    conversation_id = Column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Store whether the message came from the user or assistant.
    role = Column(String(20), nullable=False)

    # Store the actual message content.
    content = Column(Text, nullable=False)

    # Record when the message was created.
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Link this message back to its conversation.
    conversation = relationship(
        "Conversation",
        back_populates="messages",
    )



class ConversationSummary(Base):
    # Store one current summary for each conversation.
    __tablename__ = "conversation_summaries"

    # Ensure that a conversation has at most one summary record.
    __table_args__ = (
        UniqueConstraint(
            "conversation_id",
            name="uq_summary_conversation_id",
        ),
    )

    # Each conversation summary has its own database ID.
    id = Column(Integer, primary_key=True, index=True)

    # Associate the summary with its conversation.
    conversation_id = Column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Store the generated summary text.
    summary_text = Column(Text, nullable=False)

    # Track the last message included in this summary.
    covered_through_message_id = Column(
        Integer,
        nullable=False,
        default=0,
    )

    # Record the query that shaped this summary.
    last_query = Column(Text, nullable=False)

    # Record when the summary was last updated.
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class Memory(Base):
    # Store durable information for future conversations.
    __tablename__ = "memories"

    # Unique identifier for each memory.
    id = Column(Integer, primary_key=True, index=True)

    # Identify whose memory this record belongs to.
    user_id = Column(String(100), nullable=False, index=True)

    # Classify the information being stored.
    memory_type = Column(String(20), nullable=False, index=True)

    # Store the actual fact, preference, goal, context, or note.
    content = Column(Text, nullable=False)

    # Assign a priority from 1 (low) to 5 (high).
    importance = Column(Integer, nullable=False, default=3)

    # Optionally record which conversation produced this memory.
    conversation_id = Column(
        String(36),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Record when the memory was created.
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Record when the memory was last updated.
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
