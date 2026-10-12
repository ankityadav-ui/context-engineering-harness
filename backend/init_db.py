
# Import the shared database configuration.
from app.context_engine.database import Base, engine

# Import every model so SQLAlchemy registers all tables.
from app.context_engine.models import (
    Conversation,
    Message,
    ConversationSummary,
    Memory,
)


def init_db():
    """Create all registered database tables."""

    # Create tables that do not already exist.
    Base.metadata.create_all(bind=engine)

    print("Database tables initialized successfully.")


# Run initialization when this file is executed directly.
if __name__ == "__main__":
    init_db()
