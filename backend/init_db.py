
from app.context_engine.database import Base, engine

# Import the models so SQLAlchemy knows which tables exist.
from app.context_engine.models import Conversation, Message


def initialize_database():
    # Create any missing tables in the configured database.
    Base.metadata.create_all(bind=engine)

    print("Conversation database tables are ready.")


if __name__ == "__main__":
    initialize_database()
