
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import os


# Find the project root, where the shared .env file lives.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Load the existing project's environment variables.
load_dotenv(PROJECT_ROOT / ".env")

# Read the PostgreSQL connection URL.
DATABASE_URL = os.getenv("DATABASE_URL")

# Stop early if the connection URL has not been configured.
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not configured.")

# Create the SQLAlchemy engine for PostgreSQL.
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

# Create a reusable factory for database sessions.
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)

# Base class for all SQLAlchemy database models.
Base = declarative_base()
