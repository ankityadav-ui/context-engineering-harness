import os

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# RAG RETRIEVAL
# ============================================================

# Distance threshold for relevance filtering.
# ChromaDB cosine distance:
# 0 = identical, 2 = opposite.
RAG_DISTANCE_THRESHOLD = float(
    os.getenv("RAG_DISTANCE_THRESHOLD", "1.8")
)

# Default number of chunks retrieved.
RAG_DEFAULT_TOP_K = int(
    os.getenv("RAG_DEFAULT_TOP_K", "5")
)

# Maximum number of chunks allowed.
RAG_MAX_TOP_K = int(
    os.getenv("RAG_MAX_TOP_K", "20")
)


# ============================================================
# CONTEXT BUDGET
# ============================================================

# Maximum characters allowed in final context.
RAG_MAX_CONTEXT_CHARS = int(
    os.getenv("RAG_MAX_CONTEXT_CHARS", "8000")
)

# Approximate tokens per character.
RAG_TOKENS_PER_CHAR = float(
    os.getenv("RAG_TOKENS_PER_CHAR", "0.25")
)


# ============================================================
# DEDUPLICATION
# ============================================================

# Minimum Jaccard similarity for duplicate detection.
RAG_DEDUP_THRESHOLD = float(
    os.getenv("RAG_DEDUP_THRESHOLD", "0.85")
)


# ============================================================
# SHORT-TERM MEMORY
# ============================================================

# Maximum recent conversation messages included in context.
MEMORY_MAX_HISTORY_MESSAGES = int(
    os.getenv("MEMORY_MAX_HISTORY_MESSAGES", "10")
)


# ============================================================
# LONG-TERM MEMORY
# ============================================================

# Default user identifier.
MEMORY_DEFAULT_USER_ID = os.getenv(
    "MEMORY_DEFAULT_USER_ID",
    "default_user",
)

# Maximum long-term memories retrieved per query.
MEMORY_MAX_RETRIEVED = int(
    os.getenv("MEMORY_MAX_RETRIEVED", "5")
)

# Maximum characters for long-term memory context.
MEMORY_MAX_CONTEXT_CHARS = int(
    os.getenv("MEMORY_MAX_CONTEXT_CHARS", "2000")
)


# ============================================================
# MEMORY EXTRACTION
# ============================================================

# Enable/disable automatic memory extraction.
MEMORY_AUTO_EXTRACT_ENABLED = (
    os.getenv(
        "MEMORY_AUTO_EXTRACT_ENABLED",
        "true",
    ).lower()
    == "true"
)

# Maximum memories extracted per chat exchange.
MEMORY_EXTRACT_MAX_PER_MESSAGE = int(
    os.getenv(
        "MEMORY_EXTRACT_MAX_PER_MESSAGE",
        "3",
    )
)

# Minimum user-message word count required for extraction.
MEMORY_EXTRACT_MIN_WORDS = int(
    os.getenv(
        "MEMORY_EXTRACT_MIN_WORDS",
        "3",
    )
)


# ============================================================
# LLM
# ============================================================

LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "gemini",
).lower().strip()

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gemini-3.6-flash",
)


# Provider-specific API keys.
LLM_API_KEY_MAP = {
    "gemini": os.getenv("GEMINI_API_KEY"),
    "groq": os.getenv("GROQ_API_KEY"),
    "mistral": os.getenv("MISTRAL_API_KEY"),
    "openrouter": os.getenv("OPENROUTER_API_KEY"),
}


# API key for the currently selected provider.
LLM_API_KEY = LLM_API_KEY_MAP.get(
    LLM_PROVIDER
)


# ============================================================
# LLM TIMEOUT & RETRY
# ============================================================

# Timeout for individual LLM requests (seconds).
LLM_TIMEOUT_SECONDS = float(
    os.getenv("LLM_TIMEOUT_SECONDS", "60")
)

# Maximum number of retries after the initial attempt.
# max_retries=2 means up to 3 total attempts.
LLM_MAX_RETRIES = int(
    os.getenv("LLM_MAX_RETRIES", "2")
)

# Delay between retries (seconds).
LLM_RETRY_DELAY_SECONDS = float(
    os.getenv("LLM_RETRY_DELAY_SECONDS", "1")
)


# ============================================================
# KNOWLEDGE GRAPH / NEO4J
# ============================================================

NEO4J_URI = os.getenv(
    "NEO4J_URI",
    "",
)

NEO4J_USERNAME = os.getenv(
    "NEO4J_USERNAME",
    "neo4j",
)

NEO4J_PASSWORD = os.getenv(
    "NEO4J_PASSWORD",
    "",
)

NEO4J_DATABASE = os.getenv(
    "NEO4J_DATABASE",
    "neo4j",
)