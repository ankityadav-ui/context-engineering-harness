import os

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# RAG RETRIEVAL
# ============================================================

# Distance threshold for relevance filtering.
# ChromaDB cosine distance: 0 = identical, 2 = opposite.
# Values below this threshold are considered relevant.
RAG_DISTANCE_THRESHOLD = float(
    os.getenv("RAG_DISTANCE_THRESHOLD", "1.8")
)

# Default top_k for retrieval
RAG_DEFAULT_TOP_K = int(
    os.getenv("RAG_DEFAULT_TOP_K", "5")
)

# Maximum top_k allowed
RAG_MAX_TOP_K = int(
    os.getenv("RAG_MAX_TOP_K", "20")
)


# ============================================================
# CONTEXT BUDGET
# ============================================================

# Maximum characters allowed in the final context.
# Chunks are added in rank order until this limit is reached.
RAG_MAX_CONTEXT_CHARS = int(
    os.getenv("RAG_MAX_CONTEXT_CHARS", "8000")
)

# Approximate tokens per character (English avg ~4 chars/token)
RAG_TOKENS_PER_CHAR = float(
    os.getenv("RAG_TOKENS_PER_CHAR", "0.25")
)


# ============================================================
# DEDUPLICATION
# ============================================================

# Minimum similarity (Jaccard) to consider two chunks duplicates.
# 1.0 = exact match only, 0.8 = allow near-duplicates.
RAG_DEDUP_THRESHOLD = float(
    os.getenv("RAG_DEDUP_THRESHOLD", "0.85")
)


# ============================================================
# SHORT-TERM MEMORY (Conversation History)
# ============================================================

# Maximum number of recent messages to include as conversation history.
MEMORY_MAX_HISTORY_MESSAGES = int(
    os.getenv("MEMORY_MAX_HISTORY_MESSAGES", "10")
)


# ============================================================
# LONG-TERM MEMORY
# ============================================================

# Default user identifier (no auth in this app).
MEMORY_DEFAULT_USER_ID = os.getenv(
    "MEMORY_DEFAULT_USER_ID", "default_user"
)

# Maximum number of long-term memories to retrieve per query.
MEMORY_MAX_RETRIEVED = int(
    os.getenv("MEMORY_MAX_RETRIEVED", "5")
)

# Maximum characters for long-term memory context injection.
MEMORY_MAX_CONTEXT_CHARS = int(
    os.getenv("MEMORY_MAX_CONTEXT_CHARS", "2000")
)


# ============================================================
# MEMORY EXTRACTION (Auto-extract from conversations)
# ============================================================

# Enable/disable automatic memory extraction after chat responses.
MEMORY_AUTO_EXTRACT_ENABLED = os.getenv(
    "MEMORY_AUTO_EXTRACT_ENABLED", "true"
).lower() == "true"

# Maximum memories to extract per single chat exchange.
MEMORY_EXTRACT_MAX_PER_MESSAGE = int(
    os.getenv("MEMORY_EXTRACT_MAX_PER_MESSAGE", "3")
)

# Minimum word count in user message to trigger extraction.
# Short messages like "hi" or "ok" are skipped.
MEMORY_EXTRACT_MIN_WORDS = int(
    os.getenv("MEMORY_EXTRACT_MIN_WORDS", "3")
)


# ============================================================
# LLM
# ============================================================

LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "gemini",
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gemini-3.6-flash",
)


LLM_API_KEY_MAP = {
    "gemini": os.getenv("GEMINI_API_KEY"),
    "groq": os.getenv("GROQ_API_KEY"),
    "mistral": os.getenv("MISTRAL_API_KEY"),
    "openrouter": os.getenv("OPENROUTER_API_KEY"),
}

LLM_API_KEY = LLM_API_KEY_MAP.get(
    LLM_PROVIDER.lower().strip()
)

# ============================================================
# KNOWLEDGE GRAPH / NEO4J
# ============================================================

NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")