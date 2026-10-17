"""Manual check: print live Pinecone index statistics.

Queries the Pinecone index, so it only runs on demand.
Run from backend/:  python scripts/manual_pinecone.py
"""

import _bootstrap  # noqa: F401 - adds backend/ to sys.path
from app.rag.pinecone_service import get_index

from manual_guard import require_live_opt_in


def print_index_statistics() -> None:
    # Request statistics from the Pinecone index.
    # The client is created lazily here, not at import time.
    stats = get_index().describe_index_stats()

    print("Pinecone index statistics:")
    print(stats)


if __name__ == "__main__":
    require_live_opt_in("query the live Pinecone index")

    print_index_statistics()
