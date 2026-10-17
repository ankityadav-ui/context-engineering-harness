"""Make the backend package importable for the stand-alone manual scripts.

The ``manual_*.py`` scripts live in ``backend/scripts/`` but import the
application as ``app.*``, which lives in ``backend/``. Running a script
directly puts ``backend/scripts/`` (not ``backend/``) on ``sys.path``, so
every script imports this module first to add the backend directory.

Import it before any ``app.*`` import::

    import _bootstrap  # noqa: F401  (adds backend/ to sys.path)
    from app.rag.embedding_service import generate_embedding
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
