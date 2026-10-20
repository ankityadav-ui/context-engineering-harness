"""Shared safeguard for the project's manual integration scripts.

Manual scripts are named ``manual_*.py`` so pytest never collects them.
They talk to live external services (PostgreSQL, Neo4j, Pinecone,
Gemini) and several of them write test data.

To avoid accidental writes or paid API calls, every manual script calls
``require_live_opt_in()`` before performing any external operation.
Opt in explicitly with the environment variable:

    Windows (cmd):   set CONTEXT_ENGINE_ALLOW_LIVE_TESTS=1
    bash / Git Bash: export CONTEXT_ENGINE_ALLOW_LIVE_TESTS=1

Then run a script, for example:

    python scripts/manual_short_term_memory.py
"""

import os


# Environment variable that must be set to a truthy value to allow
# a manual script to touch external services.
LIVE_OPT_IN_ENV_VAR = "CONTEXT_ENGINE_ALLOW_LIVE_TESTS"

_TRUE_VALUES = {"1", "true", "yes", "on"}


def live_tests_enabled() -> bool:
    """Return True when the operator has opted into live operations."""

    value = os.getenv(LIVE_OPT_IN_ENV_VAR, "")
    return value.strip().lower() in _TRUE_VALUES


def require_live_opt_in(action: str) -> None:
    """Abort unless the operator explicitly opted into live operations.

    ``action`` describes what the script would do (and why it needs
    explicit permission), so the message is actionable.
    """

    if live_tests_enabled():
        return

    raise SystemExit(
        "\n"
        "This is a manual integration script (disabled by default).\n"
        f"It would perform a live operation: {action}.\n\n"
        "To allow it, opt in explicitly and run the script again:\n"
        f"  Windows (cmd):   set {LIVE_OPT_IN_ENV_VAR}=1\n"
        f"  bash:            export {LIVE_OPT_IN_ENV_VAR}=1\n"
    )
