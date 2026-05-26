"""Default logging setup for the fmls_parser library.

Library code uses `logging.getLogger("fmls.*")`. Importing this module sets
up a sensible default handler if the application hasn't configured one.
"""

from __future__ import annotations

import logging
import os
import sys


def setup_logging(level: str | None = None) -> None:
    """Idempotent — safe to call multiple times."""
    lvl_name = (level or os.getenv("FMLS_LOG_LEVEL") or "INFO").upper()
    lvl = getattr(logging, lvl_name, logging.INFO)

    root = logging.getLogger("fmls")
    if root.handlers:
        root.setLevel(lvl)
        return
    h = logging.StreamHandler(sys.stderr)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s", datefmt="%H:%M:%S")
    h.setFormatter(fmt)
    root.addHandler(h)
    root.setLevel(lvl)
    root.propagate = False
