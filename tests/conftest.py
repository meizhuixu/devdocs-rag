"""Pytest fixtures shared across the suite.

Sets `RERANKER_TYPE=identity` *before* any test imports `devdocs_rag.config`,
so the test process never loads the cross-encoder weights. Production keeps
`cross_encoder` as the default; the M2 probe explicitly opts into it.
"""

from __future__ import annotations

import os

# Must run at import time, before any test module imports config / settings.
os.environ.setdefault("RERANKER_TYPE", "identity")
