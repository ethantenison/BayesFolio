"""Expose isolated rank-risk experiment imports for synthetic unit tests.

Usage: loaded by pytest. No financial I/O, allocations or network at import.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.extend(
    [
        str(HERE.parent / "approved_scope_v1"),
        str(HERE.parent / "approved_scope_review_v2"),
        str(HERE.parent / "allocation_numerics_v2"),
    ]
)
