"""Configure isolated research imports without changing any sealed sources.

Usage: automatically loaded by pytest in allocation_numerics_v2.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "approved_scope_v1"))

import study_common  # noqa: E402, F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
