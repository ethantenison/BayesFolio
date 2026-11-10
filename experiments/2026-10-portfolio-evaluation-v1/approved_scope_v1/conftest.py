"""Configure research-only import paths and one-thread numerical controls.

Usage: loaded by pytest for this isolated experiment; no I/O or fits performed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import study_common  # noqa: E402, F401
