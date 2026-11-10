"""Expose isolated Schwab benchmark modules for synthetic tests.

Usage: loaded automatically by pytest; no financial writes or network calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
