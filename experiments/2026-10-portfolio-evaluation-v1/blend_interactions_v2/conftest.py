"""Expose the isolated interaction study to pytest; no financial I/O.

Usage: automatically loaded by pytest in this experiment directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
