"""Expose only isolated expected-rank experiment modules to pytest.

Usage: loaded by pytest automatically. No financial writes, fits or network.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
