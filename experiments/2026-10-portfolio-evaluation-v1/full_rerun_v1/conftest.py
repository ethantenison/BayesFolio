"""Locate read-only experiment adapters for isolated full-rerun unit tests."""

import sys
from pathlib import Path

_framework = Path(__file__).resolve().parent.parent
sys.path.extend([str(_framework), str(_framework / "selection_refresh_v1")])
