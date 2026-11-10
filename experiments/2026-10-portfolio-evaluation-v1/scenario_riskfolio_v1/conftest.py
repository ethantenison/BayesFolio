"""Expose only the isolated scenario/Riskfolio research modules for tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/bayesfolio-scenario-riskfolio-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/bayesfolio-scenario-riskfolio-cache")
sys.path.insert(0, str(Path(__file__).resolve().parent))
