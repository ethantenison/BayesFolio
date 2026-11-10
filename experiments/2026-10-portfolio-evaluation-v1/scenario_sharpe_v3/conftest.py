"""Expose sign-safe Sharpe research modules without importing fitting code."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/bayesfolio-scenario-riskfolio-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/bayesfolio-scenario-riskfolio-cache")
HERE = Path(__file__).resolve().parent
sys.path.extend([str(HERE), str(HERE.parent / "scenario_riskfolio_v1")])
