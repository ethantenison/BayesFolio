"""Load isolated cached-comparison imports for deterministic tests.

Usage: loaded automatically by pytest in minimum_seven_hybrid_v1.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
