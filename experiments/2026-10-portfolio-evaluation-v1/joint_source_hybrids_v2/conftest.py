"""Expose the isolated joint-source comparison to synthetic tests.

Usage: automatically loaded by pytest in joint_source_hybrids_v2.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
