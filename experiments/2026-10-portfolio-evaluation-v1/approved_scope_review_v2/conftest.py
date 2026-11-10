"""Import the repaired research allocator without editing sealed modules.

Usage: automatically loaded by pytest in approved_scope_review_v2.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "approved_scope_v1"))
sys.path.insert(0, str(HERE.parent / "allocation_numerics_v2"))
sys.path.insert(0, str(HERE))
