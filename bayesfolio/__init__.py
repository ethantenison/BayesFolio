"""BayesFolio package."""

from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message=r'Field name "schema" in ".*" shadows an attribute in parent ".*"',
    category=UserWarning,
)
