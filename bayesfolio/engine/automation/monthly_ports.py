"""Structural runtime-I/O boundary for monthly portfolio execution.

The engine owns monthly computation and sequencing but receives every runtime
side effect through :class:`MonthlyPortfolioRunIO`. Implementations load feature
tables containing decimal-return targets and persist contract, scenario,
prediction, and weight artifacts without making modeling decisions.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol

import pandas as pd
from pydantic import BaseModel

from bayesfolio.contracts.results.report import ArtifactPointer


class MonthlyPortfolioRunIO(Protocol):
    """Side-effect capabilities required by the monthly engine workflow."""

    def start_run(self, *, run_name: str) -> AbstractContextManager[object]:
        """Start the configured run-tracking context.

        Args:
            run_name: Stable label for the monthly run.

        Returns:
            Context manager whose lifetime covers the complete monthly run.
        """

        ...

    def log_contract(self, name: str, payload: BaseModel) -> None:
        """Log one validated contract payload.

        Args:
            name: Stable artifact name without a file extension.
            payload: Validated Pydantic contract to serialize.
        """

        ...

    def log_source_revision(self) -> None:
        """Log the current source revision when one can be discovered."""

        ...

    def load_parquet(self, uri: str) -> pd.DataFrame:
        """Load a feature table from a parquet path or URI.

        Args:
            uri: Parquet artifact location returned by feature construction.

        Returns:
            Feature table whose return-valued columns use decimal units.
        """

        ...

    def save_csv(self, frame: pd.DataFrame, relative_path: str) -> ArtifactPointer:
        """Persist a table as index-free CSV beneath the configured run root.

        Args:
            frame: Table to serialize without its pandas index.
            relative_path: Relative output path beneath the artifact root.

        Returns:
            Pointer containing path, SHA-256 digest, and byte size.
        """

        ...

    def save_json_contract(self, payload: BaseModel, relative_path: str) -> ArtifactPointer:
        """Persist a contract as deterministic indented JSON.

        Args:
            payload: Validated Pydantic contract to serialize.
            relative_path: Relative output path beneath the artifact root.

        Returns:
            Pointer containing path, SHA-256 digest, and byte size.
        """

        ...
