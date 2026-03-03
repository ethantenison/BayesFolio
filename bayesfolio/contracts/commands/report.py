from __future__ import annotations

from typing import Literal

from bayesfolio.contracts.base import SchemaName, VersionedContract


class ReportCommand(VersionedContract):
    """Command to assemble a final portfolio report."""

    schema: Literal[SchemaName.REPORT_COMMAND] = SchemaName.REPORT_COMMAND
    schema_version: Literal["0.1.0"] = "0.1.0"
    run_id: str
    include_artifacts: bool = True
