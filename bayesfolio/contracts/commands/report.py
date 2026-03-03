from __future__ import annotations

from pydantic import Field

from bayesfolio.contracts.base import SchemaName, VersionedContract


class ReportCommand(VersionedContract):
    """Command to assemble a final portfolio report."""

    schema: SchemaName = Field(default=SchemaName.REPORT_COMMAND, const=True)
    schema_version: str = Field(default="0.1.0", const=True)
    run_id: str
    include_artifacts: bool = True
