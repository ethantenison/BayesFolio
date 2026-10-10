"""Engine-owned port for future brokerage performance-history adapters."""

from __future__ import annotations

from datetime import date
from typing import Protocol

from bayesfolio.contracts.commands.performance import PerformancePeriod


class PerformancePeriodReader(Protocol):
    """Read externally measured, cash-flow-aware portfolio return periods."""

    def get_performance_periods(
        self,
        *,
        account_reference: str,
        start_date: date,
        end_date: date,
        benchmark_id: str,
    ) -> list[PerformancePeriod]:
        """Return aligned portfolio, benchmark, P/L, and fee observations.

        Args:
            account_reference: Opaque non-sensitive adapter account reference.
            start_date: Inclusive measurement start date.
            end_date: Inclusive measurement end date.
            benchmark_id: Benchmark requested for aligned return measurement.

        Returns:
            Ordered, non-overlapping performance periods.
        """

        ...
