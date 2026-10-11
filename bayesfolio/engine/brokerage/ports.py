"""Engine-owned structural ports for future brokerage account adapters."""

from __future__ import annotations

from typing import Protocol

from bayesfolio.contracts.results.brokerage import BrokerageAccountSnapshot


class BrokerageAccountReader(Protocol):
    """Read-only account interface implemented by a future broker adapter."""

    def get_account_snapshot(self, *, account_reference: str) -> BrokerageAccountSnapshot:
        """Return a point-in-time account snapshot without placing orders.

        Args:
            account_reference: Opaque non-sensitive adapter account reference.

        Returns:
            Validated account balances and positions.
        """

        ...
