"""Commands for non-executing brokerage rebalance planning."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import Field

from bayesfolio.contracts.base import SchemaName, VersionedContract
from bayesfolio.contracts.results.brokerage import BrokerageAccountSnapshot
from bayesfolio.contracts.results.monthly_decision import MonthlyPortfolioDecisionResult


class BuildRebalancePlanCommand(VersionedContract):
    """Request a dollar-notional plan from approved weights and holdings.

    Attributes:
        schema: Stable schema identifier.
        schema_version: Semantic version of this contract.
        plan_id: Caller-provided idempotency and audit identifier.
        decision: Approved monthly portfolio decision.
        account: Read-only brokerage account snapshot.
        min_trade_notional: Absolute dollar threshold below which a deviation
            becomes a hold instruction.
    """

    schema: Literal[SchemaName.REBALANCE_PLAN_COMMAND] = SchemaName.REBALANCE_PLAN_COMMAND
    schema_version: Literal["0.1.0"] = "0.1.0"
    plan_id: str = Field(min_length=1)
    decision: MonthlyPortfolioDecisionResult
    account: BrokerageAccountSnapshot
    min_trade_notional: Decimal = Field(default=Decimal("0"), ge=0, allow_inf_nan=False)
