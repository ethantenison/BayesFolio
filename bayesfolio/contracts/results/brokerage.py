"""Read-only brokerage state and non-executing rebalance plan contracts."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from bayesfolio.contracts.base import ContractModel, SchemaName, VersionedContract


class BrokeragePosition(ContractModel):
    """One long-only brokerage position valued in account currency.

    Attributes:
        asset: Tradable asset identifier.
        quantity: Non-negative position quantity.
        market_price: Latest price per unit.
        market_value: Current total market value.
        cost_basis: Optional current cost basis.
        unrealized_gain_loss: Optional unrealized gain or loss amount.
    """

    asset: str = Field(min_length=1)
    quantity: Decimal = Field(ge=0, allow_inf_nan=False)
    market_price: Decimal = Field(ge=0, allow_inf_nan=False)
    market_value: Decimal = Field(ge=0, allow_inf_nan=False)
    cost_basis: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    unrealized_gain_loss: Decimal | None = Field(default=None, allow_inf_nan=False)


class BrokerageAccountSnapshot(VersionedContract):
    """Read-only IRA account state used for planning and measurement.

    Attributes:
        schema: Stable schema identifier.
        schema_version: Semantic version of this contract.
        account_reference: Opaque non-sensitive adapter reference. It must not
            contain a full brokerage account number.
        as_of: Time at which values and positions were observed.
        currency: ISO-style account currency code.
        net_liquidation_value: Total account value.
        cash_value: Settled and unsettled cash included in total value.
        other_value: Other assets or liabilities included in total value.
        positions: Long-only security positions.
    """

    schema: Literal[SchemaName.BROKERAGE_ACCOUNT_SNAPSHOT] = SchemaName.BROKERAGE_ACCOUNT_SNAPSHOT
    schema_version: Literal["0.1.0"] = "0.1.0"
    account_reference: str = Field(min_length=1)
    as_of: AwareDatetime
    currency: str = Field(default="USD", pattern=r"^[A-Z]{3}$")
    net_liquidation_value: Decimal = Field(gt=0, allow_inf_nan=False)
    cash_value: Decimal = Field(ge=0, allow_inf_nan=False)
    other_value: Decimal = Field(default=Decimal("0"), allow_inf_nan=False)
    positions: list[BrokeragePosition] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_snapshot_totals(self) -> "BrokerageAccountSnapshot":
        """Require unique positions and a reconciled account total."""

        assets = [position.asset for position in self.positions]
        if len(set(assets)) != len(assets):
            raise ValueError("brokerage positions must have unique asset identifiers")
        components = (
            self.cash_value
            + self.other_value
            + sum(
                (position.market_value for position in self.positions),
                start=Decimal("0"),
            )
        )
        tolerance = max(Decimal("0.02"), self.net_liquidation_value * Decimal("0.000001"))
        if abs(components - self.net_liquidation_value) > tolerance:
            raise ValueError(
                "account components must reconcile to net_liquidation_value; "
                f"components={components}, net_liquidation_value={self.net_liquidation_value}"
            )
        return self


class RebalanceInstruction(ContractModel):
    """Dollar-notional planning instruction that is not a brokerage order.

    Attributes:
        asset: Asset identifier.
        action: Suggested buy, sell, or hold direction.
        current_weight: Current account weight as a decimal.
        target_weight: Approved target weight as a decimal.
        current_notional: Current market value.
        target_notional: Target market value.
        trade_notional: Absolute planned trade value; zero for holds.
    """

    asset: str
    action: Literal["buy", "sell", "hold"]
    current_weight: Decimal = Field(ge=0, allow_inf_nan=False)
    target_weight: Decimal = Field(ge=0, allow_inf_nan=False)
    current_notional: Decimal = Field(ge=0, allow_inf_nan=False)
    target_notional: Decimal = Field(ge=0, allow_inf_nan=False)
    trade_notional: Decimal = Field(ge=0, allow_inf_nan=False)


class RebalancePlanResult(VersionedContract):
    """Auditable non-executing plan derived from final portfolio weights.

    Attributes:
        schema: Stable schema identifier.
        schema_version: Semantic version of this contract.
        plan_id: Idempotency and audit identifier from the command.
        account_reference: Opaque account reference to which the plan applies.
        account_as_of: Snapshot timestamp used for current holdings.
        decision_reviewed_at: Human portfolio-review timestamp.
        strategy_id: Approved portfolio strategy identifier.
        candidate_run_label: Source monthly candidate run.
        currency: Account currency.
        net_liquidation_value: Account value used for target notionals.
        min_trade_notional: Threshold applied to buy/sell deviations.
        instructions: Per-asset planning instructions.
        estimated_cash_after_trades: Cash estimate after planned notionals.
        executable: Always false; this contract is not an order request.
        diagnostics: Human-readable planning limitations and warnings.
    """

    schema: Literal[SchemaName.REBALANCE_PLAN_RESULT] = SchemaName.REBALANCE_PLAN_RESULT
    schema_version: Literal["0.1.0"] = "0.1.0"
    plan_id: str
    account_reference: str
    account_as_of: AwareDatetime
    decision_reviewed_at: AwareDatetime
    strategy_id: str
    candidate_run_label: str
    currency: str
    net_liquidation_value: Decimal = Field(gt=0, allow_inf_nan=False)
    min_trade_notional: Decimal = Field(ge=0, allow_inf_nan=False)
    instructions: list[RebalanceInstruction] = Field(min_length=1)
    estimated_cash_after_trades: Decimal = Field(ge=0, allow_inf_nan=False)
    executable: Literal[False] = False
    diagnostics: list[str] = Field(default_factory=list)
