"""Pure dollar-notional rebalance planning from finalized portfolio weights."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from bayesfolio.contracts.commands.brokerage import BuildRebalancePlanCommand
from bayesfolio.contracts.results.brokerage import RebalanceInstruction, RebalancePlanResult


def build_rebalance_plan(command: BuildRebalancePlanCommand) -> RebalancePlanResult:
    """Build a non-executing rebalance plan for an approved decision.

    The planner deliberately emits dollar notionals rather than quantities or
    order types. Fractional-share support, price protection, tax lots, market
    hours, settlement, and broker authorization belong to a future concrete
    brokerage adapter and execution workflow.

    Args:
        command: Approved decision, current account snapshot, and trade threshold.

    Returns:
        Read-only per-asset buy, sell, and hold planning instructions.

    Raises:
        ValueError: If the decision is not approved, final weights are absent,
            the account snapshot predates review, or target weights are invalid.
    """

    decision = command.decision
    if decision.decision != "approved" or decision.finalized_weights is None:
        raise ValueError("A rebalance plan requires an approved decision with finalized weights.")
    if command.account.as_of < decision.reviewed_at:
        raise ValueError("The brokerage snapshot must be observed at or after portfolio review.")

    optimization = decision.finalized_weights
    if len(set(optimization.asset_order)) != len(optimization.asset_order):
        raise ValueError("Finalized asset_order must contain unique assets.")
    if len(optimization.asset_order) != len(optimization.weights):
        raise ValueError("Finalized asset_order and weights must have equal lengths.")

    target_weights = {
        asset: Decimal(str(weight))
        for asset, weight in zip(optimization.asset_order, optimization.weights, strict=True)
    }
    if any(weight < 0 for weight in target_weights.values()):
        raise ValueError("Finalized weights must be non-negative.")
    if abs(sum(target_weights.values(), start=Decimal("0")) - Decimal("1")) > Decimal("0.00000001"):
        raise ValueError("Finalized weights must sum to 1.0 within 1e-8.")

    current_values = {position.asset: position.market_value for position in command.account.positions}
    assets = [*optimization.asset_order]
    assets.extend(sorted(set(current_values) - set(target_weights)))
    instructions: list[RebalanceInstruction] = []
    total_buys = Decimal("0")
    total_sells = Decimal("0")

    for asset in assets:
        current_notional = current_values.get(asset, Decimal("0"))
        current_weight = current_notional / command.account.net_liquidation_value
        target_weight = target_weights.get(asset, Decimal("0"))
        target_notional = command.account.net_liquidation_value * target_weight
        difference = target_notional - current_notional
        if abs(difference) < command.min_trade_notional:
            action: Literal["buy", "sell", "hold"] = "hold"
            trade_notional = Decimal("0")
        elif difference > 0:
            action = "buy"
            trade_notional = difference
            total_buys += trade_notional
        else:
            action = "sell"
            trade_notional = -difference
            total_sells += trade_notional

        instructions.append(
            RebalanceInstruction(
                asset=asset,
                action=action,
                current_weight=current_weight,
                target_weight=target_weight,
                current_notional=current_notional,
                target_notional=target_notional,
                trade_notional=trade_notional,
            )
        )

    estimated_cash = command.account.cash_value + total_sells - total_buys
    if estimated_cash < Decimal("-0.01"):
        raise ValueError(
            "Planned buys exceed available cash plus planned sells; reduce min_trade_notional or revise target weights."
        )
    if estimated_cash < 0:
        estimated_cash = Decimal("0")
    return RebalancePlanResult(
        plan_id=command.plan_id,
        account_reference=command.account.account_reference,
        account_as_of=command.account.as_of,
        decision_reviewed_at=decision.reviewed_at,
        strategy_id=decision.strategy_id,
        candidate_run_label=decision.candidate_run_label,
        currency=command.account.currency,
        net_liquidation_value=command.account.net_liquidation_value,
        min_trade_notional=command.min_trade_notional,
        instructions=instructions,
        estimated_cash_after_trades=estimated_cash,
        executable=False,
        diagnostics=[
            "Planning output only; no brokerage orders were created or submitted.",
            "Future execution must resolve quantities, tax lots, order types, price limits, and broker authorization.",
        ],
    )
