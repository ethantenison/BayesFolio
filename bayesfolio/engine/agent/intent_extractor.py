from __future__ import annotations

from bayesfolio.schemas.contracts.intent import OptimizationIntent


def extract_intent(payload: dict[str, str | float | bool]) -> OptimizationIntent:
    """Build OptimizationIntent from structured key-value payload."""

    return OptimizationIntent(
        objective=str(payload.get("objective", "Sharpe")),
        risk_measure=str(payload.get("risk_measure", "CVaR")),
        long_only=bool(payload.get("long_only", True)),
        min_weight=float(payload.get("min_weight", 0.0)),
        max_weight=float(payload.get("max_weight", 0.35)),
    )
