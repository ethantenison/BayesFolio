from __future__ import annotations

from bayesfolio.contracts.chat.intent import ParsedIntent


def extract_intent(payload: dict[str, str | float | bool]) -> ParsedIntent:
    """Build ParsedIntent from structured key-value payload."""

    return ParsedIntent(
        objective=str(payload.get("objective", "Sharpe")),
        risk_measure=str(payload.get("risk_measure", "CVaR")),
        long_only=bool(payload.get("long_only", True)),
        min_weight=float(payload.get("min_weight", 0.0)),
        max_weight=float(payload.get("max_weight", 0.35)),
    )
