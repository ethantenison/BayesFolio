"""Guided step-by-step portfolio wizard for BayesFolio.

This module implements a conversational wizard that collects portfolio
parameters through a series of focused questions. Unlike the free-form MVP
chat, each step asks for exactly one thing so the user never has to compose
a complex request.

Layer: engine orchestration
Boundary responsibility: Wizard state machine and LLM-backed field extraction.
Does not perform file I/O or MLflow calls.

Key inputs/outputs:
    - Input: User free-form replies to step-specific prompts.
    - Output: ``GuidedChatState`` tracking collected values and current step;
      ``HistoricalMvpRequest`` (for ``run_historical_mvp_pipeline``) when
      the wizard reaches ``GuidedChatStep.COMPLETE``.
      Weights are decimal (0.35 = 35%).

Execution Flow:
    1. ``ASSETS``      — LLM extracts tickers; validates ≥ 3.
    2. ``OBJECTIVE``   — LLM maps goal description → Riskfolio objective code.
    3. ``RISK``        — LLM maps tolerance description → risk measure code.
    4. ``CONSTRAINTS`` — LLM extracts position limits (skippable).
    5. ``CONFIRM``     — Summary shown; user confirms to proceed.
    6. ``COMPLETE``    — Call ``build_request_from_state`` to get the request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from enum import StrEnum

from bayesfolio.engine.agent.ticker_extractor import (
    extract_constraints_with_llm,
    extract_objective_with_llm,
    extract_risk_preference_with_llm,
    extract_tickers_with_llm,
)
from bayesfolio.engine.mvp_historical_chat import HistoricalMvpRequest

_DEFAULT_OBJECTIVE = "Sharpe"
_DEFAULT_RISK_MEASURE = "MV"
_DEFAULT_MAX_WEIGHT = 0.35
_MIN_TICKERS = 3
_LOOKBACK_YEARS = 5

# Noise tokens to filter when LLM is unavailable for ticker extraction.
_TICKER_NOISE: frozenset[str] = frozenset(
    {
        "MY",
        "ID",
        "IT",
        "IN",
        "AND",
        "OR",
        "AT",
        "OF",
        "FOR",
        "TO",
        "WITH",
        "THE",
        "AN",
        "A",
        "IS",
        "BE",
        "BY",
        "AS",
        "ON",
        "UP",
        "DO",
        "IF",
        "WE",
        "ME",
        "HE",
        "NO",
        "SO",
        "GO",
        "AM",
        "US",
    }
)


class GuidedChatStep(StrEnum):
    """Ordered steps of the guided portfolio wizard."""

    ASSETS = "assets"
    OBJECTIVE = "objective"
    RISK = "risk"
    CONSTRAINTS = "constraints"
    CONFIRM = "confirm"
    COMPLETE = "complete"


@dataclass(frozen=True)
class GuidedChatState:
    """Immutable snapshot of the guided wizard conversation state.

    Attributes:
        step: Current wizard step.
        tickers: Collected ETF tickers in uppercase. Empty before ASSETS completes.
        objective: Riskfolio objective code (e.g. ``Sharpe``).
        risk_measure: Riskfolio risk measure code (e.g. ``MV``).
        max_weight: Maximum per-asset weight as a decimal (0.35 = 35%).
        min_weight: Minimum per-asset weight as a decimal (0.0 = no floor).
    """

    step: GuidedChatStep = GuidedChatStep.ASSETS
    tickers: list[str] = field(default_factory=list)
    objective: str = _DEFAULT_OBJECTIVE
    risk_measure: str = _DEFAULT_RISK_MEASURE
    max_weight: float = _DEFAULT_MAX_WEIGHT
    min_weight: float = 0.0


def initial_prompt() -> str:
    """Return the opening wizard message shown at session start.

    Returns:
        Markdown-formatted assistant message introducing the wizard and
        presenting the first step prompt.
    """
    return (
        "Welcome to the **Guided Portfolio Builder**! "
        "I'll walk you through a few quick questions to configure your portfolio.\n\n"
        + _step_prompt(GuidedChatStep.ASSETS)
    )


def advance_guided_chat(
    state: GuidedChatState,
    user_message: str,
) -> tuple[GuidedChatState, str]:
    """Process a user reply and advance the wizard to the next step.

    Args:
        state: Current wizard state.
        user_message: User's reply for the active step.

    Returns:
        Tuple of ``(new_state, bot_response)`` where ``bot_response`` is a
        markdown-formatted message to display. If the step cannot advance
        (e.g. too few tickers), the state is returned unchanged with a
        clarification prompt.
    """
    if state.step == GuidedChatStep.ASSETS:
        return _handle_assets(state, user_message)
    if state.step == GuidedChatStep.OBJECTIVE:
        return _handle_objective(state, user_message)
    if state.step == GuidedChatStep.RISK:
        return _handle_risk(state, user_message)
    if state.step == GuidedChatStep.CONSTRAINTS:
        return _handle_constraints(state, user_message)
    if state.step == GuidedChatStep.CONFIRM:
        return _handle_confirm(state, user_message)
    return state, "The wizard is already complete."


def build_request_from_state(state: GuidedChatState) -> HistoricalMvpRequest:
    """Convert a completed wizard state into an optimization pipeline request.

    Uses a rolling 5-year lookback ending today for the date range, as the
    wizard does not ask the user for explicit dates.

    Args:
        state: Wizard state that has reached ``GuidedChatStep.COMPLETE``.

    Returns:
        ``HistoricalMvpRequest`` ready for ``run_historical_mvp_pipeline``.
        Weights are in decimal units (0.35 = 35%).

    Raises:
        ValueError: If ``state.step`` is not ``COMPLETE`` or tickers are empty.
    """
    if state.step != GuidedChatStep.COMPLETE:
        msg = f"Wizard must be COMPLETE to build a request; got {state.step!r}."
        raise ValueError(msg)
    if not state.tickers:
        msg = "Cannot build request: no tickers were collected."
        raise ValueError(msg)

    end_date = date.today()
    start_date = end_date - timedelta(days=365 * _LOOKBACK_YEARS)
    # nea must always be < n_assets to allow non-equal weighting; target ~half
    # the ticker count so concentrated positions are reachable.
    nea = max(3, len(state.tickers) // 2)
    return HistoricalMvpRequest(
        tickers=state.tickers,
        start_date=start_date,
        end_date=end_date,
        objective=state.objective,
        risk_measure=state.risk_measure,
        max_weight=state.max_weight,
        min_weight=state.min_weight,
        nea=nea,
    )


# ---------------------------------------------------------------------------
# Private step handlers
# ---------------------------------------------------------------------------


def _handle_assets(state: GuidedChatState, message: str) -> tuple[GuidedChatState, str]:
    tickers, _status = extract_tickers_with_llm(message)
    fallback_note = ""
    if not tickers:
        tickers = _ticker_fallback(message)
        if tickers:
            fallback_note = "\n\n_(LLM unavailable — tickers extracted by pattern matching.)_"

    if not tickers:
        return state, (
            "I couldn't identify any ETF tickers in your message. "
            "Please list them clearly — for example: _SPY, QQQ, TLT, IEF, GLD_."
        )

    if len(tickers) < _MIN_TICKERS:
        joined = ", ".join(tickers)
        return state, (
            f"I found **{joined}**, but need at least {_MIN_TICKERS} ETFs for diversification. Please add more tickers."
        )

    new_state = replace(state, step=GuidedChatStep.OBJECTIVE, tickers=tickers)
    return new_state, (
        f"Got it! Portfolio universe: **{', '.join(tickers)}**{fallback_note}\n\n"
        + _step_prompt(GuidedChatStep.OBJECTIVE)
    )


def _handle_objective(state: GuidedChatState, message: str) -> tuple[GuidedChatState, str]:
    objective, _status = extract_objective_with_llm(message)
    if objective is None:
        objective = _objective_keyword_fallback(message)

    if objective is None:
        return state, (
            "I wasn't sure how to interpret that goal. Please describe it — for example:\n"
            "- *Maximize risk-adjusted returns*\n"
            "- *Maximize total return*\n"
            "- *Minimize risk*\n"
            "- *Balance return and risk*"
        )

    new_state = replace(state, step=GuidedChatStep.RISK, objective=objective)
    return new_state, f"Objective set to **{objective}**.\n\n" + _step_prompt(GuidedChatStep.RISK)


def _handle_risk(state: GuidedChatState, message: str) -> tuple[GuidedChatState, str]:
    risk_measure, _status = extract_risk_preference_with_llm(message)
    if risk_measure is None:
        risk_measure = _risk_keyword_fallback(message)

    if risk_measure is None:
        return state, (
            "I couldn't determine your risk preference. Please answer with:\n"
            "- **Low** — minimize variance\n"
            "- **Moderate** — limit tail losses\n"
            "- **High** — tolerate larger drawdowns"
        )

    new_state = replace(state, step=GuidedChatStep.CONSTRAINTS, risk_measure=risk_measure)
    return new_state, f"Risk measure set to **{risk_measure}**.\n\n" + _step_prompt(GuidedChatStep.CONSTRAINTS)


def _handle_constraints(state: GuidedChatState, message: str) -> tuple[GuidedChatState, str]:
    if _is_skip(message):
        new_state = replace(state, step=GuidedChatStep.CONFIRM)
        return new_state, "Using default constraints (35% max per ETF, long-only).\n\n" + _confirm_prompt(new_state)

    constraints, _status = extract_constraints_with_llm(message)

    max_weight = state.max_weight
    if isinstance(constraints.get("max_weight"), float):
        max_weight = constraints["max_weight"]  # type: ignore[assignment]

    new_state = replace(state, step=GuidedChatStep.CONFIRM, max_weight=max_weight)
    constraint_note = f"Max position: **{max_weight:.0%}**"
    return new_state, f"Constraints set — {constraint_note}.\n\n" + _confirm_prompt(new_state)


def _handle_confirm(state: GuidedChatState, message: str) -> tuple[GuidedChatState, str]:
    if _is_affirmative(message):
        return replace(state, step=GuidedChatStep.COMPLETE), "Starting portfolio optimization..."
    return state, (
        "Just say **yes**, **go**, or **confirm** to run the optimization, "
        "or reload the page to start over.\n\n" + _confirm_prompt(state)
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _step_prompt(step: GuidedChatStep) -> str:
    """Return the question prompt for the given wizard step."""
    prompts: dict[GuidedChatStep, str] = {
        GuidedChatStep.ASSETS: (
            "**Step 1 — Assets**\n\n"
            "Which ETFs would you like to include? Please name at least 3.\n\n"
            "_Example: SPY, QQQ, TLT, IEF, GLD_"
        ),
        GuidedChatStep.OBJECTIVE: (
            "**Step 2 — Objective**\n\n"
            "What is your main goal for this portfolio?\n\n"
            "- Maximize risk-adjusted returns *(Sharpe)*\n"
            "- Maximize total return *(aggressive)*\n"
            "- Minimize risk *(conservative)*\n"
            "- Balance return and risk *(utility)*"
        ),
        GuidedChatStep.RISK: (
            "**Step 3 — Risk**\n\n"
            "How much risk or potential loss are you willing to tolerate?\n\n"
            "- **Low** — minimize variance, most conservative\n"
            "- **Moderate** — limit tail losses (CVaR)\n"
            "- **High** — tolerate larger drawdowns (CDaR)"
        ),
        GuidedChatStep.CONSTRAINTS: (
            "**Step 4 — Constraints** *(optional)*\n\n"
            "Any position limits or restrictions?\n\n"
            "- Max position size *(e.g. 'max 30% per ETF')*\n"
            "- Short selling allowed?\n\n"
            "Say **skip** to use defaults (35% cap, long-only)."
        ),
    }
    return prompts.get(step, "")


def _confirm_prompt(state: GuidedChatState) -> str:
    """Render a summary of the current wizard settings for the CONFIRM step."""
    lines = [
        "**Ready to optimize — here is your configuration:**\n",
        f"- **Assets**: {', '.join(state.tickers)}",
        f"- **Objective**: {state.objective}",
        f"- **Risk measure**: {state.risk_measure}",
        f"- **Max position**: {state.max_weight:.0%}",
        f"- **Date range**: last {_LOOKBACK_YEARS} years\n",
        "Say **yes** to run the optimization, or reload the page to start over.",
    ]
    return "\n".join(lines)


def _is_skip(message: str) -> bool:
    """Return ``True`` when the user intends to skip the constraints step."""
    return bool(re.match(r"^(skip|default|defaults|no|none|n/?a)$", message.strip().lower()))


def _is_affirmative(message: str) -> bool:
    """Return ``True`` when the user confirms they want to proceed."""
    tokens = r"yes|y|go|run|confirm|ok|okay|proceed|start|do\s+it|yep|sure"
    return bool(re.match(rf"^({tokens})$", message.strip().lower()))


def _ticker_fallback(message: str) -> list[str]:
    """Extract 2-6 char uppercase tokens as tickers when LLM is unavailable.

    Args:
        message: User message text.

    Returns:
        Deduplicated list of candidate tickers with common noise filtered out.
    """
    raw = re.findall(r"\b[A-Z]{2,6}\b", message.upper())
    seen: list[str] = []
    for token in raw:
        if token not in _TICKER_NOISE and token not in seen:
            seen.append(token)
    return seen


def _objective_keyword_fallback(message: str) -> str | None:
    """Map keywords in a message to a Riskfolio objective code.

    Args:
        message: User description of portfolio objective.

    Returns:
        Objective code string, or ``None`` when no match is found.
    """
    lowered = message.lower()
    if re.search(r"\b(max\s*ret|maximum\s*return|aggressive|highest\s*return)\b", lowered):
        return "MaxRet"
    if re.search(r"\b(min\s*risk|minimum\s*risk|conservative|low\s*risk|safest)\b", lowered):
        return "MinRisk"
    if re.search(r"\b(utility|balance|trade.?off|risk.?avers)\b", lowered):
        return "Utility"
    if re.search(r"\b(sharpe|risk.?adjust|risk\s+adjusted)\b", lowered):
        return "Sharpe"
    return None


def _risk_keyword_fallback(message: str) -> str | None:
    """Map keywords in a message to a Riskfolio risk measure code.

    Args:
        message: User description of risk tolerance.

    Returns:
        Risk measure code string, or ``None`` when no match is found.
    """
    lowered = message.lower()
    if re.search(r"\b(low|conserv|minimal|safe|least|mv)\b", lowered):
        return "MV"
    if re.search(r"\b(moderate|medium|balanced|mid|cvar)\b", lowered):
        return "CVaR"
    if re.search(r"\b(high|aggress|maximum|large|significant|drawdown|cdar)\b", lowered):
        return "CDaR"
    return None
