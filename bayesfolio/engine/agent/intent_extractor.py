from __future__ import annotations

import json
import os
from urllib import error as urllib_error
from urllib import request as urllib_request

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


def extract_intent_overrides_from_text(message: str, timeout_seconds: float = 8.0) -> dict[str, str | float | int]:
    """Extract optimization-intent overrides from free text using an optional LLM.

    This function is fail-soft by design: when no LLM credentials are
    available, or when any request/parse error occurs, it returns an empty
    dictionary and the caller should use deterministic rule-based parsing.

    Args:
        message: User free-form request text.
        timeout_seconds: HTTP timeout in seconds for the LLM request.

    Returns:
        Sanitized overrides dictionary with zero or more keys among:
        ``objective``, ``risk_measure``, ``min_weight``, ``max_weight``,
        and ``nea`` (number of effective assets).
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None or not message.strip():
        return {}

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("BAYESFOLIO_INTENT_MODEL", "gpt-4o-mini")

    system_prompt = (
        "Extract portfolio optimization parameters from user text. "
        "Return ONLY a JSON object with optional keys: objective, risk_measure, "
        "min_weight, max_weight, nea, number_effective_assets. "
        "Use decimals for weights (0.35 = 35%). If not specified, omit key."
    )
    payload = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        "temperature": 0,
    }

    body = json.dumps(payload).encode("utf-8")
    request_obj = urllib_request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(request_obj, timeout=timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except (urllib_error.URLError, TimeoutError, json.JSONDecodeError):
        return {}

    choices = response_payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return {}

    message_content = choices[0].get("message", {}).get("content", "")
    if not isinstance(message_content, str) or not message_content.strip():
        return {}

    try:
        raw_overrides = json.loads(message_content)
    except json.JSONDecodeError:
        return {}

    if not isinstance(raw_overrides, dict):
        return {}

    return _sanitize_intent_overrides(raw_overrides)


def _sanitize_intent_overrides(raw_overrides: dict[str, object]) -> dict[str, str | float | int]:
    """Validate and normalize LLM-provided intent overrides.

    Args:
        raw_overrides: Raw key-value mapping returned by the LLM.

    Returns:
        Normalized subset of supported override keys.
    """

    cleaned: dict[str, str | float | int] = {}

    objective = raw_overrides.get("objective")
    if isinstance(objective, str) and objective.strip():
        cleaned["objective"] = objective.strip()

    risk_measure = raw_overrides.get("risk_measure")
    if isinstance(risk_measure, str) and risk_measure.strip():
        cleaned["risk_measure"] = risk_measure.strip()

    min_weight = _normalize_weight(raw_overrides.get("min_weight"))
    if min_weight is not None:
        cleaned["min_weight"] = min_weight

    max_weight = _normalize_weight(raw_overrides.get("max_weight"))
    if max_weight is not None:
        cleaned["max_weight"] = max_weight

    nea_raw = raw_overrides.get("nea", raw_overrides.get("number_effective_assets"))
    if isinstance(nea_raw, int) and nea_raw >= 1:
        cleaned["nea"] = nea_raw
    elif isinstance(nea_raw, float) and nea_raw.is_integer() and nea_raw >= 1:
        cleaned["nea"] = int(nea_raw)
    elif isinstance(nea_raw, str) and nea_raw.strip().isdigit() and int(nea_raw) >= 1:
        cleaned["nea"] = int(nea_raw)

    return cleaned


def _normalize_weight(raw_value: object) -> float | None:
    """Normalize a weight-like value to decimal units.

    Args:
        raw_value: Candidate value from an untrusted source.

    Returns:
        Decimal weight in [0, 1], or ``None`` when invalid.
    """

    value: float
    if isinstance(raw_value, int | float):
        value = float(raw_value)
    elif isinstance(raw_value, str):
        stripped = raw_value.strip().replace("%", "")
        try:
            value = float(stripped)
        except ValueError:
            return None
        if "%" in raw_value:
            value = value / 100.0
    else:
        return None

    if value > 1.0 and value <= 100.0:
        value = value / 100.0

    if value < 0.0 or value > 1.0:
        return None
    return value
