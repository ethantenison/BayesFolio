"""LLM-based field extractors for the guided portfolio wizard.

Layer: engine/agent
Boundary responsibility: HTTP calls to an OpenAI-compatible API to extract
structured portfolio parameters from natural language. All functions are
fail-soft and return a status string alongside the parsed result.

Key inputs/outputs:
    - Inputs: Free-form user text strings.
    - Outputs: Typed dicts / lists with a status code. Weights are always
      returned as decimals (0.30 = 30%).
"""

from __future__ import annotations

import json
import os
from urllib import error as urllib_error
from urllib import request as urllib_request


def extract_tickers_with_llm(
    message: str,
    timeout_seconds: float = 8.0,
) -> tuple[list[str], str]:
    """Extract ETF ticker symbols from free-form text using an LLM.

    Fail-soft: returns an empty list and a status code on any failure so
    callers can apply a rule-based fallback.

    Args:
        message: User message that may name ETFs by ticker or description.
        timeout_seconds: HTTP timeout in seconds for the LLM request.

    Returns:
        Tuple of ``(tickers, status)`` where ``tickers`` is a deduplicated
        list of uppercase ticker strings and ``status`` is ``ok`` or one of
        the failure codes documented in ``_llm_json_call``.
    """
    if not message.strip():
        return [], "empty_message"
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        return [], "missing_openai_api_key"

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("BAYESFOLIO_INTENT_MODEL", "gpt-4o-mini")

    system_prompt = (
        "Extract ETF ticker symbols from the user's text. "
        "Return ONLY a JSON object with key 'tickers' containing a list of uppercase ticker strings. "
        "If the user names an ETF by its full name or description, infer the standard US ticker symbol. "
        'Example: {"tickers": ["SPY", "QQQ", "TLT"]}'
    )
    payload: dict[str, object] = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        "temperature": 0,
    }
    result, status = _llm_json_call(payload, base_url=base_url, api_key=api_key, timeout_seconds=timeout_seconds)
    if result is None:
        return [], status

    tickers_raw = result.get("tickers", [])
    if not isinstance(tickers_raw, list) or not tickers_raw:
        return [], "no_tickers_found"
    tickers = [str(t).strip().upper() for t in tickers_raw if str(t).strip()]
    return (tickers, "ok") if tickers else ([], "no_tickers_found")


def extract_objective_with_llm(
    message: str,
    timeout_seconds: float = 8.0,
) -> tuple[str | None, str]:
    """Map a natural-language portfolio goal to a Riskfolio objective code.

    Args:
        message: User description of their portfolio objective.
        timeout_seconds: HTTP timeout in seconds for the LLM request.

    Returns:
        Tuple of ``(objective, status)`` where ``objective`` is one of
        ``Sharpe``, ``MaxRet``, ``MinRisk``, ``Utility``, or ``None`` on
        failure.
    """
    if not message.strip():
        return None, "empty_message"
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        return None, "missing_openai_api_key"

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("BAYESFOLIO_INTENT_MODEL", "gpt-4o-mini")

    system_prompt = (
        "Map the user's portfolio goal to exactly one Riskfolio objective code. "
        "Sharpe = maximize risk-adjusted return (Sharpe ratio). "
        "MaxRet = maximize total return regardless of risk. "
        "MinRisk = minimize risk, most conservative. "
        "Utility = balance return and risk with a risk-aversion coefficient. "
        "Return ONLY a JSON object with key 'objective' set to one of: Sharpe, MaxRet, MinRisk, Utility."
    )
    payload: dict[str, object] = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        "temperature": 0,
    }
    result, status = _llm_json_call(payload, base_url=base_url, api_key=api_key, timeout_seconds=timeout_seconds)
    if result is None:
        return None, status

    valid = {"Sharpe", "MaxRet", "MinRisk", "Utility"}
    objective = result.get("objective")
    if isinstance(objective, str) and objective in valid:
        return objective, "ok"
    return None, "invalid_objective"


def extract_risk_preference_with_llm(
    message: str,
    timeout_seconds: float = 8.0,
) -> tuple[str | None, str]:
    """Map a natural-language risk description to a Riskfolio risk measure.

    Args:
        message: User description of their risk tolerance.
        timeout_seconds: HTTP timeout in seconds for the LLM request.

    Returns:
        Tuple of ``(risk_measure, status)`` where ``risk_measure`` is one of
        ``MV``, ``CVaR``, ``CDaR``, or ``None`` on failure.
    """
    if not message.strip():
        return None, "empty_message"
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        return None, "missing_openai_api_key"

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("BAYESFOLIO_INTENT_MODEL", "gpt-4o-mini")

    system_prompt = (
        "Map the user's risk tolerance to one Riskfolio risk measure. "
        "MV (Mean-Variance) = low risk tolerance, minimize portfolio variance, most conservative. "
        "CVaR (Conditional Value-at-Risk) = moderate tolerance, limits tail losses. "
        "CDaR (Conditional Drawdown-at-Risk) = higher tolerance, focuses on drawdown control. "
        "Return ONLY a JSON object with key 'risk_measure' set to one of: MV, CVaR, CDaR."
    )
    payload: dict[str, object] = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        "temperature": 0,
    }
    result, status = _llm_json_call(payload, base_url=base_url, api_key=api_key, timeout_seconds=timeout_seconds)
    if result is None:
        return None, status

    valid = {"MV", "CVaR", "CDaR"}
    risk_measure = result.get("risk_measure")
    if isinstance(risk_measure, str) and risk_measure in valid:
        return risk_measure, "ok"
    return None, "invalid_risk_measure"


def extract_constraints_with_llm(
    message: str,
    timeout_seconds: float = 8.0,
) -> tuple[dict[str, float | bool], str]:
    """Extract portfolio constraints from natural language.

    Args:
        message: User description of portfolio constraints such as position
            limits or short-selling rules.
        timeout_seconds: HTTP timeout in seconds for the LLM request.

    Returns:
        Tuple of ``(constraints, status)`` where ``constraints`` may contain
        ``max_weight`` (decimal in (0, 1]) and ``long_only`` (bool).
        Returns an empty dict on failure; the caller should use defaults.
    """
    if not message.strip():
        return {}, "empty_message"
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        return {}, "missing_openai_api_key"

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("BAYESFOLIO_INTENT_MODEL", "gpt-4o-mini")

    system_prompt = (
        "Extract portfolio constraints from the user's text. "
        "Return ONLY a JSON object with optional keys: "
        "'max_weight' (decimal 0-1, e.g. 0.30 for a 30% cap per position), "
        "'long_only' (bool, true means no short selling). "
        "Convert percentages to decimals. Omit keys not mentioned."
    )
    payload: dict[str, object] = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        "temperature": 0,
    }
    result, status = _llm_json_call(payload, base_url=base_url, api_key=api_key, timeout_seconds=timeout_seconds)
    if result is None:
        return {}, status

    constraints: dict[str, float | bool] = {}

    max_weight_raw = result.get("max_weight")
    if isinstance(max_weight_raw, int | float) and max_weight_raw > 0:
        value = float(max_weight_raw)
        if value > 1.0:
            value = value / 100.0
        if 0.0 < value <= 1.0:
            constraints["max_weight"] = value

    long_only_raw = result.get("long_only")
    if isinstance(long_only_raw, bool):
        constraints["long_only"] = long_only_raw

    return constraints, "ok"


def _llm_json_call(
    payload: dict[str, object],
    *,
    base_url: str,
    api_key: str,
    timeout_seconds: float,
) -> tuple[dict[str, object] | None, str]:
    """Execute a single OpenAI-compatible chat completion and parse the response body.

    Args:
        payload: Full request body for the ``chat/completions`` endpoint.
        base_url: OpenAI-compatible API base URL without trailing slash.
        api_key: Bearer token for authorization.
        timeout_seconds: HTTP timeout in seconds.

    Returns:
        Tuple of ``(parsed_object, status)`` where the first element is
        ``None`` on any failure. Status codes: ``ok``,
        ``http_error_<code>``, ``url_error``, ``invalid_response_json``,
        ``no_choices``, ``empty_content``, ``invalid_content_json``,
        ``non_object_json``.
    """
    body = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout_seconds) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        return None, f"http_error_{exc.code}"
    except (urllib_error.URLError, TimeoutError):
        return None, "url_error"
    except json.JSONDecodeError:
        return None, "invalid_response_json"

    choices = response_data.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return None, "no_choices"

    content = choices[0].get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        return None, "empty_content"

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None, "invalid_content_json"

    if not isinstance(parsed, dict):
        return None, "non_object_json"

    return parsed, "ok"
