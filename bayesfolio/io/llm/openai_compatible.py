"""OpenAI-compatible chat-completions transport for structured JSON output.

This adapter owns HTTP request and response parsing only. It can call a local
Qwen server or another compatible provider, but it contains no portfolio logic
and does not persist prompts or responses.
"""

from __future__ import annotations

import json
from urllib import error as urllib_error
from urllib import request as urllib_request


class OpenAICompatibleModelError(RuntimeError):
    """Raised when an OpenAI-compatible endpoint cannot return model text."""


class OpenAICompatibleChatModel:
    """Synchronous JSON-generation adapter for chat-completions endpoints.

    Args:
        base_url: API base URL ending before ``/chat/completions``.
        model: Provider model identifier or local model path.
        api_key: Optional bearer token; local endpoints may omit it.
        timeout_seconds: HTTP timeout in seconds.
        request_options: Additional provider-specific scalar request fields.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        request_options: dict[str, str | int | float | bool | None] | None = None,
    ) -> None:
        """Initialize an OpenAI-compatible synchronous model adapter."""

        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if not model.strip():
            raise ValueError("model must not be empty")
        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._request_options = dict(request_options or {})

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        seed: int,
        max_output_tokens: int,
    ) -> str:
        """Request one JSON-only response from the configured model endpoint.

        Args:
            system_prompt: Immutable role and procedure instructions.
            user_prompt: Canonical point-in-time input packet and schema.
            temperature: Requested provider sampling temperature.
            seed: Requested provider seed when supported.
            max_output_tokens: Maximum generated token count.

        Returns:
            Raw assistant response text.

        Raises:
            OpenAICompatibleModelError: If transport or response parsing fails.
        """

        protected_keys = {"model", "messages", "temperature", "seed", "max_tokens"}
        overlap = protected_keys & set(self._request_options)
        if overlap:
            raise ValueError(f"request_options cannot override core fields: {sorted(overlap)}")

        payload: dict[str, object] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_output_tokens,
            **self._request_options,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request_obj = urllib_request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib_request.urlopen(request_obj, timeout=self._timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            raise OpenAICompatibleModelError(f"model endpoint returned HTTP {exc.code}") from exc
        except (urllib_error.URLError, TimeoutError) as exc:
            raise OpenAICompatibleModelError("model endpoint was unavailable or timed out") from exc
        except json.JSONDecodeError as exc:
            raise OpenAICompatibleModelError("model endpoint returned invalid JSON") from exc

        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise OpenAICompatibleModelError("model response did not contain choices")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise OpenAICompatibleModelError("model response choice was not an object")
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise OpenAICompatibleModelError("model response did not contain a message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise OpenAICompatibleModelError("model response content was empty")
        return content
