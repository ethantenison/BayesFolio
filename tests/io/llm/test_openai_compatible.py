"""Tests for the OpenAI-compatible chat transport."""

from __future__ import annotations

import json
from typing import Any

from bayesfolio.io.llm.openai_compatible import OpenAICompatibleChatModel


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_chat_model_sends_reproducible_settings_and_returns_content(monkeypatch: Any) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: Any, *, timeout: float) -> _FakeResponse:
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse({"choices": [{"message": {"content": '{"answer": true}'}}]})

    monkeypatch.setattr("bayesfolio.io.llm.openai_compatible.urllib_request.urlopen", fake_urlopen)
    model = OpenAICompatibleChatModel(
        base_url="http://127.0.0.1:8080/v1/",
        model="local-qwen",
        timeout_seconds=30.0,
        request_options={"enable_thinking": False},
    )

    content = model.generate_json(
        system_prompt="system",
        user_prompt="user",
        temperature=0.0,
        seed=16_638,
        max_output_tokens=2_000,
    )

    assert content == '{"answer": true}'
    assert captured["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert captured["timeout"] == 30.0
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "local-qwen"
    assert payload["temperature"] == 0.0
    assert payload["seed"] == 16_638
    assert payload["max_tokens"] == 2_000
    assert payload["enable_thinking"] is False
