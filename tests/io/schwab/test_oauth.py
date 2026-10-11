"""Tests for the Schwab OAuth authorization and refresh lifecycle."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

from bayesfolio.io.schwab.credentials import SchwabClientCredentials
from bayesfolio.io.schwab.oauth import SchwabAuthorizationError, SchwabOAuthClient, SchwabTokenBundle


class MemoryTokenStore:
    """Store serialized OAuth tokens in memory for deterministic tests."""

    def __init__(self, token_json: str | None = None) -> None:
        self.token_json = token_json

    def load(self) -> str | None:
        return self.token_json

    def save(self, token_json: str) -> None:
        self.token_json = token_json


class FakeResponse:
    """Minimal successful requests response."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeSession:
    """Record token requests and return queued responses."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.posts: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.posts.append({"url": url, **kwargs})
        return FakeResponse(self.responses.pop(0))


class FailedTokenSession:
    """Return one OAuth error response with an attached HTTP status."""

    def post(self, url: str, **kwargs: Any) -> requests.Response:
        response = requests.Response()
        response.status_code = 401
        response.url = url
        response._content = (  # noqa: SLF001 - construct a requests response fixture
            b'{"error":"invalid_client","error_description":"Client authentication failed"}'
        )
        return response


def test_authorization_callback_exchanges_code_and_persists_tokens() -> None:
    now = datetime(2026, 10, 11, 16, 0, tzinfo=UTC)
    store = MemoryTokenStore()
    session = FakeSession(
        [
            {
                "access_token": "access-value",
                "refresh_token": "refresh-value",
                "token_type": "Bearer",
                "expires_in": 1800,
                "refresh_token_expires_in": 604800,
                "scope": "api",
            }
        ]
    )
    oauth = SchwabOAuthClient(
        credentials=SchwabClientCredentials("client-value", "secret-value"),
        token_store=store,
        redirect_uri="https://127.0.0.1:8182/callback",
        session=session,
        clock=lambda: now,
    )

    authorization = oauth.begin_authorization()
    authorization_query = parse_qs(urlsplit(authorization.url).query)
    assert authorization_query["client_id"] == ["client-value"]
    assert "secret-value" not in authorization.url
    oauth.complete_authorization(
        redirected_url=(f"https://127.0.0.1:8182/callback?code=authorization-code&state={authorization.state}"),
        expected_state=authorization.state,
    )

    assert store.token_json is not None
    stored = SchwabTokenBundle.from_json(store.token_json)
    assert stored.access_token == "access-value"
    assert stored.refresh_token == "refresh-value"
    assert session.posts[0]["data"]["grant_type"] == "authorization_code"
    assert session.posts[0]["auth"] == ("client-value", "secret-value")


def test_expired_access_token_refreshes_without_reauthorization() -> None:
    now = datetime(2026, 10, 11, 16, 0, tzinfo=UTC)
    previous = SchwabTokenBundle(
        access_token="expired-access",
        refresh_token="existing-refresh",
        token_type="Bearer",
        scope="api",
        access_expires_at=now - timedelta(seconds=1),
        refresh_expires_at=now + timedelta(days=1),
    )
    store = MemoryTokenStore(previous.to_json())
    session = FakeSession(
        [
            {
                "access_token": "new-access",
                "token_type": "Bearer",
                "expires_in": 1800,
                "scope": "api",
            }
        ]
    )
    oauth = SchwabOAuthClient(
        credentials=SchwabClientCredentials("client-value", "secret-value"),
        token_store=store,
        redirect_uri="https://127.0.0.1:8182/callback",
        session=session,
        clock=lambda: now,
    )

    access_token = oauth.get_access_token()

    assert access_token == "new-access"
    assert session.posts[0]["data"] == {
        "grant_type": "refresh_token",
        "refresh_token": "existing-refresh",
    }
    assert store.token_json is not None
    assert SchwabTokenBundle.from_json(store.token_json).refresh_token == "existing-refresh"


def test_token_exchange_reports_safe_oauth_error_details() -> None:
    oauth = SchwabOAuthClient(
        credentials=SchwabClientCredentials("client-value", "secret-value"),
        token_store=MemoryTokenStore(),
        redirect_uri="https://127.0.0.1:8182/callback",
        session=FailedTokenSession(),
    )

    authorization = oauth.begin_authorization()
    with pytest.raises(SchwabAuthorizationError) as exc_info:
        oauth.complete_authorization(
            redirected_url=(f"https://127.0.0.1:8182/callback?code=authorization-code&state={authorization.state}"),
            expected_state=authorization.state,
        )

    message = str(exc_info.value)
    assert message == ("Schwab token exchange failed (HTTP 401): invalid_client: Client authentication failed.")
    assert "secret-value" not in message
    assert "authorization-code" not in message
