"""Perform Schwab OAuth authorization and refresh without exposing secrets.

The adapter uses the authorization-code flow, persists tokens through an
injected secure store, and returns only short-lived access tokens to the
read-only account client.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from urllib.parse import parse_qs, urlencode, urlsplit

import requests

from bayesfolio.io.schwab.credentials import SchwabClientCredentials, SchwabTokenStore


class SchwabAuthorizationError(RuntimeError):
    """Report a failed Schwab OAuth operation without including secret payloads."""


class SchwabAuthorizationRequired(SchwabAuthorizationError):
    """Indicate that the user must repeat Schwab's interactive consent flow."""


class OAuthResponse(Protocol):
    """Minimal HTTP response surface required by the OAuth adapter."""

    def raise_for_status(self) -> None:
        """Raise when the HTTP response is unsuccessful."""

        ...

    def json(self) -> object:
        """Return the decoded response body."""

        ...


class OAuthSession(Protocol):
    """HTTP session boundary used for Schwab token requests."""

    def post(
        self,
        url: str,
        *,
        auth: tuple[str, str],
        data: dict[str, str],
        headers: dict[str, str],
        timeout: float,
    ) -> OAuthResponse:
        """Send one form-encoded token request."""

        ...


@dataclass(frozen=True, slots=True)
class SchwabAuthorizationRequest:
    """Interactive Schwab authorization request details.

    Attributes:
        url: Schwab authorization URL to open in a browser.
        state: Random CSRF token that must match the returned callback.
    """

    url: str
    state: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class SchwabTokenBundle:
    """Persisted Schwab OAuth tokens and expiration metadata.

    Attributes:
        access_token: Short-lived bearer token, excluded from representations.
        refresh_token: Refresh credential, excluded from representations.
        token_type: OAuth token type, normally ``Bearer``.
        scope: Scope string returned by Schwab, when present.
        access_expires_at: UTC expiration time for the access token.
        refresh_expires_at: UTC refresh-token expiration when Schwab reports it.
    """

    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)
    token_type: str
    scope: str | None
    access_expires_at: datetime
    refresh_expires_at: datetime | None

    def to_json(self) -> str:
        """Serialize the token bundle for encrypted Keychain persistence."""

        payload = asdict(self)
        payload["access_expires_at"] = self.access_expires_at.isoformat()
        payload["refresh_expires_at"] = (
            self.refresh_expires_at.isoformat() if self.refresh_expires_at is not None else None
        )
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, token_json: str) -> SchwabTokenBundle:
        """Deserialize a validated token bundle from secure storage.

        Args:
            token_json: JSON previously emitted by :meth:`to_json`.

        Returns:
            Parsed token bundle with timezone-aware expiration values.

        Raises:
            SchwabAuthorizationRequired: If stored token data is incomplete or invalid.
        """

        try:
            payload = json.loads(token_json)
            access_expires_at = datetime.fromisoformat(payload["access_expires_at"])
            refresh_expires_raw = payload.get("refresh_expires_at")
            refresh_expires_at = (
                datetime.fromisoformat(refresh_expires_raw) if refresh_expires_raw is not None else None
            )
            bundle = cls(
                access_token=_required_string(payload, "access_token"),
                refresh_token=_required_string(payload, "refresh_token"),
                token_type=_required_string(payload, "token_type"),
                scope=_optional_string(payload, "scope"),
                access_expires_at=access_expires_at,
                refresh_expires_at=refresh_expires_at,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SchwabAuthorizationRequired("Stored Schwab OAuth tokens are invalid; authorize again.") from exc
        if bundle.access_expires_at.tzinfo is None or (
            bundle.refresh_expires_at is not None and bundle.refresh_expires_at.tzinfo is None
        ):
            raise SchwabAuthorizationRequired("Stored Schwab OAuth token timestamps must include a timezone.")
        return bundle


class SchwabOAuthClient:
    """Manage Schwab authorization-code exchange and access-token refresh.

    Args:
        credentials: Schwab application's client identifier and secret.
        token_store: Secure persistence for OAuth token JSON.
        redirect_uri: Exact HTTPS callback registered for the Schwab app.
        session: Optional HTTP session for tests or connection reuse.
        clock: UTC time source used for deterministic expiry checks.
        timeout_seconds: Network timeout for token requests.
    """

    def __init__(
        self,
        *,
        credentials: SchwabClientCredentials,
        token_store: SchwabTokenStore,
        redirect_uri: str,
        session: OAuthSession | None = None,
        clock: Callable[[], datetime] | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        _validate_https_redirect(redirect_uri)
        self._credentials = credentials
        self._token_store = token_store
        self._redirect_uri = redirect_uri
        self._session = session or requests.Session()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._timeout_seconds = timeout_seconds
        self._authorization_url = "https://api.schwabapi.com/v1/oauth/authorize"
        self._token_url = "https://api.schwabapi.com/v1/oauth/token"

    @property
    def redirect_uri(self) -> str:
        """Return the exact callback registered with Schwab."""

        return self._redirect_uri

    def begin_authorization(self) -> SchwabAuthorizationRequest:
        """Create a browser URL and random state for Schwab user consent."""

        state = secrets.token_urlsafe(32)
        query = urlencode(
            {
                "client_id": self._credentials.client_id,
                "redirect_uri": self._redirect_uri,
                "response_type": "code",
                "state": state,
            }
        )
        return SchwabAuthorizationRequest(url=f"{self._authorization_url}?{query}", state=state)

    def complete_authorization(self, *, redirected_url: str, expected_state: str) -> None:
        """Validate a Schwab callback, exchange its code, and persist tokens.

        Args:
            redirected_url: Complete callback URL copied from the browser.
            expected_state: State returned by :meth:`begin_authorization`.

        Raises:
            SchwabAuthorizationError: If callback validation or token exchange fails.
        """

        callback = urlsplit(redirected_url.strip())
        expected = urlsplit(self._redirect_uri)
        if (callback.scheme, callback.netloc, callback.path) != (
            expected.scheme,
            expected.netloc,
            expected.path,
        ):
            raise SchwabAuthorizationError("The OAuth callback does not match the registered Schwab redirect URI.")
        query = parse_qs(callback.query)
        if "error" in query:
            raise SchwabAuthorizationError("Schwab declined or failed the authorization request.")
        returned_state = _single_query_value(query, "state")
        if not secrets.compare_digest(returned_state, expected_state):
            raise SchwabAuthorizationError("Schwab OAuth state validation failed; restart authorization.")
        code = _single_query_value(query, "code")
        token_payload = self._post_token(
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self._redirect_uri,
            }
        )
        try:
            bundle = _bundle_from_response(token_payload, now=self._now())
        except ValueError as exc:
            raise SchwabAuthorizationError("Schwab returned an invalid token response.") from exc
        self._token_store.save(bundle.to_json())

    def get_access_token(self) -> str:
        """Return a valid bearer token, refreshing it when possible.

        Returns:
            Current access-token secret for an immediate Schwab request.

        Raises:
            SchwabAuthorizationRequired: If no refreshable authorization exists.
        """

        token_json = self._token_store.load()
        if token_json is None:
            raise SchwabAuthorizationRequired("Schwab authorization has not been completed.")
        bundle = SchwabTokenBundle.from_json(token_json)
        now = self._now()
        if now < bundle.access_expires_at - timedelta(seconds=60):
            return bundle.access_token
        if bundle.refresh_expires_at is not None and now >= bundle.refresh_expires_at:
            raise SchwabAuthorizationRequired("Schwab refresh authorization expired; authorize again.")

        token_payload = self._post_token(
            {
                "grant_type": "refresh_token",
                "refresh_token": bundle.refresh_token,
            }
        )
        try:
            refreshed = _bundle_from_response(token_payload, now=now, previous=bundle)
        except ValueError as exc:
            raise SchwabAuthorizationError("Schwab returned an invalid refresh response.") from exc
        self._token_store.save(refreshed.to_json())
        return refreshed.access_token

    def _post_token(self, form: Mapping[str, str]) -> dict[str, Any]:
        try:
            response = self._session.post(
                self._token_url,
                auth=(self._credentials.client_id, self._credentials.client_secret),
                data=dict(form),
                headers={"Accept": "application/json"},
                timeout=self._timeout_seconds,
            )
        except requests.RequestException as exc:
            raise SchwabAuthorizationError("Could not reach Schwab's token service.") from exc
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise SchwabAuthorizationError(
                _safe_token_failure_message(
                    response,
                    secret_values=(
                        self._credentials.client_secret,
                        *form.values(),
                    ),
                )
            ) from exc
        except requests.RequestException as exc:
            raise SchwabAuthorizationError("Schwab's token service rejected the request.") from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise SchwabAuthorizationError("Schwab's token service returned unreadable data.") from exc
        if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
            raise SchwabAuthorizationError("Schwab returned an invalid token response.")
        return payload

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("Schwab OAuth clock must return a timezone-aware datetime")
        return now


def _safe_token_failure_message(response: OAuthResponse, *, secret_values: tuple[str, ...]) -> str:
    """Describe a failed token request without including credential material."""

    status_code = getattr(response, "status_code", None)
    status_suffix = f" (HTTP {status_code})" if isinstance(status_code, int) else ""
    try:
        payload = response.json()
    except (ValueError, requests.RequestException):
        payload = None
    if not isinstance(payload, dict):
        return f"Schwab token exchange failed{status_suffix}."

    safe_parts: list[str] = []
    for key in ("error", "error_description"):
        value = payload.get(key)
        if not isinstance(value, str):
            continue
        normalized = " ".join(value.split())[:240]
        if normalized and not any(secret and secret in normalized for secret in secret_values):
            safe_parts.append(normalized)
    detail = ": ".join(safe_parts)
    if detail:
        return f"Schwab token exchange failed{status_suffix}: {detail}."
    return f"Schwab token exchange failed{status_suffix}."


def _bundle_from_response(
    payload: Mapping[str, Any],
    *,
    now: datetime,
    previous: SchwabTokenBundle | None = None,
) -> SchwabTokenBundle:
    access_token = _required_string(payload, "access_token")
    refresh_token = _optional_string(payload, "refresh_token")
    if refresh_token is None and previous is not None:
        refresh_token = previous.refresh_token
    if refresh_token is None:
        raise SchwabAuthorizationError("Schwab's token response omitted a refresh token.")
    expires_in = _positive_seconds(payload, "expires_in")
    refresh_expires_in = _optional_positive_seconds(payload, "refresh_token_expires_in")
    if refresh_expires_in is not None:
        refresh_expires_at = now + timedelta(seconds=refresh_expires_in)
    elif previous is not None:
        refresh_expires_at = previous.refresh_expires_at
    else:
        refresh_expires_at = None
    return SchwabTokenBundle(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type=_optional_string(payload, "token_type") or "Bearer",
        scope=_optional_string(payload, "scope"),
        access_expires_at=now + timedelta(seconds=expires_in),
        refresh_expires_at=refresh_expires_at,
    )


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"OAuth field {key!r} must be a non-empty string")
    return value


def _optional_string(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"OAuth field {key!r} must be a non-empty string when present")
    return value


def _positive_seconds(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool):
        raise ValueError(f"OAuth field {key!r} must be positive seconds")
    try:
        seconds = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"OAuth field {key!r} must be positive seconds") from exc
    if seconds <= 0:
        raise ValueError(f"OAuth field {key!r} must be positive seconds")
    return seconds


def _optional_positive_seconds(payload: Mapping[str, Any], key: str) -> int | None:
    if payload.get(key) is None:
        return None
    return _positive_seconds(payload, key)


def _single_query_value(query: Mapping[str, list[str]], key: str) -> str:
    values = query.get(key)
    if values is None or len(values) != 1 or not values[0]:
        raise SchwabAuthorizationError(f"Schwab OAuth callback omitted a valid {key!r} value.")
    return values[0]


def _validate_https_redirect(redirect_uri: str) -> None:
    parsed = urlsplit(redirect_uri)
    if parsed.scheme != "https" or not parsed.netloc or parsed.fragment or parsed.query:
        raise ValueError("Schwab redirect_uri must be an HTTPS URL without query parameters or fragments")
