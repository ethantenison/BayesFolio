"""Tests for macOS-backed Schwab credential and token persistence."""

from __future__ import annotations

from bayesfolio.io.schwab.credentials import (
    AppleKeychainCredentialStore,
    AppleKeychainDefaultAccountStore,
    AppleKeychainTokenStore,
    SchwabClientCredentials,
    SchwabDefaultAccount,
)


class FakeKeychainBackend:
    """Return deterministic Keychain values and record writes."""

    def __init__(self) -> None:
        self.values = {
            ("bayesfolio.schwab.client-credentials", "BayesFolio OAuth Client"): (
                '{"client_id":"client-id-value","client_secret":"client-secret-value"}'
            ),
            ("bayesfolio.schwab.oauth-token", "BayesFolio"): '{"token":"value"}',
        }
        self.writes: list[tuple[str, str, str]] = []

    def get_password(self, *, service: str, account: str) -> str | None:
        return self.values.get((service, account))

    def set_password(self, *, service: str, account: str, value: str) -> None:
        self.writes.append((service, account, value))
        self.values[(service, account)] = value


def test_keychain_credentials_load_without_repr_leak() -> None:
    backend = FakeKeychainBackend()

    credentials = AppleKeychainCredentialStore(backend=backend).load()

    assert credentials.client_id == "client-id-value"
    assert credentials.client_secret == "client-secret-value"
    assert "client-secret-value" not in repr(credentials)


def test_keychain_credentials_write_one_serialized_value() -> None:
    backend = FakeKeychainBackend()
    store = AppleKeychainCredentialStore(backend=backend)

    store.save(
        SchwabClientCredentials(
            client_id="client-id-value",
            client_secret="client-secret-value",
        )
    )

    payload = '{"client_id":"client-id-value","client_secret":"client-secret-value"}'
    assert backend.writes[-1] == (
        "bayesfolio.schwab.client-credentials",
        "BayesFolio OAuth Client",
        payload,
    )


def test_keychain_token_store_writes_serialized_token() -> None:
    backend = FakeKeychainBackend()
    store = AppleKeychainTokenStore(backend=backend)

    store.save('{"access_token":"secret-token"}')

    assert backend.writes[-1] == (
        "bayesfolio.schwab.oauth-token",
        "BayesFolio",
        '{"access_token":"secret-token"}',
    )


def test_keychain_token_store_reads_serialized_token() -> None:
    backend = FakeKeychainBackend()

    token_json = AppleKeychainTokenStore(backend=backend).load()

    assert token_json == '{"token":"value"}'


def test_keychain_default_account_round_trip() -> None:
    backend = FakeKeychainBackend()
    store = AppleKeychainDefaultAccountStore(backend=backend)

    store.save(
        SchwabDefaultAccount(
            alias="retirement_ira",
            account_reference="opaque-account-hash",
            masked_account_number="…1234",
        )
    )

    preference = store.load()
    assert preference is not None
    assert preference.alias == "retirement_ira"
    assert preference.account_reference == "opaque-account-hash"
    assert preference.masked_account_number == "…1234"
    assert "opaque-account-hash" not in repr(preference)
