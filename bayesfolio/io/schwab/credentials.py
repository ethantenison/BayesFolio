"""Load Schwab OAuth credentials and tokens from macOS Keychain.

Client credentials and OAuth tokens use separate app-specific generic-password
entries. Secret values are written through the native Security framework so
they do not appear in process arguments, shell history, or temporary files.
"""

from __future__ import annotations

import ctypes
import json
import sys
from dataclasses import dataclass, field
from typing import Protocol


class SchwabCredentialError(RuntimeError):
    """Report missing or inaccessible Schwab secrets without exposing them."""


@dataclass(frozen=True, slots=True)
class SchwabClientCredentials:
    """OAuth client credentials retrieved from secure local storage.

    Attributes:
        client_id: Schwab application client identifier.
        client_secret: Schwab application secret, excluded from representations.
    """

    client_id: str
    client_secret: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class SchwabDefaultAccount:
    """Locally saved identity for the preferred linked Schwab account.

    Attributes:
        alias: Human-readable local name such as ``retirement_ira``.
        account_reference: Opaque Schwab API hash, excluded from representations.
        masked_account_number: Redacted account-number suffix for confirmation.
    """

    alias: str
    account_reference: str = field(repr=False)
    masked_account_number: str


class SchwabTokenStore(Protocol):
    """Persistence boundary for the serialized Schwab OAuth token bundle."""

    def load(self) -> str | None:
        """Return the stored token JSON, or ``None`` when not yet authorized."""

        ...

    def save(self, token_json: str) -> None:
        """Persist token JSON without exposing it in command arguments or logs.

        Args:
            token_json: Serialized OAuth token bundle containing secrets.
        """

        ...


class _KeychainBackend(Protocol):
    def get_password(self, *, service: str, account: str) -> str | None:
        """Return one password value, or ``None`` when the item is absent."""

        ...

    def set_password(self, *, service: str, account: str, value: str) -> None:
        """Create or replace one password value."""

        ...


class _MacOSKeychainBackend:
    """Read and write generic passwords through the native Security framework."""

    _ITEM_NOT_FOUND = -25300

    def __init__(self) -> None:
        _require_macos()
        self._security = ctypes.CDLL("/System/Library/Frameworks/Security.framework/Security")
        self._core_foundation = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        self._configure_signatures()

    def get_password(self, *, service: str, account: str) -> str | None:
        """Read a UTF-8 password from the user's default Keychain."""

        service_bytes = service.encode()
        account_bytes = account.encode()
        password_length = ctypes.c_uint32()
        password_data = ctypes.c_void_p()
        status = self._security.SecKeychainFindGenericPassword(
            None,
            len(service_bytes),
            service_bytes,
            len(account_bytes),
            account_bytes,
            ctypes.byref(password_length),
            ctypes.byref(password_data),
            None,
        )
        if status == self._ITEM_NOT_FOUND:
            return None
        self._raise_for_status(status)
        try:
            raw_value = ctypes.string_at(password_data, password_length.value)
            return raw_value.decode()
        except UnicodeDecodeError as exc:
            raise SchwabCredentialError("The macOS Keychain entry is not valid UTF-8.") from exc
        finally:
            if password_data.value is not None:
                self._security.SecKeychainItemFreeContent(None, password_data)

    def set_password(self, *, service: str, account: str, value: str) -> None:
        """Create or replace a UTF-8 password in the default Keychain."""

        service_bytes = service.encode()
        account_bytes = account.encode()
        value_bytes = value.encode()
        value_buffer = ctypes.create_string_buffer(value_bytes)
        value_pointer = ctypes.cast(value_buffer, ctypes.c_void_p)
        item_ref = ctypes.c_void_p()
        status = self._security.SecKeychainFindGenericPassword(
            None,
            len(service_bytes),
            service_bytes,
            len(account_bytes),
            account_bytes,
            None,
            None,
            ctypes.byref(item_ref),
        )
        if status == self._ITEM_NOT_FOUND:
            status = self._security.SecKeychainAddGenericPassword(
                None,
                len(service_bytes),
                service_bytes,
                len(account_bytes),
                account_bytes,
                len(value_bytes),
                value_pointer,
                None,
            )
            self._raise_for_status(status)
            return
        self._raise_for_status(status)
        try:
            status = self._security.SecKeychainItemModifyAttributesAndData(
                item_ref,
                None,
                len(value_bytes),
                value_pointer,
            )
            self._raise_for_status(status)
        finally:
            if item_ref.value is not None:
                self._core_foundation.CFRelease(item_ref)

    def _configure_signatures(self) -> None:
        self._security.SecKeychainFindGenericPassword.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._security.SecKeychainFindGenericPassword.restype = ctypes.c_int32
        self._security.SecKeychainAddGenericPassword.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._security.SecKeychainAddGenericPassword.restype = ctypes.c_int32
        self._security.SecKeychainItemModifyAttributesAndData.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        self._security.SecKeychainItemModifyAttributesAndData.restype = ctypes.c_int32
        self._security.SecKeychainItemFreeContent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._security.SecKeychainItemFreeContent.restype = ctypes.c_int32
        self._core_foundation.CFRelease.argtypes = [ctypes.c_void_p]
        self._core_foundation.CFRelease.restype = None

    @staticmethod
    def _raise_for_status(status: int) -> None:
        if status != 0:
            raise SchwabCredentialError(f"macOS Keychain operation failed with status {status}.")


class AppleKeychainCredentialStore:
    """Persist Schwab client credentials as one generic Keychain password.

    Args:
        service: Keychain service label for the serialized credentials.
        account: Keychain account label.
        backend: Injectable secure-store backend used for deterministic tests.
    """

    def __init__(
        self,
        *,
        service: str = "bayesfolio.schwab.client-credentials",
        account: str = "BayesFolio OAuth Client",
        backend: _KeychainBackend | None = None,
    ) -> None:
        self._service = service
        self._account = account
        self._backend = backend or _MacOSKeychainBackend()

    def load(self) -> SchwabClientCredentials:
        """Return the Schwab client credentials stored in macOS Keychain.

        Returns:
            Client identifier and secret from the app-specific Keychain item.

        Raises:
            SchwabCredentialError: If the entry is missing, unreadable, or invalid.
        """

        stored_value = self._backend.get_password(service=self._service, account=self._account)
        if stored_value is None:
            raise SchwabCredentialError(
                "Schwab client credentials are not configured. Run "
                "'uv run --locked python scripts/connect_schwab.py credentials'."
            )
        try:
            payload = json.loads(stored_value)
            client_id = payload["client_id"].strip()
            client_secret = payload["client_secret"].strip()
        except (json.JSONDecodeError, KeyError, AttributeError, TypeError) as exc:
            raise SchwabCredentialError("The Schwab client-credentials Keychain entry is invalid.") from exc
        if not client_id or not client_secret:
            raise SchwabCredentialError("The Schwab client-credentials Keychain entry is incomplete.")
        return SchwabClientCredentials(client_id=client_id, client_secret=client_secret)

    def save(self, credentials: SchwabClientCredentials) -> None:
        """Store credentials without placing either value in process arguments.

        Args:
            credentials: Schwab OAuth client identifier and secret.

        Raises:
            SchwabCredentialError: If Keychain rejects the update.
        """

        client_id = credentials.client_id.strip()
        client_secret = credentials.client_secret.strip()
        if not client_id or not client_secret:
            raise ValueError("Schwab client ID and client secret must not be empty")
        payload = json.dumps(
            {"client_id": client_id, "client_secret": client_secret},
            separators=(",", ":"),
        )
        self._backend.set_password(service=self._service, account=self._account, value=payload)


class AppleKeychainTokenStore:
    """Persist Schwab OAuth tokens as one generic macOS Keychain password.

    Args:
        service: Keychain service label for the serialized token bundle.
        account: Keychain account label.
        backend: Injectable secure-store backend used for deterministic tests.
    """

    def __init__(
        self,
        *,
        service: str = "bayesfolio.schwab.oauth-token",
        account: str = "BayesFolio",
        backend: _KeychainBackend | None = None,
    ) -> None:
        self._service = service
        self._account = account
        self._backend = backend or _MacOSKeychainBackend()

    def load(self) -> str | None:
        """Return stored token JSON without writing secret material to logs."""

        token_json = self._backend.get_password(service=self._service, account=self._account)
        return token_json or None

    def save(self, token_json: str) -> None:
        """Store token JSON through standard input rather than a process argument.

        Args:
            token_json: Serialized OAuth token bundle containing secrets.

        Raises:
            SchwabCredentialError: If Keychain rejects the token update.
        """

        if not token_json:
            raise ValueError("token_json must not be empty")
        self._backend.set_password(service=self._service, account=self._account, value=token_json)


class AppleKeychainDefaultAccountStore:
    """Persist the preferred Schwab account identity in macOS Keychain.

    Args:
        service: Keychain service label for the serialized preference.
        account: Keychain account label.
        backend: Injectable secure-store backend used for deterministic tests.
    """

    def __init__(
        self,
        *,
        service: str = "bayesfolio.schwab.default-account",
        account: str = "BayesFolio",
        backend: _KeychainBackend | None = None,
    ) -> None:
        self._service = service
        self._account = account
        self._backend = backend or _MacOSKeychainBackend()

    def load(self) -> SchwabDefaultAccount | None:
        """Return the saved account preference, or ``None`` when unset.

        Raises:
            SchwabCredentialError: If the saved preference is invalid.
        """

        stored_value = self._backend.get_password(service=self._service, account=self._account)
        if stored_value is None:
            return None
        try:
            payload = json.loads(stored_value)
            alias = payload["alias"].strip()
            account_reference = payload["account_reference"].strip()
            masked_account_number = payload["masked_account_number"].strip()
        except (json.JSONDecodeError, KeyError, AttributeError, TypeError) as exc:
            raise SchwabCredentialError("The saved Schwab default-account entry is invalid.") from exc
        if not alias or not account_reference or not masked_account_number.startswith("…"):
            raise SchwabCredentialError("The saved Schwab default-account entry is incomplete.")
        return SchwabDefaultAccount(
            alias=alias,
            account_reference=account_reference,
            masked_account_number=masked_account_number,
        )

    def save(self, preference: SchwabDefaultAccount) -> None:
        """Save the preferred linked account without exposing its opaque hash.

        Args:
            preference: Alias, opaque account reference, and redacted suffix.
        """

        alias = preference.alias.strip()
        account_reference = preference.account_reference.strip()
        masked_account_number = preference.masked_account_number.strip()
        if not alias or not account_reference or not masked_account_number.startswith("…"):
            raise ValueError("default Schwab account preference is incomplete")
        payload = json.dumps(
            {
                "alias": alias,
                "account_reference": account_reference,
                "masked_account_number": masked_account_number,
            },
            separators=(",", ":"),
        )
        self._backend.set_password(service=self._service, account=self._account, value=payload)


def _require_macos() -> None:
    if sys.platform != "darwin":
        raise SchwabCredentialError("macOS Keychain access requires macOS.")
