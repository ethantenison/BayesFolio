"""Authorize and inspect a read-only Schwab account connection.

Usage:
    uv run --locked python scripts/connect_schwab.py credentials
    uv run --locked python scripts/connect_schwab.py authorize
    uv run --locked python scripts/connect_schwab.py accounts
    uv run --locked python scripts/connect_schwab.py set-default
    uv run --locked python scripts/connect_schwab.py snapshot

The authorization command opens Schwab in the default browser. After consent,
copy the complete callback URL from the browser address bar into the hidden
prompt. The authorization code and all resulting tokens stay local.
"""

from __future__ import annotations

import argparse
import getpass
import sys
import webbrowser

from bayesfolio.application import make_default_schwab_account_access
from bayesfolio.io.schwab.account_reader import SchwabAccountDataError, SchwabLinkedAccount
from bayesfolio.io.schwab.credentials import (
    AppleKeychainCredentialStore,
    SchwabClientCredentials,
    SchwabCredentialError,
    SchwabDefaultAccount,
)
from bayesfolio.io.schwab.oauth import SchwabAuthorizationError


def main() -> int:
    """Run the requested local Schwab authorization or account-read action."""

    parser = argparse.ArgumentParser(description="Read-only Schwab connection for BayesFolio")
    parser.add_argument(
        "action",
        choices=("credentials", "authorize", "accounts", "set-default", "snapshot"),
    )
    args = parser.parse_args()
    try:
        if args.action == "credentials":
            _save_credentials()
            return 0
        access = make_default_schwab_account_access()
        if args.action == "authorize":
            authorization = access.begin_authorization()
            print("Opening Schwab authorization in your browser.")
            print("Select the Schwab accounts you want BayesFolio to read.")
            if not webbrowser.open(authorization.url):
                print(f"Open this URL manually:\n{authorization.url}")
            redirected_url = getpass.getpass(
                "After Schwab redirects, paste the complete callback URL here (input is hidden): "
            )
            access.complete_authorization(
                redirected_url=redirected_url,
                expected_state=authorization.state,
            )
            print("Schwab authorization saved securely in macOS Keychain.")
            _print_accounts(access.list_linked_accounts())
            return 0
        if args.action == "accounts":
            _print_accounts(access.list_linked_accounts(), default=access.get_default_account())
            return 0

        accounts = access.list_linked_accounts()
        if args.action == "set-default":
            selected = _select_account(accounts)
            alias = input("Account alias [retirement_ira]: ").strip() or "retirement_ira"
            access.set_default_account(account=selected, alias=alias)
            print(f"Saved {alias!r} ({selected.masked_account_number}) as the default Schwab account.")
            return 0

        selected = _resolve_default_account(accounts, default=access.get_default_account())
        snapshot = access.get_account_snapshot(account_reference=selected.account_reference)
        print(snapshot.model_dump_json(indent=2))
        return 0
    except (SchwabAccountDataError, SchwabAuthorizationError, SchwabCredentialError, ValueError) as exc:
        print(f"Schwab connection failed: {exc}", file=sys.stderr)
        return 1


def _save_credentials() -> None:
    """Prompt for Schwab app credentials and save them in macOS Keychain."""

    print("Copy these values from the Schwab app entry in Apple Passwords.")
    print("Both prompts are hidden; the terminal will look blank while you paste.")
    client_id = getpass.getpass("Paste Schwab client ID, then press Return: ").strip()
    client_secret = getpass.getpass("Paste Schwab client secret, then press Return: ").strip()
    AppleKeychainCredentialStore().save(
        SchwabClientCredentials(client_id=client_id, client_secret=client_secret),
    )
    print("Schwab client credentials saved securely in macOS Keychain.")


def _print_accounts(
    accounts: list[SchwabLinkedAccount],
    *,
    default: SchwabDefaultAccount | None = None,
) -> None:
    if not accounts:
        print("No Schwab accounts are linked to this application.")
        return
    print("Linked Schwab accounts:")
    for index, account in enumerate(accounts, start=1):
        default_label = ""
        if default is not None and default.account_reference == account.account_reference:
            default_label = f" (default: {default.alias})"
        print(f"  {index}. {account.masked_account_number}{default_label}")


def _select_account(accounts: list[SchwabLinkedAccount]) -> SchwabLinkedAccount:
    if not accounts:
        raise SchwabAccountDataError("No Schwab accounts are linked to this application.")
    if len(accounts) == 1:
        return accounts[0]
    _print_accounts(accounts)
    selection = input("Choose the account number to snapshot: ").strip()
    try:
        return accounts[int(selection) - 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("Account selection must be one of the displayed numbers.") from exc


def _resolve_default_account(
    accounts: list[SchwabLinkedAccount],
    *,
    default: SchwabDefaultAccount | None,
) -> SchwabLinkedAccount:
    if default is not None:
        for account in accounts:
            if account.account_reference == default.account_reference:
                print(f"Using default Schwab account {default.alias!r} ({account.masked_account_number}).")
                return account
        print("The saved default account is no longer linked; choose another account.")
    return _select_account(accounts)


if __name__ == "__main__":
    raise SystemExit(main())
