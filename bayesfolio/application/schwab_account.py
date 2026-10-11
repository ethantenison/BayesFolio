"""Compose BayesFolio's supported read-only Schwab account workflow.

This application root wires macOS Keychain storage, Schwab OAuth, the GET-only
Trader API client, and the engine-owned brokerage snapshot boundary. Monetary
values emitted by the workflow are USD amounts.
"""

from __future__ import annotations

from dataclasses import dataclass

from bayesfolio.contracts.results.brokerage import BrokerageAccountSnapshot
from bayesfolio.io.schwab import (
    AppleKeychainCredentialStore,
    AppleKeychainDefaultAccountStore,
    AppleKeychainTokenStore,
    SchwabAuthorizationRequest,
    SchwabBrokerageAccountReader,
    SchwabDefaultAccount,
    SchwabLinkedAccount,
    SchwabOAuthClient,
    SchwabReadOnlyClient,
)

DEFAULT_SCHWAB_REDIRECT_URI = "https://127.0.0.1:8182/callback"


@dataclass(frozen=True, slots=True)
class SchwabAccountAccess:
    """Application service for authorization and read-only account snapshots.

    Attributes:
        oauth: Schwab authorization and token-refresh adapter.
        client: GET-only Schwab account-data client.
        reader: Adapter producing validated BayesFolio brokerage snapshots.
        default_account_store: Secure local preference for the default account.
    """

    oauth: SchwabOAuthClient
    client: SchwabReadOnlyClient
    reader: SchwabBrokerageAccountReader
    default_account_store: AppleKeychainDefaultAccountStore

    def begin_authorization(self) -> SchwabAuthorizationRequest:
        """Return a browser authorization request for the Schwab account owner."""

        return self.oauth.begin_authorization()

    def complete_authorization(self, *, redirected_url: str, expected_state: str) -> None:
        """Complete Schwab consent and save OAuth tokens in macOS Keychain.

        Args:
            redirected_url: Complete URL shown after Schwab redirects the browser.
            expected_state: State returned by :meth:`begin_authorization`.
        """

        self.oauth.complete_authorization(
            redirected_url=redirected_url,
            expected_state=expected_state,
        )

    def list_linked_accounts(self) -> list[SchwabLinkedAccount]:
        """Return redacted identifiers for accounts selected during consent."""

        return self.client.list_linked_accounts()

    def get_account_snapshot(self, *, account_reference: str) -> BrokerageAccountSnapshot:
        """Return one current, reconciled Schwab account snapshot.

        Args:
            account_reference: Opaque hash returned by :meth:`list_linked_accounts`.

        Returns:
            Validated read-only account state in USD.
        """

        return self.reader.get_account_snapshot(account_reference=account_reference)

    def get_default_account(self) -> SchwabDefaultAccount | None:
        """Return the saved default linked account, if one is configured."""

        return self.default_account_store.load()

    def set_default_account(self, *, account: SchwabLinkedAccount, alias: str) -> None:
        """Save one discovered account as the local default.

        Args:
            account: Redacted account identity returned by Schwab discovery.
            alias: Human-readable local name for the account.
        """

        self.default_account_store.save(
            SchwabDefaultAccount(
                alias=alias,
                account_reference=account.account_reference,
                masked_account_number=account.masked_account_number,
            )
        )


def make_default_schwab_account_access(
    *,
    redirect_uri: str = DEFAULT_SCHWAB_REDIRECT_URI,
) -> SchwabAccountAccess:
    """Build the default macOS-backed read-only Schwab account service.

    Args:
        redirect_uri: Exact HTTPS callback registered in the Schwab application.

    Returns:
        Composed authorization, discovery, and snapshot service.
    """

    credentials = AppleKeychainCredentialStore().load()
    oauth = SchwabOAuthClient(
        credentials=credentials,
        token_store=AppleKeychainTokenStore(),
        redirect_uri=redirect_uri,
    )
    client = SchwabReadOnlyClient(token_provider=oauth)
    reader = SchwabBrokerageAccountReader(client=client)
    return SchwabAccountAccess(
        oauth=oauth,
        client=client,
        reader=reader,
        default_account_store=AppleKeychainDefaultAccountStore(),
    )
