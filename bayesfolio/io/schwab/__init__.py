"""Read-only Charles Schwab authentication and account-data adapters.

The package owns external OAuth, macOS credential storage, and Schwab HTTP
access. It emits validated BayesFolio contracts and never exposes order-entry
operations.
"""

from bayesfolio.io.schwab.account_reader import (
    SchwabBrokerageAccountReader,
    SchwabLinkedAccount,
    SchwabReadOnlyClient,
)
from bayesfolio.io.schwab.credentials import (
    AppleKeychainCredentialStore,
    AppleKeychainDefaultAccountStore,
    AppleKeychainTokenStore,
    SchwabClientCredentials,
    SchwabCredentialError,
    SchwabDefaultAccount,
)
from bayesfolio.io.schwab.oauth import (
    SchwabAuthorizationError,
    SchwabAuthorizationRequest,
    SchwabAuthorizationRequired,
    SchwabOAuthClient,
    SchwabTokenBundle,
)

__all__ = [
    "AppleKeychainCredentialStore",
    "AppleKeychainDefaultAccountStore",
    "AppleKeychainTokenStore",
    "SchwabAuthorizationError",
    "SchwabAuthorizationRequest",
    "SchwabAuthorizationRequired",
    "SchwabBrokerageAccountReader",
    "SchwabClientCredentials",
    "SchwabCredentialError",
    "SchwabDefaultAccount",
    "SchwabLinkedAccount",
    "SchwabOAuthClient",
    "SchwabReadOnlyClient",
    "SchwabTokenBundle",
]
