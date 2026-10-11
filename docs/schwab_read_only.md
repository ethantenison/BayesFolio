# Read-Only Schwab Account Access

Status: Active

Audience: Users, contributors, and coding agents

Canonical topic: Charles Schwab OAuth setup and read-only account retrieval

Last verified: 2026-10-11

## Scope

BayesFolio can read user-authorized retail Schwab account balances, long-only
positions, and transactions through Trader API - Individual. It cannot place,
replace, or cancel orders. Rebalance results remain non-executing planning
outputs.

## Schwab Application

Create a Production application with these values:

- API product: `Accounts and Trading Production`
- Order limit: `0`
- Callback URL: `https://127.0.0.1:8182/callback`

The callback URL is registered before BayesFolio is authorized. The current
local flow does not run a public web service: after Schwab redirects to the
loopback callback, copy the complete URL from the browser address bar into the
hidden terminal prompt. A browser connection error at the loopback address does
not invalidate the authorization response contained in that URL.

## Save Client Credentials

An Apple Passwords entry is a good secure copy of the Schwab application
credentials. However, some iCloud Passwords records are not exposed to the
macOS command-line Keychain API, even when they are visible in the Passwords
app. BayesFolio therefore uses a dedicated generic Keychain entry.

Run:

```bash
uv run --locked python scripts/connect_schwab.py credentials
```

Copy the following values from the Schwab app entry in Apple Passwords when
prompted:

- client ID
- client secret

Both prompts are hidden, so no characters appear while typing or pasting. Press
Return after each value. BayesFolio stores both in the
`bayesfolio.schwab.client-credentials` generic-password entry in the macOS login
keychain, under the `BayesFolio OAuth Client` account label. The original
Passwords entry can remain as a backup.

Do not use the Schwab brokerage password. Do not put the client secret,
authorization code, access token, or refresh token in source files, `.env`
files, shell history, screenshots, logs, or chat messages.

BayesFolio stores OAuth tokens separately in the
`bayesfolio.schwab.oauth-token` generic-password entry. Secret values are never
supplied as command arguments.

## Authorize

After saving the credentials, run from the repository root:

```bash
uv run --locked python scripts/connect_schwab.py authorize
```

Log into Schwab in the opened browser and select the accounts that BayesFolio
should be allowed to read. Linking multiple accounts is supported; the
`set-default` command below records which linked account the monthly workflow
should treat as the preferred portfolio. Copy the complete redirected URL into
the hidden prompt. The script validates the OAuth state before exchanging the
authorization code.

## Inspect Linked Accounts and a Snapshot

```bash
uv run --locked python scripts/connect_schwab.py accounts
uv run --locked python scripts/connect_schwab.py set-default
uv run --locked python scripts/connect_schwab.py snapshot
```

Account discovery displays only masked account-number suffixes. The snapshot
command uses the saved default when one exists; otherwise it asks which linked
account to read. `set-default` assigns a local alias such as `retirement_ira`
and stores the opaque account reference in macOS Keychain rather than a project
file. Snapshot output contains sensitive portfolio values; do not paste it into
public issues or logs.

## Security Boundary

The Schwab adapter provides only explicit GET calls for account discovery,
balances, positions, and transactions. It has no generic HTTP request method and
no order-entry methods. The application receives validated
`BrokerageAccountSnapshot` objects rather than credentials or raw authorization
material.

If Schwab authorization expires, repeat the `authorize` command. Revoking the
application in Schwab invalidates BayesFolio access.
