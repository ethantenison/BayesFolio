"""Read Schwab account state and map it into BayesFolio contracts.

The HTTP client deliberately exposes GET operations only. The adapter uses
Schwab's opaque account hashes, discards full account numbers after discovery,
and reports all monetary values in account currency (currently USD).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol
from urllib.parse import quote

import requests

from bayesfolio.contracts.results.brokerage import BrokerageAccountSnapshot, BrokeragePosition

_ALLOWED_READ_PATH = re.compile(r"^/accounts(?:/accountNumbers|/[^/]+(?:/transactions)?)?$")


class SchwabAccountDataError(RuntimeError):
    """Report an invalid or unavailable Schwab account-data response."""


class AccessTokenProvider(Protocol):
    """Provide a current Schwab bearer token to the read-only HTTP client."""

    def get_access_token(self) -> str:
        """Return a valid access token for one immediate request."""

        ...


class AccountResponse(Protocol):
    """Minimal HTTP response surface required by account retrieval."""

    def raise_for_status(self) -> None:
        """Raise when the HTTP response is unsuccessful."""

        ...

    def json(self) -> object:
        """Return the decoded response body."""

        ...


class AccountSession(Protocol):
    """GET-only HTTP session boundary for Schwab account data."""

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, str] | None,
        timeout: float,
    ) -> AccountResponse:
        """Send one authenticated account-data GET request."""

        ...


@dataclass(frozen=True, slots=True)
class SchwabLinkedAccount:
    """Redacted linked-account identity returned during discovery.

    Attributes:
        account_reference: Schwab's opaque hash used for API calls.
        masked_account_number: Last four account-number characters for local selection.
    """

    account_reference: str
    masked_account_number: str


class SchwabReadOnlyClient:
    """GET-only client for Schwab balances, positions, and transactions.

    The class has no generic request method and never calls ``Session.request``;
    every network operation uses ``Session.get`` with a bearer token.

    Args:
        token_provider: OAuth provider returning a current access token.
        session: Optional HTTP session for tests or connection reuse.
        timeout_seconds: Network timeout for account-data requests.
        base_url: Schwab Trader API base URL.
    """

    def __init__(
        self,
        *,
        token_provider: AccessTokenProvider,
        session: AccountSession | None = None,
        timeout_seconds: float = 20.0,
        base_url: str = "https://api.schwabapi.com/trader/v1",
    ) -> None:
        self._token_provider = token_provider
        self._session = session or requests.Session()
        self._timeout_seconds = timeout_seconds
        self._base_url = base_url.rstrip("/")

    def list_linked_accounts(self) -> list[SchwabLinkedAccount]:
        """Return opaque hashes and masked identifiers for linked accounts."""

        payload = self._get_json("/accounts/accountNumbers")
        if not isinstance(payload, list):
            raise SchwabAccountDataError("Schwab account discovery returned an invalid response.")
        accounts: list[SchwabLinkedAccount] = []
        for item in payload:
            if not isinstance(item, dict):
                raise SchwabAccountDataError("Schwab account discovery returned an invalid account entry.")
            account_reference = _required_text(item, "hashValue")
            account_number = _required_text(item, "accountNumber")
            masked = f"…{account_number[-4:]}" if len(account_number) >= 4 else "…"
            accounts.append(
                SchwabLinkedAccount(
                    account_reference=account_reference,
                    masked_account_number=masked,
                )
            )
        return accounts

    def get_account(self, *, account_reference: str) -> dict[str, Any]:
        """Return one linked account with current balances and positions.

        Args:
            account_reference: Schwab account hash from :meth:`list_linked_accounts`.

        Returns:
            Raw JSON object from Schwab's account endpoint.
        """

        encoded_reference = _encoded_account_reference(account_reference)
        payload = self._get_json(f"/accounts/{encoded_reference}", params={"fields": "positions"})
        if not isinstance(payload, dict):
            raise SchwabAccountDataError("Schwab account retrieval returned an invalid response.")
        return payload

    def get_transactions(
        self,
        *,
        account_reference: str,
        start_date: date,
        end_date: date,
        transaction_types: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return account transactions in an inclusive requested date window.

        Args:
            account_reference: Schwab account hash from account discovery.
            start_date: Inclusive transaction start date.
            end_date: Inclusive transaction end date; must not precede start.
            transaction_types: Optional Schwab transaction-type filters.

        Returns:
            Raw transaction objects for later cash-flow classification.

        Raises:
            ValueError: If the date range is invalid.
        """

        if end_date < start_date:
            raise ValueError("end_date must be on or after start_date")
        encoded_reference = _encoded_account_reference(account_reference)
        params: dict[str, str] = {
            "startDate": f"{start_date.isoformat()}T00:00:00.000Z",
            "endDate": f"{end_date.isoformat()}T23:59:59.999Z",
        }
        if transaction_types:
            params["types"] = ",".join(transaction_types)
        payload = self._get_json(f"/accounts/{encoded_reference}/transactions", params=params)
        if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
            raise SchwabAccountDataError("Schwab transaction retrieval returned an invalid response.")
        return payload

    def _get_json(
        self,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
    ) -> object:
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("Schwab request path must be an absolute API path")
        if _ALLOWED_READ_PATH.fullmatch(path) is None:
            raise ValueError("Schwab read-only client rejected a non-allowlisted API path")
        try:
            response = self._session.get(
                f"{self._base_url}{path}",
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self._token_provider.get_access_token()}",
                },
                params=dict(params) if params is not None else None,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            raise SchwabAccountDataError(
                "Schwab account-data retrieval failed without displaying credentials or response contents."
            ) from exc


class SchwabBrokerageAccountReader:
    """Implement BayesFolio's account-reader protocol with Schwab account data.

    Args:
        client: GET-only Schwab account client.
        clock: UTC observation-time source used for deterministic tests.
    """

    def __init__(
        self,
        *,
        client: SchwabReadOnlyClient,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._client = client
        self._clock = clock or (lambda: datetime.now(UTC))

    def get_account_snapshot(self, *, account_reference: str) -> BrokerageAccountSnapshot:
        """Return a reconciled, point-in-time Schwab account snapshot.

        Args:
            account_reference: Opaque Schwab account hash, never a full account number.

        Returns:
            Validated USD balances and long-only positions.

        Raises:
            SchwabAccountDataError: If Schwab data is incomplete, short, or invalid.
        """

        payload = self._client.get_account(account_reference=account_reference)
        account = payload.get("securitiesAccount", payload)
        if not isinstance(account, dict):
            raise SchwabAccountDataError("Schwab response omitted a securities account object.")
        balances = account.get("currentBalances")
        if not isinstance(balances, dict):
            raise SchwabAccountDataError("Schwab response omitted current account balances.")

        positions = _map_positions(account.get("positions", []))
        cash_value = _decimal(balances.get("cashBalance", 0), field_name="cashBalance")
        net_liquidation_value = _decimal(
            balances.get("liquidationValue"),
            field_name="liquidationValue",
        )
        if cash_value < 0:
            raise SchwabAccountDataError("Negative cash balances are outside the read-only IRA snapshot contract.")
        positions_value = sum((position.market_value for position in positions), start=Decimal("0"))
        other_value = net_liquidation_value - cash_value - positions_value
        observed_at = self._clock()
        if observed_at.tzinfo is None:
            raise ValueError("Schwab account-reader clock must return a timezone-aware datetime")
        return BrokerageAccountSnapshot(
            account_reference=account_reference,
            as_of=observed_at,
            currency="USD",
            net_liquidation_value=net_liquidation_value,
            cash_value=cash_value,
            other_value=other_value,
            positions=positions,
        )


@dataclass(slots=True)
class _PositionAccumulator:
    quantity: Decimal = Decimal("0")
    market_value: Decimal = Decimal("0")
    cost_basis: Decimal | None = Decimal("0")


def _map_positions(raw_positions: object) -> list[BrokeragePosition]:
    if not isinstance(raw_positions, list):
        raise SchwabAccountDataError("Schwab positions must be a list.")
    accumulators: dict[str, _PositionAccumulator] = {}
    for raw_position in raw_positions:
        if not isinstance(raw_position, dict):
            raise SchwabAccountDataError("Schwab returned an invalid position entry.")
        short_quantity = _decimal(raw_position.get("shortQuantity", 0), field_name="shortQuantity")
        if short_quantity > 0:
            raise SchwabAccountDataError("Short positions are outside the read-only IRA snapshot contract.")
        quantity = _decimal(raw_position.get("longQuantity", 0), field_name="longQuantity")
        if quantity <= 0:
            continue
        instrument = raw_position.get("instrument")
        if not isinstance(instrument, dict):
            raise SchwabAccountDataError("Schwab position omitted instrument metadata.")
        symbol = _required_text(instrument, "symbol").upper()
        market_value = _decimal(raw_position.get("marketValue"), field_name="marketValue")
        if market_value < 0:
            raise SchwabAccountDataError("Negative position values are outside the long-only snapshot contract.")
        average_price_raw = raw_position.get("averageLongPrice", raw_position.get("averagePrice"))
        cost_basis = (
            _decimal(average_price_raw, field_name="averageLongPrice") * quantity
            if average_price_raw is not None
            else None
        )

        accumulator = accumulators.setdefault(symbol, _PositionAccumulator())
        accumulator.quantity += quantity
        accumulator.market_value += market_value
        if accumulator.cost_basis is None or cost_basis is None:
            accumulator.cost_basis = None
        else:
            accumulator.cost_basis += cost_basis

    positions: list[BrokeragePosition] = []
    for symbol, accumulator in sorted(accumulators.items()):
        market_price = accumulator.market_value / accumulator.quantity
        unrealized = accumulator.market_value - accumulator.cost_basis if accumulator.cost_basis is not None else None
        positions.append(
            BrokeragePosition(
                asset=symbol,
                quantity=accumulator.quantity,
                market_price=market_price,
                market_value=accumulator.market_value,
                cost_basis=accumulator.cost_basis,
                unrealized_gain_loss=unrealized,
            )
        )
    return positions


def _decimal(value: object, *, field_name: str) -> Decimal:
    if value is None or isinstance(value, bool):
        raise SchwabAccountDataError(f"Schwab field {field_name!r} must be numeric.")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise SchwabAccountDataError(f"Schwab field {field_name!r} must be numeric.") from exc
    if not result.is_finite():
        raise SchwabAccountDataError(f"Schwab field {field_name!r} must be finite.")
    return result


def _required_text(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SchwabAccountDataError(f"Schwab field {key!r} must be a non-empty string.")
    return value.strip()


def _encoded_account_reference(account_reference: str) -> str:
    if not account_reference:
        raise ValueError("account_reference must not be empty")
    return quote(account_reference, safe="")
