"""Tests for GET-only Schwab account access and contract mapping."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pytest

from bayesfolio.io.schwab.account_reader import SchwabBrokerageAccountReader, SchwabReadOnlyClient


class StaticTokenProvider:
    """Return a fixed bearer token for local HTTP-client tests."""

    def get_access_token(self) -> str:
        return "access-token"


class FakeResponse:
    """Minimal successful requests response."""

    def __init__(self, payload: object) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


class GetOnlySession:
    """Record GET calls and return responses by path suffix."""

    def __init__(self) -> None:
        self.gets: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.gets.append({"url": url, **kwargs})
        if url.endswith("/accounts/accountNumbers"):
            return FakeResponse([{"accountNumber": "12345678", "hashValue": "opaque-hash"}])
        if url.endswith("/transactions"):
            return FakeResponse([{"activityId": 42, "type": "DIVIDEND_OR_INTEREST"}])
        return FakeResponse(
            {
                "securitiesAccount": {
                    "currentBalances": {
                        "cashBalance": 50.0,
                        "liquidationValue": 1050.0,
                    },
                    "positions": [
                        {
                            "longQuantity": 10.0,
                            "shortQuantity": 0.0,
                            "marketValue": 1000.0,
                            "averageLongPrice": 90.0,
                            "instrument": {"symbol": "spy", "assetType": "EQUITY"},
                        }
                    ],
                }
            }
        )


def test_account_discovery_masks_plain_account_number_and_uses_get_only() -> None:
    session = GetOnlySession()
    client = SchwabReadOnlyClient(
        token_provider=StaticTokenProvider(),
        session=session,
    )

    accounts = client.list_linked_accounts()

    assert accounts[0].account_reference == "opaque-hash"
    assert accounts[0].masked_account_number == "…5678"
    assert session.gets[0]["headers"]["Authorization"] == "Bearer access-token"
    assert not hasattr(client, "post")
    assert not hasattr(client, "request")


def test_account_reader_maps_and_reconciles_schwab_snapshot() -> None:
    session = GetOnlySession()
    client = SchwabReadOnlyClient(
        token_provider=StaticTokenProvider(),
        session=session,
    )
    reader = SchwabBrokerageAccountReader(
        client=client,
        clock=lambda: datetime(2026, 10, 11, 16, 0, tzinfo=UTC),
    )

    snapshot = reader.get_account_snapshot(account_reference="opaque-hash")

    assert snapshot.account_reference == "opaque-hash"
    assert snapshot.net_liquidation_value == Decimal("1050.0")
    assert snapshot.cash_value == Decimal("50.0")
    assert snapshot.other_value == Decimal("0.0")
    assert snapshot.positions[0].asset == "SPY"
    assert snapshot.positions[0].cost_basis == Decimal("900.00")
    assert snapshot.positions[0].unrealized_gain_loss == Decimal("100.00")


def test_transactions_use_get_with_bounded_dates() -> None:
    session = GetOnlySession()
    client = SchwabReadOnlyClient(
        token_provider=StaticTokenProvider(),
        session=session,
    )

    transactions = client.get_transactions(
        account_reference="opaque-hash",
        start_date=date(2026, 9, 1),
        end_date=date(2026, 9, 30),
        transaction_types=["TRADE", "DIVIDEND_OR_INTEREST"],
    )

    assert transactions == [{"activityId": 42, "type": "DIVIDEND_OR_INTEREST"}]
    assert session.gets[0]["params"] == {
        "startDate": "2026-09-01T00:00:00.000Z",
        "endDate": "2026-09-30T23:59:59.999Z",
        "types": "TRADE,DIVIDEND_OR_INTEREST",
    }


def test_read_only_client_rejects_order_paths_before_network_access() -> None:
    session = GetOnlySession()
    client = SchwabReadOnlyClient(
        token_provider=StaticTokenProvider(),
        session=session,
    )

    with pytest.raises(ValueError, match="non-allowlisted"):
        client._get_json("/accounts/opaque-hash/orders")

    assert session.gets == []
