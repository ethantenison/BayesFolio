from __future__ import annotations

from datetime import date

import pandas as pd

from bayesfolio.contracts.results.optimize import OptimizeResult
from bayesfolio.contracts.ui.universe import UniverseRecord
from bayesfolio.engine.mvp_historical_chat import (
    DataQualityResult,
    HistoricalMvpResult,
    assess_data_quality,
    parse_chat_request,
    run_historical_mvp_chat_turn,
)


def test_parse_chat_request_extracts_tickers_dates_and_settings() -> None:
    request = parse_chat_request(
        "Build portfolio for SPY, QQQ, TLT from 2020-01-01 to 2024-12-31 objective maxret risk cvar"
    )

    assert request.tickers == ["SPY", "QQQ", "TLT"]
    assert request.start_date == date(2020, 1, 1)
    assert request.end_date == date(2024, 12, 31)
    assert request.objective == "MaxRet"
    assert request.risk_measure == "CVaR"


def test_parse_chat_request_uses_default_dates_when_missing() -> None:
    request = parse_chat_request("tickers: SPY, IEF", today=date(2026, 3, 5))

    assert request.start_date == date(2021, 3, 6)
    assert request.end_date == date(2026, 3, 5)


def test_assess_data_quality_flags_stale_and_insufficient_assets() -> None:
    returns = pd.DataFrame(
        {
            "SPY": [0.01, 0.02, 0.03, 0.02],
            "TLT": [0.00, 0.00, 0.00, 0.00],
            "GLD": [0.01, None, 0.02, None],
        }
    )

    quality = assess_data_quality(returns, min_observations=4)

    assert quality.pass_gate is False
    assert "TLT" in quality.stale_assets
    assert "GLD" in quality.insufficient_history_assets
    assert quality.n_assets == 3
    assert quality.n_periods == 4


def test_assess_data_quality_passes_for_clean_panel() -> None:
    returns = pd.DataFrame(
        {
            "SPY": [0.01, 0.02, 0.00, 0.03],
            "QQQ": [0.02, 0.01, 0.01, 0.02],
        }
    )

    quality = assess_data_quality(returns, min_observations=4)

    assert quality.pass_gate is True
    assert quality.stale_assets == []
    assert quality.insufficient_history_assets == []


def test_run_historical_mvp_chat_turn_executes_tool_cycle(monkeypatch) -> None:
    def _fake_pipeline(request, progress=None):
        if progress is not None:
            progress("fake run")
        return HistoricalMvpResult(
            request=request,
            universe=UniverseRecord(asset_order=["SPY", "QQQ"], n_observations=60, return_unit="decimal"),
            data_quality=DataQualityResult(
                pass_gate=True,
                n_periods=60,
                n_assets=2,
                missing_rate_by_asset={"SPY": 0.0, "QQQ": 0.0},
                stale_assets=[],
                insufficient_history_assets=[],
            ),
            optimize_result=OptimizeResult(asset_order=["SPY", "QQQ"], weights=[0.6, 0.4]),
            report_markdown="### Historical MVP Portfolio Report\n- Top weights: SPY: 60.0%, QQQ: 40.0%",
            weights_table=pd.DataFrame(
                {
                    "asset": ["SPY", "QQQ"],
                    "weight": [0.6, 0.4],
                }
            ),
            agent_logs=["fake run"],
            warnings=[],
            features_result=None,
        )

    monkeypatch.setattr("bayesfolio.engine.mvp_historical_chat.run_historical_mvp_pipeline", _fake_pipeline)

    turn = run_historical_mvp_chat_turn("Build portfolio for SPY, QQQ from 2020-01-01 to 2024-12-31")

    assert turn.tool_results
    assert turn.tool_results[-1].success is True
    assert "report_markdown" in turn.tool_results[-1].payload
    assert turn.assistant_message is not None
    assert "Historical MVP Portfolio Report" in turn.assistant_message.content
