from __future__ import annotations

from datetime import date

import pandas as pd

from bayesfolio.contracts.results.backtest import BacktestResult
from bayesfolio.contracts.results.optimize import OptimizeResult
from bayesfolio.contracts.ui.universe import UniverseRecord
from bayesfolio.core.settings import RiskfolioConfig
from bayesfolio.engine.mvp_historical_chat import (
    DataQualityResult,
    HistoricalMvpRequest,
    HistoricalMvpResult,
    assess_data_quality,
    parse_chat_request,
    run_historical_mvp_chat_turn,
    run_historical_mvp_pipeline,
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


def test_parse_chat_request_extracts_nea_and_upperlng_overrides() -> None:
    request = parse_chat_request("Build portfolio for SPY, QQQ nea 8 upperlng 0.25 from 2020-01-01 to 2024-12-31")

    assert request.nea == 8
    assert request.max_weight == 0.25


def test_parse_chat_request_extracts_max_weight_percent_alias() -> None:
    request = parse_chat_request("Build portfolio for SPY, QQQ from 2020-01-01 to 2024-12-31 max_weight=35%")

    assert request.max_weight == 0.35


def test_parse_chat_request_does_not_treat_article_as_ticker() -> None:
    request = parse_chat_request(
        "Build a portfolio for SPY, IJR, VNQ, VWO, VEA, VNQI, IEF, LQD, EWX, VWOB "
        "from 2022-01-01 to 2025-12-31 objective sharpe risk cvar with max weight of 35% and nea of 8"
    )

    assert request.tickers == ["SPY", "IJR", "VNQ", "VWO", "VEA", "VNQI", "IEF", "LQD", "EWX", "VWOB"]
    assert request.max_weight == 0.35
    assert request.nea == 8


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
            portfolio_metrics={
                "cumulative_return": 0.12,
                "annualized_return": 0.08,
                "annualized_volatility": 0.11,
                "max_drawdown": -0.09,
                "sharpe_ratio": 0.73,
                "sortino_ratio": 1.05,
                "calmar_ratio": 0.89,
            },
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
    assert "metrics" in turn.tool_results[-1].payload
    metrics_payload = turn.tool_results[-1].payload["metrics"]
    assert metrics_payload["max_drawdown"] <= 0.0
    assert "sortino_ratio" in metrics_payload
    assert "calmar_ratio" in metrics_payload
    assert turn.assistant_message is not None
    assert "Historical MVP Portfolio Report" in turn.assistant_message.content


def test_run_historical_mvp_pipeline_uses_riskfolio_defaults(monkeypatch) -> None:
    defaults = RiskfolioConfig()
    captured: dict[str, object] = {}

    returns_matrix = pd.DataFrame(
        {
            "SPY": [0.01, 0.02, -0.01, 0.005],
            "QQQ": [0.015, 0.01, -0.02, 0.01],
        },
        index=pd.date_range("2020-01-31", periods=4, freq="BME"),
    )
    universe = UniverseRecord(asset_order=["SPY", "QQQ"], n_observations=4, return_unit="decimal")

    def _fake_universe_agent(request: HistoricalMvpRequest) -> tuple[UniverseRecord, pd.DataFrame]:
        return universe, returns_matrix

    def _fake_optimize(returns: pd.DataFrame, request) -> OptimizeResult:
        captured["opt_request"] = request
        return OptimizeResult(asset_order=["SPY", "QQQ"], weights=[0.5, 0.5])

    def _fake_backtest(realized_returns: pd.DataFrame, optimization: OptimizeResult) -> BacktestResult:
        return BacktestResult(
            cumulative_return=0.01,
            annualized_return=0.03,
            annualized_volatility=0.04,
            sharpe_ratio=0.75,
            max_drawdown=-0.02,
            calmar_ratio=1.50,
            sortino_ratio=1.10,
        )

    monkeypatch.setattr("bayesfolio.engine.mvp_historical_chat._run_universe_agent", _fake_universe_agent)
    monkeypatch.setattr("bayesfolio.engine.mvp_historical_chat.optimize_from_historical_returns", _fake_optimize)
    monkeypatch.setattr("bayesfolio.engine.mvp_historical_chat.run_weighted_backtest", _fake_backtest)

    request = HistoricalMvpRequest(
        tickers=["SPY", "QQQ"],
        start_date=date(2020, 1, 1),
        end_date=date(2020, 4, 30),
        build_features=False,
    )

    _ = run_historical_mvp_pipeline(request=request)

    opt_request = captured["opt_request"]
    assert opt_request.max_weight == defaults.upperlng
    assert opt_request.nea == defaults.nea
