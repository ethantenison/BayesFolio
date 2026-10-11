"""Tests for monthly application dependency composition."""

from __future__ import annotations

from typing import cast

import pytest

from bayesfolio.application import monthly_portfolio as application
from bayesfolio.contracts.commands.monthly_portfolio import (
    MonthlyPortfolioArtifactsConfig,
    MonthlyPortfolioCommand,
)
from bayesfolio.engine.automation.monthly_portfolio import MonthlyPortfolioRuntimeDependencies
from bayesfolio.engine.automation.monthly_ports import MonthlyPortfolioRunIO
from bayesfolio.engine.features.dataset_builder import ArtifactStoreProtocol, FeatureProviders


def _command() -> MonthlyPortfolioCommand:
    return MonthlyPortfolioCommand.model_construct(
        artifacts=MonthlyPortfolioArtifactsConfig(artifact_root="artifacts/test-monthly")
    )


def _dependencies(*, run_io: MonthlyPortfolioRunIO | None = None) -> MonthlyPortfolioRuntimeDependencies:
    return MonthlyPortfolioRuntimeDependencies(
        feature_providers=cast(FeatureProviders, object()),
        artifact_store=cast(ArtifactStoreProtocol, object()),
        run_io=run_io,
    )


def test_build_default_dependencies_assembles_providers_store_and_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _command()
    providers = cast(FeatureProviders, object())
    store = cast(ArtifactStoreProtocol, object())
    runtime_io = cast(MonthlyPortfolioRunIO, object())
    calls: dict[str, object] = {}

    def _providers(*, cache_root):
        calls["cache_root"] = cache_root
        return providers

    def _store(*, base_dir):
        calls["base_dir"] = base_dir
        return store

    def _runtime(config):
        calls["artifacts_config"] = config
        return runtime_io

    monkeypatch.setattr(application, "make_default_feature_providers", _providers)
    monkeypatch.setattr(application, "ParquetArtifactStore", _store)
    monkeypatch.setattr(application, "build_monthly_portfolio_run_io", _runtime)

    dependencies = application.build_default_monthly_portfolio_dependencies(command)

    assert dependencies.feature_providers is providers
    assert dependencies.artifact_store is store
    assert dependencies.run_io is runtime_io
    assert calls == {
        "cache_root": "artifacts/cache",
        "base_dir": "artifacts/test-monthly",
        "artifacts_config": command.artifacts,
    }


def test_complete_dependencies_preserves_legacy_provider_and_store(monkeypatch: pytest.MonkeyPatch) -> None:
    command = _command()
    legacy = _dependencies()
    runtime_io = cast(MonthlyPortfolioRunIO, object())
    monkeypatch.setattr(application, "build_monthly_portfolio_run_io", lambda config: runtime_io)

    completed = application.complete_monthly_portfolio_dependencies(command, legacy)

    assert completed.feature_providers is legacy.feature_providers
    assert completed.artifact_store is legacy.artifact_store
    assert completed.run_io is runtime_io


def test_complete_dependencies_keeps_complete_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    command = _command()
    runtime_io = cast(MonthlyPortfolioRunIO, object())
    complete = _dependencies(run_io=runtime_io)

    monkeypatch.setattr(
        application,
        "build_monthly_portfolio_run_io",
        lambda config: pytest.fail("complete dependencies should not be rebuilt"),
    )

    assert application.complete_monthly_portfolio_dependencies(command, complete) is complete


def test_application_runner_completes_dependencies_before_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    command = _command()
    completed = _dependencies(run_io=cast(MonthlyPortfolioRunIO, object()))
    expected_result = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        application,
        "complete_monthly_portfolio_dependencies",
        lambda command, dependencies: completed,
    )

    def _execute(command, *, dependencies):
        captured["command"] = command
        captured["dependencies"] = dependencies
        return expected_result

    monkeypatch.setattr(application, "execute_monthly_portfolio", _execute)

    result = application.run_monthly_portfolio(command)

    assert result is expected_result
    assert captured == {"command": command, "dependencies": completed}


def test_engine_entry_point_warns_and_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    from bayesfolio.engine.automation import monthly_portfolio as engine

    command = _command()
    expected_result = object()
    monkeypatch.setattr(application, "run_monthly_portfolio", lambda command, dependencies=None: expected_result)

    with pytest.deprecated_call(match="bayesfolio.application.run_monthly_portfolio"):
        result = engine.run_monthly_portfolio(command)

    assert result is expected_result
