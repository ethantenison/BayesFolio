"""Read-only artifact adapters for the isolated portfolio evaluation framework.

Hashes exactly the bytes parsed, validates task/date ordering and target units,
and never imports a fitting runner or refreshes data. All prices/returns are
converted to decimal monthly values. Generated outputs belong to the new folder.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


class Reader:
    """Track immutable input bytes independently of evaluation results.

    Attributes:
        hashes: Absolute input paths mapped to the SHA-256 of parsed bytes.
    """

    def __init__(self) -> None:
        self.hashes: dict[str, str] = {}

    def read(self, path: Path) -> Any:
        """Parse a supported local artifact while recording its exact fingerprint.

        Args:
            path: Existing CSV, JSON, parquet, NPZ or UTF-8 text artifact.

        Returns:
            Parsed value; NPZ arrays are copied with pickle explicitly forbidden.
        """
        payload = path.read_bytes()
        name = str(path.resolve())
        digest = hashlib.sha256(payload).hexdigest()
        if name in self.hashes and self.hashes[name] != digest:
            raise ValueError(f"Input changed during evaluation: {path}")
        self.hashes[name] = digest
        buffer = io.BytesIO(payload)
        if path.suffix == ".parquet":
            return pd.read_parquet(buffer)
        if path.suffix == ".csv":
            return pd.read_csv(buffer)
        if path.suffix == ".json":
            return json.loads(payload)
        if path.suffix == ".npz":
            with np.load(buffer, allow_pickle=False) as arrays:
                return {key: arrays[key].copy() for key in arrays.files}
        return payload.decode()

    def verify(self) -> None:
        """Raise if any source changed since it was parsed; never rewrite inputs."""
        for name, digest in self.hashes.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Evaluation source changed: {name}")


def aligned_matrix(frame: pd.DataFrame, column: str, dates: list[str], assets: list[str]) -> np.ndarray:
    """Require exact keys before constructing a months-by-assets decimal matrix.

    Args:
        frame: Source table with unique date and asset_id keys.
        column: Numeric column to extract.
        dates: Explicit chronological forecast-origin grid, ISO dates.
        assets: Explicit asset order used by all covariance and weight arrays.

    Returns:
        Complete finite float64 matrix; no imputation, dropping or positional joins.
    """
    expected = pd.MultiIndex.from_product([dates, sorted(assets)], names=["date", "asset_id"])
    indexed = frame.set_index(["date", "asset_id"]).sort_index()
    if indexed.index.has_duplicates or not indexed.index.equals(expected):
        raise ValueError("Artifact dates/assets are missing, duplicated or unexpected")
    matrix = frame.pivot(index="date", columns="asset_id", values=column).reindex(index=dates, columns=assets)
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite {column}; no observation can be silently dropped")
    return values


def verify_legacy_audit(reader: Reader, root: Path) -> int:
    """Verify the completed study's existing evidence seal without updating it.

    Args:
        reader: Input provenance tracker.
        root: Original repaired study root, read-only.

    Returns:
        Number of old evidence files whose hashes still match the published audit.
    """
    audit = reader.read(root / "followup/results/all_joint/independent_audit.json")
    if not audit.get("passed") or audit.get("unresolved"):
        raise ValueError("Original joint evidence was not successfully audited")
    repository = root.resolve().parents[1]
    for name, expected in audit["source_sha256"].items():
        path = repository / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Original audited evidence changed: {name}")
        reader.hashes[str(path.resolve())] = actual
    return len(audit["source_sha256"])


def load_context(reader: Reader, root: Path) -> dict:
    """Cache shared training-only baselines and resolved return conversion once.

    Args:
        reader: Provenance-tracking local input reader.
        root: Existing repaired study directory, accessed read-only.

    Returns:
        Date/task mapping, raw target panel, expanding means/common historical
        covariance, fixed probes and realized RF. The historical target equals
        (1+total_return)/(1+RF)-1, not arithmetic total_return minus RF.
    """
    study = reader.read(root / "study.json")
    assets = [asset for family in study["families"].values() for asset in family]
    dates = reader.read(root / "artifacts/v1/scored_dates_24.txt").splitlines()
    if dates != sorted(set(dates)) or len(dates) != 24 or len(assets) != 15:
        raise ValueError("Unexpected approved date grid or task universe")
    panel = reader.read(root / "artifacts/v2/global.parquet")
    panel["date"] = pd.to_datetime(panel.date).dt.strftime("%Y-%m-%d")
    all_dates = sorted(panel.date.unique().tolist())
    targets = aligned_matrix(panel, "y_excess_lead", all_dates, assets)
    positions = [all_dates.index(date) for date in dates]
    if positions != list(range(84, 108)):
        raise ValueError("Expected exactly 84+k resolved training dates")
    baseline = np.stack([targets[:position].mean(axis=0) for position in positions])
    historical_covariance = np.stack([LedoitWolf().fit(targets[:position]).covariance_ for position in positions])
    raw_path = root / "artifacts/v1/raw/INDEX_IRX.parquet"
    tail_path = root / "artifacts/target_audit_v1/INDEX_IRX.parquet"
    raw_meta = reader.read(raw_path.parent / "sources.json")
    tail_meta = reader.read(tail_path.parent / "sources.json")
    raw, tail = reader.read(raw_path)["Adj Close"], reader.read(tail_path)["Adj Close"]
    if reader.hashes[str(raw_path.resolve())] != raw_meta["^IRX"]["sha256"]:
        raise ValueError("Unregistered risk-free source")
    if reader.hashes[str(tail_path.resolve())] != tail_meta["^IRX"]["sha256"]:
        raise ValueError("Unregistered forward risk-free source")
    annual_percent = pd.concat([raw, tail[tail.index > raw.index.max()]])
    if annual_percent.index.has_duplicates or not annual_percent.index.is_monotonic_increasing:
        raise ValueError("Risk-free source chronology differs")
    daily = (annual_percent / 100 / 252).reindex(pd.bdate_range(
        annual_percent.index.min(), annual_percent.index.max())).ffill()
    rf = []
    for date in dates:
        point = pd.Timestamp(date)
        endpoint = point + pd.offsets.BMonthEnd(1)
        if point < daily.index.min() or endpoint > daily.index.max():
            raise ValueError("Incomplete risk-free outcome endpoint")
        interval = daily.loc[(daily.index > point) & (daily.index <= endpoint)]
        if interval.empty or interval.isna().any():
            raise ValueError("Missing integrated risk-free return")
        rf.append(np.expm1(interval.sum()))
    probes = {"equal_weight": np.ones(len(assets)) / len(assets)}
    for family, tickers in study["families"].items():
        probes[f"equal_weight_{family}"] = np.array([1 / len(tickers) if asset in tickers else 0 for asset in assets])
    equity = set(study["families"]["us_equity"] + study["families"]["international_equity"])
    probes["fixed_60_40"] = np.array([0.6 / len(equity) if asset in equity else 0.4 / (len(assets) - len(equity))
                                    for asset in assets])
    return {"assets": assets, "dates": dates, "families": study["families"], "baseline_mean": baseline,
            "historical_covariance": historical_covariance, "actual": targets[positions],
            "risk_free": np.asarray(rf), "fixed_probes": probes,
            "return_convention": "discounted_excess=(1+total)/(1+RF)-1"}


def load_run(reader: Reader, root: Path, model_id: str, seed: int, context: dict) -> dict:
    """Read and independently align one saved full-universe run without refitting.

    Args:
        reader: Tracking local reader.
        root: Read-only repaired study directory.
        model_id: Full joint model identifier in followup/runs.
        seed: Original fitting seed.
        context: Shared dates, assets and frozen actuals.

    Returns:
        Marginal Gaussian forecasts and saved strictly primary GP allocations.
    """
    run = root / "followup/runs" / f"{model_id}_s{seed}"
    manifest = reader.read(run / "manifest.json")
    dates, assets = context["dates"], context["assets"]
    if manifest["modeling"]["seed"] != seed or manifest["data"]["training_universe"] != assets:
        raise ValueError("Saved task universe or seed differs")
    if manifest["feature_sha256"] != reader.hashes[str((root / "artifacts/v2/global.parquet").resolve())]:
        raise ValueError("Saved data fingerprint differs")
    frame = reader.read(run / "gp_window_predictions.csv")
    actual = aligned_matrix(frame, "y_true", dates, assets)
    if not np.allclose(actual, context["actual"], rtol=0, atol=1e-12):
        raise ValueError("Forecast labels differ from frozen repaired targets")
    weights = reader.read(run / "portfolio_weights.csv")
    selected = weights[weights.strategy.eq("gp_scenarios_riskfolio")].copy()
    extra = selected[~selected.asset_id.isin(assets)]
    if not np.allclose(extra.weight, 0, rtol=0, atol=1e-12):
        raise ValueError("Unexpected nonzero assets/cash in saved allocation")
    selected = selected[selected.asset_id.isin(assets)]
    allocation = aligned_matrix(selected, "weight", dates, assets)
    if allocation.min() < -1e-9 or not np.allclose(allocation.sum(axis=1), 1, rtol=0, atol=1e-7):
        raise ValueError("Invalid saved weights; no normalization or replacement")
    diagnostics = reader.read(run / "riskfolio_optimization_diagnostics.csv")
    primary = diagnostics[diagnostics.strategy.eq("gp_scenarios_riskfolio")]
    if (len(primary) != len(dates) or set(primary.date) != set(dates)
            or not primary.status.eq("solved").all() or not primary.fallback_stage.eq("primary").all()):
        raise ValueError("Saved portfolio has missing/failed/fallback solves")
    return {"run": run, "manifest": manifest, "actual": actual,
            "mean": aligned_matrix(frame, "y_pred", dates, assets),
            "std": aligned_matrix(frame, "y_std", dates, assets), "legacy_weights": allocation}


def load_covariance(reader: Reader, run: dict, context: dict, mode: str,
                    analytic_root: Path | None, seed: int) -> tuple[np.ndarray | None, dict]:
    """Distinguish exact predictive covariance from Monte Carlo approximations.

    Args:
        reader: Tracking reader.
        run: Saved run and matching marginal predictions.
        context: Explicit chronological dates and asset order.
        mode: none, scenarios, or analytic; no automatic fallback between modes.
        analytic_root: Optional directory with sSEED.npz and sSEED.json.
        seed: Fitting seed for exact-covariance artifact names.

    Returns:
        Covariance array or None, with unavoidable precision/source labels.
    """
    if mode == "none":
        return None, {"source": "unavailable", "exact_gp_joint_nll_available": False}
    if mode == "scenarios":
        matrices, counts = [], []
        for date in context["dates"]:
            frame = reader.read(run["run"] / f"gp_scenarios_{date}.csv")
            if set(frame.columns) != set(context["assets"]) or len(frame) <= len(context["assets"]):
                raise ValueError("Incomplete joint scenario universe/sample count")
            values = frame[context["assets"]].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite saved joint scenarios")
            matrices.append(np.cov(values, rowvar=False, ddof=1))
            counts.append(len(values))
        return np.stack(matrices), {"source": "gaussian_plugin_sample_covariance_of_saved_joint_scenarios",
            "exact_gp_joint_nll_available": False, "monte_carlo_approximation": True,
            "scenario_counts": counts, "mean_source": "saved_analytic_gp_mean"}
    if mode != "analytic" or analytic_root is None:
        raise ValueError("Analytic mode requires an explicit covariance artifact directory")
    arrays = reader.read(analytic_root / f"s{seed}.npz")
    meta = reader.read(analytic_root / f"s{seed}.json")
    if (arrays["dates"].tolist() != context["dates"] or arrays["assets"].tolist() != context["assets"]
            or not np.allclose(arrays["mean"], run["mean"], rtol=0, atol=1e-12)):
        raise ValueError("Exact covariance axes/means differ from saved predictions")
    prediction_path = str((run["run"] / "gp_window_predictions.csv").resolve())
    if (meta.get("kind") != "observed_predictive" or meta.get("source") != "analytic_gp"
            or meta.get("units") != "discounted_excess_decimal_squared"
            or meta.get("prediction_sha256") != reader.hashes[prediction_path]):
        raise ValueError("Unverified exact-covariance semantics/provenance")
    covariance = arrays["covariance"]
    if not np.allclose(np.diagonal(covariance, axis1=-2, axis2=-1), run["std"]**2, rtol=1e-7, atol=1e-12):
        raise ValueError("Exact predictive diagonal disagrees with saved observation SD")
    return covariance, {"source": "analytic_gp_observed_predictive", "exact_gp_joint_nll_available": True,
                        "monte_carlo_approximation": False}
