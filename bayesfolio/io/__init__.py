"""Input/output and persistence utilities."""

from bayesfolio.io.artifact_store import ParquetArtifactStore
from bayesfolio.io.artifacts import (
    save_dataframe_csv,
    save_json_contract,
    save_plotly_figure_html,
    save_plotly_figures_html,
)
from bayesfolio.io.backends import (
    ArtifactBackend,
    FsspecArtifactBackend,
    LocalArtifactBackend,
    default_artifact_root_uri,
    make_artifact_backend,
    resolve_backend_and_key,
)
from bayesfolio.io.fingerprints import sha256_fingerprint
from bayesfolio.io.mlflow_logger import log_artifact_ref, log_contract
from bayesfolio.io.parquet_store import write_parquet_with_metadata
from bayesfolio.io.providers import EtfFeaturesProvider, MacroProvider, ReturnsProvider
from bayesfolio.io.report_artifacts import persist_report_diagnostic_figures

__all__ = [
    "log_artifact_ref",
    "log_contract",
    "ParquetArtifactStore",
    "ArtifactBackend",
    "LocalArtifactBackend",
    "FsspecArtifactBackend",
    "default_artifact_root_uri",
    "make_artifact_backend",
    "resolve_backend_and_key",
    "save_dataframe_csv",
    "save_json_contract",
    "save_plotly_figure_html",
    "save_plotly_figures_html",
    "persist_report_diagnostic_figures",
    "EtfFeaturesProvider",
    "MacroProvider",
    "ReturnsProvider",
    "sha256_fingerprint",
    "write_parquet_with_metadata",
]
