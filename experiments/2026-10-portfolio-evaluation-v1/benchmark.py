"""Verify and time batched Gaussian evaluation against an independent SciPy loop.

Synthetic benchmark only, never investment evidence or a GP fitting benchmark.
Arrays are raw decimal returns/covariance. No input files or subprocess pool.

Usage:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
        .venv/bin/python experiments/2026-10-portfolio-evaluation-v1/benchmark.py
"""

from __future__ import annotations

import json
import time

import numpy as np
from metrics import joint_gaussian_scores, paired_block_bootstrap
from scipy.stats import multivariate_normal


def main() -> None:
    """Print measured speed and numerical parity for a fixed-seed synthetic workload."""
    rng = np.random.default_rng(20261020)
    months, assets = 1000, 15
    mean = rng.normal(0, 0.01, (months, assets))
    actual = rng.normal(0, 0.03, (months, assets))
    factors = rng.normal(0, 0.005, (months, assets, assets))
    covariance = factors @ factors.swapaxes(-1, -2) + np.eye(assets)[None] * 0.0001
    started = time.perf_counter()
    vectorized = joint_gaussian_scores(actual, mean, covariance)["joint_nll"]
    batched_seconds = time.perf_counter() - started
    started = time.perf_counter()
    reference = np.array([-multivariate_normal.logpdf(y, mu, cov) for y, mu, cov in
                          zip(actual, mean, covariance, strict=True)])
    loop_seconds = time.perf_counter() - started
    np.testing.assert_allclose(vectorized, reference, rtol=0, atol=1e-8)
    started = time.perf_counter()
    paired_block_bootstrap(actual[:24, :3], actual[24:48, :3])
    bootstrap_seconds = time.perf_counter() - started
    print(json.dumps({"synthetic_only": True, "months": months, "assets": assets,
        "batched_joint_nll_seconds": batched_seconds, "scipy_reference_loop_seconds": loop_seconds,
        "speedup_vs_reference_loop": loop_seconds / batched_seconds,
        "maximum_absolute_nll_difference": float(np.max(np.abs(vectorized - reference))),
        "paired_2000_replicate_bootstrap_seconds": bootstrap_seconds,
        "gp_fits": 0, "worker_processes": 1}, indent=2))


if __name__ == "__main__":
    main()
