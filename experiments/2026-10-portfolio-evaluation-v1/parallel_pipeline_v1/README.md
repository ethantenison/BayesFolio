# Parallel forecast production and portfolio evaluation

This isolated pipeline connects the accepted 15-ETF baseline to the deterministic evaluation framework. Forecast months run independently in a global pool of up to 14 workers, followed by independent monthly allocation solves and then batched portfolio statistics. Earlier studies, datasets and evaluation outputs remain unchanged.

## Frozen model and data

The baseline keeps IWM, excludes IJR and REIT/commodity ETF targets, and uses the accepted 14 non-time inputs plus t_index. It does not add hy_spread_chg_1p. Rank remains two, the optimizer iteration cap remains 15, and fitting seeds remain 27/127/227. Each forecast origin trains on the original expanding 84+k monthly observations per ETF and uses its original deterministic window seed. Months do not have to finish chronologically to obey this cutoff.

The four family sizes remain 4/4/4/3, and all 15 tasks are fitted together in this baseline integration run. This is not yet a comparison of joint, regional and family architectures.

## Execution stages

1. Fit one seed/month and check legacy predictions/scenarios against the saved baseline at absolute tolerance 1e-8. Reuse that fit in the complete batch.
2. Complete all 72 independent monthly fits, with one Torch/BLAS thread per worker. The process ceiling is cores minus two, capped at 14, and bounded by observed worker peak memory with a 25% safety margin and a 48-GiB research budget.
3. Only after all forecasts are valid, solve all 72 independent monthly allocation jobs. Each job produces the three registered allocation policies below.
4. Assemble exact predictive covariance artifacts and evaluate all months, portfolios and seeds together using the existing pure numerical functions.

Saved checkpoints include source and artifact hashes. Resume skips only matching complete jobs and rejects changed artifacts. Failed jobs never obtain replacement predictions or carried-forward weights. One task-scoped file lock prevents duplicate launches for the same output directory.

## Predictive covariance export

The original model-building and fitting code is reused unchanged. The old fast-variance predictions and 256 scenarios remain available for parity checks. After fitting, a separate readout clears prediction caches, disables GPyTorch fast_pred_var, and forces dense Cholesky conditioning. It exports the full observed predictive covariance in raw target units, including likelihood noise and the fitted per-task outcome unscaling.

Exact here means a Gaussian conditional at fitted hyperparameters, not integration over hyperparameter uncertainty. The latent covariance is also queried to verify that the difference is the expected positive diagonal observation noise. No task-kernel matrix or sample covariance is substituted for the predictive matrix. Numerical warnings from the unchanged legacy fitter are distinct from the exported covariance's positive-definiteness check; a fitted model is not assumed to have converged merely because maxiter was reached.

The analytic readout can differ numerically from legacy fast predictive variances. Those differences are measured for every month and summarized in timing.json. This isolates readout differences from fitting differences rather than silently attributing them to parallelism. The evaluator checks exported means, task/date ordering, marginal variances and prediction-file hashes before scoring joint NLL and fixed-reference portfolio distributions.

## Allocation policies

| Policy | Objective | Risk and concentration controls |
| --- | --- | --- |
| Legacy scenario MaxRet | Maximize mean of the saved legacy scenarios | Long-only, sum one, 35% ETF cap, at least three effective assets; no variance budget. Reimplemented as the equivalent deterministic convex problem, not a new return forecast. |
| Risk rank with cap | Maximize a fixed rank score of analytic GP means | Same long-only/budget/diversification controls, 35% cap, common historical variance budget. |
| Risk rank without cap | Same rank objective | Remove only the 35% cap; retain at least three effective assets and the same variance budget. |

The common covariance is Ledoit-Wolf estimated from resolved target labels strictly before each forecast origin. The research variance budget is the common equal-weight portfolio variance at that origin. It is not a live investor risk target. The same covariance and risk budget are used for both cap variants. Rank scores have a fixed 1e-8 squared-weight tie-breaker; that coefficient is not optimized after inspecting results.

CLARABEL must return optimal status and weights must independently satisfy the constraints. There is no weight clipping, renormalization, constraint relaxation, alternate solver, previous-weight fallback or equal-weight fallback. A unit-budget upper bound of one is omitted in the no-cap solve because it is already implied by nonnegativity and full investment.

Equal-weight, fixed 60/40 and linear-rank controls are evaluated separately. The GP covariance is scored for calibration and joint/fixed-portfolio NLL; the rank allocator initially uses common historical covariance to keep the risk control comparable. This does not yet establish whether allocating with GP covariance is better.

## Run and verify

From the repository root:

```sh
PYTHONPATH=experiments/2026-10-portfolio-evaluation-v1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=/private/tmp/bayesfolio-mpl .venv/bin/python experiments/2026-10-portfolio-evaluation-v1/parallel_pipeline_v1/run_pipeline.py --execute
PYTHONPATH=experiments/2026-10-portfolio-evaluation-v1 .venv/bin/python experiments/2026-10-portfolio-evaluation-v1/parallel_pipeline_v1/run_pipeline.py --verify-only
PYTHONPATH=experiments/2026-10-portfolio-evaluation-v1 .venv/bin/python -m pytest -q experiments/2026-10-portfolio-evaluation-v1
.venv/bin/ruff check --no-force-exclude experiments/2026-10-portfolio-evaluation-v1
```

Use `--smoke-only` for the parity gate alone. Outputs are stored under this subfolder's outputs/baseline_parallel_v1 directory. A source change requires a new versioned run name; it cannot silently resume old results. A completed execution is verification-only on rerun.

COMPLETED.json seals monthly checkpoints, analytic artifacts, evaluation tables and timings. Each checkpoint additionally seals its individual predictions, scenarios, diagnostics and optimizer logs. The three-stage status is available in status.json. Evaluation tables retain the earlier hierarchy, gross-cost label, explicit return conversion, fixed reference portfolios and paired bootstrap conventions. Three seeds remain 24 market months, not 72 independent observations.

Tests check raw-unit dense conditioning against an independent manual calculation, observation-noise inclusion, unchanged model parameters, strict allocation failures, resource limits, original month indices, unordered completion and hash-valid resume. Pylance tooling is not available in this environment; tests, Ruff and provenance verification are the runnable checks.

The work is research-only and remains exploratory on reused evaluation months. No winner is automatically promoted, no production APIs change, and no Git commit or push is performed. Removing this new subfolder rolls back the producer while leaving earlier evaluation artifacts intact.

## Numerical references

[GPyTorch settings](https://docs.gpytorch.ai/en/latest/settings.html) documents fast_pred_var as the LOVE approximation and max_cholesky_size as the control for Cholesky versus iterative solves. The [official LOVE tutorial](https://docs.gpytorch.ai/en/latest/examples/02_Scalable_Exact_GPs/Simple_GP_Regression_With_LOVE_Fast_Variances_and_Sampling.html) distinguishes the standard predictive covariance calculation from its accelerated approximation.
