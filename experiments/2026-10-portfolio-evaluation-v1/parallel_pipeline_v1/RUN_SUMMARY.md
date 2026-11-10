# Parallel baseline validation results

The three-stage pipeline completed the accepted 15-ETF joint baseline for seeds 27, 127 and 227 across all 24 monthly forecast origins. It retained IWM, the accepted 14 non-time inputs plus t_index, rank two and maxiter 15. No HY-change input was added. Existing studies and the earlier evaluator outputs remain unchanged.

## Execution and verification

| Check | Result |
| --- | --- |
| Global worker pool | 14 processes, one numerical thread per process |
| Independent monthly fits | 72 completed; the initial smoke fit was reused |
| Allocation stage | 72 monthly jobs, three policies each, 216 optimal solves |
| Fallback weights or relaxed constraints | None |
| Main fit stage | 266.54 seconds, including 71 fresh fits and one cached smoke fit |
| Parallel allocation stage | 0.09 seconds |
| Main execution | 267.66 seconds; the separate initial smoke took 23.72 seconds |
| Largest observed worker peak RSS | 2.73 GiB |
| Legacy forecasts versus saved baseline | Maximum absolute difference zero across all 72 months, for predictions and scenarios |
| Dense versus legacy predictive means | Maximum absolute difference below 1.0e-16 |
| Dense versus legacy predictive SD | Maximum absolute difference below 3.6e-16 |
| Tests and lint | 44 tests passed; Ruff clean |
| Completed rerun | Verification only; no new fitting |

These timings are an integration measurement, not a cold serial-versus-parallel benchmark. The dense readout is separately saved, includes likelihood noise and raw outcome unscaling, and is evaluated as analytic Gaussian covariance rather than covariance estimated from scenarios.

All 72 optimization logs ended at the registered 15-iteration cap. Successful pipeline verification is not a claim that these GP fits converged or that the model configuration is optimal. Pylance tooling was unavailable; numerical tests, lint and source/artifact checks were completed.

## Portfolio results

Results are means across three seeds. Returns and drawdowns use reconstructed total-return NAV; Sharpe uses arithmetic excess returns. These are gross results without an explicit trading-cost series. All policies use the same 15-ETF investable universe, although fixed reference weights can assign zero to some ETFs.

| Allocation policy | Conventional Sharpe | Annualized excess volatility | Compounded total return | Maximum total NAV drawdown |
| --- | --- | --- | --- | --- |
| Equal weight | 1.258 | 7.47% | 30.64% | -4.30% |
| Fixed 60/40 | 1.284 | 8.01% | 32.84% | -4.58% |
| Legacy scenario MaxRet | 1.772 | 13.22% | 70.32% | -7.82% |
| Linear rank weights | 1.555 | 9.04% | 42.88% | -5.19% |
| Risk rank with 35% cap | 1.616 | 7.62% | 38.30% | -5.62% |
| Risk rank without per-ETF cap | 1.539 | 7.81% | 37.47% | -5.87% |

The risk-aware policies use the same training-only common covariance and equal-weight reference variance budget. Removing the 35% cap retains at least three effective assets; it does not remove diversification or risk constraints. Their lower volatility and lower absolute return cannot be interpreted as a pure improvement over the higher-risk legacy policy. No allocation policy was automatically promoted.

## Exact predictive risk calibration

For the fixed equal-weight reference, the full analytic GP covariance gives:

| Seed | Error RMS divided by predicted SD RMS | Mean squared standardized error | Coverage of nominal 95% interval |
| --- | --- | --- | --- |
| 27 | 1.521 | 2.691 | 83.3% |
| 127 | 1.514 | 2.479 | 83.3% |
| 227 | 1.538 | 2.720 | 83.3% |

The calibration targets are one for the two error-scale ratios and 95% for interval coverage. The portfolio uncertainty appears understated on these evaluation months, consistent with the earlier scenario-covariance approximation. This is a diagnostic based on only 24 reused market months, not a precise population-calibration estimate or untouched-holdout confirmation.

Joint NLL, marginal diagnostics, fixed-portfolio NLL/CRPS, within-family ranking, monthly return paths and paired benchmark comparisons are in `outputs/baseline_parallel_v1/evaluation`. Joint Gaussian NLL per complete 15-ETF monthly vector averages -38.847, -38.713 and -39.058 for the three seeds respectively; lower is better on the same task universe and units.

## Scope and next comparison

This run validates the parallel producer and exact-covariance evaluator on one accepted joint baseline. It does not yet compare the full joint model against regional or four-family architectures. Those can now use the same staged workflow and metric definitions. The current calibration warning should remain visible during that comparison rather than being hidden by aggregate IC or marginal NLL.

Runnable commands and covariance/allocation assumptions are in README.md. `outputs/baseline_parallel_v1/COMPLETED.json` seals the completed artifacts, and `--verify-only` checks their recorded source and output hashes without starting fits.
