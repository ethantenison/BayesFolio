# Paired portfolio policies for all ETF configurations

Record both approved allocation variants for every one of the 499 configurations
in the ongoing full covariance rerun, without restarting or modifying its GP fits.

| Variant | Joint GP configurations | Family configurations in the common 15 ETF portfolio |
| --- | --- | --- |
| Risk budget only | No effective-count minimum | No effective-count minimum |
| Risk budget plus effective count | Effective count at least 10 | Effective count at least 3 |

Both variants are long-only and fully invested, with no explicit ETF position
cap, no CVaR, no turnover objective, no constraint relaxation and no failed-weight
fallback. They use identical forecasts, rank objective, covariance source and
absolute historical equal-weight variance budget. The same tiny deterministic
tie-break term remains in both objectives; it is not a hard diversification floor.
Risk-budget-only can allocate 100% to one ETF when the risk constraint permits it.

The diversified variant intentionally has different thresholds for joint and
family candidates. Do not attribute differences between those rows solely to
forecast quality. Risk-budget-only uses identical constraints across all candidates.

## Complete recordings

Every configuration receives both variants using the common historical covariance.
The 35 full joint GPs also receive both variants using their complete observed GP
predictive covariance, including likelihood noise. A family configuration does
not supply missing cross-family GP covariance; its forecasts replace only its
family in the same 15-ETF panel, with the other-family anchors held fixed. The
minimum of three applies to that whole portfolio, not inside a three-ETF family.

Each variant retains all three seeds, all 24 months, every ETF weight, conventional
Sharpe, gross return, volatility, drawdown, concentration and portfolio-risk
calibration. A failed month has explicit status and null weights; later months
are still evaluated. Aggregate performance is unavailable unless all months
succeed. There is no averaging over only successful months or seeds.

Outputs after completion:

- `outputs/all_499_policy_metrics.csv`: all 499 configurations with forecast
  metrics and both portfolio variants side by side. Superseded original
  joint-minimum-three allocation columns are excluded.
- `outputs/all_portfolio_policies.csv`: 1,068 configuration/covariance/policy rows.
- `outputs/portfolios/<configuration>.json`: complete seed/month weights,
  outcomes, solver failures and risk-calibration diagnostics.
- `outputs/monthly_weights/<configuration>.csv`: weights for every month and
  ETF, with failure status and the exact minimum-effective setting.

Fixed-reference distribution scores and joint predictive NLL remain in the
unchanged upstream evaluation artifacts. Adaptive-weight calibration is labeled
as a decision diagnostic, not a proper same-portfolio distribution comparison.
Return conversion preserves the discounted-excess target convention; Sharpe uses
arithmetic excess returns. Reused months are exploratory, not untouched holdout.

## Automatic execution without interrupting fits

The existing fitting runner and its frozen source hashes are left untouched.
This separate one-shot follower waits for that entire 499-study to finish,
release its writer lock, and pass its own final verification. Only then does
this scorer launch its worker pool, avoiding competition with the GP fits.
The former minimum-three results in the upstream study remain legacy controls;
this directory is authoritative for the newly approved joint-minimum-ten pairing.

From the repository root:

```sh
.venv/bin/pytest -q experiments/2026-10-portfolio-evaluation-v1/portfolio_policy_comparison_v1
.venv/bin/ruff check experiments/2026-10-portfolio-evaluation-v1/portfolio_policy_comparison_v1
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/portfolio_policy_comparison_v1/run_comparison.py --pilot-only
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/portfolio_policy_comparison_v1/launch_background.py
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/portfolio_policy_comparison_v1/run_comparison.py --status
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/portfolio_policy_comparison_v1/run_comparison.py --verify-only
```

The detached launcher resumes checkpointed fits if needed and starts the
dependent policy follower without tying either job to a chat tool session.
It records PIDs and log paths in `background_jobs.json`, uses single-writer
locks, and installs no cron or recurring jobs. Progress is in each job's
`outputs/status.json`; detached logs are in each directory's `background.log`.

`run_comparison.py --execute` requires the upstream study to be complete;
`--follow` queues the same scoring now. `--pilot-only` checks the four joint-policy cases
on a previously completed baseline; it is not a substitute for all 499 results.
Its outputs are kept separately as `pilot.json`.

Policy source/context contracts, upstream completion seals and predictive-array
hashes protect the recordings. Per-configuration checkpoints support resumption.
No production files, original fit artifacts, or Git state are changed. Rollback
is stopping this follower and setting aside this directory; the massive fit run
continues independently and no completed fits are discarded. A complete policy
result requires this directory's `COMPLETED.json` and successful verification.
