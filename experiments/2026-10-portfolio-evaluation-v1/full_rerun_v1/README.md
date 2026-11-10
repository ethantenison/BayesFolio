# Full ETF configuration rerun

Refit every one of the 499 eligible historical feature configurations and retain
complete predictive covariance and evaluation outputs. No shortlist gate and no
reuse of fits from the prior study. The two joint configurations containing
`hy_spread_chg_1p` remain excluded; historical family masks are unchanged.

## Portfolio constraints

There is **no explicit ETF position cap**. The previous 35% upper bound is not
included in either allocation problem. Long-only weights, full investment,
effective asset count at least three, and a variance budget remain. Consequently,
removing the position cap does not authorize concentrating the entire portfolio
in one ETF. There are no sector/family weight caps introduced here.

The historical-risk allocator uses expanding training-only Ledoit–Wolf covariance.
For each joint GP predicting all 15 ETFs, a second allocator uses that model's
complete observed predictive covariance, including likelihood noise. Both maximize
forecast ranks and share the same **absolute** monthly risk budget: the historical
equal-weight portfolio's variance. The GP budget is not widened to match each
model's declared uncertainty. An infeasible or inaccurate solve is recorded as
failed; no previous weights, clipping, normalization, or relaxation is substituted.

Family candidates replace only their family's forecasts in the same 15-ETF panel,
holding the previously documented other-family anchors fixed. Their common-risk
portfolio scores are conditional on those anchors. Own-family joint NLL and fixed
equal-weight distribution scores use their full GP covariance. A full-portfolio
GP-native allocation is unavailable for a single-family GP; no missing cross-family
dependence is invented. This is not an exhaustive search over four-family
combinations and does not identify a globally optimal assembled portfolio.

## Fixed model and evaluation controls

Rank 2, maxiter 15, seeds 27/127/227, 84 initial training months expanding to
84+k, and the same 24 forecast origins May 2024–April 2026. Keep IWM and
`t_index`; omit IJR and REIT/commodity targets. There are 48 U.S.-equity, 186
international-equity, 105 U.S.-fixed-income, 125 international-fixed-income, and
35 joint configurations: 499 × 3 × 24 = 35,928 fresh monthly fits.

All fits finish before batch portfolio scoring. One bounded process pool uses
at most cores minus two/fourteen, one numerical thread per process, and a 48 GiB
research memory budget with a 25% safety allowance. The previous measured peak
gives an initial 13-worker ceiling; the new late-window smoke can reduce it for
the largest masks. Queued jobs are bounded rather than submitting
all 35,928 futures at once. Checkpoints are partitioned by configuration to avoid
repeated full-directory scans. Original window seeds are retained on resume.

Each fit retains mean, SD, complete raw observed covariance, likelihood-noise
diagonal, scenarios, optimizer diagnostics, and training coverage. Replayed means
and SDs must match the original saved forecasts to 1e-8. The legacy entrypoint is
used only for fitting/export; its former portfolio optimizer is not invoked.

Every configuration receives marginal NLL/CRPS, joint NLL, coverage and widths,
error/volatility calibration, RMSE, expanding-mean R-squared, monthly rank IC,
fixed-reference distribution scores, and common-risk portfolio performance.
Full joint models additionally receive GP-risk allocations and their calibration.
Chronological paths, three separate seeds, and full weights are retained. Sharpe
uses conventional arithmetic excess returns after the explicit discounted-excess
target conversion. Returns are gross unless costs are supplied; turnover is not
an objective. All months remain reused exploratory data, not untouched holdout.

`outputs/all_499_configuration_metrics.csv` will contain every configuration,
including explicit missing performance for failed allocations. Detailed complete
seed records are in `outputs/evaluation/`; reusable exact arrays are in
`outputs/analytic/`. No production model is promoted automatically.

## Run and verify

From the repository root:

```sh
.venv/bin/pytest -q experiments/2026-10-portfolio-evaluation-v1/full_rerun_v1
.venv/bin/ruff check experiments/2026-10-portfolio-evaluation-v1/full_rerun_v1
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/full_rerun_v1/full_rerun.py --smoke-only
/usr/bin/caffeinate -i .venv/bin/python experiments/2026-10-portfolio-evaluation-v1/full_rerun_v1/full_rerun.py --execute
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/full_rerun_v1/full_rerun.py --status
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/full_rerun_v1/full_rerun.py --verify-only
```

`--smoke-only` runs representative smallest/largest feature masks and a largest
joint model's final month, with exact replay checks. These are fresh fits in this
new run and are reused only when resuming the same immutable contract. Execute
continues every remaining configuration without selecting a shortlist.

The run holds a single-writer lock. Source/data/context contracts and per-month
artifact hashes protect resumption. Do not edit frozen executable sources during
the run. Failures stop fitting and preserve valid checkpoints; rerunning the same
command resumes. `status.json` and the captured run log distinguish ongoing work
from completion. Only `COMPLETED.json` plus successful verification confirms the
full study finished. Original studies and production code are read-only. Rollback
is stopping this runner and setting aside this new folder; no old data need repair.

The first preflight found a legacy command-string serialization issue: empty
custom macro lists had lost their empty CLI argument. Its partial fit artifacts
and sources are preserved under `preflight_empty_argument_failure/` and
`preflight_sources/`; no bulk run launched. The new command adapter restores the
explicit empty argument, never defaults to unwanted predictors. Existing v1/v2
data panels have identical date/task axes and identical shared numeric columns;
v2 adds columns without revising the original observations.
