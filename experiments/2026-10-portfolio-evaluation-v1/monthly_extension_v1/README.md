# Four-month append-only comparison

This isolated extension adds exactly 24 fits: broad14 and compact5 joint models,
three original seeds, and four origins (May–August 2026). Their realized return
months are June–September 2026. October is incomplete and excluded.

Rank 2, maxiter 15, the original kernels/transforms/window seed convention,
expanding 84+k-month training windows and dense observed covariance are retained.
The only fit-worker adaptation extends its date-index assertion from 0–23 to
24–27. No prior fit, forecast, weight, return, or NAV observation is overwritten.
The old CLI's allocation flags are inert: the fit-only adapter never invokes that
legacy portfolio optimization. Existing files and all sealed evidence stay read-only.

Prediction compact fractions, in family order, remain 0.5 / 0 / 0.375 / 0.25.
Observed covariance remains the 50/50 broad/compact arithmetic blend, not a
recalibrated or prediction-weight-matched blend.

The four curves retain their economic definitions:

- Hard rank: long-only, fully invested, minimum effective assets 7, no ETF cap,
  unchanged expanding historical equal-weight variance ceiling.
- GP scenarios: 5,000 deterministic joint Gaussian draws per month/seed;
  no-CVaR maximum signed Sharpe, effective assets 10, ETF cap 25%. Reuses the
  sign-safe Riskfolio MV / direct signed-Sharpe solver without changing objectives.
- Schwab aggressive: the frozen documented proxy schedule, with cash earning
  the same RF proxy. No invented annual target changes.
- Simple momentum: adjusted price at t-1 divided by price at t-12 minus one;
  top seven ETFs, equally weighted, alphabetical tie-break.

Prices, labels and RF come from the sealed monthly-data-extension-v1 dataset.
Returns are decimal discounted excess; total return is reconstructed as
`(1 + w @ actual) * (1 + RF) - 1`. Portfolio statistics are calculated after
monthly jobs finish. NAV starts at $10,000, no contributions/withdrawals.
Gross of trading costs/taxes. Three seeds share the same market history; shaded
seed ranges are not confidence intervals. Research selection used earlier results.
The signed negative-mean Sharpe branch is nonconvex; multistart agreement is not
a global-optimality certificate.

## Run and verify

From the repository root, prefix commands with
`UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache PYTHONDONTWRITEBYTECODE=1`.

```sh
uv run --locked --no-sync python experiments/2026-10-portfolio-evaluation-v1/monthly_extension_v1/run_fits.py --execute --workers 14
uv run --locked --no-sync python experiments/2026-10-portfolio-evaluation-v1/monthly_extension_v1/run_extension.py --execute --workers 14
uv run --locked --no-sync python experiments/2026-10-portfolio-evaluation-v1/monthly_extension_v1/run_fits.py --verify
uv run --locked --no-sync python experiments/2026-10-portfolio-evaluation-v1/monthly_extension_v1/run_extension.py --verify
uv run --locked --no-sync pytest -q experiments/2026-10-portfolio-evaluation-v1/monthly_extension_v1/test_extension.py
uv run --locked --no-sync ruff check experiments/2026-10-portfolio-evaluation-v1/monthly_extension_v1
```

Completed fits checkpoint independently. Resume with the same fit command if an
interrupted run lacks FITS_COMPLETED.json; completed checkpoints are reused.
Final verification checks hashes of prior evidence and every original array
prefix, plus complete return-to-wealth reconstruction. Rollback is simply to
continue referencing the unchanged scenario_sharpe_v3 comparison.
