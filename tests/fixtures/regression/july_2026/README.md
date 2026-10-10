# July 2026 pipeline regression fixture

Status: active regression evidence.

This fixture is the reproducibility oracle for the accepted July 2026
portfolio. It supports two independent gates: the optimizer-only test starts
from recorded posterior scenarios, and the full-pipeline test starts from the
frozen feature artifact and executes maintained as-of alignment, multitask GP
fitting, posterior sampling, and Riskfolio optimization.

The historical run began from commit `469413e` with local changes. Those changes
were recovered in commit `948ba4e`, whose numerical environment and source
reproduce the recorded predictions and scenarios to CSV round-trip precision.
The production monthly configuration now expresses the recovered behavior
without importing the experiment runner.

The source artifacts came from
`notebooks/outputs/20260702_july_portfolio_full3`. `features.parquet` preserves
the exact upstream feature panel, `expected_predictions.csv` preserves the GP
forecast output, and `scenarios.csv` is a byte-for-byte copy of the posterior
scenario matrix. Expected weights retain the source values and row order; only
the unnamed first CSV header was renamed to `asset`. `config.json` records
hashes, model controls, tolerances, and reference package versions.

The historical run used Riskfolio `alpha=0.5`. Riskfolio's default is `0.05`,
so this value must be explicit to reproduce the recorded CVaR portfolio. The
general production default remains `0.05`.

The optimizer-only gate uses the recorded scenario matrix and therefore keeps
near-exact weight tolerances. The full GP pipeline traverses platform-specific
linear algebra and posterior sampling, so its cross-platform gate allows
`0.005` absolute forecast/scenario error, `0.01` dimensionless standardized
score error, `0.005` per-asset weight error, and `0.03` total weight L1 error.
These correspond to 50 basis points of monthly return, 50 basis points per
allocation, and 3 percentage points of aggregate allocation drift.

Run the focused gate with:

```console
uv run --locked pytest -q tests/regression/test_july_2026_optimizer_parity.py
uv run --locked pytest -q tests/regression/test_july_2026_full_pipeline_parity.py
```
