# July 2026 optimizer regression fixture

Status: active regression evidence.

This fixture is the optimizer-stage reproducibility oracle for the July 2026
portfolio. It verifies that the recorded posterior scenario matrix still
produces the recorded portfolio weights through the production Riskfolio
adapter. It does not claim to replay or validate the historical Gaussian
process fit: the complete historical source state for that upstream run is not
available as a clean patch, so `scenarios.csv` is the authoritative optimizer
input.

The source artifacts came from
`notebooks/outputs/20260702_july_portfolio_full3`. `scenarios.csv` is a
byte-for-byte copy of `july_gp_posterior_scenarios.csv`. The expected weights
retain the source values and row order; only the unnamed first CSV header was
renamed to `asset` so the regression test can load it explicitly. `config.json`
records both source and fixture hashes, the reference package versions, and the
full optimization request.

The historical run used Riskfolio `alpha=0.5`. Riskfolio's default is `0.05`,
so this value must be explicit to reproduce the recorded CVaR portfolio. The
general production default remains `0.05`.

Run the focused gate with:

```console
uv run --locked pytest -q tests/regression/test_july_2026_optimizer_parity.py
```
