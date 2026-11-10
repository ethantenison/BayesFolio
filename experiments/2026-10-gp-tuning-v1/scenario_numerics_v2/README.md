# Scenario allocator numerical repair v2

This is a versioned root-cause numerical fix, not a new investment strategy.
The original campaign is sealed and is never edited. New evidence is written
to `outputs/` here. No GP refits are needed.

Negative scenario returns, negative asset means and negative feasible portfolio
means are valid. No return is filtered, clipped, shifted or required to exceed
zero. The objective stays **signed mean / sample standard deviation**, rf=0,
under long-only/full investment, effective assets >=10, and a 25% position cap.

The positive-Sharpe branch previously normalized `mu @ x = 1` and minimized
variance. Tiny feasible means forced enormous auxiliary variables, causing
`optimal_inaccurate`. `positive_sharpe.py` uses the equivalent unit-risk cone
normalization with dimensionless inputs, not a mean-normalization denominator.
The auxiliary unit-risk cone imposes no variance budget on final weights.
Strict optimal status, duality gap, primal/dual residuals and raw economic
constraints must all pass. Inaccurate status is still rejected.

The nonpositive-Sharpe branch retains direct signed-ratio multistart. It is
explicitly not claimed to prove a nonconvex global optimum. There is no previous
portfolio, minimum-variance or constraint-relaxation fallback.

## Reuse: do not call the old sealed allocator

For new experiments, import `allocate_scenarios` from **this** `allocator.py`.
It dispatches both signs correctly and records negative-observation counts.
For the campaign's three-seed/28-origin case schema, use **this**
`allocator.evaluate_case(case, NEW_output_root)` entry point. It wires the
corrected allocator into deterministic scoring and fingerprints its code.
Use process-level parallelism, one case per process. The sealed old runner
remains intentionally unchanged; rerunning it only verifies historical evidence.

## Execute and verify this repair

From `/Users/et/Documents/BayesFolio`, with `PYTHONDONTWRITEBYTECODE=1`,
`MPLCONFIGDIR=/private/tmp/bayesfolio-gp-tuning-mpl`,
`XDG_CACHE_HOME=/private/tmp/bayesfolio-gp-tuning-cache`, and
`UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache`:

```sh
uv run --locked --no-sync pytest -q experiments/2026-10-gp-tuning-v1/scenario_numerics_v2
uv run --locked --no-sync ruff check experiments/2026-10-gp-tuning-v1/scenario_numerics_v2
uv run --locked --no-sync python experiments/2026-10-gp-tuning-v1/scenario_numerics_v2/repair.py --execute
uv run --locked --no-sync python experiments/2026-10-gp-tuning-v1/scenario_numerics_v2/repair.py --verify
```

Every previously successful allocation remains bitwise unchanged. Only the 12
missing seed-month allocations are filled. All ten incomplete cases are then
rescored over their full paths. Tracks 2/3 and baseline weights remain unchanged;
scenario selection and integrated reports are recalculated with all 36 cases.
The original failure evidence and all new certificates are preserved.

Rollback is simply to use the original report directory; no original files or
production code are modified. Neither version auto-promotes a production model.
