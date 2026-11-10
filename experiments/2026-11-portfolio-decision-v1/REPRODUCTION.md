# Reproduce the November decision

From `/Users/et/Documents/BayesFolio`:

```sh
UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --locked --no-sync python experiments/2026-11-portfolio-decision-v1/run_decision.py --verify
```

Verification uses saved per-expert, per-seed moments. It recalculates the causal
variance ceiling from the versioned dataset, reapplies fixed family blends,
pools seeds before allocation, solves once, and independently checks the raw
CSV weights against the original constrained conic optimizer. It does not fetch
data, fit GPs, submit orders, or change production defaults.

The matching `--execute --workers 14` command assembles the workflow. Passed
checkpoints are reused; a completed decision is verified rather than rerun.
The locked project environment and its `uv.lock` fingerprint are protected.

## Evidence map

- `PREREGISTRATION.json`, `PREREGISTRATION.sha256`, `PLAN.md`: frozen menu,
  blocks, limits, policy and conservative promotion rule, before new scoring.
- `HISTORICAL_DECISION.json`, `outputs/historical/ALL_RESULTS.json`: all 30
  historical cases (five controls and 25 new stable blends), eligibility,
  Pareto frontier, deterministic selection and 15-case shortlist.
- `outputs/reversal_analysis.md` and `.json`: retrospective explanation of the
  earlier/later reversal; these are diagnostics, not untouched holdout evidence.
- `data/COMPLETED.json`, `data/coverage_audit.json`, `data/dataset/global.parquet`:
  source timestamps, source hashes, cutoff/coverage checks and sealed extension.
  `data/failures/` preserves resolved download/vintage failures.
- `FIT_CONTRACT.json`, `FITS_COMPLETED.json`: exact input/model contract and
  18 fit attempts. `outputs/fits/*/attempt_*/` contains attempt metadata,
  fitted parameters and analytical forecasts. `outputs/fit_checkpoints/`
  records job identities and raw artifact fingerprints.
- `outputs/recent_components/`: three model panels, three seeds, two origins.
  November targets are unknown and preserved as such.
- `outputs/finalists/`: matched 53-month paths, predictions, covariance,
  portfolios and raw optimizer evidence for every finalist; November live
  moments and weights are saved separately from historical outcomes.
- `outputs/benchmarks/`: the two standing benchmark paths and overlap audit.
- `outputs/NOVEMBER_MOMENTS.npz`, `NOVEMBER_WEIGHTS.csv`, `CONFIG.json`:
  exact selected live inputs, raw target weights and reproducible policy.
- `VALIDATION.json` and `outputs/INDEPENDENT_*AUDIT.json`: availability,
  original-conic parity, raw constraints and independent selection checks.
- `EXPERIMENT_LEDGER.json`: finite campaign counts and preserved integration
  failures; `STATUS.json` records the current state and exact next action.
- `COMPLETED.json`: SHA256 fingerprints of the complete delivery, source code,
  data and numerical artifacts. Existing evidence is separately hash-protected
  by the preregistration and fit contract. Log/lock/cache files are not sealed.

The decision was fixed on 52 previously inspected months. October was appended
only after selection; it did not change the recipe. Nothing here constitutes an
untouched prospective validation or assumes an October 30 closing fill was
available after observing October 30 closing data.
