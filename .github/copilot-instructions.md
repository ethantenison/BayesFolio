# GitHub Copilot Instructions for BayesFolio

BayesFolio is a research-grade Bayesian portfolio optimization framework built around:
- Multitask (Hadamard) Gaussian Processes with uncertainty quantification
- Scenario-based portfolio optimization (Riskfolio-Lib integration)
- Rolling time-series CV, backtesting, and diagnostics
- Transparent, reproducible, and responsible AI practices

These instructions are **STRICT**. Generated code must follow them.

---

## Language & Style

- Python **3.13+**
- Use **native typing**: `list[str]`, `dict[str, float]`, `X | None`
- Do NOT use `typing.List`, `Optional`, `Union`
- Use `from __future__ import annotations` in new modules
- Prefer clarity over cleverness
- Avoid hidden state; prefer explicit inputs/outputs

### Docstrings
- All **public** functions/classes require **Google-style docstrings**
- Include shape/units expectations for arrays/DataFrames (especially returns)

### Numeric / Finance Conventions (IMPORTANT)
- Returns must be clearly documented as either:
  - **decimal** (0.02 = 2%), or
  - **percent points** (2.0 = 2%)
- Never silently mix units. If converting, do it explicitly and document it.
- Time indexing must avoid look-ahead bias. Every feature/label must be aligned.

---

## Code Quality & Linting (STRICT)

This repository uses **ruff** with strict pre-commit hooks.
All generated code MUST pass linting.

### Must Avoid
- Unused variables (F841)
- Unused imports (F401)
- Line length > 120 (E501)
- Imports not at top of file (E402)
- Bare `except:` (E722)
- Shadowing builtins (A001/A002/A003)

### Formatting Rules
- Max line length: **120**
- Organize imports: stdlib → third-party → local
- No commented-out dead code
- No debug prints in library code; use `logging` when needed

### Determinism
- Fix RNG seeds when randomness affects results (torch / numpy / random)
- Provide `seed` in configs where relevant
- Avoid nondeterministic ops unless explicitly justified

---

## Architecture Rules (STRICT)

### Separation of Concerns
- Workflows/orchestrators coordinate only — **no heavy computation**
- Heavy logic lives in dedicated helpers/modules

### No Cross-Imports Between Domains
Avoid circular coupling. Do NOT introduce cross-imports between:
- `data/` (ingestion, cleaning, panels)
- `features/` (feature engineering)
- `models/` (GPs, baselines)
- `portfolio/` (optimization, constraints, scenario engines)
- `backtesting/` (walk-forward, CV, performance attribution)
- `reporting/` (plots, artifacts, report generation)
- `apps/` (future agent/RAG UI layer)

Use interfaces/adapters if two domains must interact.

### Where Code Goes
- Data ingestion & panels → `bayesfolio/features/`
- Feature engineering → `bayesfolio/features/`
- GP models, kernels, likelihoods, means → `bayesfolio/models/`
- Scaling/normalization utilities → `bayesfolio/models/scaling.py` or `bayesfolio/preprocessing/`
- CV and walk-forward splitting → `bayesfolio/backtesting/cv.py`
- Backtesting engine → `bayesfolio/backtesting/`
- Portfolio optimization + Riskfolio adapters → `bayesfolio/portfolio/`
- Evaluation metrics → `bayesfolio/evaluation/`
- Visualization → `bayesfolio/visualization/`
- Config schemas → `bayesfolio/configs/`
- Experiment tracking → `bayesfolio/mlflow/` (optional)
- Tests → `tests/unit/`, `tests/integration/`, `tests/regression/`

---

## Config & Schemas

- All configs and “results that cross boundaries” must be **Pydantic** models
- `*Config` defines behavior; include defaults that are safe and reproducible
- Any transport/result schema must be **JSON-serializable**
- NEVER use `pickle`/`dill` for saved artifacts that need interoperability
- Add `schema_version: str` to transport schemas if intended to persist across releases

---

## Gaussian Process + Uncertainty Rules (CRITICAL)

### Scaling / Transforms
- Scaling must be explicit and invertible:
  - `transform_x`, `fit_y`, `inverse_y`, `inverse_std`
- If scaling is **per_task**, inverse transforms must respect task alignment.
- When using predictive covariance:
  - If `y_orig = D * y_scaled + mu`, then `Cov_orig = D Cov_scaled D`
- Never “use uncertainty” as a scalar per asset when joint covariance is available.
  Prefer the **full posterior covariance** for scenario generation.

### Posterior Scenario Generation
- If you use GP posterior scenarios for portfolio risk:
  - sample from the **joint predictive distribution**
  - preserve cross-asset covariance
- Make scenario sampling functions take:
  - `mean: torch.Tensor (n_assets,)`
  - `cov: torch.Tensor (n_assets, n_assets)`
  - `asset_order: list[str]`
  - and output `pd.DataFrame` scenarios with stable column order

### Look-Ahead Safety
- Features must not leak future info.
- Labels such as `y_excess_lead` must be aligned so that:
  - training uses only information available at time t
  - predicting t+1 uses only features known at t

---

## Riskfolio Integration Rules (CRITICAL)

BayesFolio uses Riskfolio-Lib as an optimizer backend.

### Historical Optimization
- `port = rp.Portfolio(returns=historical_returns)`
- `assets_stats(method_mu=..., method_cov=...)`
- Use `hist=True` for scenario-based risk measures (CVaR/EVaR/etc.) when appropriate.

### GP Posterior Scenario Optimization
- GP scenario matrix is treated as a **scenario return panel**
- Use:
  - `port = rp.Portfolio(returns=scenario_returns)`
  - `assets_stats(method_mu="hist", method_cov="hist")` (or set `port.mu` explicitly)
  - `optimization(hist=True, rm="CVaR", ...)`

### Unit Consistency
- Ensure GP scenarios are in the same return units (decimal vs percent) as historical.
- Provide helper checks:
  - compare scenario mean/std vs historical mean/std for sanity.

### Constraints & Universe
- If an asset is modeled in the GP but excluded from the portfolio:
  - exclude **after** posterior extraction (subselect mean/cov / scenario columns)
  - do NOT break asset alignment

---

## Testing Expectations

- Helpers require unit tests
- CV/backtesting requires integration tests
- Any schema changes require regression tests
- Tests must be deterministic (fixed seeds)
- Use small synthetic data for fast tests

Suggested test focus areas:
- Scaling round-trip correctness (`fit_y` + `inverse_y`, `inverse_std`)
- Posterior covariance unscaling correctness
- Scenario sampling PSD robustness (jitter / Cholesky fallback)
- Asset ordering stability (task_idx sort)
- Riskfolio adapter correctness (hist portfolio vs GP scenario portfolio)

---

## Performance & Research Integrity

- Avoid accidental quadratic operations over long panels
- Avoid copying large DataFrames unless necessary
- Cache reusable computations explicitly (no hidden global caches)
- Make expensive operations obvious and documented

---

## Responsible AI & Transparency

BayesFolio outputs must be explainable and auditable:
- record model assumptions (kernel, rank, likelihood)
- record scaling choice (global vs per_task)
- record dataset cutoffs and dates used
- include uncertainty diagnostics (posterior std, correlations, stress scenarios)
- do not claim guaranteed performance

---

## Defaults (When Unsure)

- Keep APIs small and composable
- Add type hints + docstrings by default

---
