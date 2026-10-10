# BayesFolio

BayesFolio is a monthly, uncertainty-aware ETF portfolio research workflow for a retirement IRA. It combines ETF and macroeconomic data, multitask Gaussian process forecasts, posterior scenarios, portfolio optimization, and out-of-sample evaluation to produce candidate portfolio weights.

BayesFolio is research and decision-support software. It does not place trades, connect to Charles Schwab, or provide financial advice.

## Status

BayesFolio is undergoing an active refactor toward a smaller, schema-first package with explicit boundaries between contracts, business logic, and external I/O.

| Capability | Status |
|---|---|
| ETF and macro feature construction | Implemented |
| Configurable and planner-assisted multitask GP workflows | Implemented |
| Fixed-default monthly forecast and optimization workflow | Implemented |
| Walk-forward backtesting and explicit final portfolio review | Implemented |
| Reproduction of the last accepted pre-refactor portfolio weights | Feature-artifact-to-weights parity implemented |
| Direct Schwab execution and holdings synchronization | Planned |
| Realized gain/loss and alpha tracking | Planned |

## What BayesFolio Does

The package supports the main research stages needed for a monthly allocation decision:

- acquire and cache ETF returns, ETF characteristics, and macroeconomic features;
- clean, align, and lag predictors to prevent look-ahead bias;
- fit uncertainty-aware multitask Gaussian process models;
- optionally evaluate alternative GP structures in time-aware experiments;
- sample joint posterior return scenarios;
- construct constrained ETF portfolio weights;
- evaluate configurations with backtests and diagnostics; and
- persist predictions, scenarios, weights, contracts, and reproducibility metadata.

The current `run_monthly_portfolio` workflow automates feature construction, strict as-of alignment, the accepted multitask GP fit, posterior forecasting, optimization, and artifact persistence. Training rows must precede the as-of date, and scoring rows must match it exactly. The application layer assembles concrete providers and runtime I/O while the engine owns computation and sequencing. Its optimizer output is always a candidate. `run_walk_forward_backtest` evaluates dated weights against explicitly paired subsequent returns, and `finalize_monthly_portfolio` releases final weights only after a recorded human approval using evidence available strictly before the candidate's as-of date.

## Why Uncertainty Matters

Monthly ETF returns are noisy, data histories are short relative to the number of possible predictors, and assets share changing market and macroeconomic drivers. A point forecast alone does not describe how uncertain the estimate is or how assets may move together.

BayesFolio uses multitask Gaussian processes to share information across related ETFs and produce a joint predictive distribution. Portfolio construction can then use posterior scenarios rather than treating one return estimate as certain. See [Prediction, Uncertainty, and Multitask Gaussian Processes](docs/first_principles.md) for the canonical rationale and limitations.

## Monthly Decision Workflow

```text
As-of date and ETF universe
    -> ETF and macro data acquisition
    -> cleaning, transformation, and time alignment
    -> multitask GP fit
    -> posterior predictions and joint scenarios
    -> constrained portfolio optimization
    -> walk-forward backtest with explicit realization dates
    -> recorded human approval or rejection
    -> finalized portfolio weights when approved
    -> manual brokerage execution
```

Model selection and tuning are optional research activities. They must use time-aware out-of-sample validation and should not silently change the fixed monthly defaults.

## Installation

BayesFolio requires Python 3.13 and uses [uv](https://docs.astral.sh/uv/) for dependency and environment management.

```bash
uv sync --locked --all-groups
```

Some data providers require network access. MLflow tracking and artifact locations are configured by the command or runtime environment used for a run.

## Running the Monthly Workflow

The current monthly entry point is programmatic:

```python
from bayesfolio.contracts.commands import MonthlyPortfolioCommand
from bayesfolio.application import run_monthly_portfolio

def run_validated_monthly_command(command: MonthlyPortfolioCommand):
    """Run an already validated monthly configuration."""
    return run_monthly_portfolio(command)
```

Inspect the command schema when constructing a configuration:

```bash
uv run --locked python -c "from bayesfolio.contracts.commands import MonthlyPortfolioCommand; print(MonthlyPortfolioCommand.model_json_schema())"
```

There is not yet a stable CLI for the full monthly workflow. Research helpers under `scripts/` are not substitutes for the canonical monthly entry point.

The former `bayesfolio.engine.automation.run_monthly_portfolio` import remains as a deprecated compatibility shim. New code should use the application entry point so concrete provider, storage, and MLflow assembly stays outside the engine.

Walk-forward evaluation and final review are separate, explicit steps:

```python
from datetime import UTC, datetime

from bayesfolio.application import finalize_monthly_portfolio
from bayesfolio.contracts.commands import (
    FinalizeMonthlyPortfolioCommand,
    WalkForwardBacktestCommand,
)
from bayesfolio.engine.backtest import build_walk_forward_period, run_walk_forward_backtest

period = build_walk_forward_period(
    historical_candidate,
    realization_date=next_month_end,
    realized_returns=next_period_returns,
)
backtest_command = WalkForwardBacktestCommand(
    strategy_id=historical_candidate.strategy_id,
    periods=[period],
)
backtest_result = run_walk_forward_backtest(backtest_command)

finalize_command = FinalizeMonthlyPortfolioCommand(
    candidate=current_candidate,
    backtest=backtest_result,
    benchmarks=benchmark_results,
    decision="approved",
    reviewer="portfolio-owner",
    reviewed_at=datetime.now(UTC),
    rationale="Reviewed the recorded strategy and benchmark evidence.",
)
decision_result = finalize_monthly_portfolio(finalize_command)
```

`backtest_command` must pair each decision date with a later realization date,
the weights fixed on the decision date, and subsequent realized returns.
`finalize_command` must contain matching strategy evidence, at least one
benchmark evaluated on the same dates, and an explicit timestamped human
decision. Rejected candidates never expose finalized weights.

## Validation

Run targeted checks first, followed by the broader suite when the change warrants it:

```bash
uv run --locked pytest -q tests/contracts
uv run --locked pytest -q tests/engine/automation tests/application tests/io/test_monthly_portfolio_runtime.py
uv run --locked ruff check bayesfolio scripts tests
uv run --locked pytest -q
```

## Reproducibility and Refactor Parity

Accepted monthly runs should record the as-of date, dataset cutoff, contract configuration, random seed, artifact fingerprints, and relevant model assumptions. Features used to predict month `t` must contain only information available before `t`, and returns use decimal units (`0.02 = 2%`).

The refactor is not complete until the refactored workflow reproduces the last accepted pre-refactor portfolio weights within an explicitly documented tolerance. Any difference must be traced to data, model, optimization, or dependency changes rather than assumed to be harmless.

The July 2026 regression evidence is versioned under
`tests/fixtures/regression/july_2026/`. The focused optimizer gate verifies the
recorded scenarios independently, while
`tests/regression/test_july_2026_full_pipeline_parity.py` runs the maintained
application workflow from the frozen feature artifact through as-of alignment,
GP fitting, posterior sampling, and Riskfolio optimization. The historical
dirty source state was recovered in commit `948ba4e`; the maintained workflow
reproduces its scenarios and predictions within `1e-12` absolute tolerance and
its accepted weights within `1e-10` per asset and `1e-9` total L1 error.

## Documentation Map

- [Documentation authority and navigation](docs/README.md)
- [First-principles modeling rationale](docs/first_principles.md)
- [Architecture guardrails](docs/copilot_architecture.md)
- [Package map and public entry points](docs/package_map.md)
- [Agent roles and repository workflow](AGENTS.md)

## Experiments

Experiments are versioned evidence, not normative package documentation. They are excluded from routine package checks unless explicitly requested. A result becomes part of current package behavior only when it is implemented, tested, and promoted into the appropriate canonical document.

## Roadmap

Planned extensions include a Charles Schwab adapter, holdings synchronization, transaction tracking, realized gain/loss reporting, and attribution of portfolio performance and alpha. These capabilities need explicit contracts and I/O boundaries before implementation; none should be inferred from the current optimization output.

## Contributor Workflow

Repository agent roles are defined in [AGENTS.md](AGENTS.md). The recommended sequence is:

1. Run the Boundary Inspector.
2. Run the Documentation Drift Inspector.
3. Run the Simplicity and Redundancy Inspector when the scope warrants it.
4. Apply the smallest in-scope patch for implementation and active-goal work.
5. Pause only for destructive work, unauthorized breaking APIs, or unresolved
   product-impact choices.
6. Re-run targeted validation and report the exact commands and results.

## Disclaimer

BayesFolio is for research and educational purposes only. It does not constitute financial, investment, tax, or legal advice. Forecasts and optimized weights can be wrong, and past or backtested performance does not guarantee future results.
