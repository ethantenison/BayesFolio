# Structured LLM Portfolio Adjustment

Status: Provisional

Audience: Users, modelers, contributors, and coding agents

Canonical topic: Point-in-time LLM review of monthly optimizer weights

Last verified: 2026-10-12

## Summary

The LLM is a constrained reviewer, not a second portfolio optimizer. It receives
the current portfolio, raw optimizer target, asset forecasts, posterior risk for
a fixed blend grid, and a dated evidence packet. It may select one blend fraction
and one timing action. Python then calculates the final weights deterministically.

This keeps the adjustment explainable and backtestable:

- research is gathered by the same ordered semantic tool calls;
- every source must have been available by the decision cutoff;
- the prompt, input packet, raw response, and model settings are fingerprinted;
- the LLM cannot introduce assets or emit arbitrary weights;
- invalid output is repaired once and then fails closed.

Pydantic contracts are the maintained boundary. PydanticAI is not required for
version 1 because provider orchestration is small and a direct model port avoids
adding framework payloads to BayesFolio contracts. A PydanticAI adapter can be
added later without changing the command or result schemas.

## Exact October Adjustment Procedure

The October 2026 recommendation followed these steps. Version 1 formalizes the
same procedure rather than reconstructing a more elaborate strategy after the
fact.

1. **Freeze the information cutoffs.** The multitask GP and optimizer used data
   through September 30, 2026. Account and research evidence was limited to
   information available by the October decision timestamp.
2. **Record the starting portfolios.** Supply current account weights and the
   raw scenario-optimized target. Assets missing from either side receive zero
   on that side of the interpolation.
3. **Measure forecast signal.** Compare each expected monthly excess return with
   its predictive standard deviation. The October means were small relative to
   uncertainty, so the forecasts were not treated as high-conviction rankings.
4. **Compare posterior portfolio outcomes.** Evaluate current, 25%, 50%, 75%,
   and 100% optimizer blends using expected return, volatility, probability of a
   negative return, and 5% CVaR from the same scenario set.
5. **Run the same ordered evidence retrievals.** Record a Schwab account snapshot,
   Federal Reserve releases, BLS releases, Treasury yields, Schwab research, and
   financial-news search. Preserve arguments, timestamps, evidence IDs, and raw
   content hashes. The LLM receives summaries but does not control retrieval.
6. **Compare evidence with the optimizer.** Assess each cited item separately as
   supporting, contradicting, mixed, or neutral. In October, sticky inflation,
   high long yields, narrow equity breadth, and emerging-market pressure reduced
   confidence in the optimizer's largest rotations.
7. **Choose from a restricted action grid.** Select one of
   ``[0.0, 0.25, 0.5, 0.75, 1.0]``, where zero retains the current portfolio and
   one adopts the raw optimizer. October selected ``0.5``.
8. **Calculate weights outside the LLM.** For each asset, Python applies
   ``adjusted = (1 - blend) * current + blend * optimizer`` and reports one-way
   turnover. The LLM never calculates or edits individual weights.
9. **Record timing separately.** The model chooses ``rebalance_now``, ``stage``,
   or ``defer``. Deferral must name an observable event. Timing is not allowed to
   alter the weight arithmetic.
10. **Persist an audit record.** Save the command, result, prompt version, model
    identifier, generation settings, validation attempts, and SHA-256 hashes.

## Maintained Entry Points

- Command: `bayesfolio.contracts.commands.GeneratePortfolioAdjustmentCommand`
- Result: `bayesfolio.contracts.results.PortfolioAdjustmentResult`
- Pure engine: `bayesfolio.engine.agent.generate_portfolio_adjustment`
- Composition root: `bayesfolio.application.run_portfolio_adjustment`
- Compatible transport: `bayesfolio.io.llm.OpenAICompatibleChatModel`
- CLI: `scripts/run_portfolio_adjustment.py`

The CLI expects a complete command JSON. It intentionally does not perform live
research: point-in-time evidence collection needs its own archive-backed backtest
adapter so current web pages cannot leak future information.

## Local Qwen Configuration

The default local adapter targets:

```text
base_url: http://127.0.0.1:8080/v1
model: /Users/et/local-llm/models/Qwen3.8-27B-4bit
endpoint: /chat/completions
temperature: 0.0
seed: 16638
max_output_tokens: 2000
enable_thinking: false
validation_attempts: 2
```

Start the existing local server before running the CLI:

```bash
cd /Users/et/local-llm
./start-server.sh
```

Then run a prepared point-in-time packet:

```bash
cd /Users/et/Documents/BayesFolio
uv run --locked python scripts/run_portfolio_adjustment.py \
  --input /path/to/adjustment-command.json \
  --output /path/to/adjustment-result.json
```

Model seeds and zero temperature improve repeatability but do not prove bitwise
determinism. The proposed backtest therefore measures repeat-run agreement.

## Proposed Backtest Configuration — Awaiting Approval

No historical LLM-adjustment backtest should begin until this configuration is
approved or revised.

### Point-in-time geometry

| Setting | Proposal |
|---|---|
| Evaluation decisions | 2024-03-31 through 2026-08-31; 30 monthly decisions |
| GP training window | Rolling trailing 36 calendar months, strictly before each model as-of date |
| Model data cutoff | Final trading-day close of month `t` |
| Research cutoff | 08:00 America/New_York on the second trading day of month `t+1` |
| Simulated execution | Closing prices on that same second trading day |
| Holding period | Execution close through the next decision's execution close |
| Research lookback | Trailing 90 calendar days |
| Account state | Each strategy's prior holdings drifted by realized asset returns |

The second-trading-day rule gives releases and evidence ingestion a realistic,
repeatable cutoff. Daily execution prices are required so the backtest does not
credit returns earned before the recommendation existed.

### GP and optimizer

| Setting | Proposal |
|---|---|
| Training universe | 18 accepted tasks; `MGK` and `BND` remain helper tasks |
| Investable universe | Existing 16-ETF accepted portfolio universe |
| GP rank | 5 |
| Covariance | `E + M + T + ET + MT + EMT` |
| Mean | Multitask constant |
| Input/outcome transforms | BoTorch normalize; stratified standardize by task |
| Minimum inferred noise | 0.0025 in transformed target variance units |
| Fit iterations | 75 |
| Posterior scenarios | 5,000 per decision |
| Base seed | 16,638, with a deterministic date-derived per-window seed |
| Optimizer | Riskfolio Classic, Sharpe objective, CVaR risk, alpha 0.50 |
| Constraints | Long only, fully invested, 25% maximum weight, NEA 10 |

### Strategies

1. **Primary baseline:** raw GP scenario optimizer, blend fraction 1.0.
2. **Deterministic shrinkage control:** fixed 0.5 blend without an LLM.
3. **LLM adjustment:** local Qwen chooses from the five-value blend grid.
4. **Context baselines:** equal weight and the existing static Schwab-style
   moderate-aggressive allocation.

The fixed 0.5 control is essential. It tests whether the LLM adds value beyond
the simple shrinkage rule used in October.

### Point-in-time evidence packet

Every month uses the exact semantic retrieval sequence enforced by the command:

1. `schwab_account_snapshot`
2. `federal_reserve_releases`
3. `bls_releases`
4. `treasury_yield_curve`
5. `schwab_market_commentary`
6. `financial_news_search`

For backtesting, these must resolve to archived or publication-dated material,
never the current version of a mutable web page. Raw responses and normalized
evidence records receive content hashes. Search queries, result limits, source
allowlists, and summarization rules must be frozen in the experiment manifest.

### Evaluation metrics

Prediction and uncertainty quality belong to the shared GP and are therefore
reported once, not attributed to different blend strategies:

- RMSE and mean absolute error of monthly excess-return means;
- Gaussian negative log predictive density and CRPS;
- central 50%, 80%, 90%, and 95% interval coverage;
- mean absolute coverage error, interval score, and interval width/sharpness;
- aggregate, by-asset, and by-decision-date slices.

Portfolio outcomes are compared separately:

- cumulative and annualized return;
- annualized volatility, Sharpe, Sortino, and Calmar ratios;
- maximum drawdown, monthly 5% CVaR, and negative-month frequency;
- average one-way turnover and number of changed positions;
- paired monthly return differences with a three-month moving-block bootstrap
  95% confidence interval.

LLM-process quality is also measured:

- blend and timing distributions;
- structured-output failure and repair rates;
- citation validity and evidence coverage;
- exact agreement across three repeat runs of the same packet;
- realized counterfactual regret versus the best blend on the frozen grid.

### Costs and decision rule

Because the account is an IRA, the primary comparison uses zero capital-gains
tax and zero commission. One- and five-basis-point turnover sensitivities will
show whether bid-ask and execution frictions could change the conclusion.

The LLM approach should not be considered better from a higher point estimate
alone. The proposed acceptance guardrails are:

- at least 25 valid out-of-sample decisions;
- no look-ahead or missing-source fallback in accepted periods;
- Sharpe and cumulative return competitive with the raw GP baseline;
- maximum drawdown no more than two percentage points worse than raw GP;
- average turnover no greater than raw GP;
- repeat-run blend agreement of at least 90%;
- evidence that Qwen beats or materially differs from fixed 0.5 shrinkage.

All manifests, input packets, structured outputs, metrics, and plots should live
under `experiments/2026-10-llm-portfolio-adjustment/`, outside the installed
`bayesfolio` package.
