# Follow-Up Experiment Analysis

## Portfolio Path Readout

The GP scenario optimizer finished ahead of both baselines, but the edge is
incremental rather than decisive:

| Strategy | Terminal Value | CAGR | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|
| GP scenarios + Riskfolio | `$11,897.87` | `18.98%` | `1.96` | `-5.56%` |
| Historical y + Riskfolio EWMA2 | `$11,809.35` | `18.09%` | `1.82` | `-6.11%` |
| Equal weight | `$11,493.12` | `14.93%` | `1.69` | `-5.33%` |

The new CAGR-to-date plot is `cagr_to_date.png`. The GP method maintained the
highest cumulative CAGR for most of the path. Its advantage came from small
monthly gains versus historical Riskfolio in May, June, July, October, December,
February, and March. It lagged in August, September, November, January, and
April.

The worst shared month was February 2026:

- GP scenarios + Riskfolio: `-5.56%`
- Historical y + Riskfolio EWMA2: `-6.11%`
- Equal weight: `-5.33%`
- GP IC: `-0.091`

That month looks like a forecast/ranking failure and a market-beta drawdown, not
only an optimizer problem. The strongest GP months by IC were March 2026
(`0.815`), December 2025 (`0.659`), April 2026 (`0.532`), June 2025 (`0.524`),
and July 2025 (`0.500`). April still lagged historical Riskfolio despite good IC,
which suggests the optimizer/scenario distribution can lose value even when the
ranking signal is present.

The GP portfolio was mostly a diversified risk-on allocation. Top weights were
usually SPY, VTV, VEA, IJR/IWM, EWX, HYG, and VWO. The final universe excluded
MGK/BND as requested. Average turnover was higher for GP (`0.229`) than
historical Riskfolio (`0.135`) and equal weight (`0.083`), so transaction-cost
sensitivity is a real next diagnostic.

## GP Setup And Learned Diagnostics

The portfolio run used the same current production-candidate setup:

- `positive_beta_prior`
- positive task covariance with `BetaPrior(2.5, 1.5)`
- Hadamard multitask GP
- rank `5`
- task-stratified standardization
- BoTorch Normalize input transform
- lengthscale-only time modulation
- no target winsorization
- max optimizer iterations `50`

Prior model diagnostics still point to the same mechanism:

- Lengthscale-only is the best time-modulation direction. IC was `0.367`, top/bottom Sharpe was `3.205`, and residual top/bottom Sharpe was `1.925`.
- No time modulation was respectable but weaker: IC `0.320`, top/bottom Sharpe `2.341`.
- Outputscale-only was poor: IC `0.235`, top/bottom Sharpe `1.146`, residual long-short Sharpe negative.
- Both time-varying outputscale and lengthscale underperformed lengthscale-only: IC `0.270`, residual IC near zero.

Learned hyperparameters from the outputscale-structure diagnostic show why:

- Evaluation-date time-varying effective lengthscales were tiny: median `0.0037`, mean `0.0042`.
- Training effective lengthscales were also small but less extreme: median `0.0108`, mean `0.0112`.
- Likelihood noise was stable around `0.00655`, so noise is not exploding month to month.
- Several component outputscales were effectively suppressed:
  - ETF Matern block mean outputscale near `0.0018`.
  - Macro RQ near `0.0094`.
  - Macro linear near `0.0162`.
  - Macro x ETF interaction near `0.0064`.
- The strongest component outputscale was the macro Matern block, mean `0.471`.
- The task covariance is very positive and low-rank-like:
  - mean off-diagonal task correlation `0.741`
  - no negative task correlations
  - first eigenvalue share about `0.760`
  - first five eigenvalues explain about `0.959`

Interpretation: the current model is finding a shared cross-asset structure and a
useful time-varying rank signal, but the task covariance is very globally
positive and some component amplitudes are collapsing. This likely helps broad
risk-on/risk-off allocation but may blunt rotations among sleeves, especially in
stress months.

## Five Follow-Up Tuning Experiments

### 1. Signed Task Covariance Portfolio Run

Run the same 12-rebalance portfolio workflow with `signed_lkj_eta_2` and
lengthscale-only time modulation.

Why: the positive task covariance forces all task correlations positive. The
current task correlation is extremely global, with eig1 share around `0.76`.
Allowing signed correlations may improve bond/equity/credit rotations and reduce
February-style shared drawdowns.

Decision metric: terminal value, Sharpe, max drawdown, mean IC, and whether GP
weights diversify away from equity sleeves during negative-rank months.

### 2. Lengthscale Floor / Prior Strength Run

Keep lengthscale-only time modulation, but constrain or regularize the
time-varying effective lengthscale away from near-zero eval values. Test a small
floor or stronger prior that prevents eval effective lengthscales around
`0.001-0.004`.

Why: the time-varying lengthscale is useful, but the learned eval lengthscales
look overly sharp on transformed inputs. A floor may preserve regime sensitivity
while reducing over-local monthly behavior.

Decision metric: February 2026 return, mean IC, residual IC, and turnover.

### 3. Component Outputscale Floor For Suppressed Blocks

Do a narrow component-scale floor experiment, not a broad outputscale wrapper.
Target only the blocks that collapsed: ETF Matern, macro RQ, macro linear, and
macro x ETF interaction.

Why: previous outputscale-only and both-time-modulation variants were harmful,
but the component diagnostics show some feature groups are effectively turned
off. A mild floor can test whether those blocks add portfolio value without
reintroducing harmful time-varying outputscale behavior.

Decision metric: residual top/bottom signal, GP-vs-historical monthly spread,
and whether IC improves in weak months without damaging strong months.

### 4. Scenario Mean Calibration / Temperature Sweep

Keep the fitted GP fixed, but transform posterior scenario means before
Riskfolio optimization: test mean scale factors such as `0.5`, `1.0`, `1.5`,
and residual-mean centering before sampling/optimization.

Why: the model-quality report says posterior means are conservative while
uncertainty is wide. The portfolio optimizer may care more about relative
scenario mean strength than raw RMSE. This is cheap and isolates optimizer
translation from GP fitting.

Decision metric: terminal value, max drawdown, turnover, and concentration. This
should be run before another expensive architecture grid.

### 5. Cost-Aware / Turnover-Penalized Riskfolio Translation

Keep the current GP scenarios, but compare final portfolio translation variants:
current CVaR Sharpe, lower `upperlng`, turnover penalty or blend with previous
weights, and maybe top-k scenario universe filtering before Riskfolio.

Why: GP average turnover is materially higher (`0.229`) than historical Riskfolio
(`0.135`). If small gross edge disappears after realistic costs, the model edge
is not portfolio-useful yet. If a turnover-aware version keeps most return with
lower drawdown/cost drag, that is the more deployable policy.

Decision metric: cost-adjusted terminal value at 5, 10, and 25 bps per unit
turnover, Sharpe, max drawdown, and turnover.

## Recommended Order

1. Scenario mean calibration / temperature sweep.
2. Cost-aware Riskfolio translation.
3. Signed task covariance portfolio run.
4. Lengthscale floor / prior strength run.
5. Component outputscale floor for suppressed blocks.

The first two are cheapest and tell us whether the current fitted model is being
translated well. The last three change the GP and should include learned
hyperparameter diagnostics in the portfolio run itself.
