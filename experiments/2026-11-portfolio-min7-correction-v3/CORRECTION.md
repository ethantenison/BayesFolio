# Corrected November recommendation: minimum seven

This supersedes the prior NE3 allocation for November. The allocator now enforces NE>=7 and at least seven holdings of0.01% or more, AFTER excluding MGK from the eligible set. It ranks all14eligible ETFs; it never makes a top7 portfolio and deletes MGK afterward.

Chosen policy: `corrected_v2_blend_NE7_MGK_excluded_before_allocation`. Holdings: 9; effective assets: 7.00000000; MGK exactly0.

All15 GP training tasks, model features, kernels, rank/maxiter settings and training windows are unchanged. No GP fits or new blend search occurred. The corrected fixed blend was compared with unchanged min7 controls using the prior frozen decision rules; old NE3 performance is not attributed to these weights.

All data were inspected previously. This is a retrospective constraint correction, not a fresh holdout.

| Policy | Full53 return | Full53 max drawdown | Sharpe | Latest12 return |
|---|---:|---:|---:|---:|
| corrected_v2_blend_NE7_MGK_excluded_before_allocation | 57.83% | -10.75% | 0.585 | 14.69% |
| v1_three_expert_reference_MGK0 | 56.16% | -11.47% | 0.562 | 14.49% |
| older_two_expert_incumbent_MGK0 | 47.78% | -13.83% | 0.441 | 12.81% |

| ETF | Target (%) |
|---|---:|
| SPY | 8.8109% |
| MGK | 0.0000% |
| VTV | 4.2105% |
| IWM | 18.0118% |
| VEA | 11.1111% |
| VWO | 15.7115% |
| VSS | 13.4113% |
| EWX | 20.3120% |
| BND | 0.0000% |
| IEF | 0.0000% |
| LQD | 0.0000% |
| HYG | 0.0000% |
| BNDX | 0.0000% |
| VWOB | 1.9102% |
| HYEM | 6.5107% |

Displayed percentages are rounded. The CSV retains raw decimal weights.

Corrected challenger passed the inherited promotion rule: True.

Data origin remains October30,2026 predicting November. A decision made afterward cannot assume an October30closing fill. No trades, production changes, commits or pushes occurred.

Verify from the repository root:

```sh
uv run --locked --no-sync python experiments/2026-11-portfolio-min7-correction-v3/run.py --verify
```
