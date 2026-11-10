# Target alignment and availability audit

All original decimal excess-return targets remain unchanged.

Observation/decision dates are not the same as the future return endpoints:

| Panel | Observation dates | Forward outcome endpoints |
| --- | --- | --- |
| Initial 84 training rows | May 2017–April 2024 | June 2017–May 2024 |
| 24 scored rows | May 2024–April 2026 | June 2024–May 2026 |

At decision month-end t, the last training row is dated t-1 and its forward outcome
has finished at t. The prediction concerns t to the next month-end. Inputs are lagged
once. This preserves the runner's existing causal timing; no extra label shift is applied.

## Independent daily-source reconstruction

| asset_id   |   independently_rebuilt_observations |   max_abs_difference_basis_points |
|:-----------|-------------------------------------:|----------------------------------:|
| SPY        |                                  108 |                        0.00624726 |
| MGK        |                                  108 |                        0.0063537  |
| VTV        |                                  108 |                        0.00512765 |
| IWM        |                                  108 |                        0.0095665  |
| VEA        |                                  108 |                        0.00609169 |
| VWO        |                                  108 |                        0.00471621 |
| VSS        |                                  108 |                        0.00483977 |
| EWX        |                                  108 |                        0.0039559  |
| BND        |                                  108 |                        0.0190258  |
| IEF        |                                  108 |                        0.0103186  |
| LQD        |                                  108 |                        0.0101728  |
| HYG        |                                  108 |                        0.0107517  |
| BNDX       |                                  108 |                        0.0109772  |
| VWOB       |                                  108 |                        0.0118003  |
| HYEM       |                                  108 |                        0.0111464  |

All 1,620 reconstructed labels differ by less than 0.02 basis points from the preserved
cache, consistent with tiny adjusted-price source-vintage differences. No target was changed.

## Source-vintage and endpoint policy

Latest adjusted-price vintage, not a point-in-time vendor-vintage replication. All preserved labels were independently checked.

Full machine-readable evidence: `target_alignment_audit.json`.
