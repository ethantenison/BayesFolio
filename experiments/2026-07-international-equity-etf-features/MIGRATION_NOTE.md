# International Equity Experiment Folder Migration

Date: 2026-07-08

The July 2026 international equity ETF feature work was initially stored under:

`experiments/2026-07-us-equity-etf-features/`

That was incorrect organization. The work belongs under this dedicated international equity experiment root:

`experiments/2026-07-international-equity-etf-features/`

Copied into this folder:

- International feature artifacts from `20260707T172312Z` and `20260707T182133Z`
- International correlation audit reports
- International block, interaction, market-kernel, expanded-feature, dynamic-feature, and remaining-feature runs
- International screen scripts
- The shared walk-forward wrapper needed by those scripts

The original files were left in place to avoid destructive changes. Treat this folder as canonical for future
international equity work. Treat any duplicated international run under the U.S. equity folder as a legacy copy.

The copied run manifests preserve historical command strings and absolute paths from the original execution root.
Those paths are lineage, not the preferred future location.
