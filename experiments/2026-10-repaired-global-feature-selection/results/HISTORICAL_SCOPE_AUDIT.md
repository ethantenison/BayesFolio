# Historical scope reconciliation

The first inventory covered October's 24-window studies. This additional audit explicitly
covers July, including valid full 12-month runs. Old metrics remain separate and invalid
hard-coded international split-group screens remain excluded.

Historical manifests audited: 1466.

Additional unique masks supported by the repaired v1 feature pool:

- us_equity: 2
- international_equity: 56
- us_fixed_income: 0
- international_fixed_income: 0

Disposition counts:

- excluded: 916
- quarantined: 157
- already_registered_october_mask: 279
- registered_july_extension: 58
- deduplicated_july_mask: 56

Detailed per-run reasons, manifest hashes, and repaired mask identifiers are recorded in
`historical_scope_audit.json`. Quarantined predictors were not validated in frozen v1;
their absence is not evidence that they are unimportant. Adding them requires a separately
versioned feature-pool extension, not modification of frozen datasets or silent substitution.
