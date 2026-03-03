# Migration Map: Old Paths → New Paths

| Old Path | New Path |
|----------|----------|
| `bayesfolio/schemas/common.py` | `bayesfolio/contracts/base.py` |
| `bayesfolio/schemas/configs/core.py` | `bayesfolio/core/settings.py` |
| `bayesfolio/schemas/contracts/optimize.py` | `bayesfolio/contracts/commands/optimize.py` + `bayesfolio/contracts/results/optimize.py` |
| `bayesfolio/schemas/contracts/backtest.py` | `bayesfolio/contracts/commands/backtest.py` + `bayesfolio/contracts/results/backtest.py` |
| `bayesfolio/schemas/contracts/forecast.py` | `bayesfolio/contracts/commands/forecast.py` + `bayesfolio/contracts/results/forecast.py` |
| `bayesfolio/schemas/contracts/report.py` | `bayesfolio/contracts/commands/report.py` + `bayesfolio/contracts/results/report.py` |
| `bayesfolio/schemas/contracts/scenarios.py` | `bayesfolio/contracts/commands/scenario.py` + `bayesfolio/contracts/ui/scenario.py` |
| `bayesfolio/schemas/contracts/universe.py` | `bayesfolio/contracts/commands/universe.py` + `bayesfolio/contracts/ui/universe.py` |
| `bayesfolio/schemas/contracts/intent.py` | `bayesfolio/contracts/chat/intent.py` |
| `bayesfolio/schemas/contracts/beliefs.py` | `bayesfolio/contracts/commands/beliefs.py` |

## Class Name Changes

| Old Class | New Class | New Location |
|-----------|-----------|-------------|
| `SchemaMetadata` | `Meta` | `bayesfolio/contracts/base.py` |
| `ArtifactFingerprint` | `ArtifactPointer` | `bayesfolio/contracts/results/report.py` |
| `ArtifactRef` | `ArtifactPointer` | `bayesfolio/contracts/results/report.py` |
| `OptimizationRequest` | `OptimizeCommand` | `bayesfolio/contracts/commands/optimize.py` |
| `OptimizationResult` | `OptimizeResult` | `bayesfolio/contracts/results/optimize.py` |
| `ForecastPayload` | `ForecastResult` | `bayesfolio/contracts/results/forecast.py` |
| `ReportBundle` | `ReportResult` | `bayesfolio/contracts/results/report.py` |
| `ScenarioPanel` | `ScenarioCommand` | `bayesfolio/contracts/commands/scenario.py` |
| `UniverseRequest` | `UniverseCommand` | `bayesfolio/contracts/commands/universe.py` |
| `UniverseSnapshot` | `UniverseRecord` | `bayesfolio/contracts/ui/universe.py` |
| `OptimizationIntent` | `ParsedIntent` | `bayesfolio/contracts/chat/intent.py` |
| `PriorBeliefs` | `BeliefsCommand` | `bayesfolio/contracts/commands/beliefs.py` |
