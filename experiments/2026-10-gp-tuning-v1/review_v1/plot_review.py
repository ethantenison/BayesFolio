"""Readable scientific comparison figures from sealed GP tuning research.

Report-only I/O adapter. Reads exact saved seed NAV paths (dollars) and benchmark
decimal total returns; writes new PNG figures and provenance under this folder.
No fits, allocations, altered returns, curve offsets or baseline promotion.
The blue reference is drawn last with higher z-order and a white outline, and
an additional dollar-difference panel exposes otherwise hidden near-overlaps.

Usage:
    uv run --locked --no-sync python <this_file>
    uv run --locked --no-sync pytest -q <this_folder>
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.patheffects as effects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "scenario_numerics_v2/outputs"
OUTPUT = HERE / "outputs"
BASELINE = "control_hard_rank"
BLUE = "#1565C0"
REFINEMENT = "track2_mean__broad_r2_i15_exp__compact_r2_i15_exp__a05000_00000_03750_01875__c05000"
THIRD_SMALL = "t3p__broad_roll60_r2_i15__m01250__v00000__4777ae4caca5"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _save(path: Path, content: bytes) -> None:
    if not path.resolve().is_relative_to(OUTPUT):
        raise ValueError("Only new review outputs are writable")
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("Cannot overwrite immutable review output")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _paths(identities: list[str]) -> tuple[dict, dict, dict]:
    context = _arrays(SOURCE / "context.npz")
    inputs = {str(SOURCE / "context.npz"): _hash(SOURCE / "context.npz")}
    paths = {}
    for identity in [BASELINE, *identities]:
        path = SOURCE / "cases" / f"{identity}.npz"
        record_path = path.with_suffix(".json")
        record = _read(record_path)
        if record["status"] != "complete" or record["artifact_sha256"] != _hash(path):
            raise ValueError("Only complete sealed portfolio paths may be plotted")
        arrays = _arrays(path)
        if arrays["wealth"].shape != (3, 29):
            raise ValueError("Exactly three seed paths and 28 months required")
        np.testing.assert_allclose(arrays["wealth"][:, 0], 10000., rtol=0, atol=1e-8)
        for key in ["dates", "assets", "seeds"]:
            np.testing.assert_array_equal(arrays[key], context[key])
        paths[identity] = arrays["wealth"].mean(axis=0)
        if abs(paths[identity][-1] - record["periods"]["all28"]["ending_value"]) > 1e-7:
            raise ValueError("NAV endpoint differs from the reported seed mean")
        inputs.update({str(path): _hash(path), str(record_path): _hash(record_path)})
    return context, paths, inputs


def _figure(context: dict, paths: dict, series: list[dict], title: str) -> tuple[plt.Figure, dict]:
    origins = context["dates"].astype("datetime64[D]")
    endings = (origins.astype("datetime64[M]") + 2).astype("datetime64[D]") - 1
    dates = np.r_[origins[:1], endings]
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(14, 10.5), sharex=True,
                                        gridspec_kw={"height_ratios": [3.1, 1.25]}, layout="constrained")
    reference = paths[BASELINE]
    differences = {}
    palette = ["#E69F00", "#009E73", "#D55E00"]
    for index, item in enumerate(series):
        path = paths[item["id"]]
        color = palette[index % len(palette)]
        top.plot(dates, path, color=color, linestyle="--", linewidth=2.1, zorder=3 + index,
                 label=f"{item['label']} | ${path[-1]:,.0f}")
        if item["difference_panel"]:
            delta = path - reference
            bottom.plot(dates, delta, color=color, linewidth=2, marker="s", markersize=3,
                        markevery=3, label=f"{item['label']} | final {delta[-1]:+,.2f}")
            differences[item["id"]] = {"ending_dollars_minus_baseline": float(delta[-1]),
                                        "largest_absolute_path_difference_dollars": float(np.abs(delta).max())}
    for label, key, color in [
        ("Schwab aggressive proxy", "benchmark_schwab_total", "#8E6BBE"),
        ("Simple momentum top 7", "benchmark_momentum_total", "#666666"),
    ]:
        path = 10000 * np.r_[1., np.cumprod(1 + context[key])]
        top.plot(dates, path, color=color, linestyle=":", linewidth=2.1, zorder=2,
                 label=f"{label} | ${path[-1]:,.0f}")
    # No curve is offset. Draw the actual blue baseline above close candidates.
    baseline_line, = top.plot(dates, reference, color=BLUE, linewidth=2.9, marker="o", markersize=3.3,
                              markevery=2, zorder=20, label=f"Current hard-rank baseline | ${reference[-1]:,.0f}")
    baseline_line.set_gid("hard_rank_baseline")
    baseline_line.set_path_effects([effects.Stroke(linewidth=4.7, foreground="white"), effects.Normal()])
    bottom.axhline(0, color=BLUE, linewidth=2, zorder=10, label="Blue baseline = $0 difference")
    top.set_title(f"{title}\nGrowth of $10,000 | June 2024–September 2026", fontsize=15)
    top.set_ylabel("Investment value ($)")
    bottom.set_ylabel("Difference vs blue ($)")
    bottom.set_title("Hard-rank alternatives minus blue baseline — no artificial curve offsets", fontsize=11)
    bottom.set_xlabel("Month-end investment value; vertical divider marks the later 12 months")
    for axis in [top, bottom]:
        axis.axvline(dates[16], color="#888888", linestyle="--", linewidth=1.2, alpha=.8)
        axis.grid(alpha=.18)
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    handles, labels = top.get_legend_handles_labels()
    top.legend([handles[-1], *handles[:-1]], [labels[-1], *labels[:-1]], loc="upper left", fontsize=9)
    bottom.legend(loc="lower left", fontsize=8.5)
    bottom.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    figure.supxlabel("Exact saved mean NAV across three GP seeds; monthly rebalancing; gross of costs/taxes.\n"
                    "Static recipes selected on first16 or already-approved controls. Previously viewed history; "
                    "later12 is not an untouched holdout.", fontsize=9)
    return figure, differences


def main() -> None:
    """Write two readable, provenance-checked figures without changing results."""
    summary = _read(SOURCE / "GLOBAL_SUMMARY.json")
    originals = summary["first16_selected_case_ids"]
    original_series = [
        {"id": identity, "label": label, "difference_panel": True}
        for identity, label in zip(originals, ["60m third GP: 25% means", "Mean refinement + compact covariance",
                                               "60m third GP: 50% means/covariance"], strict=True)
    ]
    track_series = [
        {"id": REFINEMENT, "label": "Track 2: small static mean refinement", "difference_panel": True},
        {"id": "control_scenarios", "label": "Track 1: blended GP scenarios, rank2 / iter15",
         "difference_panel": False},
        {"id": THIRD_SMALL, "label": "Track 3: 12.5% third GP, rolling60", "difference_panel": True},
    ]
    all_ids = list(dict.fromkeys([*originals, *(row["id"] for row in track_series)]))
    context, paths, inputs = _paths(all_ids)
    inputs[str(SOURCE / "GLOBAL_SUMMARY.json")] = _hash(SOURCE / "GLOBAL_SUMMARY.json")
    inputs[str(SOURCE / "COMPLETED.json")] = _hash(SOURCE / "COMPLETED.json")
    plots = {}
    for name, series, title in [
        ("GLOBAL_COMPARISON_VISIBLE", original_series, "Original global comparison — blue baseline drawn on top"),
        ("THREE_TRACK_COMPARISON", track_series, "Three-track review — practical alternatives to current hard rank"),
    ]:
        figure, differences = _figure(context, paths, series, title)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=160)
        plt.close(figure)
        destination = OUTPUT / f"{name}.png"
        _save(destination, buffer.getvalue())
        plots[name] = {"series": series, "path": str(destination), "sha256": _hash(destination),
                       "hard_rank_dollar_differences": differences}
    for name, expected in inputs.items():
        if _hash(Path(name)) != expected:
            raise ValueError("Sealed input changed during figure rendering")
    metadata = {"passed": True, "blue_baseline_drawn_on_top": True, "curve_offsets": False,
                "GP_fits": 0, "allocations": 0, "input_sha256": inputs, "plots": plots,
                "implementation_sha256": _hash(Path(__file__))}
    _save(OUTPUT / "VERIFIED.json", (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode())
    print("Verified exact saved NAVs; blue baseline visible above overlaps; no fits or allocations")


if __name__ == "__main__":
    main()
