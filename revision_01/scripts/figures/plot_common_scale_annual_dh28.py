#!/usr/bin/env python3
"""Regenerate the Revision 01 common-scale annual DH>28 figure.

The default input, summary, and output paths resolve within the public
``revision_01`` package. Optional paths make the script convenient for checks
that write the figure to a temporary directory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
REVISION_ROOT = SCRIPT.parents[2]
DEFAULT_SOURCE = (
    REVISION_ROOT / "summary_outputs/d07_common_scale_annual_dh28_source.csv"
)
DEFAULT_SUMMARY = (
    REVISION_ROOT / "summary_outputs/d07_common_scale_annual_dh28_summary.csv"
)
DEFAULT_OUTPUT = REVISION_ROOT / "figures/fig_s_common_scale_annual_dh28.pdf"

COMPARATORS = (
    "diagnostic_reference",
    "paperb_pmv_relax",
    "paperb_adaptive_band_relax",
)
COMPARATOR_LABELS = {
    "diagnostic_reference": "Fixed-setpoint reference",
    "paperb_pmv_relax": "Building-mean PMV policy",
    "paperb_adaptive_band_relax": "Adaptive-band benchmark",
}
ROLES = ("present_typical", "future_extreme")
ROLE_LABELS = {
    "present_typical": "Present-typical selector role",
    "future_extreme": "Late-century-hot-extreme selector role",
}
VALUE_COLUMN = "delta_worst_zone_degree_hours_gt_28c"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_and_validate(source_path: Path, summary_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(source_path)
    summary = pd.read_csv(summary_path)

    required_source = {
        "anchor_id",
        "anchor_role",
        "city",
        "scenario",
        "comparator",
        VALUE_COLUMN,
        "comparator_label",
        "selector_role_label",
    }
    required_summary = {
        "comparator",
        "comparator_label",
        "n_scenarios",
        "mean_delta_dh_above_28c",
    }
    require(required_source.issubset(source.columns), "D07 source schema is incomplete")
    require(required_summary.issubset(summary.columns), "D07 summary schema is incomplete")
    require(len(source) == 72, "D07 source must contain 72 scenario-comparator rows")
    require(source["anchor_id"].nunique() == 24, "D07 source must contain 24 anchors")
    require(tuple(summary["comparator"]) == COMPARATORS, "D07 summary comparator order changed")
    require(source[VALUE_COLUMN].notna().all(), "D07 source contains missing thermal values")

    for comparator in COMPARATORS:
        comparator_rows = source.loc[source["comparator"].eq(comparator)]
        require(len(comparator_rows) == 24, f"{comparator}: expected 24 scenario rows")
        require(
            set(comparator_rows["anchor_role"]) == set(ROLES),
            f"{comparator}: selector-role coverage changed",
        )
        for role in ROLES:
            require(
                len(comparator_rows.loc[comparator_rows["anchor_role"].eq(role)]) == 12,
                f"{comparator}/{role}: expected 12 rows",
            )
        computed_mean = float(comparator_rows[VALUE_COLUMN].mean())
        recorded_mean = float(
            summary.loc[
                summary["comparator"].eq(comparator), "mean_delta_dh_above_28c"
            ].iloc[0]
        )
        require(
            np.isclose(computed_mean, recorded_mean, rtol=0.0, atol=1e-9),
            f"{comparator}: source and summary means differ",
        )

    return source, summary


def render(source: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    cache = Path(tempfile.mkdtemp(prefix="paperb_rev01_d07_mpl_"))
    os.environ["MPLCONFIGDIR"] = str(cache)
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 8.0,
                "axes.labelsize": 8.5,
                "axes.titlesize": 9.0,
                "xtick.labelsize": 7.5,
                "ytick.labelsize": 8.0,
                "legend.fontsize": 7.2,
                "pdf.fonttype": 42,
            }
        )
        figure, axis = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
        y_positions = {
            "diagnostic_reference": 2.0,
            "paperb_pmv_relax": 1.0,
            "paperb_adaptive_band_relax": 0.0,
        }
        colors = {"present_typical": "#2878B5", "future_extreme": "#D95F02"}
        markers = {"present_typical": "o", "future_extreme": "s"}
        jitter = np.linspace(-0.16, 0.16, 12)

        for comparator in COMPARATORS:
            for role in ROLES:
                subset = source.loc[
                    source["comparator"].eq(comparator)
                    & source["anchor_role"].eq(role)
                ].sort_values("anchor_id")
                axis.scatter(
                    subset[VALUE_COLUMN],
                    y_positions[comparator] + jitter,
                    s=23,
                    marker=markers[role],
                    facecolor=colors[role],
                    edgecolor="white",
                    linewidth=0.45,
                    alpha=0.88,
                    zorder=3,
                )
            mean = float(
                summary.loc[
                    summary["comparator"].eq(comparator),
                    "mean_delta_dh_above_28c",
                ].iloc[0]
            )
            axis.scatter(mean, y_positions[comparator], marker="D", s=48, color="black", zorder=5)
            axis.annotate(
                f"mean {mean:+.1f}",
                xy=(mean, y_positions[comparator]),
                xytext=(6, 7),
                textcoords="offset points",
                fontsize=7.0,
                color="black",
            )

        values = source[VALUE_COLUMN].to_numpy(dtype=float)
        span = float(np.max(values) - np.min(values))
        pad = max(20.0, 0.05 * span)
        axis.set_xlim(float(np.min(values) - pad), float(np.max(values) + pad))
        axis.axvline(0.0, color="0.25", linewidth=0.9, zorder=2)
        axis.grid(axis="x", color="0.88", linewidth=0.7, zorder=1)
        axis.set_yticks([2.0, 1.0, 0.0])
        axis.set_yticklabels([COMPARATOR_LABELS[item] for item in COMPARATORS])
        axis.set_ylim(-0.45, 2.45)
        axis.set_xlabel(
            r"Learned-p90 minus comparator annual worst-zone DH above 28$^{\circ}$C ($^{\circ}$C h)"
        )
        axis.set_title("Primary annual thermal comparison on one common horizontal scale")
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.tick_params(axis="y", length=0)
        handles = [
            Line2D(
                [0],
                [0],
                marker=markers[role],
                linestyle="none",
                markerfacecolor=colors[role],
                markeredgecolor="white",
                markersize=5.5,
                label=ROLE_LABELS[role],
            )
            for role in ROLES
        ]
        handles.append(
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="none",
                color="black",
                markersize=5.5,
                label="24-scenario mean",
            )
        )
        axis.legend(handles=handles, loc="lower right", frameon=False, ncol=1)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_path,
            format="pdf",
            dpi=300,
            metadata={
                "Title": "Common-scale annual DH28 comparison",
                "Creator": "Paper B Rev01 D07",
                "CreationDate": None,
                "ModDate": None,
            },
        )
        plt.close(figure)
    finally:
        shutil.rmtree(cache, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source, summary = load_and_validate(args.source, args.summary)
    render(source, summary, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
