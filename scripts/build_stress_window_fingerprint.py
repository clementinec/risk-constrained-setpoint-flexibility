#!/usr/bin/env python3
"""Build a compact stress-window controller fingerprint from audit summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "Probabilities_ENB" / "paperB_control"
FIG_DIR = PAPER / "asset_write" / "figs"
TABLE_DIR = PAPER / "asset_write" / "tables"

STRATEGY_LABELS = {
    "diagnostic_reference": "Reference",
    "paperb_adaptive_band_relax": "Adaptive",
    "paperb_pmv_exceedance_guard_relax": "PMV guard",
    "paperb_ppd_guard_relax": "PPD guard",
    "paperb_gate_tail_asym_relax": "Mean TSV tail",
    "paperb_p90_tail_asym_relax": "p90 TSV tail",
    "paperb_mu_relax": "Expected TSV",
}
ORDER = ["Reference", "Adaptive", "PMV guard", "PPD guard", "Mean TSV tail", "p90 TSV tail", "Expected TSV"]

BLUE_RED = LinearSegmentedColormap.from_list("blue_red", ["#1f5b7a", "#f7f7f7", "#a83f38"], N=256)
TAIL = LinearSegmentedColormap.from_list("tail", ["#f4f7f8", "#b8d6df", "#4b8aa7", "#244a62"], N=256)
BURDEN = LinearSegmentedColormap.from_list("burden", ["#f6f4ef", "#d9c9a5", "#9f7d45", "#60451e"], N=256)
EVENT = LinearSegmentedColormap.from_list("event", ["#f7f7f7", "#c8c5dd", "#7f76ae", "#3d326d"], N=256)


def load_summaries(summary_files: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in summary_files]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["strategy"].isin(STRATEGY_LABELS)].copy()
    df["strategy_label"] = df["strategy"].map(STRATEGY_LABELS)
    return df


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    ref = df[df["strategy"] == "diagnostic_reference"]
    if len(ref) != 1:
        raise ValueError("Expected exactly one diagnostic_reference row in the audit summaries.")
    ref_row = ref.iloc[0]
    rows = []
    for _, row in df.iterrows():
        occupied = float(row["occupied_steps"])
        rows.append(
            {
                "strategy_label": row["strategy_label"],
                "hvac_elec_delta_pct": (float(row["hvac_electricity_kwh"]) - float(ref_row["hvac_electricity_kwh"]))
                / float(ref_row["hvac_electricity_kwh"])
                * 100.0,
                "peak_hvac_delta_pct": (float(row["peak_hvac_electric_kw_15min"]) - float(ref_row["peak_hvac_electric_kw_15min"]))
                / float(ref_row["peak_hvac_electric_kw_15min"])
                * 100.0,
                "max_tail_pct": float(row["max_zone_p_tail_ge_0p20_pct_occ"]),
                "pmv_violation_pct": float(row["pmv_violation_pct_occ"]),
                "relax_density_pct": float(row["relax_action_count"]) / occupied * 100.0,
                "hold_density_pct": float(row["guard_hold_count"]) / occupied * 100.0,
                "warm_protect_density_pct": float(row["warm_protection_count"]) / occupied * 100.0,
            }
        )
    out = pd.DataFrame(rows)
    out["strategy_label"] = pd.Categorical(out["strategy_label"], categories=ORDER, ordered=True)
    return out.sort_values("strategy_label")


def draw(table: pd.DataFrame, output_name: str) -> None:
    cols = [
        ("hvac_elec_delta_pct", "HVAC elec.\nchange", "%", BLUE_RED, TwoSlopeNorm(vmin=-3, vcenter=0, vmax=1)),
        ("peak_hvac_delta_pct", "Peak HVAC\nchange", "%", BLUE_RED, TwoSlopeNorm(vmin=-3, vcenter=0, vmax=1)),
        ("max_tail_pct", "Max-zone\nhigh tail", "% occ.", TAIL, Normalize(vmin=0, vmax=35)),
        ("pmv_violation_pct", "PMV\nviolation", "% occ.", BURDEN, Normalize(vmin=0, vmax=90)),
        ("relax_density_pct", "Relax\nactions", "% occ.", EVENT, Normalize(vmin=0, vmax=60)),
        ("hold_density_pct", "Guard\nholds", "% occ.", EVENT, Normalize(vmin=0, vmax=40)),
        ("warm_protect_density_pct", "Warm\nprotection", "% occ.", EVENT, Normalize(vmin=0, vmax=25)),
    ]
    strategies = list(table["strategy_label"].astype(str))
    fig, ax = plt.subplots(figsize=(10.2, 4.25))
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(len(strategies) - 0.5, -0.5)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([c[1] for c in cols], fontsize=8.8)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_yticks(np.arange(len(strategies)))
    ax.set_yticklabels(strategies, fontsize=9)
    ax.tick_params(axis="y", length=0)
    for i, (_, row) in enumerate(table.iterrows()):
        for j, (col, _, unit, cmap, norm) in enumerate(cols):
            value = float(row[col])
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor=cmap(norm(value)),
                    edgecolor="white",
                    linewidth=1.2,
                )
            )
            text = f"{value:.1f}" if abs(value) < 100 else f"{value:.0f}"
            if unit:
                text = f"{text}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8.1, color="#111111")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color("#222222")
    ax.set_title(
        "Stress-window controller fingerprint: Guangzhou SSP5-8.5 late-2080s heatwave-extreme year, July 8-28",
        fontsize=11.8,
        pad=22,
    )
    ax.text(
        len(cols) - 0.5,
        len(strategies) - 0.05,
        "All energy changes are relative to the fixed reference schedule with diagnostic outputs.",
        ha="right",
        va="top",
        fontsize=7.8,
        color="#333333",
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_DIR / f"{output_name}_source.csv", index=False)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.77, bottom=0.15)
    for suffix in [".png", ".pdf"]:
        fig.savefig(FIG_DIR / f"{output_name}{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, nargs="+")
    parser.add_argument("--output-name", default="fig6_stress_window_fingerprint")
    args = parser.parse_args()
    df = load_summaries(args.summary)
    table = build_table(df)
    draw(table, args.output_name)
    print(f"[fingerprint] wrote {args.output_name} to {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
