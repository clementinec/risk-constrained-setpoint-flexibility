#!/usr/bin/env python3
"""Build a stress-window controller state map from retained trace parquet files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "Probabilities_ENB" / "paperB_control"
FIG_DIR = PAPER / "asset_write" / "figs"

STRATEGY_LABELS = {
    "diagnostic_reference": "Reference",
    "paperb_adaptive_band_relax": "Adaptive",
    "paperb_pmv_exceedance_guard_relax": "PMV guard",
    "paperb_ppd_guard_relax": "PPD guard",
    "paperb_gate_tail_asym_relax": "Mean TSV tail",
    "paperb_p90_tail_asym_relax": "p90 TSV tail",
    "paperb_mu_relax": "Expected TSV",
}
ORDER = [
    "diagnostic_reference",
    "paperb_adaptive_band_relax",
    "paperb_pmv_exceedance_guard_relax",
    "paperb_ppd_guard_relax",
    "paperb_gate_tail_asym_relax",
    "paperb_p90_tail_asym_relax",
    "paperb_mu_relax",
]


def load_window(trace_dirs: list[Path], weather: str, start: tuple[int, int], end: tuple[int, int]) -> pd.DataFrame:
    frames = []
    for strategy in ORDER:
        matches: list[Path] = []
        for trace_dir in trace_dirs:
            matches.extend(sorted(trace_dir.glob(f"{weather}_*_{strategy}.parquet")))
        if not matches:
            continue
        df = pd.read_parquet(matches[0])
        start_key = start[0] * 100 + start[1]
        end_key = end[0] * 100 + end[1]
        key = df["month"] * 100 + df["day"]
        df = df[(key >= start_key) & (key <= end_key)].copy()
        df["strategy_label"] = STRATEGY_LABELS[strategy]
        df["strategy_order"] = ORDER.index(strategy)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No retained traces found for {weather} in {trace_dirs}")
    out = pd.concat(frames, ignore_index=True)
    out["date_hour"] = (
        out["month"].astype(str).str.zfill(2)
        + "-"
        + out["day"].astype(str).str.zfill(2)
        + " "
        + out["hour"].astype(int).astype(str).str.zfill(2)
    )
    return out


def hourly_state(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if "max_zone_tail_probability" not in df.columns:
        warm_cols = [c for c in df.columns if c.startswith("zone_") and c.endswith("_warm_tail")]
        max_tail_parts = []
        for warm in warm_cols:
            cold = warm.replace("_warm_tail", "_cold_tail")
            if cold in df.columns:
                max_tail_parts.append(df[warm].astype(float) + df[cold].astype(float))
        if max_tail_parts:
            df = df.copy()
            df["max_zone_tail_probability"] = pd.concat(max_tail_parts, axis=1).max(axis=1)
        else:
            df = df.copy()
            df["max_zone_tail_probability"] = df["discomfort_probability"].astype(float)
    df = df.copy()
    df["timestamp"] = pd.to_datetime(
        {
            "year": 2001,
            "month": df["month"].astype(int),
            "day": df["day"].astype(int),
            "hour": df["hour"].astype(int).clip(lower=0, upper=23),
        }
    )
    df["time_bin"] = df["timestamp"].dt.floor("3h")
    group_cols = ["strategy_label", "strategy_order", "time_bin"]
    hourly = (
        df.groupby(group_cols, as_index=False)
        .agg(
            cooling_setpoint=("cooling_setpoint_c", "mean"),
            heating_setpoint=("heating_setpoint_c", "mean"),
            max_tail=("max_zone_tail_probability", "mean"),
            hold=("paperb_hold_guard", "sum"),
            action=("action_direction", lambda s: float(np.nansum(np.abs(s)))),
            oat=("outdoor_temp_c", "mean"),
            occupied=("occupied", "max"),
        )
    )
    hourly["date_hour"] = hourly["time_bin"].dt.strftime("%m-%d %H")
    time_order = list(hourly.sort_values("time_bin")["date_hour"].drop_duplicates())
    return hourly, time_order


def build_state_map(df: pd.DataFrame, weather: str, output_name: str) -> None:
    hourly, time_order = hourly_state(df)
    strategies = [STRATEGY_LABELS[s] for s in ORDER if STRATEGY_LABELS[s] in set(hourly["strategy_label"])]
    strategy_order = {s: i for i, s in enumerate(strategies)}
    n_time = len(time_order)
    matrix = np.full((len(strategies), n_time), np.nan)
    tail = np.full_like(matrix, np.nan, dtype=float)
    hold = np.zeros_like(matrix, dtype=float)
    action = np.zeros_like(matrix, dtype=float)
    for _, row in hourly.iterrows():
        i = strategy_order[row["strategy_label"]]
        j = time_order.index(row["date_hour"])
        # Cooling relaxation is the visual state; reference remains near fixed cooling setpoint.
        matrix[i, j] = row["cooling_setpoint"]
        tail[i, j] = row["max_tail"]
        hold[i, j] = row["hold"]
        action[i, j] = row["action"]

    cmap = LinearSegmentedColormap.from_list("setpoint_state", ["#305f72", "#f5f7f7", "#a94442"])
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12.2, 7.25),
        gridspec_kw={"height_ratios": [0.38, 2.05, 1.05, 2.05]},
        constrained_layout=True,
    )
    # Outdoor strip.
    oat = hourly.groupby("date_hour")["oat"].mean().reindex(time_order)
    axes[0].imshow(
        oat.to_numpy()[None, :],
        aspect="auto",
        cmap="inferno",
        norm=Normalize(vmin=oat.min(), vmax=oat.max()),
    )
    axes[0].set_yticks([])
    axes[0].set_xticks([])
    axes[0].set_title("Outdoor dry-bulb temperature", loc="left", fontsize=9, fontweight="bold")

    im = axes[1].imshow(matrix, aspect="auto", cmap=cmap, norm=Normalize(vmin=23.5, vmax=24.6))
    axes[1].set_yticks(np.arange(len(strategies)))
    axes[1].set_yticklabels(strategies)
    tick_idx = np.linspace(0, n_time - 1, 8, dtype=int)
    axes[1].set_xticks([])
    axes[1].set_title("Cooling setpoint state", loc="left", fontsize=9, fontweight="bold")
    for ax in [axes[1], axes[2], axes[3]]:
        ax.set_xticks(np.arange(-0.5, n_time, 4), minor=True)
        ax.set_yticks(np.arange(-0.5, len(strategies), 1), minor=True)
        ax.grid(which="minor", color="white", lw=0.42, alpha=0.42)
        ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=axes[1], shrink=0.85, pad=0.01)
    cbar.set_label(r"Cooling setpoint ($^\circ$C)")

    guard_cmap = LinearSegmentedColormap.from_list("guard_activity", ["#f7f7f7", "#d6c18a", "#7f6632"], N=256)
    guard = np.where(hold > 0, 1.0, 0.0)
    im_guard = axes[2].imshow(guard, aspect="auto", cmap=guard_cmap, norm=Normalize(vmin=0.0, vmax=1.0))
    axes[2].set_yticks(np.arange(len(strategies)))
    axes[2].set_yticklabels(strategies, fontsize=8)
    axes[2].set_xticks([])
    axes[2].set_title("Hold/protection guard active", loc="left", fontsize=9, fontweight="bold")
    cbar_guard = fig.colorbar(im_guard, ax=axes[2], shrink=0.85, pad=0.01)
    cbar_guard.set_ticks([0, 1])
    cbar_guard.set_ticklabels(["off", "on"])

    tail_cmap = LinearSegmentedColormap.from_list(
        "tail_prob", ["#f3f6f7", "#b7d4dd", "#4d8ca8", "#254b63"], N=256
    )
    im2 = axes[3].imshow(tail, aspect="auto", cmap=tail_cmap, norm=Normalize(vmin=0.0, vmax=0.45))
    axes[3].set_yticks(np.arange(len(strategies)))
    axes[3].set_yticklabels(strategies)
    axes[3].set_xticks(tick_idx)
    axes[3].set_xticklabels([time_order[i][:5] for i in tick_idx], rotation=0, ha="center", fontsize=8)
    axes[3].set_title("Max-zone discomfort-tail probability", loc="left", fontsize=9, fontweight="bold")
    cbar2 = fig.colorbar(im2, ax=axes[3], shrink=0.85, pad=0.01)
    cbar2.set_label(r"Max-zone $p_{tail}$")
    readable = (
        weather.replace("guangzhou_ssp585", "Guangzhou SSP5-8.5")
        .replace("guangzhou_ssp245", "Guangzhou SSP2-4.5")
        .replace("late_2080s", "late-2080s")
        .replace("mid_2050s", "mid-2050s")
        .replace("near_2030s", "near-2030s")
        .replace("baseline_2020s", "2020s")
        .replace("heatwave_extreme", "heatwave-extreme")
        .replace("_", " ")
    )
    fig.suptitle(
        f"Stress-window controller state map: {readable}, July 8-28",
        fontsize=12.5,
        y=1.02,
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf"]:
        fig.savefig(FIG_DIR / f"{output_name}{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True, nargs="+")
    parser.add_argument("--weather", required=True)
    parser.add_argument("--start", default="7-8")
    parser.add_argument("--end", default="7-28")
    parser.add_argument("--output-name", default="fig5_stress_window_state_map")
    args = parser.parse_args()
    start = tuple(int(x) for x in args.start.split("-"))
    end = tuple(int(x) for x in args.end.split("-"))
    df = load_window(args.trace_dir, args.weather, start, end)
    build_state_map(df, args.weather, args.output_name)
    print(f"[state-map] wrote {args.output_name} to {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
