#!/usr/bin/env python3
"""Build interim Paper B figures from retained full-matrix summaries."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "Probabilities_ENB" / "paperB_control"
RUN_ROOT = PAPER / "runs" / "full_matrix_realmet"
P90_RUN_ROOT = PAPER / "runs" / "full_matrix_p90_realmet"
FIG_DIR = PAPER / "asset_write" / "figs"
TABLE_DIR = PAPER / "asset_write" / "tables"

STRATEGY_LABELS = {
    "paperb_adaptive_band_relax": "Adaptive",
    "paperb_pmv_exceedance_guard_relax": "PMV guard",
    "paperb_ppd_guard_relax": "PPD guard",
    "paperb_gate_tail_asym_relax": "Mean TSV tail",
    "paperb_p90_tail_asym_relax": "p90 TSV tail",
    "paperb_mu_relax": "Expected TSV",
}
STRATEGY_ORDER = [
    "Adaptive",
    "PMV guard",
    "PPD guard",
    "Mean TSV tail",
    "p90 TSV tail",
    "Expected TSV",
]
STRATEGY_SHORT = {
    "Adaptive": "Adaptive",
    "PMV guard": "PMV",
    "PPD guard": "PPD",
    "Mean TSV tail": "Mean tail",
    "p90 TSV tail": "p90 tail",
    "Expected TSV": "Exp. TSV",
}
COLORS = {
    "Adaptive": "#4C78A8",
    "PMV guard": "#F58518",
    "PPD guard": "#B279A2",
    "Mean TSV tail": "#54A24B",
    "p90 TSV tail": "#2F8F6B",
    "Expected TSV": "#E45756",
}
DIV_CMAP = LinearSegmentedColormap.from_list(
    "paperb_diverging", ["#2f6f9f", "#f7f7f7", "#b64a3b"], N=256
)
ENERGY_CMAP = LinearSegmentedColormap.from_list(
    "paperb_energy", ["#1f5b7a", "#eef3f4", "#a83f38"], N=256
)
RISK_CMAP = LinearSegmentedColormap.from_list(
    "paperb_risk", ["#2f6f9f", "#f7f7f7", "#b6653f"], N=256
)
TAIL_BURDEN_CMAP = LinearSegmentedColormap.from_list(
    "paperb_tail_burden", ["#f7f7f7", "#d8d8d8", "#9c9c9c", "#404040"], N=256
)
TIME_ORDER = ["baseline_2020s", "near_2030s", "mid_2050s", "late_2080s"]
TIME_LABELS = {
    "baseline_2020s": "2020s",
    "near_2030s": "2030s",
    "mid_2050s": "2050s",
    "late_2080s": "2080s",
}
ROLE_ORDER = ["typical", "hot", "heatwave_extreme"]
ROLE_LABELS = {"typical": "Typical", "hot": "Hot", "heatwave_extreme": "Heatwave"}


def parse_weather(stem: str) -> dict[str, object]:
    match = re.match(
        r"(.+?)_(ssp\d+)_(baseline_2020s|near_2030s|mid_2050s|late_2080s)_"
        r"(typical|hot|heatwave_extreme)_(\d{4})$",
        stem,
    )
    if not match:
        return {"time_slice": "", "role": "", "year": np.nan}
    return {
        "time_slice": match.group(3),
        "role": match.group(4),
        "year": int(match.group(5)),
    }


def load_deltas() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    reference_rows: dict[tuple[str, str, str], tuple[pd.Series, pd.Series]] = {}
    for summary in sorted(RUN_ROOT.glob("*/summary/medium_office_trace_summary.csv")):
        shard = summary.parents[1].name
        city, pathway = shard.rsplit("_", 1)
        df = pd.read_csv(summary)
        for weather, group in df.groupby("weather"):
            ref = group[group["strategy"] == "reference"]
            diag = group[group["strategy"] == "diagnostic_reference"]
            if len(ref) != 1 or len(diag) != 1:
                continue
            ref_row = ref.iloc[0]
            diag_row = diag.iloc[0]
            reference_rows[(city, pathway, weather)] = (ref_row, diag_row)
            meta = parse_weather(weather)
            for _, row in group[group["strategy"].isin(STRATEGY_LABELS)].iterrows():
                rec = make_delta_record(city, pathway, weather, row, ref_row, diag_row, meta)
                rows.append(rec)
    for summary in sorted(P90_RUN_ROOT.glob("*/summary/medium_office_trace_summary.csv")):
        shard = summary.parents[1].name
        city, pathway = shard.rsplit("_", 1)
        df = pd.read_csv(summary)
        for _, row in df[df["strategy"].isin(STRATEGY_LABELS)].iterrows():
            weather = row["weather"]
            ref_pair = reference_rows.get((city, pathway, weather))
            if ref_pair is None:
                continue
            ref_row, diag_row = ref_pair
            rows.append(make_delta_record(city, pathway, weather, row, ref_row, diag_row, parse_weather(weather)))
    out = pd.DataFrame(rows)
    out["strategy_label"] = pd.Categorical(
        out["strategy_label"], categories=STRATEGY_ORDER, ordered=True
    )
    out["time_slice"] = pd.Categorical(out["time_slice"], categories=TIME_ORDER, ordered=True)
    return out


def make_delta_record(
    city: str,
    pathway: str,
    weather: str,
    row: pd.Series,
    ref_row: pd.Series,
    diag_row: pd.Series,
    meta: dict[str, object],
) -> dict[str, object]:
    rec: dict[str, object] = {
        "city": city.title(),
        "pathway": pathway.upper(),
        "weather": weather,
        "strategy": row["strategy"],
        "strategy_label": STRATEGY_LABELS[row["strategy"]],
        **meta,
    }
    for metric in [
        "electricity_kwh",
        "hvac_electricity_kwh",
        "cooling_electricity_kwh",
        "heating_electricity_kwh",
        "natural_gas_kwh",
        "peak_hvac_electric_kw_15min",
    ]:
        base = float(ref_row[metric])
        value = float(row[metric])
        rec[f"delta_pct_{metric}"] = (
            np.nan if abs(base) < 1e-12 else (value - base) / base * 100.0
        )
    for metric in [
        "pmv_violation_pct_occ",
        "adaptive_violation_pct_occ",
        "mean_zone_p_tail_ge_0p20_pct_occ",
        "max_zone_p_tail_ge_0p20_pct_occ",
    ]:
        rec[f"delta_{metric}"] = float(row[metric]) - float(diag_row[metric])
        rec[f"abs_{metric}"] = float(row[metric])
    if "p90_zone_p_tail_ge_0p20_pct_occ" in row.index:
        rec["abs_p90_zone_p_tail_ge_0p20_pct_occ"] = float(row["p90_zone_p_tail_ge_0p20_pct_occ"])
    else:
        rec["abs_p90_zone_p_tail_ge_0p20_pct_occ"] = np.nan
    return rec


def savefig(name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf"]:
        plt.savefig(FIG_DIR / f"{name}{suffix}", dpi=300, bbox_inches="tight")
    plt.close()


def controller_schematic() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, fc: str = "#f7f7f7", ec: str = "#333333", fs: float = 9.2, weight: str = "normal") -> None:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.05))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight=weight, linespacing=1.18)

    # Fixed simulation layer.
    box(0.05, 0.66, 0.18, 0.10, "Future weather\n12 files per pathway", "#eef3f4")
    box(0.05, 0.51, 0.18, 0.10, "DOE Medium Office\nfixed archetype", "#eef3f4")
    box(0.05, 0.36, 0.18, 0.10, "Occupancy + people\nreal-met heat gain", "#eef3f4")
    ax.text(0.14, 0.82, "Fixed reference layer", ha="center", fontsize=10.5, fontweight="bold")

    # Controller layer.
    box(0.36, 0.56, 0.25, 0.16, "Relaxation-first controller\nrequest setpoint relief\nwithin actuator bounds", "#f8f6f0", fs=9.5, weight="bold")
    box(0.38, 0.34, 0.21, 0.12, "Guard decision\nrelax / hold / protect", "#ffffff", fs=9.2)
    ax.annotate("", xy=(0.485, 0.56), xytext=(0.485, 0.46), arrowprops=dict(arrowstyle="->", lw=1.2, color="#333333"))
    ax.text(0.485, 0.79, "Same actuator and dwell rules", ha="center", fontsize=9.4, color="#333333")

    # Output layer.
    box(0.75, 0.66, 0.19, 0.10, "Energy components\nand peak HVAC", "#f4f4f4")
    box(0.75, 0.51, 0.19, 0.10, "PMV / adaptive\ncomfort violations", "#f4f4f4")
    box(0.75, 0.36, 0.19, 0.10, "TSV-tail exposure\nzone resolved", "#f4f4f4")
    ax.text(0.845, 0.82, "Common evaluation layer", ha="center", fontsize=10.5, fontweight="bold")

    for y in [0.71, 0.56, 0.41]:
        ax.annotate("", xy=(0.36, 0.64), xytext=(0.23, y), arrowprops=dict(arrowstyle="->", lw=1.0, color="#333333"))
    for y in [0.71, 0.56, 0.41]:
        ax.annotate("", xy=(0.75, y), xytext=(0.61, 0.64), arrowprops=dict(arrowstyle="->", lw=1.0, color="#333333"))

    ax.text(0.50, 0.25, "Guard signal is the experimental variable", ha="center", fontsize=10.2, fontweight="bold")
    guard_y = 0.09
    guard_xs = np.linspace(0.075, 0.795, len(STRATEGY_ORDER))
    guard_w = 0.125
    for x, label in zip(guard_xs, STRATEGY_ORDER):
        box(x, guard_y, guard_w, 0.09, STRATEGY_SHORT[label], fc=COLORS[label], ec=COLORS[label], fs=8.2)
        ax.patches[-1].set_alpha(0.18)
    ax.text(
        0.5,
        0.025,
        "Savings are attributed to threshold-enabled setpoint relaxation, not to changes in the building, weather, or actuator model.",
        ha="center",
        fontsize=8.8,
        color="#222222",
    )
    savefig("fig1_controller_logic_schematic")


def controller_state_machine() -> None:
    fig, ax = plt.subplots(figsize=(10.4, 4.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    xs = [0.10, 0.34, 0.58]
    labels = [
        ("Occupied timestep", "15-min update"),
        ("Relaxation request", "toward energy-saving bounds"),
        ("Guard screen", "threshold and direction check"),
    ]
    ax.plot([0.07, 0.68], [0.67, 0.67], color="#303030", lw=1.3)
    for x, (head, sub) in zip(xs, labels):
        ax.plot([x, x], [0.61, 0.73], color="#303030", lw=1.3)
        ax.text(x, 0.79, head, ha="center", va="center", fontsize=10.6, fontweight="bold")
        ax.text(x, 0.55, sub, ha="center", va="center", fontsize=8.6, color="#333333")
    ax.annotate("", xy=(0.70, 0.67), xytext=(0.66, 0.67), arrowprops=dict(arrowstyle="->", lw=1.3, color="#303030"))

    outcome_x = 0.79
    outcomes = [
        (0.78, "Relax", "step toward 20 / 26 C bounds", "#7aa37b"),
        (0.61, "Hold", "keep current setpoint", "#b69b63"),
        (0.44, "Protect", "step toward 23.25 C bound", "#bf746b"),
    ]
    for y, head, sub, color in outcomes:
        ax.plot([0.72, 0.95], [y, y], color=color, lw=5.2, solid_capstyle="butt", alpha=0.58)
        ax.text(outcome_x, y + 0.045, head, ha="left", va="bottom", fontsize=10.4, fontweight="bold", color="#111111")
        ax.text(outcome_x, y - 0.042, sub, ha="left", va="top", fontsize=8.7, color="#333333")
    ax.plot([0.70, 0.72], [0.67, 0.78], color="#303030", lw=1.0)
    ax.plot([0.70, 0.72], [0.67, 0.61], color="#303030", lw=1.0)
    ax.plot([0.70, 0.72], [0.67, 0.44], color="#303030", lw=1.0)
    ax.text(0.695, 0.82, "no active risk", ha="right", fontsize=8.2, color="#333333")
    ax.text(0.695, 0.49, "active risk", ha="right", fontsize=8.2, color="#333333")

    guard_y = 0.135
    ax.text(0.50, 0.27, "Only the guard signal changes across controller variants", ha="center", fontsize=10.2, fontweight="bold")
    guard_xs = np.linspace(0.075, 0.795, len(STRATEGY_ORDER))
    guard_w = 0.125
    for x, label in zip(guard_xs, STRATEGY_ORDER):
        ax.plot([x, x + guard_w], [guard_y, guard_y], color=COLORS[label], lw=5.0, alpha=0.55, solid_capstyle="butt")
        ax.text(x + guard_w / 2.0, guard_y - 0.045, STRATEGY_SHORT[label], ha="center", va="center", fontsize=8.0)
    ax.text(0.50, 0.035, "The same request, dwell limits, and actuator bounds are used for every guard.", ha="center", fontsize=8.5, color="#333333")
    savefig("fig1b_controller_state_machine")


def tradeoff(df: pd.DataFrame) -> None:
    agg = (
        df.groupby(["city", "pathway", "strategy_label"], observed=True)
        .agg(
            facility=("delta_pct_electricity_kwh", "mean"),
            mean_tail=("delta_mean_zone_p_tail_ge_0p20_pct_occ", "mean"),
            max_tail_abs=("abs_max_zone_p_tail_ge_0p20_pct_occ", "mean"),
            peak=("delta_pct_peak_hvac_electric_kw_15min", "mean"),
        )
        .reset_index()
    )
    row_order = sorted((agg["city"] + " " + agg["pathway"].str.replace("SSP", "SSP ")).unique())
    agg["row"] = agg["city"] + " " + agg["pathway"].str.replace("SSP", "SSP ")
    mat = (
        agg.pivot_table(index="row", columns="strategy_label", values="facility", observed=True)
        .reindex(index=row_order, columns=STRATEGY_ORDER)
        .to_numpy(dtype=float)
    )
    tail = (
        agg.pivot_table(index="row", columns="strategy_label", values="mean_tail", observed=True)
        .reindex(index=row_order, columns=STRATEGY_ORDER)
        .to_numpy(dtype=float)
    )
    tail_abs = (
        agg.pivot_table(index="row", columns="strategy_label", values="max_tail_abs", observed=True)
        .reindex(index=row_order, columns=STRATEGY_ORDER)
        .to_numpy(dtype=float)
    )
    fig, ax = plt.subplots(figsize=(9.2, max(5.4, 0.42 * len(row_order) + 2.2)))
    energy_norm = TwoSlopeNorm(vmin=-8.0, vcenter=0.0, vmax=2.0)
    risk_norm = TwoSlopeNorm(vmin=-8.0, vcenter=0.0, vmax=14.0)
    ax.set_xlim(-0.5, len(STRATEGY_ORDER) - 0.5)
    ax.set_ylim(len(row_order) - 0.5, -1.18)
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=8.6)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    ax.set_xticks(np.arange(-0.5, len(STRATEGY_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_order), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.35)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            energy = mat[i, j]
            risk = tail[i, j]
            if not np.isfinite(energy) or not np.isfinite(risk):
                continue
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, i - 0.5),
                    0.5,
                    1.0,
                    facecolor=ENERGY_CMAP(energy_norm(energy)),
                    edgecolor="none",
                )
            )
            ax.add_patch(
                plt.Rectangle(
                    (j, i - 0.5),
                    0.5,
                    1.0,
                    facecolor=RISK_CMAP(risk_norm(risk)),
                    edgecolor="none",
                )
            )
            if np.isfinite(tail_abs[i, j]) and tail_abs[i, j] >= 20.0:
                ax.add_patch(
                    plt.Rectangle(
                        (j, i - 0.5),
                        0.5,
                        1.0,
                        facecolor="none",
                        edgecolor="#ffffff",
                        hatch="////",
                        lw=0.0,
                    )
                )
    for i in range(len(row_order) + 1):
        ax.plot([-0.5, len(STRATEGY_ORDER) - 0.5], [i - 0.5, i - 0.5], color="white", lw=1.35)
    for j in range(len(STRATEGY_ORDER) + 1):
        ax.plot([j - 0.5, j - 0.5], [-0.5, len(row_order) - 0.5], color="white", lw=1.35)
    for j in range(len(STRATEGY_ORDER)):
        ax.plot([j, j], [-0.5, len(row_order) - 0.5], color="#222222", lw=0.42, alpha=0.38)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat[i, j]
            if not np.isfinite(value):
                continue
            ax.text(j - 0.24, i, f"{value:.1f}", ha="center", va="center", fontsize=7.0, color="#111111")
            tail_delta = tail[i, j]
            if np.isfinite(tail_delta):
                ax.text(j + 0.24, i, f"{tail_delta:+.1f}", ha="center", va="center", fontsize=7.0, color="#111111")
    sm_energy = plt.cm.ScalarMappable(norm=energy_norm, cmap=ENERGY_CMAP)
    sm_risk = plt.cm.ScalarMappable(norm=risk_norm, cmap=RISK_CMAP)
    cbar_tail = fig.colorbar(sm_risk, ax=ax, shrink=0.78, pad=0.012, location="right")
    cbar_tail.set_label("Right half: mean-zone high-tail change (pp)")
    cbar_energy = fig.colorbar(sm_energy, ax=ax, shrink=0.78, pad=0.070, location="right")
    cbar_energy.set_label("Left half: facility electricity change (%)")
    for j, label in enumerate(STRATEGY_ORDER):
        ax.text(j, -1.02, STRATEGY_SHORT[label], ha="center", va="center", fontsize=9.2)
        ax.text(j - 0.24, -0.64, "E", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#333333")
        ax.text(j + 0.24, -0.64, "T", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#333333")
    ax.set_title(
        "Annual energy-risk matrix: savings, mean tail change, and any-zone hatch",
        fontsize=12.5,
        pad=18,
    )
    fig.subplots_adjust(left=0.16, right=0.80, top=0.84, bottom=0.08)
    savefig("fig2_annual_performance_matrix")


def component_decomposition(df: pd.DataFrame) -> None:
    agg = (
        df.groupby(["city", "strategy_label"], observed=True)
        .agg(
            cooling=("delta_pct_cooling_electricity_kwh", "mean"),
            heating=("delta_pct_heating_electricity_kwh", "mean"),
            hvac=("delta_pct_hvac_electricity_kwh", "mean"),
            gas=("delta_pct_natural_gas_kwh", "mean"),
        )
        .reset_index()
    )
    cities = sorted(agg["city"].unique())
    metrics = ["cooling", "heating", "hvac", "gas"]
    metric_labels = ["Cooling\nelec.", "Heating\nelec.", "HVAC\nelec.", "Gas"]
    norm = TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40)
    fig, axes = plt.subplots(
        len(cities), 1, figsize=(8.8, 1.65 * len(cities)), sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for ax, city in zip(axes, cities):
        sub = agg[agg["city"] == city].set_index("strategy_label").reindex(STRATEGY_ORDER)
        mat = sub[metrics].to_numpy(dtype=float)
        im = ax.imshow(mat, aspect="auto", cmap=DIV_CMAP, norm=norm)
        ax.set_yticks(np.arange(len(STRATEGY_ORDER)))
        ax.set_yticklabels(STRATEGY_ORDER, fontsize=9)
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_title(city, loc="left", fontsize=11, fontweight="bold", pad=4)
        ax.set_xticks(np.arange(-0.5, len(metrics), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(STRATEGY_ORDER), 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.4)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                value = mat[i, j]
                if not np.isfinite(value):
                    continue
                if abs(value) >= 0.5:
                    color = "white" if value <= -32 or value >= 28 else "#222222"
                    ax.text(
                        j,
                        i,
                        f"{value:.0f}",
                        ha="center",
                        va="center",
                        fontsize=7.4,
                        color=color,
                    )
    cbar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
    cbar.set_label("Change vs reference (%), clipped at +/-40")
    fig.suptitle("Energy-component fingerprint by city and guard signal", y=1.01, fontsize=13)
    savefig("fig3_energy_component_fingerprint")


def time_slice_trends(df: pd.DataFrame) -> None:
    agg = (
        df.groupby(["city", "pathway", "time_slice", "strategy_label"], observed=True)
        .agg(
            facility=("delta_pct_electricity_kwh", "mean"),
            max_tail=("delta_max_zone_p_tail_ge_0p20_pct_occ", "mean"),
        )
        .reset_index()
    )
    agg["city_pathway"] = agg["city"] + " " + agg["pathway"].str.replace("SSP", "SSP ")
    row_order = sorted(agg["city_pathway"].unique())
    fig, axes = plt.subplots(
        1, len(STRATEGY_ORDER), figsize=(13.2, max(4.8, 0.34 * len(row_order) + 1.6)),
        sharey=True, constrained_layout=True
    )
    norm = TwoSlopeNorm(vmin=-8, vcenter=0, vmax=2)
    for ax, strategy in zip(axes, STRATEGY_ORDER):
        sub = agg[agg["strategy_label"] == strategy]
        mat = (
            sub.pivot_table(index="city_pathway", columns="time_slice", values="facility", observed=True)
            .reindex(index=row_order, columns=TIME_ORDER)
            .to_numpy(dtype=float)
        )
        tail = (
            sub.pivot_table(index="city_pathway", columns="time_slice", values="max_tail", observed=True)
            .reindex(index=row_order, columns=TIME_ORDER)
            .to_numpy(dtype=float)
        )
        im = ax.imshow(mat, aspect="auto", cmap=ENERGY_CMAP, norm=norm)
        ax.set_title(strategy, fontsize=10, fontweight="bold")
        ax.set_xticks(np.arange(len(TIME_ORDER)))
        ax.set_xticklabels([TIME_LABELS[t] for t in TIME_ORDER], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(row_order)))
        ax.set_yticklabels(row_order, fontsize=8)
        ax.set_xticks(np.arange(-0.5, len(TIME_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(row_order), 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        # Tail-risk overlay: hollow marker grows with positive max-zone tail penalty.
        for i in range(tail.shape[0]):
            for j in range(tail.shape[1]):
                value = tail[i, j]
                if not np.isfinite(value):
                    continue
                if value > 0.25:
                    size = 18 + min(value, 8) * 10
                    ax.scatter(j, i, s=size, facecolors="none", edgecolors="#1b1b1b", linewidths=0.8)
                elif value < -0.25:
                    ax.scatter(j, i, s=22, marker=".", color="#1b1b1b")
    cbar = fig.colorbar(im, ax=axes, shrink=0.70, pad=0.012)
    cbar.set_label("Facility electricity change vs reference (%)")
    fig.suptitle("Time-slice matrix: energy savings with localized tail-risk overlay", y=1.03, fontsize=13)
    savefig("fig4_time_slice_matrix")


def weather_role_robustness(df: pd.DataFrame) -> None:
    agg = (
        df.groupby(["city", "pathway", "role", "strategy_label"], observed=True)
        .agg(
            facility=("delta_pct_electricity_kwh", "mean"),
            mean_tail_delta=("delta_mean_zone_p_tail_ge_0p20_pct_occ", "mean"),
            max_tail_abs=("abs_max_zone_p_tail_ge_0p20_pct_occ", "mean"),
        )
        .reset_index()
    )
    agg["city_pathway"] = agg["city"] + " " + agg["pathway"].str.replace("SSP", "SSP ")
    row_order = sorted(agg["city_pathway"].unique())
    fig, axes = plt.subplots(
        2,
        len(STRATEGY_ORDER),
        figsize=(13.2, max(6.8, 0.46 * len(row_order) + 1.7)),
        sharey=False,
        constrained_layout=False,
    )
    energy_norm = TwoSlopeNorm(vmin=-8, vcenter=0, vmax=2)
    tail_norm = TwoSlopeNorm(vmin=-8.0, vcenter=0.0, vmax=14.0)
    for j_strategy, strategy in enumerate(STRATEGY_ORDER):
        ax_energy = axes[0, j_strategy]
        ax_tail = axes[1, j_strategy]
        sub = agg[agg["strategy_label"] == strategy]
        energy = (
            sub.pivot_table(index="city_pathway", columns="role", values="facility", observed=True)
            .reindex(index=row_order, columns=ROLE_ORDER)
            .to_numpy(dtype=float)
        )
        mean_tail_delta = (
            sub.pivot_table(index="city_pathway", columns="role", values="mean_tail_delta", observed=True)
            .reindex(index=row_order, columns=ROLE_ORDER)
            .to_numpy(dtype=float)
        )
        max_tail_abs = (
            sub.pivot_table(index="city_pathway", columns="role", values="max_tail_abs", observed=True)
            .reindex(index=row_order, columns=ROLE_ORDER)
            .to_numpy(dtype=float)
        )
        for ax, mat, cmap, norm, title_prefix in [
            (ax_energy, energy, ENERGY_CMAP, energy_norm, "Electricity change"),
            (ax_tail, mean_tail_delta, RISK_CMAP, tail_norm, "Mean-zone high-tail change"),
        ]:
            im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)
            ax.set_xlim(-0.5, len(ROLE_ORDER) - 0.5)
            ax.set_ylim(len(row_order) - 0.5, -0.5)
            ax.set_xticks(np.arange(len(ROLE_ORDER)))
            ax.set_xticklabels([ROLE_LABELS[r] for r in ROLE_ORDER], rotation=45, ha="right", fontsize=8)
            ax.set_yticks(np.arange(len(row_order)))
            if j_strategy == 0:
                ax.set_yticklabels(row_order, fontsize=7.2)
            else:
                ax.set_yticklabels([])
            ax.tick_params(axis="both", length=0)
            ax.set_xticks(np.arange(-0.5, len(ROLE_ORDER), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(row_order), 1), minor=True)
            ax.grid(which="minor", color="white", lw=1.0)
            ax.tick_params(which="minor", bottom=False, left=False)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    value = mat[i, j]
                    if not np.isfinite(value):
                        continue
                    if title_prefix.startswith("Electricity") and value <= -6.5:
                        ax.text(
                            j,
                            i,
                            f"{value:.0f}",
                            ha="center",
                            va="center",
                            fontsize=6.7,
                            fontweight="bold",
                            color="#ffffff",
                        )
                    elif title_prefix.startswith("Mean") and abs(value) >= 6.0:
                        ax.text(
                            j,
                            i,
                            f"{value:.0f}",
                            ha="center",
                            va="center",
                            fontsize=6.7,
                            fontweight="bold",
                            color="#ffffff" if value > 8 or value < -5 else "#111111",
                        )
            if title_prefix.startswith("Mean"):
                for i in range(max_tail_abs.shape[0]):
                    for j in range(max_tail_abs.shape[1]):
                        if np.isfinite(max_tail_abs[i, j]) and max_tail_abs[i, j] >= 20.0:
                            ax.add_patch(
                                plt.Rectangle(
                                    (j - 0.5, i - 0.5),
                                    1.0,
                                    1.0,
                                    facecolor="none",
                                    edgecolor="#ffffff",
                                    hatch="////",
                                    lw=0.0,
                                )
                            )
        ax_energy.set_title(strategy, fontsize=10, fontweight="bold")
        if j_strategy == 0:
            ax_energy.set_ylabel("Electricity\nchange", fontsize=9, fontweight="bold")
            ax_tail.set_ylabel("Mean high-tail\nchange", fontsize=9, fontweight="bold")
        ax_energy.set_xticklabels([])
    sm_energy = plt.cm.ScalarMappable(norm=energy_norm, cmap=ENERGY_CMAP)
    sm_tail = plt.cm.ScalarMappable(norm=tail_norm, cmap=RISK_CMAP)
    fig.subplots_adjust(left=0.17, right=0.75, top=0.90, bottom=0.14, wspace=0.05, hspace=0.08)
    cax_energy = fig.add_axes([0.79, 0.60, 0.018, 0.24])
    cax_tail = fig.add_axes([0.79, 0.22, 0.018, 0.24])
    cbar1 = fig.colorbar(sm_energy, cax=cax_energy)
    cbar1.set_label("Electricity change vs reference (%)", labelpad=12)
    cbar2 = fig.colorbar(sm_tail, cax=cax_tail)
    cbar2.set_label("Mean-zone high-tail change (pp)", labelpad=12)
    fig.suptitle("Weather-role robustness: electricity savings and mean tail change", y=0.985, fontsize=13)
    savefig("fig4b_weather_role_robustness")


def write_tables(df: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / "figure_source_deltas.csv", index=False)
    city_table = (
        df.groupby(["city", "pathway", "strategy_label"], observed=True)
        .agg(
            facility_elec_pct=("delta_pct_electricity_kwh", "mean"),
            hvac_elec_pct=("delta_pct_hvac_electricity_kwh", "mean"),
            cooling_elec_pct=("delta_pct_cooling_electricity_kwh", "mean"),
            heating_elec_pct=("delta_pct_heating_electricity_kwh", "mean"),
            pmv_violation_pp=("delta_pmv_violation_pct_occ", "mean"),
            adaptive_violation_pp=("delta_adaptive_violation_pct_occ", "mean"),
            mean_tail_pp=("delta_mean_zone_p_tail_ge_0p20_pct_occ", "mean"),
            max_tail_pp=("delta_max_zone_p_tail_ge_0p20_pct_occ", "mean"),
        )
        .reset_index()
    )
    city_table.to_csv(TABLE_DIR / "figure_source_city_pathway_means.csv", index=False)


def main() -> int:
    df = load_deltas()
    write_tables(df)
    controller_schematic()
    controller_state_machine()
    tradeoff(df)
    component_decomposition(df)
    time_slice_trends(df)
    weather_role_robustness(df)
    print(f"[figures] completed shards: {df['city'].nunique()} cities, {df[['city','pathway']].drop_duplicates().shape[0]} city-pathways")
    print(f"[figures] wrote figures to {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
