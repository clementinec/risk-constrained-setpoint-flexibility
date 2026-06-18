#!/usr/bin/env python3
"""Summarize Paper B closed-loop smoke traces.

This script reports the quantities that matter for the first controller
diagnostic: energy, tail exposure, setpoint relaxation, and protection actions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


P_TAIL_SCREEN = 0.20


def pct(mask: pd.Series | np.ndarray) -> float:
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean() * 100.0)


def finite_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.mean()) if values.notna().any() else float("nan")


def finite_quantile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.quantile(q)) if values.notna().any() else float("nan")


def zone_columns(df: pd.DataFrame, suffix: str) -> list[str]:
    return [col for col in df.columns if col.startswith("zone_") and col.endswith(suffix)]


def summarize_strategy(df: pd.DataFrame) -> dict[str, object]:
    strategy = str(df["strategy"].iloc[0])
    weather = str(df["weather"].iloc[0])
    occ = df[df["occupied"].astype(bool)].copy()
    if occ.empty:
        occ = df.copy()

    p_disc = pd.to_numeric(occ.get("discomfort_probability"), errors="coerce")
    warm_tail = pd.to_numeric(occ.get("warm_discomfort_probability"), errors="coerce")
    cold_tail = pd.to_numeric(occ.get("cold_discomfort_probability"), errors="coerce")
    max_zone_disc = None
    zone_p_cols = zone_columns(occ, "_p_disc")
    if zone_p_cols:
        max_zone_disc = occ[zone_p_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)

    electricity_kwh = (
        pd.to_numeric(df["electricity_facility_j"], errors="coerce").sum() / 3.6e6
        if "electricity_facility_j" in df
        else float("nan")
    )
    gas_kwh = (
        pd.to_numeric(df["natural_gas_facility_j"], errors="coerce").sum() / 3.6e6
        if "natural_gas_facility_j" in df
        else float("nan")
    )

    action_dir = pd.to_numeric(occ.get("action_direction"), errors="coerce")
    request_dir = pd.to_numeric(occ.get("paperb_requested_direction"), errors="coerce")
    hold_guard = pd.to_numeric(occ.get("paperb_hold_guard"), errors="coerce").fillna(0)
    request_source = occ.get("paperb_request_source", pd.Series("", index=occ.index)).fillna("")
    adaptive_violation = np.full(len(occ), False)
    adaptive_80_violation = np.full(len(occ), False)
    adaptive_90_violation = np.full(len(occ), False)
    if {"mean_operative_temp_c", "comfort_low_c", "comfort_high_c"}.issubset(occ.columns):
        top = pd.to_numeric(occ["mean_operative_temp_c"], errors="coerce")
        low = pd.to_numeric(occ["comfort_low_c"], errors="coerce")
        high = pd.to_numeric(occ["comfort_high_c"], errors="coerce")
        adaptive_violation = (top < low) | (top > high)
    if {"mean_operative_temp_c", "adaptive_80_low_c", "adaptive_80_high_c"}.issubset(occ.columns):
        top = pd.to_numeric(occ["mean_operative_temp_c"], errors="coerce")
        low = pd.to_numeric(occ["adaptive_80_low_c"], errors="coerce")
        high = pd.to_numeric(occ["adaptive_80_high_c"], errors="coerce")
        adaptive_80_violation = (top < low) | (top > high)
    if {"mean_operative_temp_c", "adaptive_90_low_c", "adaptive_90_high_c"}.issubset(occ.columns):
        top = pd.to_numeric(occ["mean_operative_temp_c"], errors="coerce")
        low = pd.to_numeric(occ["adaptive_90_low_c"], errors="coerce")
        high = pd.to_numeric(occ["adaptive_90_high_c"], errors="coerce")
        adaptive_90_violation = (top < low) | (top > high)

    out: dict[str, object] = {
        "weather": weather,
        "paperb_met": finite_mean(df["paperb_met"]) if "paperb_met" in df else float("nan"),
        "paperb_people_activity_w_per_person": finite_mean(
            df["paperb_people_activity_w_per_person"]
        )
        if "paperb_people_activity_w_per_person" in df
        else float("nan"),
        "strategy": strategy,
        "occupied_steps": int(len(occ)),
        "electricity_kwh": electricity_kwh,
        "natural_gas_kwh": gas_kwh,
        "mean_top_c": finite_mean(occ["mean_operative_temp_c"]),
        "mean_pmv": finite_mean(occ["mean_pmv"]),
        "mean_ppd_pct": finite_mean(occ["mean_ppd_pct"]) if "mean_ppd_pct" in occ else float("nan"),
        "ppd_gt_10_pct": pct(pd.to_numeric(occ.get("mean_ppd_pct"), errors="coerce") > 10.0)
        if "mean_ppd_pct" in occ
        else float("nan"),
        "ppd_gt_25_pct": pct(pd.to_numeric(occ.get("mean_ppd_pct"), errors="coerce") > 25.0)
        if "mean_ppd_pct" in occ
        else float("nan"),
        "ppd_gt_50_pct": pct(pd.to_numeric(occ.get("mean_ppd_pct"), errors="coerce") > 50.0)
        if "mean_ppd_pct" in occ
        else float("nan"),
        "pmv_abs_gt_0_5_pct": pct(pd.to_numeric(occ["mean_pmv"], errors="coerce").abs() > 0.5),
        "pmv_abs_gt_1_0_pct": pct(pd.to_numeric(occ["mean_pmv"], errors="coerce").abs() > 1.0),
        "adaptive_violation_pct": pct(adaptive_violation),
        "adaptive_80_violation_pct": pct(adaptive_80_violation),
        "adaptive_90_violation_pct": pct(adaptive_90_violation),
        "mean_adaptive_warm_slack_80_c": finite_mean(
            pd.to_numeric(occ.get("adaptive_80_high_c"), errors="coerce")
            - pd.to_numeric(occ.get("mean_operative_temp_c"), errors="coerce")
        )
        if {"adaptive_80_high_c", "mean_operative_temp_c"}.issubset(occ.columns)
        else float("nan"),
        "mean_adaptive_warm_slack_90_c": finite_mean(
            pd.to_numeric(occ.get("adaptive_90_high_c"), errors="coerce")
            - pd.to_numeric(occ.get("mean_operative_temp_c"), errors="coerce")
        )
        if {"adaptive_90_high_c", "mean_operative_temp_c"}.issubset(occ.columns)
        else float("nan"),
        "mean_p_tail": finite_mean(p_disc),
        "p95_p_tail": finite_quantile(p_disc, 0.95),
        "p_tail_ge_0_20_pct": pct(p_disc >= P_TAIL_SCREEN),
        "mean_warm_tail": finite_mean(warm_tail),
        "mean_cold_tail": finite_mean(cold_tail),
        "cooling_sp_mean_c": finite_mean(occ["cooling_setpoint_c"]),
        "cooling_sp_max_c": finite_quantile(occ["cooling_setpoint_c"], 1.0),
        "heating_sp_mean_c": finite_mean(occ["heating_setpoint_c"]),
        "heating_sp_min_c": finite_quantile(occ["heating_setpoint_c"], 0.0),
        "warm_requests_pct": pct(request_dir > 0),
        "cold_requests_pct": pct(request_dir < 0),
        "no_risk_requests_pct": pct(request_dir == 0),
        "warm_protection_actions": int((action_dir == 1).sum()),
        "cold_protection_actions": int((action_dir == -1).sum()),
        "energy_relax_actions": int((action_dir == 2).sum()),
        "guard_hold_records": int((hold_guard == 1).sum()),
        "idle_actions": int((action_dir == 0).sum()),
        "dominant_request_source": (
            str(request_source.value_counts().idxmax()) if not request_source.empty else ""
        ),
    }
    if max_zone_disc is not None:
        out["max_zone_p_tail_ge_0_20_pct"] = pct(max_zone_disc >= P_TAIL_SCREEN)
        out["max_zone_mean_p_tail"] = finite_mean(max_zone_disc)
    else:
        out["max_zone_p_tail_ge_0_20_pct"] = float("nan")
        out["max_zone_mean_p_tail"] = float("nan")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.trace, low_memory=False)
    group_cols = ["strategy"]
    if "weather" in df.columns:
        group_cols = ["weather", "strategy"]
    if "paperb_met" in df.columns:
        group_cols = [col for col in group_cols if col != "strategy"] + [
            "paperb_met",
            "strategy",
        ]
    if "paperb_people_activity_w_per_person" in df.columns:
        group_cols = [col for col in group_cols if col != "strategy"] + [
            "paperb_people_activity_w_per_person",
            "strategy",
        ]
    rows = [summarize_strategy(group) for _, group in df.groupby(group_cols, sort=True)]
    summary = pd.DataFrame(rows)
    if not summary.empty and "reference" in set(summary["strategy"]):
        key_cols = [
            col
            for col in [
                "weather",
                "paperb_met",
                "paperb_people_activity_w_per_person",
            ]
            if col in summary.columns and summary[col].notna().any()
        ]
        if key_cols:
            ref_lookup = (
                summary.loc[
                    summary["strategy"] == "reference",
                    key_cols + ["electricity_kwh"],
                ]
                .set_index(key_cols)["electricity_kwh"]
                .to_dict()
            )
            summary["electricity_change_vs_reference_pct"] = summary.apply(
                lambda row: (
                    (
                        row["electricity_kwh"]
                        / ref_lookup[tuple(row[col] for col in key_cols)]
                        - 1.0
                    )
                    * 100.0
                    if tuple(row[col] for col in key_cols) in ref_lookup
                    and np.isfinite(ref_lookup[tuple(row[col] for col in key_cols)])
                    and ref_lookup[tuple(row[col] for col in key_cols)] > 0
                    else float("nan")
                ),
                axis=1,
            )
        elif "weather" in summary.columns:
            ref_lookup = (
                summary.loc[summary["strategy"] == "reference", ["weather", "electricity_kwh"]]
                .set_index("weather")["electricity_kwh"]
                .to_dict()
            )
            summary["electricity_change_vs_reference_pct"] = summary.apply(
                lambda row: (
                    (row["electricity_kwh"] / ref_lookup[row["weather"]] - 1.0) * 100.0
                    if row["weather"] in ref_lookup
                    and np.isfinite(ref_lookup[row["weather"]])
                    and ref_lookup[row["weather"]] > 0
                    else float("nan")
                ),
                axis=1,
            )
        else:
            reference_e = float(
                summary.loc[summary["strategy"] == "reference", "electricity_kwh"].iloc[0]
            )
            if np.isfinite(reference_e) and reference_e > 0:
                summary["electricity_change_vs_reference_pct"] = (
                    (summary["electricity_kwh"] / reference_e - 1.0) * 100.0
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"[summary] wrote {args.output}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
