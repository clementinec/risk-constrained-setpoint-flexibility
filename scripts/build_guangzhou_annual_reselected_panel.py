#!/usr/bin/env python3
"""Build the re-selected Guangzhou annual collapse panel.

The Paper B annual collapse check uses one typical year and one heat-stress
year per decade. The stress year follows the existing CMIP weather selector:
heatwave_extreme is the year with maximum annual hours at or above 35 C, with
annual maximum outdoor temperature as tie-breaker. If hot and heatwave_extreme
select the same year, the year is retained once and marked as a shared stress
selector.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEATHER_PANEL = ROOT / "HPH_Carbon_Entitlement" / "data" / "interim" / "cmip_weather_panel.csv"
DEFAULT_SUMMARY = (
    ROOT
    / "Probabilities_ENB"
    / "paperB_control"
    / "diagnostics"
    / "guangzhou_ssp585_annual_collapse_summary.csv"
)
DEFAULT_OUT_DIR = ROOT / "Probabilities_ENB" / "paperB_control" / "diagnostics"

TIME_ORDER = {
    "baseline_2020s": 0,
    "near_2030s": 1,
    "mid_2050s": 2,
    "late_2080s": 3,
}
PAPERB_STRATEGIES = [
    "reference",
    "diagnostic_reference",
    "paperb_adaptive_band_relax",
    "paperb_pmv_exceedance_guard_relax",
    "paperb_ppd_guard_relax",
    "paperb_mu_relax",
    "paperb_gate_tail_asym_relax",
]


def normalize_weather_name(city: str, scenario: str, time_slice: str, severity: str, year: int) -> str:
    return f"{city.lower()}_{scenario}_{time_slice}_{severity}_{year}"


def build_manifest(weather_panel: pd.DataFrame, city: str, scenario: str) -> pd.DataFrame:
    panel = weather_panel[
        (weather_panel["city"].eq(city)) & (weather_panel["scenario_raw"].eq(scenario))
    ].copy()
    if panel.empty:
        raise RuntimeError(f"No weather-panel rows for {city=} and {scenario=}")

    rows: list[dict[str, object]] = []
    for time_slice, group in panel.groupby("time_slice", sort=False):
        typical = group[group["severity"].eq("typical")]
        heatwave = group[group["severity"].eq("heatwave_extreme")]
        hot = group[group["severity"].eq("hot")]
        if typical.empty or heatwave.empty:
            raise RuntimeError(f"Missing typical or heatwave row for {time_slice}")

        typical_row = typical.iloc[0]
        heatwave_row = heatwave.iloc[0]
        hot_row = hot.iloc[0] if not hot.empty else None

        for label, source_row in [
            ("typical", typical_row),
            ("stress", heatwave_row),
        ]:
            year = int(source_row["weather_year"])
            severity = str(source_row["severity"])
            selector = str(source_row["selector"])
            if label == "stress" and hot_row is not None and int(hot_row["weather_year"]) == year:
                selector = "shared_max_CDD18_and_max_hours_temp_ge_35"
            rows.append(
                {
                    "panel_role": label,
                    "country": source_row["country"],
                    "city": source_row["city"],
                    "scenario": source_row["scenario_raw"],
                    "scenario_label": source_row["scenario"],
                    "time_slice": source_row["time_slice"],
                    "year_window": source_row["year_window"],
                    "selected_weather_year": year,
                    "source_severity": severity,
                    "paperb_weather": normalize_weather_name(
                        str(source_row["city"]),
                        str(source_row["scenario_raw"]),
                        str(source_row["time_slice"]),
                        severity,
                        year,
                    ),
                    "mean_T_out": float(source_row["mean_T_out"]),
                    "CDD18_hourly": float(source_row["CDD18_hourly"]),
                    "max_T_out": float(source_row["max_T_out"]),
                    "hours_temp_ge_35": int(source_row["hours_temp_ge_35"]),
                    "selector": selector,
                    "hot_year_same_as_stress": bool(
                        label == "stress" and hot_row is not None and int(hot_row["weather_year"]) == year
                    ),
                }
            )

    manifest = pd.DataFrame(rows)
    manifest["time_order"] = manifest["time_slice"].map(TIME_ORDER)
    manifest["role_order"] = manifest["panel_role"].map({"typical": 0, "stress": 1})
    return manifest.sort_values(["time_order", "role_order"]).reset_index(drop=True)


def build_outputs(manifest: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> tuple[Path, Path, Path]:
    missing = sorted(set(manifest["paperb_weather"]) - set(summary["weather"]))
    if missing:
        raise RuntimeError("Annual controller summary is missing selected weather rows: " + ", ".join(missing))

    selected = summary[
        summary["weather"].isin(manifest["paperb_weather"])
        & summary["strategy"].isin(PAPERB_STRATEGIES)
    ].copy()
    selected = selected.merge(
        manifest[
            [
                "paperb_weather",
                "panel_role",
                "time_slice",
                "year_window",
                "selected_weather_year",
                "source_severity",
                "selector",
                "hot_year_same_as_stress",
                "mean_T_out",
                "CDD18_hourly",
                "max_T_out",
                "hours_temp_ge_35",
                "time_order",
                "role_order",
            ]
        ],
        left_on="weather",
        right_on="paperb_weather",
        how="left",
    )
    selected = selected.sort_values(["time_order", "role_order", "strategy"]).reset_index(drop=True)

    diag_cols = [
        "paperb_weather",
        "panel_role",
        "time_slice",
        "selected_weather_year",
        "source_severity",
        "selector",
        "hot_year_same_as_stress",
        "mean_T_out",
        "CDD18_hourly",
        "max_T_out",
        "hours_temp_ge_35",
        "mean_p_tail_occ",
        "mean_zone_p_tail_ge_0p20_pct_occ",
        "max_zone_p_tail_ge_0p20_pct_occ",
        "pmv_abs_gt_0p5_pct_occ",
        "pmv_abs_gt_1p0_pct_occ",
        "reference_electricity_kwh",
    ]
    diag = selected[selected["strategy"].eq("diagnostic_reference")][diag_cols].copy()
    diag = diag.rename(columns={col: f"diagnostic_{col}" for col in diag_cols if col not in {"paperb_weather", "panel_role", "time_slice", "selected_weather_year", "source_severity", "selector", "hot_year_same_as_stress", "mean_T_out", "CDD18_hourly", "max_T_out", "hours_temp_ge_35", "reference_electricity_kwh"}})

    wide = selected.pivot_table(
        index="paperb_weather",
        columns="strategy",
        values="electricity_savings_pct_vs_reference",
        aggfunc="first",
    )
    wide.columns = [f"savings_pct_{col}" for col in wide.columns]
    compact = diag.merge(wide.reset_index(), on="paperb_weather", how="left")
    compact = compact.merge(
        manifest[["paperb_weather", "time_order", "role_order"]],
        on="paperb_weather",
        how="left",
    ).sort_values(["time_order", "role_order"]).drop(columns=["time_order", "role_order"])

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "guangzhou_ssp585_annual_reselected_8case_manifest.csv"
    selected_path = out_dir / "guangzhou_ssp585_annual_reselected_8case_summary.csv"
    compact_path = out_dir / "guangzhou_ssp585_annual_reselected_8case_compact.csv"
    manifest.drop(columns=["time_order", "role_order"]).to_csv(manifest_path, index=False)
    selected.to_csv(selected_path, index=False)
    compact.to_csv(compact_path, index=False)
    return manifest_path, selected_path, compact_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weather-panel", type=Path, default=DEFAULT_WEATHER_PANEL)
    parser.add_argument("--annual-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--city", default="Guangzhou")
    parser.add_argument("--scenario", default="ssp585")
    args = parser.parse_args()

    weather_panel = pd.read_csv(args.weather_panel)
    summary = pd.read_csv(args.annual_summary)
    manifest = build_manifest(weather_panel, args.city, args.scenario)
    paths = build_outputs(manifest, summary, args.out_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
