#!/usr/bin/env python3
"""Build the Paper B full-weather manifest from the CMIP weather panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PANEL = ROOT / "HPH_Carbon_Entitlement" / "data" / "interim" / "cmip_weather_panel.csv"
DEFAULT_EPW_ROOT = ROOT / "HPH_Carbon_Entitlement" / "weather" / "cmip_selected_years"
DEFAULT_OUT = (
    ROOT
    / "Probabilities_ENB"
    / "paperB_control"
    / "diagnostics"
    / "paperb_full_144_weather_manifest.csv"
)


def weather_stem(city: str, scenario: str, time_slice: str, severity: str, year: int) -> str:
    return f"{city.lower()}_{scenario}_{time_slice}_{severity}_{year}"


def build_manifest(panel: pd.DataFrame, epw_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in panel.to_dict("records"):
        city = str(row["city"])
        scenario = str(row["scenario_raw"])
        time_slice = str(row["time_slice"])
        severity = str(row["severity"])
        year = int(row["weather_year"])
        stem = weather_stem(city, scenario, time_slice, severity, year)
        epw_path = epw_root / scenario / city / f"{stem}.epw"
        rows.append(
            {
                "country": row["country"],
                "city": city,
                "scenario": scenario,
                "scenario_label": row["scenario"],
                "time_slice": time_slice,
                "year_window": row["year_window"],
                "severity": severity,
                "weather_year": year,
                "weather_stem": stem,
                "epw_path": str(epw_path),
                "mean_T_out": row["mean_T_out"],
                "CDD18_hourly": row["CDD18_hourly"],
                "max_T_out": row["max_T_out"],
                "hours_temp_ge_35": row["hours_temp_ge_35"],
                "humidity_metric": row["humidity_metric"],
                "selector": row["selector"],
                "stage_smoke": False,
                "stage_typical": severity == "typical",
                "stage_full": True,
            }
        )
    manifest = pd.DataFrame(rows)
    missing = [path for path in manifest["epw_path"] if not Path(path).exists()]
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"Missing EPW files:\n{preview}")
    order = {
        "baseline_2020s": 0,
        "near_2030s": 1,
        "mid_2050s": 2,
        "late_2080s": 3,
    }
    severity_order = {"typical": 0, "hot": 1, "heatwave_extreme": 2}
    manifest["time_order"] = manifest["time_slice"].map(order)
    manifest["severity_order"] = manifest["severity"].map(severity_order)
    return manifest.sort_values(
        ["country", "city", "scenario", "time_order", "severity_order"]
    ).drop(columns=["time_order", "severity_order"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--epw-root", type=Path, default=DEFAULT_EPW_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    panel = pd.read_csv(args.panel)
    manifest = build_manifest(panel, args.epw_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out, index=False)
    print(f"[manifest] wrote {len(manifest)} rows: {args.out}")
    print(manifest.groupby(["country", "city", "scenario"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
