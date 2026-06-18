#!/usr/bin/env python3
"""Run PPD guard threshold sweep for the Guangzhou seed case."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


CONTROL_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CONTROL_DIR.parents[1]
RUNNER = CONTROL_DIR / "scripts" / "run_medium_office_paperB_control.py"
SUMMARY = CONTROL_DIR / "scripts" / "summarize_closed_loop_smoke.py"
MODEL_SOURCE = (
    WORKSPACE_ROOT
    / "Probabilities_ENB/paperA_rebuild/runs/diagnostic_reference_zone_raw_full/models/control_predictors.joblib"
)
WEATHER = (
    WORKSPACE_ROOT
    / "HPH_Carbon_Entitlement/weather/cmip_selected_years/ssp585/Guangzhou/"
    / "guangzhou_ssp585_mid_2050s_heatwave_extreme_2059.epw"
)


def run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=CONTROL_DIR / "runs" / "guangzhou_ppd_guard_sweep",
    )
    parser.add_argument("--hold", type=float, default=10.0)
    parser.add_argument("--protect", nargs="+", type=float, default=[20.0, 25.0, 50.0])
    args = parser.parse_args()

    for threshold in args.protect:
        label = str(threshold).replace(".", "p")
        out_dir = args.output_root / f"ppd_protect_{label}"
        model_dir = out_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(MODEL_SOURCE, model_dir / "control_predictors.joblib")
        run(
            [
                sys.executable,
                str(RUNNER),
                "--output-dir",
                str(out_dir),
                "--weather",
                str(WEATHER),
                "--begin-month",
                "7",
                "--begin-day",
                "1",
                "--end-month",
                "7",
                "--end-day",
                "14",
                "--strategies",
                "reference",
                "diagnostic_reference",
                "paperb_pmv_relax",
                "paperb_pmv_exceedance_guard_relax",
                "paperb_ppd_guard_relax",
                "paperb_gate_tail_asym_relax",
                "--skip-train",
                "--skip-plot",
                "--paperb-save-heat-c",
                "20.0",
                "--paperb-save-cool-c",
                "26.0",
                "--paperb-warm-protect-cool-c",
                "23.25",
                "--paperb-cold-protect-heat-c",
                "23.25",
                "--paperb-tighten-dwell-steps",
                "4",
                "--paperb-relax-dwell-steps",
                "1",
                "--paperb-pmv-threshold",
                "0.5",
                "--paperb-pmv-extreme-threshold",
                "1.0",
                "--paperb-ppd-hold-threshold",
                str(args.hold),
                "--paperb-ppd-protect-threshold",
                str(threshold),
            ]
        )
        run(
            [
                sys.executable,
                str(SUMMARY),
                "--trace",
                str(out_dir / "traces" / "medium_office_control_traces.csv"),
                "--output",
                str(out_dir / "summary" / "paperb_closed_loop_guard_summary.csv"),
            ]
        )

    print(f"[done] PPD guard sweep outputs: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
