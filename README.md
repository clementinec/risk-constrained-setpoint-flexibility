# Risk-Constrained Setpoint Flexibility in Office Buildings Under Future Weather

This repository contains the processed reproducibility package for the study
`Risk-Constrained Setpoint Flexibility in Office Buildings Under Future Weather`.

The study compares relaxation-first HVAC controller guards under the same DOE Medium
Office model, weather files, actuator bounds, and dwell rules. The retained outputs
support the reported 144-weather-case panel and 1008 annual controller simulations.

## Contents

- `scripts/`: analysis, EnergyPlus orchestration, and figure-generation scripts.
- `diagnostics/`: weather manifests, controller sweep diagnostics, and intermediate
  summary tables.
- `results/annual_summaries/`: annual summary CSV files from the full controller panel
  and TSV-tail threshold sensitivity sweeps.
- `results/stress_window_summaries/`: summary CSV files for the three-week Guangzhou
  stress-window process checks.
- `asset_write/figs/`: manuscript figure files.
- `asset_write/tables/`: manuscript source tables.
- `CHECKSUMS_SHA256.txt`: SHA-256 checksums for retained scripts, processed outputs,
  figures, and tables.

## Not Included

Full EnergyPlus run directories are not redistributed because they are large and mostly
contain intermediate simulator output. The repository instead provides processed annual
summary tables, stress-window summary tables, weather manifests, scripts, and checksums
for auditability. Timestep-level controller traces are not redistributed.

Manuscript source files, submission PDFs, cover letters, and Editorial Manager upload
text are not included in this repository. The repository is limited to reproducibility
assets rather than submission documents.

The ASHRAE Global Thermal Comfort Database II is not redistributed because of licensing
constraints. Scripts expect the user to provide licensed access to the required comfort
database and local EnergyPlus/weather-file paths.

## Main Reproducibility Entry Points

- Full annual matrix runner: `scripts/run_full_matrix_shards.py`
- Controller simulation engine: `scripts/run_medium_office_paperB_control.py`
- Main figure builder: `scripts/build_paperb_figures.py`
- Stress-window state-map builder: `scripts/build_stress_window_state_map.py`
- Stress-window fingerprint builder: `scripts/build_stress_window_fingerprint.py`

The key processed table for the TSV-tail threshold sweep is:

`asset_write/tables/tail_threshold_sensitivity.csv`
