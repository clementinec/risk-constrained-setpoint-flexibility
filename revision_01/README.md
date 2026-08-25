# Revision 01 reproducibility addendum

This directory is the lightweight reproducibility addendum for the first
revision of *Delivered-Energy and Operative-Temperature Trade-offs for
Learned-Probability-Informed HVAC Supervision in a Weather Stress Grid*. It
records the matched supervisory-policy tests added in response to peer review.

The revision executed 136 new annual EnergyPlus cells: 64 core cells (M1--M3),
24 adaptive running-mean clamp cells (C1), and 48 full-grid learned-contribution
cells (M4--M5). All 136 passed the frozen technical and output-validation gates.
One separate annual submitted-settings parity sentinel also passed; it is not
counted among the scientific cells.

## Experiments

| ID | New policy or perturbation | Matched comparator | Cells | Question |
|---|---|---|---:|---|
| M1 | spatial p90 absolute-PMV policy | building-mean PMV policy | 24 | Effect of spatial selection with the PMV signal and otherwise matched rules |
| M2 | eight one-factor learned-p90 parameter variants | submitted learned-p90 policy | 32 | Local sensitivity to spatial quantile, probability thresholds, and relaxation bounds |
| M3 | learned-signal strengths `alpha=0.5` and `alpha=0` | submitted learned-p90 policy | 8 | Four-sentinel signal-dependence screen |
| C1 | adaptive running mean clamped to 10--33.5 C | unclamped adaptive-band benchmark | 24 | Sensitivity to applying the adaptive equation outside its stated running-mean range |
| M4 | building-mean learned tail probabilities | spatial p90 learned-tail policy | 24 | Incremental effect of p90 spatial selection with the same learned probability signal |
| M5 | neutral probability signal (`alpha=0`) with the p90 decision rules | submitted learned-p90 policy | 24 | Dependence of annual outcomes on the learned probabilities |

The weather cases are a stress-test grid. Present-typical and late-century
hot-extreme years were selected by different rules, so their contrast is not a
causal estimate of a future-climate effect.

## Directory map

- `configs/`: exact frozen variant manifests. They contain no local paths.
- `scripts/runner/`: exact Rev01 EnergyPlus runner and fail-closed policy helper.
- `scripts/automation/`: exact plan compilers, batch launcher, parity validator,
  and size-efficient CSV-to-Zstandard-Parquet storage workflow.
- `scripts/validation/`: exact accepted-cell output validator used by the batches.
- `scripts/synthesis/`: exact reviewer-evidence and learned-contribution synthesis
  scripts.
- `summary_outputs/`: compact CSV evidence only: 136 matched cell rows, grouped
  summaries, M4/M5 branch transitions, metric definitions, and reviewer mapping.
- `weather/`: the exact 24 native EPW inputs, a path-free weather manifest,
  recovered source-lineage and daily-delta metadata, and the weather-data
  licence and citations.
- `RUN_INVENTORY.csv`: human-readable M1--M5/C1 execution inventory.
- `RUN_PROVENANCE.json`: portable hashes and closure status without workstation
  paths.
- `REVISION_NOTES.md`: main findings and interpretation boundaries.
- `CHECKSUMS_SHA256.txt`: hashes of every other retained file in this addendum.

## Reproduction inputs and exclusions

The package now includes the exact 24 EPWs used by the accepted panel and every
Revision 01 rerun. `weather/weather_manifest.csv` gives their path-free scenario
roles, sizes, full-file hashes, retained payload hashes, and recovered lineage.
The broader 2025--2100 forecast tables and source workbooks are not redistributed;
their path-free filenames and hashes are retained in
`weather/provenance/source_lineage_manifest.csv`.

The package does not include EnergyPlus working directories, SQL/EIO files,
timestep traces, the trained model binary, the accepted Medium Office IDF, the
third-party thermal-comfort database, or local-path-bearing frozen run matrices
and closures. The model, IDF, and accepted 96-cell comparator panel must still be
supplied separately. Their hashes must match `RUN_PROVENANCE.json`.

The exact scripts retain generic macOS defaults for EnergyPlus 25.1 and temporary
cache directories. For another installation, provide all CLI paths explicitly.
The plan compilers create machine-specific matrices from the accepted comparator
matrix; these generated matrices should not be committed because they record
local input paths.

## Compact evidence entry points

- `summary_outputs/matched_experiment_variant_summary.csv`: grouped results for
  every M1--M5/C1 contrast and stress-test stratum.
- `summary_outputs/matched_cell_metrics.csv`: one row for each of the 136 matched
  scientific cells.
- `summary_outputs/learned_contribution_summary.csv`: M4/M5 aggregate signal and
  spatial-selection results.
- `summary_outputs/learned_contribution_cell_metrics.csv`: 48 scenario-level M4/M5
  contrasts.
- `summary_outputs/feedback_branch_outcome_transitions.csv`: M4/M5 branch and
  action-outcome transition counts.

Sign conventions differ deliberately between two entry points. The all-experiment
`matched_experiment_variant_summary.csv` records each newly run variant minus its
accepted comparator. The manuscript-facing `learned_contribution_summary.csv` reports
M4/M5 as learned-p90 minus learned building mean or neutral null. `REVISION_NOTES.md`
enumerates the conventions for M1--M5/C1.

Delivered site energy means facility electricity plus `NaturalGas:Facility`.
It is not primary/source energy, carbon, cost, or emissions. Operative-temperature
degree-hours are descriptive warm- and cold-side exposure diagnostics, not
observed comfort, satisfaction, health outcomes, or validated safety thresholds.
