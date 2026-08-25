# Revision 01 reproducibility addendum

This directory is the lightweight reproducibility addendum for the first
revision of *Delivered-Energy and Operative-Temperature Trade-offs for
Learned-Probability-Informed HVAC Supervision in a Weather Stress Grid*. It
records the matched supervisory-policy tests and compact accepted-panel
diagnostics added in response to peer review, including the D13 retained-trace
characterization of the learned-p90 spatial sentinel and the C2
city-design-condition sizing sensitivity.

The revision executed 148 new annual EnergyPlus cells: 136 attribution and
robustness cells (64 M1--M3, 24 C1, and 48 M4--M5) plus 12 C2
city-design-condition cells. All 148 passed the frozen technical and
output-validation gates. One separate annual submitted-settings parity sentinel
also passed; it is not counted among the scientific cells.

## Experiments

| ID | New policy or perturbation | Matched comparator | Cells | Question |
|---|---|---|---:|---|
| M1 | spatial p90 absolute-PMV policy | building-mean PMV policy | 24 | Effect of spatial selection with the PMV signal and otherwise matched rules |
| M2 | eight one-factor learned-p90 parameter variants | submitted learned-p90 policy | 32 | Local sensitivity to spatial quantile, probability thresholds, and relaxation bounds |
| M3 | learned-signal strengths `alpha=0.5` and `alpha=0` | submitted learned-p90 policy | 8 | Four-sentinel signal-dependence screen |
| C1 | adaptive running mean clamped to 10--33.5 C | unclamped adaptive-band benchmark | 24 | Sensitivity to applying the adaptive equation outside its stated running-mean range |
| M4 | building-mean learned tail probabilities | spatial p90 learned-tail policy | 24 | Incremental effect of p90 spatial selection with the same learned probability signal |
| M5 | neutral probability signal (`alpha=0`) with the p90 decision rules | submitted learned-p90 policy | 24 | Dependence of annual outcomes on the learned probabilities |
| C2 | city-matched location and annual design-day objects for three representative cities | accepted Denver sizing contract under the same EPW and policy | 12 | Sensitivity of capacities and policy contrasts to the design-condition sizing contract |

The weather cases are a stress-test grid. Present-typical and late-century
hot-extreme years were selected by different rules, so their contrast is not a
causal estimate of a future-climate effect.

## Directory map

- `configs/`: exact frozen variant manifests and the path-free 12-row C2 public
  matrix metadata.
- `scripts/runner/`: exact Rev01 EnergyPlus runner and fail-closed policy helper.
- `scripts/automation/`: exact plan compilers, batch launchers (including the
  C2-enabled fail-closed launcher), parity validator, and size-efficient
  CSV-to-Zstandard-Parquet storage workflow.
- `scripts/validation/`: exact accepted-cell output validator used by the batches.
- `scripts/synthesis/`: exact reviewer-evidence, learned-contribution, and C2
  synthesis scripts.
- `scripts/diagnostics/`: portable D13 rebuild and compact-package verification
  script. A full rebuild requires the separately retained, hash-pinned traces.
- `scripts/figures/`: portable script for regenerating the supplemental
  common-scale annual thermal figure from packaged CSV inputs.
- `summary_outputs/`: compact CSV evidence only: 136 M1--M5/C1 matched rerun-cell
  rows, the C2 12-cell/12-comparator processed sensitivity tables,
  accepted-panel cold-side metrics and paired contrasts, common-scale annual
  thermal source data, grouped summaries, M4/M5 branch transitions, D13
  scenario/group/zone-frequency tables, metric definitions, and reviewer mapping.
- `diagnostics/d13_spatial_trace/`: D13 report, definitions, public provenance,
  exact output hashes, and raw-trace exclusion boundary.
- `diagnostics/c2_city_ddy/`: C2 source provenance, scientific/public closures,
  full report, definitions, and private/raw-payload exclusion boundary.
- `figures/`: the common-scale annual worst-zone degree-hour figure regenerated
  from the packaged D07 source and summary tables.
- `weather/`: the exact 24 native EPW inputs, a path-free weather manifest,
  recovered source-lineage and daily-delta metadata, and the weather-data
  licence and citations.
- `RUN_INVENTORY.csv`: human-readable M1--M5/C1/C2 execution inventory.
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
raw C2 DDYs or derived/private C2 IDFs, the third-party thermal-comfort database,
or local-path-bearing frozen run matrices and closures. The model, IDF, raw DDYs,
and accepted 96-cell comparator panel must still be supplied separately. Their
hashes must match the public provenance records.
The public D13 input manifest records path-relative trace and cell-result identities
and hashes, but does not redistribute those source artifacts.

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
- `summary_outputs/d03_primary_cold_cell_metrics.csv`: accepted 96-cell,
  four-policy cold-side degree-hour metrics at 18 and 16 C; these are derived
  diagnostics and do not add EnergyPlus runs.
- `summary_outputs/d03_primary_cold_paired_cells.csv`: 72 scenario-level
  learned-p90-minus-comparator cold-side contrasts.
- `summary_outputs/d03_primary_cold_policy_summary.csv` and
  `d03_primary_cold_paired_summary.csv`: policy-level and paired cold-side
  summaries used in the revised manuscript and Supplement.
- `summary_outputs/d07_common_scale_annual_dh28_source.csv`: 72 accepted-panel
  annual learned-p90-minus-comparator values displayed on one horizontal scale.
- `summary_outputs/d07_common_scale_annual_dh28_summary.csv`: the corresponding
  three-comparator summary.
- `scripts/figures/plot_common_scale_annual_dh28.py`: regenerates
  `figures/fig_s_common_scale_annual_dh28.pdf` using only those two packaged D07
  CSV files.
- `summary_outputs/d13_group_summary.csv`: pooled and weather-role/pathway D13
  identity, persistence, and thermal-concordance results.
- `summary_outputs/d13_scenario_summary.csv` and
  `d13_zone_frequency.csv`: scenario-level findings and all 15 zone-selection
  frequencies.
- `summary_outputs/d13_input_manifest.csv` and
  `d13_reconstruction_validation.csv`: hash-pinned source identities and the
  independent 16,704-record selector-parity result.
- `diagnostics/d13_spatial_trace/D13_SPATIAL_TRACE_REPORT.md`: complete D13
  definitions, counts, findings, and interpretation boundary.
- `scripts/diagnostics/build_d13_spatial_trace.py`: verifies the compact package
  without raw traces or rebuilds all D13 tables when the complete review archive
  is supplied.
- `summary_outputs/c2_cell_metrics.csv`: 24 absolute rows covering the 12 C2
  cells and their 12 accepted-Denver comparators.
- `summary_outputs/c2_matched_city_ddy_minus_denver.csv` and
  `c2_within_contract_policy_minus_fixed.csv`: sizing substitutions and
  within-contract policy effects for all predeclared endpoints.
- `summary_outputs/c2_differential_of_contrasts.csv` and the two `c2_policy_*`
  tables: attribution/ranking sensitivity to the sizing contract.
- `diagnostics/c2_city_ddy/C2_SYNTHESIS_REPORT.md`: complete C2 tables, findings,
  and interpretation boundaries; the sibling README maps code, metadata, and
  closure evidence.

From the repository root, regenerate the supplemental figure with:

```sh
python3 revision_01/scripts/figures/plot_common_scale_annual_dh28.py
```

The retained D07 PDF was generated with Matplotlib 3.9.4, which is pinned in
this addendum's `requirements.txt`. Byte-identical PDF regeneration is defined
under that canonical plotting dependency; other Matplotlib releases can render
the same data and geometry with different PDF object serialization.

Verify the compact D13 package without the excluded timestep traces with:

```sh
python3 revision_01/scripts/diagnostics/build_d13_spatial_trace.py \
  --verify-package
```

Sign conventions differ deliberately between two entry points. The all-experiment
`matched_experiment_variant_summary.csv` records each newly run variant minus its
accepted comparator. The manuscript-facing `learned_contribution_summary.csv` reports
M4/M5 as learned-p90 minus learned building mean or neutral null. C2 reports
city-design-condition minus accepted-Denver substitutions and separately reports
policy-minus-fixed effects within each sizing contract. `REVISION_NOTES.md`
enumerates the conventions for M1--M5/C1/C2.

Delivered site energy means facility electricity plus `NaturalGas:Facility`.
It is not primary/source energy, carbon, cost, or emissions. Operative-temperature
degree-hours are descriptive warm- and cold-side exposure diagnostics, not
observed comfort, satisfaction, health outcomes, or validated safety thresholds.
