# D13 spatial-sentinel trace diagnostic

D13 is a retained-output diagnostic over the 24 accepted learned-p90 annual
scenarios. It adds no EnergyPlus run. At every occupied 15-minute record it
reconstructs the actual zone nearest the linear spatial 0.90 quantile of the 15
zone learned-tail probabilities, using the accepted zone order for the first-index
tie break.

The contemporaneous warmest zone is the zone with maximum operative temperature,
`(air temperature + mean radiant temperature) / 2`, at that record. The
retrospective worst warm-exposure zone is the single zone with the largest
full-year occupied operative-temperature degree-hours above 28 C in that
scenario; it was not available to the supervisor. A switch opportunity is an
adjacent pair of 15-minute records with both records occupied. An unoccupied gap
or selected-identity change ends a persistence episode.

## Published evidence

- `../../summary_outputs/d13_input_manifest.csv`: 24 path-relative source
  identities and trace/result hashes. The referenced source artifacts are not
  redistributed.
- `../../summary_outputs/d13_scenario_summary.csv`: one row per learned-p90
  scenario.
- `../../summary_outputs/d13_group_summary.csv`: pooled, selector-role, and SSP
  summaries.
- `../../summary_outputs/d13_zone_frequency.csv`: 15-zone selection frequencies
  by scenario and over all scenarios.
- `../../summary_outputs/d13_reconstruction_validation.csv`: exact parity-trace
  selector reconstruction record.
- `D13_SPATIAL_TRACE_REPORT.md`: human-readable definitions and findings.
- `D13_PUBLIC_PROVENANCE.json`: public-output hashes, upstream closure identities,
  scope, gates, and exclusions.

All public tables are exact CSV copies of the validated source diagnostic. The
source archive also retained Zstandard-Parquet pairs, but they are omitted here
because the CSVs are compact and byte-bound to the public provenance.

## Verify the compact package

From the repository root:

```sh
python3 revision_01/scripts/diagnostics/build_d13_spatial_trace.py \
  --verify-package
```

This mode needs no raw trace. It checks the published hashes, coverage, selector
parity record, key pooled counts, and interpretation denominators.

## Rebuild from the complete review archive

The timestep traces remain outside this lightweight repository. If the separately
retained `paperB_ENB_Rebuild` review archive is available, rebuild to a new
directory with:

```sh
python3 revision_01/scripts/diagnostics/build_d13_spatial_trace.py \
  --review-root /path/to/paperB_ENB_Rebuild \
  --output-dir /tmp/paperb_d13_rebuild
```

The script fails closed on the accepted 96-cell matrix, Fresh96 closure, accepted
runner, all 24 learned-p90 trace and cell-result hashes, trace shapes, learned-tail
identities, reconstructed warm-protection branches, and the independent parity
trace. A successful rebuild must be byte-identical to all six published outputs.

## Interpretation boundary

The selected identity is a dynamic high-tail spatial sentinel. Its stronger
top-three-warmest concordance during warm protection documents thermal alignment,
but it is not a direct warmest-zone selector, a static retrospective worst-zone
selector, or independently zonal actuation. The branch request still drives
synchronized building-level setpoint schedules.
