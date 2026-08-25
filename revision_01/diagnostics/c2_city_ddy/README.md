# C2 city-design-condition sensitivity

C2 is a predeclared 12-cell sensitivity for Reviewer 3 comment 3. It replaces
only the accepted Denver `Site:Location` and two annual sizing-design-day
objects with the geographically matched companion objects for Beijing,
Guangzhou, and Phoenix. The annual EPW, Medium Office model, HVAC topology,
controller code and settings, and four policy labels are otherwise held fixed.

The scientific batch `REV01-C2-012-20260825-v2` completed 12/12 cells. Each
city contains fixed, PMV, adaptive-band, and learned-p90 policies. These 12
cells supplement the 136 M1--M5/C1 revision cells, giving 148 new scientific
EnergyPlus cells in Revision 01. The 12 accepted-Denver comparator cells were
reused as hash-pinned evidence and were not rerun or counted again.

## Public artifacts

- `C2_DDY_PROVENANCE.json` records source URLs, retrieval times, byte sizes,
  SHA-256 identities, selected design-day metadata, and semantic-diff gates.
  It does not redistribute the source DDYs or derived IDFs.
- `../../configs/c2_city_ddy_public_matrix.csv` is the 12-row frozen matrix
  with the two workstation path columns removed. Hashes and all scientific
  bindings are retained.
- `../../configs/variant_manifest_c2_city_ddy.json` is the exact path-free
  variant manifest.
- `../../summary_outputs/c2_*.csv` contains absolute endpoints, matched
  sizing substitutions, within-contract policy-minus-fixed contrasts,
  differential-of-contrasts, and policy rankings.
- `C2_SYNTHESIS_REPORT.md` is the full human-readable report.
- `C2_SYNTHESIS_CLOSURE.json` is the exact, path-sanitized scientific closure
  produced by the frozen synthesizer; `C2_PUBLIC_PROVENANCE.json` binds the
  retained public subset to the plan, batch, freeze, and synthesis hashes.
- The path-sanitized compiler and exact C2-enabled fail-closed batch launcher
  and synthesizer are under `../../scripts/`. The compiler requires explicit
  paths to excluded inputs. The launcher/synthesizer retain archive-relative
  defaults from the executed code; pass explicit CLI paths in another
  installation. `C2_PUBLIC_PROVENANCE.json` records both the executed compiler
  hash and the sanitized public compiler hash.

The retained CSVs use the synthesizer's 12-significant-digit serialization.
An independent comparison with the canonical internal Parquet tables found a
maximum absolute difference of 4.99e-7 and maximum relative difference of
4.68e-12, with no sign or ranking changes. CSV is retained here because these
wide, small-row tables are more compact in CSV than Parquet.

## Headline result and boundary

The city design conditions materially changed capacities and absolute warm-side
outcomes in the hot-climate sentinels. Nevertheless, the sign of every active
policy's annual delivered-site-energy contrast versus fixed remained unchanged
across the nine city-policy comparisons, and so did the sign of every annual
degree-hour-above-28 C contrast. Five of nine 336-hour degree-hour-above-28 C
contrast directions changed, showing that event-window thermal rankings are
more sizing-sensitive than the annual headline.

This is a representative current city-design-condition sensitivity, not a
future-climate resizing experiment, an additional building/HVAC/GCM test, or
proof of simultaneous equipment-component saturation. Setpoint-not-met hours
and degree-hours are mismatch diagnostics only. Delivered site energy is
facility electricity plus `NaturalGas:Facility`; the gas meter includes service
water heating. Operative-temperature degree-hours are descriptive physical
exposure diagnostics, not observed satisfaction, health, or safety outcomes.

## Excluded payloads

The public addendum intentionally excludes raw DDYs, derived/private IDFs,
local-path-bearing frozen matrices and closures, EnergyPlus working folders,
SQL/EIO files, timestep traces, and third-party licensed input payloads.
