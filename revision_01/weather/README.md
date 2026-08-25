# Weather archive for Revision 01

This directory contains the exact 24 EnergyPlus Weather (`.epw`) files used by
the accepted controller panel and by all 136 Revision 01 scientific cells. The
files are preserved byte-for-byte from the frozen `rev01_core64` run inputs.
Their full-file SHA-256 hashes match the hashes embedded in the accepted run
matrices and frozen batch closure.

## Contents

- `files/`: native EPW files, arranged by SSP and city.
- `weather_manifest.csv`: one path-free row per EPW, including the simulation
  role, selected year, record count, byte count, full-file hash, retained
  weather-payload hash, model lineage, and source-data DOI.
- `provenance/source_lineage_manifest.csv`: path-free identifiers and hashes for
  the 12 preserved city--scenario source workbooks and hourly forecast tables.
- `provenance/climate_delta_metadata_48.csv`: the 48 recovered variable-level
  daily-delta records.
- `WEATHER_DATA_LICENSE.md`: licensing and attribution requirements.

## Construction and selection

The archived metadata identifies one global climate model,
`MPI-ESM1-2-LR`, no regional climate model (`RCM=N/A`), a 1991--2010
observational reference period, and a daily change-factor construction.
Temperature and pressure changes are additive; surface-wind and global
horizontal irradiance changes are multiplicative. The recorded observational
baseline filenames belong to a locally prepared `ISD_complete_solar` family.

For each city and SSP, the present-typical file is the 2025--2029 candidate
nearest the median annual cooling-degree-hour accumulation above 18 C. The
late-century-hot-extreme file is the 2080--2089 candidate with the greatest
number of hourly dry-bulb records at or above 35 C, with annual maximum
temperature and cooling-degree-hour accumulation used as tie-breakers.
Consequently, these are two deliberately different selector roles in a weather
stress grid, not a factorial isolation of a future-climate effect.

Four selected source years are leap years and retain 8,784 EPW records. The
paper's fixed 365-day EnergyPlus RunPeriod omits 29 February and evaluates
8,760 formal hours in those cases.

## Provenance boundary

The preserved workbooks recover the GCM, SSP, reference period, daily-delta
mode, observational-source filename, and four variable transforms. They do not
recover the exact CMIP6 member/variant, native grid, upstream dataset version,
exact station identifiers, independent solar-completion lineage, or a complete
field-by-field generation specification. Those fields are marked as not
recovered rather than inferred. The EPWs nevertheless make the exact downstream
weather inputs used in the reported simulations directly auditable and reusable.

## Source citations

Weather-generation method:

- Hongshan Guo and Kanxuan He (2026), *Scenario-Conditioned Actual
  Meteorological Years (sAMY): A Stochastic Weather Generator Using
  Multi-Decadal Observations*, Energy and Buildings, 117508.
  https://doi.org/10.1016/j.enbuild.2026.117508
- Hongshan Guo and Kanxuan He (2026), *Input quality, not statistical
  complexity, determines climate-adapted weather file fidelity: A causal
  decomposition of degree-day errors*, Energy, 353, 140867.
  https://doi.org/10.1016/j.energy.2026.140867

Climate-model inputs:

- MPI-M MPI-ESM1.2-LR ScenarioMIP SSP2-4.5:
  https://doi.org/10.22033/ESGF/CMIP6.6693
- MPI-M MPI-ESM1.2-LR ScenarioMIP SSP5-8.5:
  https://doi.org/10.22033/ESGF/CMIP6.6705

Recorded observational baseline family:

- NOAA/NCEI Integrated Surface Database:
  https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database

## Integrity check

From the repository root, verify the complete Revision 01 addendum with:

```sh
shasum -a 256 -c revision_01/CHECKSUMS_SHA256.txt
```

The test suite also checks that the weather manifest contains exactly 24 unique
files and that every manifest size and SHA-256 value matches the archived EPW.
