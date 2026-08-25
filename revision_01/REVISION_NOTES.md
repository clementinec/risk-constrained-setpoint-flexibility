# Revision 01 run findings

Date: 2026-08-25

## Execution closure

- Core attribution and parameter batch: 64/64 PASS (M1: 24; M2: 32; M3: 8).
- Adaptive running-mean clamp batch: 24/24 PASS (C1).
- Full-grid learned-contribution batch: 48/48 PASS (M4: 24; M5: 24).
- City-design-condition sizing sensitivity: 12/12 PASS (C2), using three
  representative cities and all four primary policies.
- Total scientific coverage: 148/148 PASS, comprising 136 attribution/robustness
  cells plus 12 C2 sizing-sensitivity cells, with zero failed cells.
- Separate submitted-settings annual parity sentinel: PASS.
- Exact selected-weather archive: 24/24 native EPWs published with matching
  accepted-run SHA-256 values, path-free selector metadata, recovered source
  lineage, daily-delta metadata, citations, and a CC BY 4.0 data notice.
- Accepted-panel cold-side diagnostic D03: 96/96 hash-pinned traces reproduced,
  covering all 24 anchors and the four primary policies; CSV/Parquet parity and
  independent checks against the active-policy synthesis both passed. This
  diagnostic reuses accepted outputs and adds no EnergyPlus runs.
- Common-scale annual thermal diagnostic D07: 72/72 primary annual contrasts
  reproduced the published three-comparator means and generated the packaged
  supplemental PDF from inspectable CSV source data.
- Spatial-sentinel trace diagnostic D13: all 24 hash-pinned learned-p90 traces and
  400,896 occupied records passed selector, trace-shape, learned-tail-identity,
  warm-branch, and parity-reconstruction gates. D13 reuses accepted outputs and
  adds no EnergyPlus run; raw traces remain outside the public addendum.
- City-design-condition sensitivity C2: 12/12 cells passed the frozen plan,
  execution, runtime-location/design-day, output, and postprocessing gates. The
  12 accepted-Denver comparators were reused as hash-pinned evidence and were not
  counted as new runs. Raw DDYs, derived IDFs, SQL/EIO, and traces remain outside
  the public addendum; path-free provenance and processed tables are packaged.

## Main attribution findings

Sign conventions follow the scientific question. M1 is spatial PMV minus
building-mean PMV; M2, M3, and C1 are revision variant minus accepted policy;
and the manuscript-facing M4/M5 contrasts are learned-p90 minus learned
building mean or neutral null. Negative degree-hour differences mean less
exposure for the first-named policy on that single temperature screen. The raw
`matched_experiment_variant_summary.csv` retains revision-variant-minus-accepted
signs for every experiment, whereas `learned_contribution_summary.csv` uses the
manuscript-facing learned-p90-minus-alternative convention for M4/M5.

1. **Spatial PMV selection produced the largest, most consistent warm-side
   protection, at a material energy cost.** Relative to building-mean PMV, the
   p90 absolute-PMV policy used 1.3907% more delivered site energy and changed
   worst-zone occupied degree-hours above 26, 28, and 30 C by -446.3731,
   -70.9403, and -6.4594 C h. The >28 C diagnostic was lower in 19 of 24 cases.

2. **Within the learned-probability-informed policy, spatial selection was
   decision-active but its annual endpoint effect was smaller and depended on
   the temperature threshold.** Learned p90 versus learned mean-tail changed
   30,744 of 400,896 occupied request branches (7.6688%), used 0.1384% more
   energy, and changed degree-hours above 26, 28, and 30 C by -59.5805,
   +0.6340, and +4.6214 C h.

3. **The learned probabilities supplied moderate >26 C protection but were not
   the isolated source of the energy savings.** Learned p90 versus the neutral
   probability null changed 44,776 occupied branches (11.1690%), used 0.1852%
   more energy, and changed degree-hours above 26, 28, and 30 C by -158.3384,
   -1.8396, and +4.5899 C h. The neutral null requested relaxation at all
   400,896 occupied decisions. Thus the probabilities altered decisions and
   thermal protection, while the surrounding relaxation policy supplied most
   of the energy-saving behavior.

4. **The attribution is signal-dependent rather than a universal spatial
   benefit.** The PMV spatial contrast was approximately an order of magnitude
   larger in delivered-energy effect than the learned p90-versus-mean contrast,
   and it showed much more consistent >28 C protection. Learned probabilities
   and p90 spatial selection should therefore be described as distinct,
   decision-active modifiers within a deterministic building-level supervisory
   policy, not as a single undifferentiated AI effect.

5. **Parameter robustness was uneven.** The local M2 variants around the
   submitted settings were small for directional-tail and probability-threshold
   changes, whereas changing the relaxation bounds produced the dominant
   energy--exposure trade-off. The panel was a four-sentinel neighborhood test,
   not an optimization exercise.

6. **Adaptive running-mean clamping had a small aggregate effect but was not
   decision-inert.** The fixed-trajectory screen changed 24,098 of 400,896
   occupied request branches (6.011%), which triggered the 24-cell closed-loop
   C1 batch. Across the matched full grid, clamping changed mean delivered site
   energy by +0.036% and mean worst-zone degree-hours above 28 C by +0.845 C h.

7. **Cold-side exposure was sparse in the accepted four-policy panel.** Median
   worst-zone occupied degree-hours below both 18 and 16 C were zero for every
   policy. Mean values below 18 C were 6.669 for fixed setpoints, 4.322 for
   building-mean PMV, 8.771 for the adaptive-band benchmark, and 8.313 for
   learned p90. Learned p90 minus fixed, PMV, and adaptive changed the mean by
   +1.644, +3.991, and -0.458 C h, respectively; the corresponding lower/near-
   zero/higher direction counts were 1/19/4, 0/19/5, and 5/19/0. The secondary
   16 C screen showed the same sparse pattern with smaller magnitudes.

8. **The common-scale figure preserves the absolute annual >28 C comparison.**
   The displayed learned-p90-minus-comparator means are +411.669 C h versus
   fixed setpoints, +11.968 C h versus building-mean PMV, and +24.707 C h versus
   the adaptive-band benchmark. All 24 scenario values per comparator share one
   horizontal scale; the CSV source and portable plotting script are packaged
   alongside the PDF.

9. **The learned-p90 spatial sentinel was dynamic and thermally aligned during
   protection without being a warmest-zone rule.** Every one of the 15 zones was
   selected in every scenario, and selected identity changed in 237,794 of
   394,632 adjacent occupied pairs (60.26%). During 43,295 warm-protection
   requests, the selected sentinel was among the three warmest zones in 76.80%
   of records, but it was exactly warmest in 20.16% and matched the scenario's
   retrospective worst-DH>28 zone in 10.14%.

10. **Annual policy conclusions were robust to the representative C2 sizing
    substitution, whereas event-window thermal rankings were more sensitive.**
    Across the nine active-policy city comparisons, changing from the accepted
    Denver design-day contract to the city-matched design-condition contract
    changed 0/9 directions for policy-minus-fixed annual delivered site energy
    and 0/9 for annual degree-hours above 28 C. It changed 5/9 directions for
    the 336-hour degree-hours-above-28 C contrast, and 49 of 144 within-city
    policy ranks changed across all declared ranking endpoints. The capacities
    and hot-climate absolute warm-side outcomes changed materially, so C2
    strengthens the annual headline while identifying a real sizing dependence
    in short-window thermal interpretation.

## Interpretation boundaries

- M1 isolates spatial aggregation within the PMV comparator. M4 isolates spatial
  aggregation within the learned probability signal. M5 is a structural signal
  null. None alone identifies a pure AI effect.
- D13 characterizes a dynamic spatial sentinel driving synchronized building-
  level actuation. It does not test independently actuated zones, and its thermal
  concordance must not be relabeled as direct warmest-zone or static retrospective
  worst-zone selection.
- Contributor-disjoint predictor performance remains the relevant transfer
  limitation. M3/M5 test supervisory-policy dependence and do not substitute for
  occupant-transfer validation, personalization, or online adaptation.
- Direction counts and scenario distributions should accompany means because
  small aggregate >28 C differences can be driven by a small number of cases.
- The primary results apply to the tested Medium Office prototype, zoning, HVAC
  topology, accepted Denver sizing contract, and weather lineage. C2 establishes
  representative current city-design-condition sensitivity for Beijing,
  Guangzhou, and Phoenix under the same building, policies, and EPWs; it is not
  an additional building/HVAC/GCM test, future-climate resizing experiment, or
  proof of simultaneous component saturation.
