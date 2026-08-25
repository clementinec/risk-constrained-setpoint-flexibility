# Revision 01 run findings

Date: 2026-08-25

## Execution closure

- Core attribution and parameter batch: 64/64 PASS (M1: 24; M2: 32; M3: 8).
- Adaptive running-mean clamp batch: 24/24 PASS (C1).
- Full-grid learned-contribution batch: 48/48 PASS (M4: 24; M5: 24).
- Total scientific coverage: 136/136 PASS, with zero failed cells.
- Separate submitted-settings annual parity sentinel: PASS.
- Exact selected-weather archive: 24/24 native EPWs published with matching
  accepted-run SHA-256 values, path-free selector metadata, recovered source
  lineage, daily-delta metadata, citations, and a CC BY 4.0 data notice.
- City-specific design-day sensitivity C2: not run. The retained panel did not
  provide simultaneous component-binding evidence or authoritative city-specific
  design-day files sufficient to authorize the predeclared sensitivity.

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

## Interpretation boundaries

- M1 isolates spatial aggregation within the PMV comparator. M4 isolates spatial
  aggregation within the learned probability signal. M5 is a structural signal
  null. None alone identifies a pure AI effect.
- The selected zone remains a spatial sentinel driving synchronized building-
  level actuation; these tests do not evaluate independently actuated zones.
- Contributor-disjoint predictor performance remains the relevant transfer
  limitation. M3/M5 test supervisory-policy dependence and do not substitute for
  occupant-transfer validation, personalization, or online adaptation.
- Direction counts and scenario distributions should accompany means because
  small aggregate >28 C differences can be driven by a small number of cases.
- The results apply to the tested Medium Office prototype, zoning, HVAC topology,
  Denver sizing contract, and weather lineage.
