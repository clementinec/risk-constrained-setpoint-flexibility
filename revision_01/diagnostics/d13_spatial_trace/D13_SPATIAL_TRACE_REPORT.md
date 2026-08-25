# D13 learned-p90 spatial-sentinel trace report

**Status: PASS.** This is a retained-output diagnostic over the 24 accepted learned-p90 primary scenarios; no EnergyPlus simulation was run.

## Definitions

At each occupied 15-minute record, the selected spatial sentinel is reconstructed as the actual zone nearest the linear spatial 0.90 quantile of the 15 zones' combined learned warm- and cold-tail probability. The accepted zone order supplies the first-index tie break.

The contemporaneous warmest zone is the zone with the maximum operative temperature, `(air + mean radiant)/2`, at that record. The cumulative worst warm-exposure zone is the single zone in each scenario with the largest full-scenario occupied operative-temperature degree-hours above 28 degC. That latter identity is retrospective and was not available to the controller.

A switch opportunity requires two adjacent 15-minute records that are both occupied. A persistence episode ends at a selected-zone change or an unoccupied gap. The warm-protection subset is reconstructed with selected total tail probability >= 0.20 and selected warm-minus-cold tail difference > 0.10; it matches the retained request branch exactly.

## Coverage and reconstruction validation

All 24 hash-pinned traces passed: 35,040 records and 16,704 occupied records per scenario, for 400,896 occupied records total and all 15 zones. The explicit zone index in the independent Revision 01 parity trace matched the reconstructed index in 16,704/16,704 occupied records (zero mismatches).

## Selected-zone identity and persistence

Every one of the 15 zones was selected at least once in every scenario. The three largest pooled selection shares were Core_bottom (12.88%), Perimeter_top_ZN_1 (12.42%), Core_mid (11.00%); the largest single-zone pooled share was only 12.88%. The sentinel therefore did not collapse to one fixed zone.

Selected-zone identity changed on 237,794/394,632 within-occupied adjacent transitions (60.26%). Across 244,058 persistence episodes, the median duration was 0.25 h, the 90th percentile was 0.75 h, and the maximum was 7.75 h. These are sentinel identities, not setpoint moves; dwell and bound logic can prevent a sentinel change from becoming an actuator change.

## Thermal concordance

Across all occupied records, the selected sentinel was the contemporaneous warmest zone in 56,708/400,896 records (14.15%) and one of the three warmest zones in 182,563/400,896 (45.54%). It matched the scenario's retrospective cumulative worst DH>28 zone in 32,896/400,896 (8.21%).

The controller entered the reconstructed warm-protection branch in 43,295 occupied records (10.80%). Within those records, the sentinel was exactly the warmest zone in 8,730/43,295 (20.16%) and was among the three warmest in 33,249/43,295 (76.80%). It matched the scenario's retrospective cumulative worst DH>28 zone in 4,388/43,295 (10.14%).

| Weather role | Switch rate | Exact warmest | Top-three warmest | Exact warmest during warm protection | Top-three during warm protection |
|---|---:|---:|---:|---:|---:|
| present_typical | 59.88% | 13.88% | 43.34% | 20.39% | 76.02% |
| future_extreme | 60.63% | 14.41% | 47.74% | 19.98% | 77.44% |

The retrospective worst DH>28 identity was distributed as Perimeter_mid_ZN_1 (12), Perimeter_bot_ZN_1 (8), Perimeter_mid_ZN_3 (3), Perimeter_top_ZN_4 (1).

## Interpretation boundary

The learned-p90 sentinel is a dynamic high-tail-risk selector, not a direct maximum-temperature selector and not a proxy for the zone that eventually accumulates the largest annual DH>28. Its closer alignment with the warmest zones during warm-protection states documents operational thermal relevance, while the lower all-hours and retrospective-worst concordance prevents treating it as independently zonal control. These diagnostics characterize spatial supervision under synchronized building-level actuation; they do not test separate zone-level actuators.
