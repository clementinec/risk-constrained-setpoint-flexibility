# C2 city-design-condition sensitivity

Generated: 2026-08-25T06:42:40+00:00

## Closure

The frozen 12-cell batch completed 12/12 with all execution, runtime-location, storage, and accepted-evidence gates passing. The alternative IDF substitutes the geographically matched companion Site:Location and two selected annual design days while retaining the identical annual EPW, building, HVAC, schedules, controller, and policy settings. Because Keep Site Location Information remains at its default No, the annual weather run uses the unchanged EPW location; the companion location makes the design-day package internally city-consistent and removes the textual Denver location/elevation from the alternative IDF.

The results are a representative current city-design-condition sensitivity. They are not future-climate resizing, proof of equipment saturation, or a multi-building/general climate-model robustness test. NaturalGas:Facility includes service-water heating. Operative-temperature degree-hours are descriptive warm-exposure diagnostics, not observed satisfaction, health, or standards compliance.

## Fixed-policy sizing changes

The capacity values are policy-invariant within every city and sizing contract; the fixed row therefore provides one nonduplicated capacity comparison per city.

| City | City-DDY DX cooling | Denver DX cooling | Delta | City-DDY fuel heat | Denver fuel heat | Delta | City-DDY electric reheat | Denver electric reheat | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Beijing | 273.81 kW | 236.93 kW | +15.57% | 71.34 kW | 96.47 kW | -26.05% | 124.77 kW | 143.91 kW | -13.30% |
| Guangzhou | 305.71 kW | 226.87 kW | +34.75% | 14.98 kW | 97.99 kW | -84.71% | 87.84 kW | 143.64 kW | -38.85% |
| Phoenix | 311.51 kW | 231.56 kW | +34.53% | 15.98 kW | 92.61 kW | -82.75% | 90.32 kW | 143.08 kW | -36.87% |

## Annual absolute outcomes

Delivered energy is reported at the site boundary. Natural gas includes service-water heating.

| City | Sizing | Policy | Electricity | Natural gas | Total site energy | DH>26 | DH>28 | DH>30 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Beijing | accepted_denver | fixed | 413740.61 kWh | 91890.83 kWh | 505631.45 kWh | 84.23 C h | 0.00 C h | 0.00 C h |
| Beijing | accepted_denver | pmv | 400087.32 kWh | 95787.32 kWh | 495874.64 kWh | 2060.56 C h | 7.01 C h | 0.00 C h |
| Beijing | accepted_denver | adaptive_band | 394618.22 kWh | 92579.16 kWh | 487197.38 kWh | 1837.62 C h | 5.53 C h | 0.00 C h |
| Beijing | accepted_denver | learned_p90 | 390381.38 kWh | 93463.55 kWh | 483844.93 kWh | 2114.86 C h | 8.06 C h | 0.00 C h |
| Beijing | city_ddy | fixed | 412528.21 kWh | 93797.82 kWh | 506326.03 kWh | 45.07 C h | 0.00 C h | 0.00 C h |
| Beijing | city_ddy | pmv | 400317.79 kWh | 97663.59 kWh | 497981.37 kWh | 2033.84 C h | 6.99 C h | 0.00 C h |
| Beijing | city_ddy | adaptive_band | 394645.55 kWh | 94375.77 kWh | 489021.32 kWh | 1841.07 C h | 5.48 C h | 0.00 C h |
| Beijing | city_ddy | learned_p90 | 390529.01 kWh | 95207.10 kWh | 485736.11 kWh | 2093.95 C h | 8.02 C h | 0.00 C h |
| Guangzhou | accepted_denver | fixed | 608391.96 kWh | 27015.95 kWh | 635407.91 kWh | 7950.73 C h | 3371.24 C h | 763.47 C h |
| Guangzhou | accepted_denver | pmv | 585786.31 kWh | 27186.08 kWh | 612972.39 kWh | 9605.31 C h | 3673.37 C h | 799.21 C h |
| Guangzhou | accepted_denver | adaptive_band | 585241.04 kWh | 27388.41 kWh | 612629.44 kWh | 9560.96 C h | 3617.01 C h | 792.12 C h |
| Guangzhou | accepted_denver | learned_p90 | 584046.74 kWh | 27381.04 kWh | 611427.78 kWh | 9794.53 C h | 3631.40 C h | 798.06 C h |
| Guangzhou | city_ddy | fixed | 611933.04 kWh | 26570.71 kWh | 638503.75 kWh | 2804.03 C h | 461.37 C h | 11.18 C h |
| Guangzhou | city_ddy | pmv | 574404.95 kWh | 26640.22 kWh | 601045.17 kWh | 5646.64 C h | 817.62 C h | 24.20 C h |
| Guangzhou | city_ddy | adaptive_band | 574323.10 kWh | 26878.86 kWh | 601201.96 kWh | 5645.21 C h | 807.35 C h | 22.43 C h |
| Guangzhou | city_ddy | learned_p90 | 572901.06 kWh | 26851.71 kWh | 599752.77 kWh | 5873.35 C h | 808.14 C h | 22.42 C h |
| Phoenix | accepted_denver | fixed | 576217.83 kWh | 25304.41 kWh | 601522.25 kWh | 5615.94 C h | 1967.66 C h | 421.30 C h |
| Phoenix | accepted_denver | pmv | 543865.15 kWh | 25155.46 kWh | 569020.60 kWh | 7649.66 C h | 2246.69 C h | 452.81 C h |
| Phoenix | accepted_denver | adaptive_band | 546778.84 kWh | 25525.62 kWh | 572304.47 kWh | 7705.10 C h | 2180.35 C h | 438.94 C h |
| Phoenix | accepted_denver | learned_p90 | 541892.59 kWh | 25427.40 kWh | 567319.99 kWh | 8089.25 C h | 2235.07 C h | 448.21 C h |
| Phoenix | city_ddy | fixed | 578857.28 kWh | 25249.23 kWh | 604106.51 kWh | 2341.76 C h | 401.23 C h | 27.82 C h |
| Phoenix | city_ddy | pmv | 535780.02 kWh | 25116.88 kWh | 560896.90 kWh | 5516.54 C h | 704.08 C h | 29.21 C h |
| Phoenix | city_ddy | adaptive_band | 547158.51 kWh | 25403.63 kWh | 572562.14 kWh | 5032.62 C h | 648.23 C h | 33.25 C h |
| Phoenix | city_ddy | learned_p90 | 534277.09 kWh | 25321.42 kWh | 559598.51 kWh | 6108.29 C h | 744.60 C h | 30.12 C h |

## Extreme-window absolute outcomes

Each event window contains exactly 336 hours (1,344 quarter-hour records), including exactly 160 occupied hours (640 records); degree-hours use occupied records only.

| City | Sizing | Policy | Electricity | Natural gas | Total site energy | DH>26 | DH>28 | DH>30 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Beijing | accepted_denver | fixed | 23452.31 kWh | 887.17 kWh | 24339.48 kWh | 33.72 C h | 0.00 C h | 0.00 C h |
| Beijing | accepted_denver | pmv | 21735.01 kWh | 884.86 kWh | 22619.87 kWh | 180.58 C h | 2.06 C h | 0.00 C h |
| Beijing | accepted_denver | adaptive_band | 21735.14 kWh | 884.86 kWh | 22620.00 kWh | 180.80 C h | 2.01 C h | 0.00 C h |
| Beijing | accepted_denver | learned_p90 | 21735.14 kWh | 884.86 kWh | 22620.00 kWh | 180.80 C h | 2.01 C h | 0.00 C h |
| Beijing | city_ddy | fixed | 22889.30 kWh | 888.36 kWh | 23777.66 kWh | 10.75 C h | 0.00 C h | 0.00 C h |
| Beijing | city_ddy | pmv | 21416.29 kWh | 884.84 kWh | 22301.13 kWh | 179.97 C h | 0.16 C h | 0.00 C h |
| Beijing | city_ddy | adaptive_band | 21416.29 kWh | 884.84 kWh | 22301.13 kWh | 179.97 C h | 0.16 C h | 0.00 C h |
| Beijing | city_ddy | learned_p90 | 21416.29 kWh | 884.84 kWh | 22301.13 kWh | 179.97 C h | 0.16 C h | 0.00 C h |
| Guangzhou | accepted_denver | fixed | 30308.40 kWh | 884.96 kWh | 31193.36 kWh | 761.75 C h | 441.75 C h | 150.58 C h |
| Guangzhou | accepted_denver | pmv | 30307.57 kWh | 884.96 kWh | 31192.52 kWh | 761.46 C h | 441.46 C h | 150.46 C h |
| Guangzhou | accepted_denver | adaptive_band | 30278.97 kWh | 884.96 kWh | 31163.93 kWh | 760.07 C h | 440.07 C h | 150.01 C h |
| Guangzhou | accepted_denver | learned_p90 | 30279.33 kWh | 884.96 kWh | 31164.28 kWh | 760.14 C h | 440.14 C h | 150.04 C h |
| Guangzhou | city_ddy | fixed | 32265.54 kWh | 890.08 kWh | 33155.63 kWh | 298.88 C h | 62.98 C h | 0.82 C h |
| Guangzhou | city_ddy | pmv | 30778.62 kWh | 887.47 kWh | 31666.09 kWh | 373.86 C h | 94.05 C h | 1.81 C h |
| Guangzhou | city_ddy | adaptive_band | 30570.68 kWh | 888.57 kWh | 31459.26 kWh | 375.96 C h | 95.35 C h | 1.84 C h |
| Guangzhou | city_ddy | learned_p90 | 30570.68 kWh | 888.57 kWh | 31459.26 kWh | 375.96 C h | 95.35 C h | 1.84 C h |
| Phoenix | accepted_denver | fixed | 31875.11 kWh | 860.87 kWh | 32735.98 kWh | 699.60 C h | 379.60 C h | 112.58 C h |
| Phoenix | accepted_denver | pmv | 31784.02 kWh | 859.68 kWh | 32643.70 kWh | 702.42 C h | 382.52 C h | 115.54 C h |
| Phoenix | accepted_denver | adaptive_band | 31595.63 kWh | 860.83 kWh | 32456.46 kWh | 684.26 C h | 364.62 C h | 109.54 C h |
| Phoenix | accepted_denver | learned_p90 | 31597.48 kWh | 860.83 kWh | 32458.31 kWh | 684.54 C h | 364.92 C h | 111.75 C h |
| Phoenix | city_ddy | fixed | 34166.64 kWh | 864.51 kWh | 35031.15 kWh | 301.82 C h | 80.88 C h | 6.84 C h |
| Phoenix | city_ddy | pmv | 31013.36 kWh | 860.89 kWh | 31874.25 kWh | 360.13 C h | 78.10 C h | 6.17 C h |
| Phoenix | city_ddy | adaptive_band | 32501.94 kWh | 863.27 kWh | 33365.21 kWh | 335.96 C h | 85.43 C h | 8.26 C h |
| Phoenix | city_ddy | learned_p90 | 30904.20 kWh | 862.04 kWh | 31766.23 kWh | 360.05 C h | 78.39 C h | 6.48 C h |

## Occupied setpoint-not-met evidence

These annual SQL diagnostics establish setpoint mismatch, not simultaneous component saturation.

| City | Sizing | Policy | Cooling not met | Heating not met | Cooling occupied DH sum | Heating occupied DH sum | Cooling occupied DH max zone | Heating occupied DH max zone |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Beijing | accepted_denver | fixed | 452.00 h | 523.25 h | 2809.32 C h | 4189.31 C h | 335.74 C h | 1458.67 C h |
| Beijing | accepted_denver | pmv | 277.00 h | 605.00 h | 1259.42 C h | 5009.46 C h | 142.16 C h | 1656.29 C h |
| Beijing | accepted_denver | adaptive_band | 434.00 h | 417.50 h | 1666.82 C h | 2997.61 C h | 320.72 C h | 1129.60 C h |
| Beijing | accepted_denver | learned_p90 | 437.00 h | 360.75 h | 1620.72 C h | 3372.02 C h | 296.67 C h | 1005.88 C h |
| Beijing | city_ddy | fixed | 306.25 h | 535.50 h | 1684.66 C h | 4325.40 C h | 197.28 C h | 1438.29 C h |
| Beijing | city_ddy | pmv | 243.50 h | 614.25 h | 1080.71 C h | 5170.14 C h | 105.79 C h | 1633.40 C h |
| Beijing | city_ddy | adaptive_band | 377.75 h | 427.50 h | 1454.46 C h | 3087.75 C h | 285.80 C h | 1113.87 C h |
| Beijing | city_ddy | learned_p90 | 374.00 h | 369.25 h | 1392.07 C h | 3567.18 C h | 258.45 C h | 1004.99 C h |
| Guangzhou | accepted_denver | fixed | 3312.75 h | 32.25 h | 100651.40 C h | 75.09 C h | 11191.87 C h | 74.76 C h |
| Guangzhou | accepted_denver | pmv | 2756.00 h | 33.25 h | 91788.99 C h | 83.62 C h | 9364.48 C h | 83.23 C h |
| Guangzhou | accepted_denver | adaptive_band | 2848.00 h | 7.25 h | 48685.98 C h | 14.08 C h | 6341.52 C h | 14.08 C h |
| Guangzhou | accepted_denver | learned_p90 | 2765.00 h | 6.50 h | 59914.56 C h | 12.47 C h | 7108.10 C h | 12.47 C h |
| Guangzhou | city_ddy | fixed | 2427.50 h | 32.00 h | 25541.43 C h | 72.85 C h | 4653.80 C h | 71.52 C h |
| Guangzhou | city_ddy | pmv | 1524.50 h | 32.50 h | 12346.35 C h | 80.55 C h | 2106.06 C h | 79.82 C h |
| Guangzhou | city_ddy | adaptive_band | 1581.75 h | 7.25 h | 9328.47 C h | 13.49 C h | 1858.69 C h | 13.49 C h |
| Guangzhou | city_ddy | learned_p90 | 1533.50 h | 6.25 h | 9160.64 C h | 12.32 C h | 1842.26 C h | 12.32 C h |
| Phoenix | accepted_denver | fixed | 3087.25 h | 17.00 h | 58856.86 C h | 22.31 C h | 7934.24 C h | 22.29 C h |
| Phoenix | accepted_denver | pmv | 2260.50 h | 18.00 h | 41481.82 C h | 23.68 C h | 5175.95 C h | 19.96 C h |
| Phoenix | accepted_denver | adaptive_band | 2413.25 h | 0.00 h | 26869.46 C h | 0.00 C h | 4379.59 C h | 0.00 C h |
| Phoenix | accepted_denver | learned_p90 | 2224.25 h | 0.00 h | 27003.90 C h | 0.00 C h | 4112.94 C h | 0.00 C h |
| Phoenix | city_ddy | fixed | 2188.25 h | 14.75 h | 14640.26 C h | 18.37 C h | 3596.99 C h | 18.32 C h |
| Phoenix | city_ddy | pmv | 1256.75 h | 14.50 h | 6372.00 C h | 22.12 C h | 1531.30 C h | 17.77 C h |
| Phoenix | city_ddy | adaptive_band | 1517.25 h | 0.00 h | 8145.59 C h | 0.00 C h | 2116.61 C h | 0.00 C h |
| Phoenix | city_ddy | learned_p90 | 1267.25 h | 0.00 h | 6075.29 C h | 0.00 C h | 1508.55 C h | 0.00 C h |

## Policy-minus-fixed contrasts

Negative values mean the policy is lower than fixed under the same sizing contract.

| City | Sizing | Policy | Annual site energy | Extreme site energy | Annual DH>28 | Extreme DH>28 | Cooling not-met time | Heating not-met time |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Beijing | accepted_denver | pmv | -9756.80 kWh | -1719.61 kWh | +7.01 C h | +2.06 C h | -175.00 h | +81.75 h |
| Beijing | accepted_denver | adaptive_band | -18434.07 kWh | -1719.48 kWh | +5.53 C h | +2.01 C h | -18.00 h | -105.75 h |
| Beijing | accepted_denver | learned_p90 | -21786.52 kWh | -1719.48 kWh | +8.06 C h | +2.01 C h | -15.00 h | -162.50 h |
| Beijing | city_ddy | pmv | -8344.66 kWh | -1476.53 kWh | +6.99 C h | +0.16 C h | -62.75 h | +78.75 h |
| Beijing | city_ddy | adaptive_band | -17304.71 kWh | -1476.53 kWh | +5.48 C h | +0.16 C h | +71.50 h | -108.00 h |
| Beijing | city_ddy | learned_p90 | -20589.92 kWh | -1476.53 kWh | +8.02 C h | +0.16 C h | +67.75 h | -166.25 h |
| Guangzhou | accepted_denver | pmv | -22435.53 kWh | -0.83 kWh | +302.13 C h | -0.29 C h | -556.75 h | +1.00 h |
| Guangzhou | accepted_denver | adaptive_band | -22778.47 kWh | -29.43 kWh | +245.78 C h | -1.67 C h | -464.75 h | -25.00 h |
| Guangzhou | accepted_denver | learned_p90 | -23980.14 kWh | -29.07 kWh | +260.16 C h | -1.60 C h | -547.75 h | -25.75 h |
| Guangzhou | city_ddy | pmv | -37458.58 kWh | -1489.54 kWh | +356.25 C h | +31.07 C h | -903.00 h | +0.50 h |
| Guangzhou | city_ddy | adaptive_band | -37301.80 kWh | -1696.37 kWh | +345.98 C h | +32.38 C h | -845.75 h | -24.75 h |
| Guangzhou | city_ddy | learned_p90 | -38750.99 kWh | -1696.37 kWh | +346.77 C h | +32.38 C h | -894.00 h | -25.75 h |
| Phoenix | accepted_denver | pmv | -32501.64 kWh | -92.28 kWh | +279.03 C h | +2.92 C h | -826.75 h | +1.00 h |
| Phoenix | accepted_denver | adaptive_band | -29217.78 kWh | -279.52 kWh | +212.69 C h | -14.97 C h | -674.00 h | -17.00 h |
| Phoenix | accepted_denver | learned_p90 | -34202.25 kWh | -277.67 kWh | +267.42 C h | -14.68 C h | -863.00 h | -17.00 h |
| Phoenix | city_ddy | pmv | -43209.61 kWh | -3156.90 kWh | +302.86 C h | -2.78 C h | -931.50 h | -0.25 h |
| Phoenix | city_ddy | adaptive_band | -31544.36 kWh | -1665.94 kWh | +247.00 C h | +4.54 C h | -671.00 h | -14.75 h |
| Phoenix | city_ddy | learned_p90 | -44507.99 kWh | -3264.92 kWh | +343.37 C h | -2.50 C h | -921.00 h | -14.75 h |

## Lower-is-better policy rankings

Rank 1 policy or tied policies for each city and sizing contract:

| Metric | Beijing city-DDY | Beijing Denver | Guangzhou city-DDY | Guangzhou Denver | Phoenix city-DDY | Phoenix Denver |
|---|---|---|---|---|---|---|
| annual_total_delivered_site_energy_kwh | learned_p90 | learned_p90 | learned_p90 | learned_p90 | learned_p90 | learned_p90 |
| extreme_336h_total_delivered_site_energy_kwh | pmv | pmv | learned_p90 | adaptive_band | learned_p90 | adaptive_band |
| annual_occupied_worst_zone_degree_hours_above_26c | fixed | fixed | fixed | fixed | fixed | fixed |
| annual_occupied_worst_zone_degree_hours_above_28c | fixed | fixed | fixed | fixed | fixed | fixed |
| annual_occupied_worst_zone_degree_hours_above_30c | fixed + pmv + adaptive_band + learned_p90 | fixed + pmv + adaptive_band + learned_p90 | fixed | fixed | fixed | fixed |
| extreme_336h_occupied_worst_zone_degree_hours_above_26c | fixed | fixed | fixed | adaptive_band | fixed | adaptive_band |
| extreme_336h_occupied_worst_zone_degree_hours_above_28c | fixed | fixed | fixed | adaptive_band | pmv | adaptive_band |
| extreme_336h_occupied_worst_zone_degree_hours_above_30c | fixed + pmv + adaptive_band + learned_p90 | fixed + pmv + adaptive_band + learned_p90 | fixed | adaptive_band | pmv | adaptive_band |
| facility_occupied_cooling_setpoint_not_met_h | pmv | pmv | pmv | pmv | pmv | learned_p90 |
| facility_occupied_heating_setpoint_not_met_h | learned_p90 | learned_p90 | learned_p90 | learned_p90 | adaptive_band + learned_p90 | adaptive_band + learned_p90 |
| cooling_occupied_unmet_degree_hours_sum | pmv | pmv | learned_p90 | adaptive_band | learned_p90 | adaptive_band |
| heating_occupied_unmet_degree_hours_sum | adaptive_band | adaptive_band | learned_p90 | learned_p90 | adaptive_band + learned_p90 | adaptive_band + learned_p90 |

## Policy-attribution robustness

The differential-of-contrasts table asks whether changing the sizing contract changes each policy-minus-fixed effect. Direction changes among the nine non-fixed city-policy comparisons were: annual site energy 0/9, annual DH>28 0/9, and extreme-period DH>28 5/9. Across the 144 matched city/metric/policy ranking entries, 49 ranks changed (ties use minimum rank).

Interpret direction changes together with their magnitudes in `differential_of_contrasts`; a sign change near zero does not automatically constitute a material reversal. Lower energy and lower degree-hours refer to different objectives and should not be collapsed into a single comfort-performance claim.

The complete city-DDY-minus-Denver substitutions for every declared endpoint are retained in `matched_city_ddy_minus_denver`; the tables above show the absolute energy/fuel, all three warm thresholds, annual occupied unmet-load evidence, and the principal policy contrasts without suppressing zero-valued outcomes.

## Files

- `cell_metrics`: absolute accepted-Denver and city-design-condition endpoints.
- `matched_city_ddy_minus_denver`: matched sizing-contract substitutions.
- `within_contract_policy_minus_fixed`: policy effects separately under each sizing contract.
- `differential_of_contrasts`: change in policy-minus-fixed effects caused by sizing substitution.
- `policy_rankings` and `policy_ranking_stability`: within-city rankings and changes.
- Long SQL sizing/unmet tables are stored as compressed Parquet only.
