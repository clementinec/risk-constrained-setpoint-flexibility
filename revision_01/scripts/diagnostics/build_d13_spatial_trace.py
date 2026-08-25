#!/usr/bin/env python3
"""Rebuild or verify the D13 learned-p90 spatial-sentinel diagnostic.

The public addendum intentionally excludes timestep traces.  ``--review-root``
rebuilds D13 from the separately retained, hash-pinned review archive, while
``--verify-package`` validates the compact public tables without those traces.
No EnergyPlus simulation is run in either mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT.parents[2]
DEFAULT_MANIFEST = PACKAGE_ROOT / "summary_outputs/d13_input_manifest.csv"
DEFAULT_PROVENANCE = (
    PACKAGE_ROOT / "diagnostics/d13_spatial_trace/D13_PUBLIC_PROVENANCE.json"
)

EXPECTED_MANIFEST_SHA256 = (
    "555d8603e515326992fc02da5f94d5762b3db6ed896111ae593d206b293f1ad3"
)
EXPECTED_MATRIX_SHA256 = (
    "fa03767f0c3d581e65e00ff6149546c44fe3e936608da73ad2f5485bf19895e0"
)
EXPECTED_FRESH96_CLOSURE_SHA256 = (
    "b571b9205a51b952852ef1e3abcd0a63d24daf336210b0e63f92e686a48ffcee"
)
EXPECTED_ACCEPTED_RUNNER_SHA256 = (
    "a9693042883ed5c20edad6fcbc757c62c7216d5abc120ad18de1a003932848a4"
)
EXPECTED_PARITY_CLOSURE_SHA256 = (
    "ca88141bfdf9344bd20ce138eee32f35d88f3f1d38b35d89bcf799710100326f"
)
EXPECTED_PARITY_TRACE_SHA256 = (
    "01d3a8eb138ccf9fd5353c7c90b1b7ad8c56f0f016a5a40d925654893ccd2282"
)

MATRIX_RELATIVE = Path(
    "06_runs/scaled/20260803_routeS96_v2_envscoped/freeze/authorization/"
    "execution/scaled_matrix_fresh96_v2_envscoped.csv"
)
FRESH96_CLOSURE_RELATIVE = Path(
    "06_runs/scaled/20260803_routeS96_v2_envscoped/FRESH96_CLOSURE.json"
)
ACCEPTED_RUNNER_RELATIVE = Path(
    "06_runs/scaled/20260803_routeS96_v2_envscoped/freeze/code/"
    "run_medium_office_enb_rebuild_v2_envscoped_warmup.py"
)
PARITY_CLOSURE_RELATIVE = Path(
    "09_rev01/04_analysis/parity_validation/rev01_parity1_20260824_v2/"
    "PARITY_CLOSURE.json"
)
PARITY_TRACE_RELATIVE = Path(
    "09_rev01/05_reruns/scientific_runs/rev01_parity1_20260824_v2/cells/"
    "REV01-PARITY-001/traces/beijing_ssp585_baseline_2020s_typical_2026_"
    "met0p854_act93p2W_paperb_p90_tail_asym_relax.parquet"
)

EXPECTED_STRATEGY = "paperb_p90_tail_asym_relax"
EXPECTED_ROWS = 35_040
EXPECTED_OCCUPIED_ROWS = 16_704
EXPECTED_CELLS = 24
EXPECTED_ZONES = 15
TIMESTEP_HOURS = 0.25
SPATIAL_QUANTILE = 0.90
WARM_EXPOSURE_THRESHOLD_C = 28.0
TAIL_THRESHOLD = 0.20
ASYMMETRY_THRESHOLD = 0.10
FLOAT_TOL = 1e-12

ZONE_LABELS = (
    "Core_bottom",
    "Core_mid",
    "Core_top",
    "Perimeter_top_ZN_3",
    "Perimeter_top_ZN_2",
    "Perimeter_top_ZN_1",
    "Perimeter_top_ZN_4",
    "Perimeter_bot_ZN_3",
    "Perimeter_bot_ZN_2",
    "Perimeter_bot_ZN_1",
    "Perimeter_bot_ZN_4",
    "Perimeter_mid_ZN_3",
    "Perimeter_mid_ZN_2",
    "Perimeter_mid_ZN_1",
    "Perimeter_mid_ZN_4",
)
ZONE_SLUGS = tuple(label.lower() for label in ZONE_LABELS)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.15g")
    return {"rows": int(len(frame)), "sha256": sha256(path)}


def selector_indices(zone_p_tail: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the actual zone nearest the linear spatial q=.90 target."""

    require(
        zone_p_tail.ndim == 2 and zone_p_tail.shape[1] == EXPECTED_ZONES,
        "invalid p-tail matrix",
    )
    targets = np.quantile(
        zone_p_tail, SPATIAL_QUANTILE, axis=1, method="linear"
    )
    indices = np.argmin(
        np.abs(zone_p_tail - targets[:, None]), axis=1
    ).astype(np.int64)
    return indices, targets


def persistence_lengths(
    indices: np.ndarray, occupied_steps: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """Return episodes, selected-zone switches, and adjacent occupied pairs."""

    require(
        len(indices) == len(occupied_steps) and len(indices) > 0,
        "invalid persistence inputs",
    )
    adjacent = np.r_[False, np.diff(occupied_steps) == 1]
    changed = np.r_[False, indices[1:] != indices[:-1]]
    starts = (~adjacent) | changed
    start_positions = np.flatnonzero(starts)
    lengths = np.diff(np.r_[start_positions, len(indices)]).astype(np.int64)
    require(int(lengths.sum()) == len(indices), "persistence coverage failure")
    return lengths, int((adjacent & changed).sum()), int(adjacent.sum())


def trace_columns() -> list[str]:
    columns = [
        "strategy",
        "weather",
        "occupied",
        "formal_weather_step",
        "sim_time_hours",
        "paperb_request_branch",
    ]
    for slug in ZONE_SLUGS:
        columns.extend(
            [
                f"zone_{slug}_p_disc",
                f"zone_{slug}_warm_tail",
                f"zone_{slug}_cold_tail",
                f"zone_{slug}_ta_c",
                f"zone_{slug}_tr_c",
            ]
        )
    return columns


def validate_review_archive(review_root: Path) -> None:
    paths_and_hashes = (
        (MATRIX_RELATIVE, EXPECTED_MATRIX_SHA256),
        (FRESH96_CLOSURE_RELATIVE, EXPECTED_FRESH96_CLOSURE_SHA256),
        (ACCEPTED_RUNNER_RELATIVE, EXPECTED_ACCEPTED_RUNNER_SHA256),
        (PARITY_CLOSURE_RELATIVE, EXPECTED_PARITY_CLOSURE_SHA256),
        (PARITY_TRACE_RELATIVE, EXPECTED_PARITY_TRACE_SHA256),
    )
    for relative, expected_hash in paths_and_hashes:
        path = review_root / relative
        require(path.is_file(), f"missing review input: {relative}")
        require(sha256(path) == expected_hash, f"hash mismatch: {relative}")

    closure = json.loads(
        (review_root / FRESH96_CLOSURE_RELATIVE).read_text(encoding="utf-8")
    )
    require(
        closure.get("closure_status") == "PASS"
        and closure.get("all_gates_passed") is True
        and closure.get("passed_cells") == 96
        and closure.get("failed_cells") == 0,
        "Fresh96 closure did not pass",
    )


def read_input_manifest(path: Path) -> pd.DataFrame:
    require(path.is_file(), f"missing D13 input manifest: {path}")
    require(sha256(path) == EXPECTED_MANIFEST_SHA256, "D13 manifest mismatch")
    manifest = pd.read_csv(path)
    require(len(manifest) == EXPECTED_CELLS, "expected 24 D13 inputs")
    require(manifest["scaled_run_id"].nunique() == EXPECTED_CELLS, "duplicate cells")
    require(manifest["anchor_id"].nunique() == EXPECTED_CELLS, "duplicate anchors")
    require(set(manifest["scenario"]) == {"ssp245", "ssp585"}, "scenario mismatch")
    require(
        set(manifest["selector_role"]) == {"present_typical", "future_extreme"},
        "selector-role mismatch",
    )
    for value in manifest["trace_path"].astype(str):
        require(not Path(value).is_absolute(), "manifest contains an absolute trace path")
    return manifest.sort_values("scaled_run_id").reset_index(drop=True)


def load_scenario(
    row: pd.Series, review_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    cell_id = str(row["scaled_run_id"])
    trace_path = review_root / str(row["trace_path"])
    result_path = review_root / str(row["cell_result_path"])
    require(trace_path.is_file(), f"{cell_id}: missing trace")
    require(result_path.is_file(), f"{cell_id}: missing cell result")
    require(sha256(trace_path) == row["trace_sha256_recorded"], f"{cell_id}: trace hash")
    require(sha256(result_path) == row["cell_result_sha256"], f"{cell_id}: result hash")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    require(result.get("all_gates_passed") is True, f"{cell_id}: gates did not pass")
    require(result.get("status") == "complete_validated", f"{cell_id}: result status")
    require(result.get("scaled_run_id") == cell_id, f"{cell_id}: result identity")
    require(result.get("strategy") == EXPECTED_STRATEGY, f"{cell_id}: strategy")
    recorded_trace = result["outputs"]["trace"]
    require(recorded_trace["sha256"] == row["trace_sha256_recorded"], f"{cell_id}: recorded hash")

    frame = pd.read_parquet(trace_path, columns=trace_columns())
    require(len(frame) == EXPECTED_ROWS, f"{cell_id}: row count")
    require(
        frame["strategy"].nunique() == 1
        and frame["strategy"].iloc[0] == EXPECTED_STRATEGY,
        f"{cell_id}: trace strategy",
    )
    require(frame["weather"].nunique() == 1, f"{cell_id}: weather varies")
    require(
        np.array_equal(
            frame["formal_weather_step"].to_numpy(),
            np.arange(1, EXPECTED_ROWS + 1),
        ),
        f"{cell_id}: formal step sequence",
    )
    require(
        np.allclose(
            np.diff(frame["sim_time_hours"].to_numpy(dtype=float)),
            TIMESTEP_HOURS,
            rtol=0.0,
            atol=1e-9,
        ),
        f"{cell_id}: timestep",
    )
    occupied_mask = frame["occupied"].fillna(False).astype(bool).to_numpy()
    require(int(occupied_mask.sum()) == EXPECTED_OCCUPIED_ROWS, f"{cell_id}: occupied rows")
    occupied = frame.loc[occupied_mask].reset_index(drop=True)

    def zone_matrix(suffix: str) -> np.ndarray:
        return occupied[
            [f"zone_{slug}_{suffix}" for slug in ZONE_SLUGS]
        ].to_numpy(dtype=float)

    p_tail = zone_matrix("p_disc")
    warm_tail = zone_matrix("warm_tail")
    cold_tail = zone_matrix("cold_tail")
    ta = zone_matrix("ta_c")
    tr = zone_matrix("tr_c")
    for name, values in (
        ("p_tail", p_tail),
        ("warm_tail", warm_tail),
        ("cold_tail", cold_tail),
        ("ta", ta),
        ("tr", tr),
    ):
        require(
            values.shape == (EXPECTED_OCCUPIED_ROWS, EXPECTED_ZONES)
            and np.isfinite(values).all(),
            f"{cell_id}: invalid {name}",
        )
    identity_error = float(np.max(np.abs(p_tail - warm_tail - cold_tail)))
    require(identity_error <= FLOAT_TOL, f"{cell_id}: tail identity")

    selected_idx, targets = selector_indices(p_tail)
    row_idx = np.arange(EXPECTED_OCCUPIED_ROWS)
    selected_p_tail = p_tail[row_idx, selected_idx]
    selected_d_tail = (warm_tail - cold_tail)[row_idx, selected_idx]
    operative = 0.5 * (ta + tr)
    warmest_idx = np.argmax(operative, axis=1)
    selected_operative = operative[row_idx, selected_idx]
    warmest_operative = operative[row_idx, warmest_idx]
    temperature_rank = 1 + np.sum(operative > selected_operative[:, None], axis=1)
    risk_rank = 1 + np.sum(p_tail > selected_p_tail[:, None], axis=1)

    dh28_by_zone = (
        np.maximum(operative - WARM_EXPOSURE_THRESHOLD_C, 0.0).sum(axis=0)
        * TIMESTEP_HOURS
    )
    worst_idx = int(np.argmax(dh28_by_zone))
    sorted_dh = np.sort(dh28_by_zone)[::-1]
    require(sorted_dh[0] > sorted_dh[1] + FLOAT_TOL, f"{cell_id}: tied worst zone")

    warm_state = (selected_p_tail >= TAIL_THRESHOLD) & (
        selected_d_tail > ASYMMETRY_THRESHOLD
    )
    recorded_warm_state = occupied["paperb_request_branch"].eq(
        "warm_protection"
    ).to_numpy()
    require(np.array_equal(warm_state, recorded_warm_state), f"{cell_id}: warm branch")

    steps = occupied["formal_weather_step"].to_numpy(dtype=np.int64)
    episodes, switches, switch_opportunities = persistence_lengths(
        selected_idx, steps
    )
    selected_counts = np.bincount(selected_idx, minlength=EXPECTED_ZONES)
    warm_counts = np.bincount(selected_idx[warm_state], minlength=EXPECTED_ZONES)
    exact_warmest = selected_idx == warmest_idx
    top3_warmest = temperature_rank <= 3
    selected_worst = selected_idx == worst_idx
    mode_idx = int(np.argmax(selected_counts))

    scenario = {
        "scaled_run_id": cell_id,
        "anchor_id": row["anchor_id"],
        "city": row["city"],
        "scenario": row["scenario"],
        "selector_role": row["selector_role"],
        "weather_year": int(row["weather_year"]),
        "occupied_rows": EXPECTED_OCCUPIED_ROWS,
        "zone_count": EXPECTED_ZONES,
        "unique_selected_zones": int(np.count_nonzero(selected_counts)),
        "modal_selected_zone": ZONE_LABELS[mode_idx],
        "modal_selected_count": int(selected_counts[mode_idx]),
        "modal_selected_share": float(selected_counts[mode_idx] / EXPECTED_OCCUPIED_ROWS),
        "switch_opportunities": switch_opportunities,
        "selected_zone_switches": switches,
        "selected_zone_switch_rate": float(switches / switch_opportunities),
        "persistence_episode_count": int(len(episodes)),
        "persistence_mean_hours": float(np.mean(episodes) * TIMESTEP_HOURS),
        "persistence_median_hours": float(np.median(episodes) * TIMESTEP_HOURS),
        "persistence_p90_hours": float(
            np.quantile(episodes, 0.90, method="linear") * TIMESTEP_HOURS
        ),
        "persistence_max_hours": float(np.max(episodes) * TIMESTEP_HOURS),
        "warmest_concordant_count": int(exact_warmest.sum()),
        "warmest_concordant_share": float(exact_warmest.mean()),
        "top3_warmest_count": int(top3_warmest.sum()),
        "top3_warmest_share": float(top3_warmest.mean()),
        "selected_temperature_rank_mean": float(np.mean(temperature_rank)),
        "selected_temperature_rank_median": float(np.median(temperature_rank)),
        "selected_to_warmest_temperature_gap_mean_c": float(
            np.mean(warmest_operative - selected_operative)
        ),
        "selected_to_warmest_temperature_gap_p90_c": float(
            np.quantile(
                warmest_operative - selected_operative, 0.90, method="linear"
            )
        ),
        "worst_cumulative_dh28_zone": ZONE_LABELS[worst_idx],
        "worst_cumulative_dh28_c_h": float(dh28_by_zone[worst_idx]),
        "worst_cumulative_dh28_margin_c_h": float(sorted_dh[0] - sorted_dh[1]),
        "worst_cumulative_zone_concordant_count": int(selected_worst.sum()),
        "worst_cumulative_zone_concordant_share": float(selected_worst.mean()),
        "warm_protection_rows": int(warm_state.sum()),
        "warm_protection_share": float(warm_state.mean()),
        "warm_protection_warmest_concordant_count": int(
            exact_warmest[warm_state].sum()
        ),
        "warm_protection_warmest_concordant_share": float(
            exact_warmest[warm_state].mean()
        ),
        "warm_protection_top3_warmest_count": int(top3_warmest[warm_state].sum()),
        "warm_protection_top3_warmest_share": float(top3_warmest[warm_state].mean()),
        "warm_protection_worst_cumulative_zone_concordant_count": int(
            selected_worst[warm_state].sum()
        ),
        "warm_protection_worst_cumulative_zone_concordant_share": float(
            selected_worst[warm_state].mean()
        ),
        "selected_p_tail_rank1_count": int((risk_rank == 1).sum()),
        "selected_p_tail_rank2_count": int((risk_rank == 2).sum()),
        "selected_p_tail_rank3_count": int((risk_rank == 3).sum()),
        "selector_target_max_abs_error": float(
            np.max(np.abs(selected_p_tail - targets))
        ),
    }
    frequencies = [
        {
            "scope": "scenario",
            "scaled_run_id": cell_id,
            "anchor_id": row["anchor_id"],
            "city": row["city"],
            "scenario": row["scenario"],
            "selector_role": row["selector_role"],
            "zone_order": zone_idx + 1,
            "zone": zone_label,
            "selected_count": int(selected_counts[zone_idx]),
            "selected_denominator": EXPECTED_OCCUPIED_ROWS,
            "selected_share": float(selected_counts[zone_idx] / EXPECTED_OCCUPIED_ROWS),
            "warm_protection_selected_count": int(warm_counts[zone_idx]),
            "warm_protection_denominator": int(warm_state.sum()),
            "warm_protection_selected_share": float(
                warm_counts[zone_idx] / warm_state.sum()
            ),
            "is_scenario_worst_cumulative_dh28_zone": zone_idx == worst_idx,
        }
        for zone_idx, zone_label in enumerate(ZONE_LABELS)
    ]
    internal = {
        "selected_counts": selected_counts,
        "warm_selected_counts": warm_counts,
        "episodes": episodes,
        "risk_rank": risk_rank,
        "selected_warmest": exact_warmest,
        "selected_top3": top3_warmest,
        "selected_worst": selected_worst,
        "warm_state": warm_state,
        "temperature_gap": warmest_operative - selected_operative,
        "switches": switches,
        "switch_opportunities": switch_opportunities,
    }
    return scenario, frequencies, internal


def summarize_group(
    group_type: str,
    group_value: str,
    cell_ids: list[str],
    internals: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    selected_counts = np.sum(
        [internals[cell]["selected_counts"] for cell in cell_ids], axis=0
    )
    warm_counts = np.sum(
        [internals[cell]["warm_selected_counts"] for cell in cell_ids], axis=0
    )
    episodes = np.concatenate([internals[cell]["episodes"] for cell in cell_ids])
    risk_rank = np.concatenate([internals[cell]["risk_rank"] for cell in cell_ids])
    selected_warmest = np.concatenate(
        [internals[cell]["selected_warmest"] for cell in cell_ids]
    )
    selected_top3 = np.concatenate(
        [internals[cell]["selected_top3"] for cell in cell_ids]
    )
    selected_worst = np.concatenate(
        [internals[cell]["selected_worst"] for cell in cell_ids]
    )
    warm_state = np.concatenate([internals[cell]["warm_state"] for cell in cell_ids])
    temperature_gap = np.concatenate(
        [internals[cell]["temperature_gap"] for cell in cell_ids]
    )
    switches = int(sum(internals[cell]["switches"] for cell in cell_ids))
    opportunities = int(
        sum(internals[cell]["switch_opportunities"] for cell in cell_ids)
    )
    total = int(selected_counts.sum())
    warm_total = int(warm_state.sum())
    mode_idx = int(np.argmax(selected_counts))
    require(total == len(cell_ids) * EXPECTED_OCCUPIED_ROWS, "group coverage")
    require(int(warm_counts.sum()) == warm_total, "warm group coverage")
    return {
        "group_type": group_type,
        "group_value": group_value,
        "scenario_count": len(cell_ids),
        "occupied_rows": total,
        "unique_selected_zones": int(np.count_nonzero(selected_counts)),
        "modal_selected_zone": ZONE_LABELS[mode_idx],
        "modal_selected_count": int(selected_counts[mode_idx]),
        "modal_selected_share": float(selected_counts[mode_idx] / total),
        "switch_opportunities": opportunities,
        "selected_zone_switches": switches,
        "selected_zone_switch_rate": float(switches / opportunities),
        "persistence_episode_count": int(len(episodes)),
        "persistence_mean_hours": float(np.mean(episodes) * TIMESTEP_HOURS),
        "persistence_median_hours": float(np.median(episodes) * TIMESTEP_HOURS),
        "persistence_p90_hours": float(
            np.quantile(episodes, 0.90, method="linear") * TIMESTEP_HOURS
        ),
        "persistence_max_hours": float(np.max(episodes) * TIMESTEP_HOURS),
        "warmest_concordant_count": int(selected_warmest.sum()),
        "warmest_concordant_share": float(selected_warmest.mean()),
        "top3_warmest_count": int(selected_top3.sum()),
        "top3_warmest_share": float(selected_top3.mean()),
        "worst_cumulative_zone_concordant_count": int(selected_worst.sum()),
        "worst_cumulative_zone_concordant_share": float(selected_worst.mean()),
        "warm_protection_rows": warm_total,
        "warm_protection_share": float(warm_total / total),
        "warm_protection_warmest_concordant_count": int(
            selected_warmest[warm_state].sum()
        ),
        "warm_protection_warmest_concordant_share": float(
            selected_warmest[warm_state].mean()
        ),
        "warm_protection_top3_warmest_count": int(
            selected_top3[warm_state].sum()
        ),
        "warm_protection_top3_warmest_share": float(
            selected_top3[warm_state].mean()
        ),
        "warm_protection_worst_cumulative_zone_concordant_count": int(
            selected_worst[warm_state].sum()
        ),
        "warm_protection_worst_cumulative_zone_concordant_share": float(
            selected_worst[warm_state].mean()
        ),
        "selected_temperature_gap_mean_c": float(np.mean(temperature_gap)),
        "selected_temperature_gap_p90_c": float(
            np.quantile(temperature_gap, 0.90, method="linear")
        ),
        "selected_p_tail_rank1_count": int((risk_rank == 1).sum()),
        "selected_p_tail_rank1_share": float((risk_rank == 1).mean()),
        "selected_p_tail_rank2_count": int((risk_rank == 2).sum()),
        "selected_p_tail_rank2_share": float((risk_rank == 2).mean()),
        "selected_p_tail_rank3_count": int((risk_rank == 3).sum()),
        "selected_p_tail_rank3_share": float((risk_rank == 3).mean()),
    }


def overall_zone_rows(
    internals: dict[str, dict[str, Any]], scenarios: pd.DataFrame
) -> list[dict[str, Any]]:
    selected_counts = np.sum(
        [data["selected_counts"] for data in internals.values()], axis=0
    )
    warm_counts = np.sum(
        [data["warm_selected_counts"] for data in internals.values()], axis=0
    )
    selected_total = int(selected_counts.sum())
    warm_total = int(warm_counts.sum())
    worst_counts = Counter(scenarios["worst_cumulative_dh28_zone"].tolist())
    return [
        {
            "scope": "all_24_scenarios",
            "scaled_run_id": "",
            "anchor_id": "",
            "city": "",
            "scenario": "",
            "selector_role": "",
            "zone_order": zone_idx + 1,
            "zone": zone_label,
            "selected_count": int(selected_counts[zone_idx]),
            "selected_denominator": selected_total,
            "selected_share": float(selected_counts[zone_idx] / selected_total),
            "warm_protection_selected_count": int(warm_counts[zone_idx]),
            "warm_protection_denominator": warm_total,
            "warm_protection_selected_share": float(
                warm_counts[zone_idx] / warm_total
            ),
            "is_scenario_worst_cumulative_dh28_zone": None,
            "worst_cumulative_dh28_scenario_count": int(worst_counts[zone_label]),
        }
        for zone_idx, zone_label in enumerate(ZONE_LABELS)
    ]


def validate_parity(review_root: Path) -> pd.DataFrame:
    closure = json.loads(
        (review_root / PARITY_CLOSURE_RELATIVE).read_text(encoding="utf-8")
    )
    require(
        closure.get("status") == "PASS"
        and closure.get("all_gates_passed") is True
        and closure.get("accepted_cell_id") == "SCALED-96-028",
        "parity closure did not pass",
    )
    columns = [f"zone_{slug}_p_disc" for slug in ZONE_SLUGS]
    frame = pd.read_parquet(
        review_root / PARITY_TRACE_RELATIVE,
        columns=["occupied", "guard_zone_index", "guard_zone", *columns],
    )
    occupied = frame.loc[
        frame["occupied"].fillna(False).astype(bool)
    ].reset_index(drop=True)
    require(len(occupied) == EXPECTED_OCCUPIED_ROWS, "parity occupied rows")
    indices, _ = selector_indices(occupied[columns].to_numpy(dtype=float))
    recorded = occupied["guard_zone_index"].to_numpy(dtype=np.int64)
    index_mismatch = indices != recorded
    labels = np.array([ZONE_LABELS[index] for index in indices], dtype=object)
    label_mismatch = labels != occupied["guard_zone"].to_numpy(dtype=object)
    require(not index_mismatch.any(), "parity index mismatch")
    require(not label_mismatch.any(), "parity label mismatch")
    return pd.DataFrame(
        [
            {
                "accepted_cell_id": "SCALED-96-028",
                "revision_parity_cell_id": "REV01-PARITY-001",
                "occupied_rows": EXPECTED_OCCUPIED_ROWS,
                "index_mismatch_count": int(index_mismatch.sum()),
                "label_mismatch_count": int(label_mismatch.sum()),
                "index_match_share": float((~index_mismatch).mean()),
                "label_match_share": float((~label_mismatch).mean()),
                "parity_trace_path": PARITY_TRACE_RELATIVE.as_posix(),
                "parity_trace_sha256": EXPECTED_PARITY_TRACE_SHA256,
                "parity_closure_path": PARITY_CLOSURE_RELATIVE.as_posix(),
                "parity_closure_sha256": EXPECTED_PARITY_CLOSURE_SHA256,
            }
        ]
    )


def render_report(
    scenarios: pd.DataFrame,
    groups: pd.DataFrame,
    zone_frequency: pd.DataFrame,
    parity: pd.DataFrame,
) -> str:
    overall = groups.loc[groups["group_value"].eq("all_24_scenarios")].iloc[0]
    overall_zones = zone_frequency.loc[
        zone_frequency["scope"].eq("all_24_scenarios")
    ].sort_values(["selected_share", "zone_order"], ascending=[False, True])
    top_zones = ", ".join(
        f"{row.zone} ({100.0 * row.selected_share:.2f}%)"
        for row in overall_zones.head(3).itertuples()
    )
    worst_counts = (
        scenarios["worst_cumulative_dh28_zone"]
        .value_counts()
        .rename_axis("zone")
        .reset_index(name="scenarios")
    )
    worst_text = ", ".join(
        f"{row.zone} ({row.scenarios})" for row in worst_counts.itertuples()
    )
    role_rows = groups.loc[groups["group_type"].eq("selector_role")]
    role_table = [
        "| Weather role | Switch rate | Exact warmest | Top-three warmest | Exact warmest during warm protection | Top-three during warm protection |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in role_rows.itertuples():
        role_table.append(
            f"| {row.group_value} | {100.0 * row.selected_zone_switch_rate:.2f}% | "
            f"{100.0 * row.warmest_concordant_share:.2f}% | "
            f"{100.0 * row.top3_warmest_share:.2f}% | "
            f"{100.0 * row.warm_protection_warmest_concordant_share:.2f}% | "
            f"{100.0 * row.warm_protection_top3_warmest_share:.2f}% |"
        )
    lines = [
        "# D13 learned-p90 spatial-sentinel trace report",
        "",
        "**Status: PASS.** This is a retained-output diagnostic over the 24 accepted learned-p90 primary scenarios; no EnergyPlus simulation was run.",
        "",
        "## Definitions",
        "",
        "At each occupied 15-minute record, the selected spatial sentinel is reconstructed as the actual zone nearest the linear spatial 0.90 quantile of the 15 zones' combined learned warm- and cold-tail probability. The accepted zone order supplies the first-index tie break.",
        "",
        "The contemporaneous warmest zone is the zone with the maximum operative temperature, `(air + mean radiant)/2`, at that record. The cumulative worst warm-exposure zone is the single zone in each scenario with the largest full-scenario occupied operative-temperature degree-hours above 28 degC. That latter identity is retrospective and was not available to the controller.",
        "",
        "A switch opportunity requires two adjacent 15-minute records that are both occupied. A persistence episode ends at a selected-zone change or an unoccupied gap. The warm-protection subset is reconstructed with selected total tail probability >= 0.20 and selected warm-minus-cold tail difference > 0.10; it matches the retained request branch exactly.",
        "",
        "## Coverage and reconstruction validation",
        "",
        f"All 24 hash-pinned traces passed: 35,040 records and 16,704 occupied records per scenario, for {int(overall.occupied_rows):,} occupied records total and all 15 zones. The explicit zone index in the independent Revision 01 parity trace matched the reconstructed index in {int(parity.iloc[0].occupied_rows):,}/{int(parity.iloc[0].occupied_rows):,} occupied records (zero mismatches).",
        "",
        "## Selected-zone identity and persistence",
        "",
        f"Every one of the 15 zones was selected at least once in every scenario. The three largest pooled selection shares were {top_zones}; the largest single-zone pooled share was only {100.0 * overall.modal_selected_share:.2f}%. The sentinel therefore did not collapse to one fixed zone.",
        "",
        f"Selected-zone identity changed on {int(overall.selected_zone_switches):,}/{int(overall.switch_opportunities):,} within-occupied adjacent transitions ({100.0 * overall.selected_zone_switch_rate:.2f}%). Across {int(overall.persistence_episode_count):,} persistence episodes, the median duration was {overall.persistence_median_hours:.2f} h, the 90th percentile was {overall.persistence_p90_hours:.2f} h, and the maximum was {overall.persistence_max_hours:.2f} h. These are sentinel identities, not setpoint moves; dwell and bound logic can prevent a sentinel change from becoming an actuator change.",
        "",
        "## Thermal concordance",
        "",
        f"Across all occupied records, the selected sentinel was the contemporaneous warmest zone in {int(overall.warmest_concordant_count):,}/{int(overall.occupied_rows):,} records ({100.0 * overall.warmest_concordant_share:.2f}%) and one of the three warmest zones in {int(overall.top3_warmest_count):,}/{int(overall.occupied_rows):,} ({100.0 * overall.top3_warmest_share:.2f}%). It matched the scenario's retrospective cumulative worst DH>28 zone in {int(overall.worst_cumulative_zone_concordant_count):,}/{int(overall.occupied_rows):,} ({100.0 * overall.worst_cumulative_zone_concordant_share:.2f}%).",
        "",
        f"The controller entered the reconstructed warm-protection branch in {int(overall.warm_protection_rows):,} occupied records ({100.0 * overall.warm_protection_share:.2f}%). Within those records, the sentinel was exactly the warmest zone in {int(overall.warm_protection_warmest_concordant_count):,}/{int(overall.warm_protection_rows):,} ({100.0 * overall.warm_protection_warmest_concordant_share:.2f}%) and was among the three warmest in {int(overall.warm_protection_top3_warmest_count):,}/{int(overall.warm_protection_rows):,} ({100.0 * overall.warm_protection_top3_warmest_share:.2f}%). It matched the scenario's retrospective cumulative worst DH>28 zone in {int(overall.warm_protection_worst_cumulative_zone_concordant_count):,}/{int(overall.warm_protection_rows):,} ({100.0 * overall.warm_protection_worst_cumulative_zone_concordant_share:.2f}%).",
        "",
        *role_table,
        "",
        f"The retrospective worst DH>28 identity was distributed as {worst_text}.",
        "",
        "## Interpretation boundary",
        "",
        "The learned-p90 sentinel is a dynamic high-tail-risk selector, not a direct maximum-temperature selector and not a proxy for the zone that eventually accumulates the largest annual DH>28. Its closer alignment with the warmest zones during warm-protection states documents operational thermal relevance, while the lower all-hours and retrospective-worst concordance prevents treating it as independently zonal control. These diagnostics characterize spatial supervision under synchronized building-level actuation; they do not test separate zone-level actuators.",
        "",
    ]
    return "\n".join(lines)


def rebuild(review_root: Path, manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    review_root = review_root.resolve()
    validate_review_archive(review_root)
    manifest = read_input_manifest(manifest_path)
    scenario_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    internals: dict[str, dict[str, Any]] = {}
    for _, row in manifest.iterrows():
        scenario, frequencies, internal = load_scenario(row, review_root)
        scenario_rows.append(scenario)
        frequency_rows.extend(frequencies)
        internals[str(row["scaled_run_id"])] = internal
    scenarios = pd.DataFrame(scenario_rows)
    require(
        len(scenarios) == EXPECTED_CELLS
        and (scenarios["unique_selected_zones"] == EXPECTED_ZONES).all(),
        "scenario coverage failure",
    )

    specs = [
        ("all", "all_24_scenarios", pd.Series(True, index=manifest.index)),
        (
            "selector_role",
            "present_typical",
            manifest["selector_role"].eq("present_typical"),
        ),
        (
            "selector_role",
            "future_extreme",
            manifest["selector_role"].eq("future_extreme"),
        ),
        ("scenario", "ssp245", manifest["scenario"].eq("ssp245")),
        ("scenario", "ssp585", manifest["scenario"].eq("ssp585")),
    ]
    group_rows = []
    for group_type, group_value, mask in specs:
        cell_ids = manifest.loc[mask, "scaled_run_id"].astype(str).tolist()
        group_rows.append(
            summarize_group(group_type, group_value, cell_ids, internals)
        )
    groups = pd.DataFrame(group_rows)

    frequency_rows.extend(overall_zone_rows(internals, scenarios))
    zone_frequency = pd.DataFrame(frequency_rows)
    zone_frequency["is_scenario_worst_cumulative_dh28_zone"] = zone_frequency[
        "is_scenario_worst_cumulative_dh28_zone"
    ].astype("boolean")
    if "worst_cumulative_dh28_scenario_count" not in zone_frequency:
        zone_frequency["worst_cumulative_dh28_scenario_count"] = np.nan
    zone_frequency = zone_frequency[
        [
            "scope",
            "scaled_run_id",
            "anchor_id",
            "city",
            "scenario",
            "selector_role",
            "zone_order",
            "zone",
            "selected_count",
            "selected_denominator",
            "selected_share",
            "warm_protection_selected_count",
            "warm_protection_denominator",
            "warm_protection_selected_share",
            "is_scenario_worst_cumulative_dh28_zone",
            "worst_cumulative_dh28_scenario_count",
        ]
    ]
    require(len(zone_frequency) == 375, "zone-frequency coverage failure")
    parity = validate_parity(review_root)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "d13_input_manifest.csv": write_csv(
            manifest, output_dir / "d13_input_manifest.csv"
        ),
        "d13_scenario_summary.csv": write_csv(
            scenarios, output_dir / "d13_scenario_summary.csv"
        ),
        "d13_group_summary.csv": write_csv(
            groups, output_dir / "d13_group_summary.csv"
        ),
        "d13_zone_frequency.csv": write_csv(
            zone_frequency, output_dir / "d13_zone_frequency.csv"
        ),
        "d13_reconstruction_validation.csv": write_csv(
            parity, output_dir / "d13_reconstruction_validation.csv"
        ),
    }
    report_path = output_dir / "D13_SPATIAL_TRACE_REPORT.md"
    report_path.write_text(
        render_report(scenarios, groups, zone_frequency, parity), encoding="utf-8"
    )
    outputs[report_path.name] = {
        "rows": len(report_path.read_text(encoding="utf-8").splitlines()),
        "sha256": sha256(report_path),
    }
    if DEFAULT_PROVENANCE.is_file():
        provenance = json.loads(DEFAULT_PROVENANCE.read_text(encoding="utf-8"))
        published_by_name = {
            Path(relative).name: record
            for relative, record in provenance["published_outputs"].items()
        }
        require(set(outputs) == set(published_by_name), "published output set mismatch")
        for name, record in outputs.items():
            require(
                record["sha256"] == published_by_name[name]["sha256"],
                f"rebuilt output differs from published D13 table: {name}",
            )
    return {"status": "PASS", "published_byte_parity": True, "outputs": outputs}


def verify_package(package_root: Path = PACKAGE_ROOT) -> dict[str, Any]:
    provenance_path = package_root / "diagnostics/d13_spatial_trace/D13_PUBLIC_PROVENANCE.json"
    require(provenance_path.is_file(), "missing public D13 provenance")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    require(provenance.get("status") == "PASS", "public D13 provenance status")
    for relative, record in provenance["published_outputs"].items():
        path = package_root / relative
        require(path.is_file(), f"missing public D13 output: {relative}")
        require(sha256(path) == record["sha256"], f"public D13 hash: {relative}")

    manifest = pd.read_csv(package_root / "summary_outputs/d13_input_manifest.csv")
    scenarios = pd.read_csv(package_root / "summary_outputs/d13_scenario_summary.csv")
    groups = pd.read_csv(package_root / "summary_outputs/d13_group_summary.csv")
    zones = pd.read_csv(package_root / "summary_outputs/d13_zone_frequency.csv")
    parity = pd.read_csv(
        package_root / "summary_outputs/d13_reconstruction_validation.csv"
    )
    require(len(manifest) == 24 and len(scenarios) == 24, "public D13 scenario coverage")
    require(len(groups) == 5 and len(zones) == 375, "public D13 table coverage")
    require(
        manifest["trace_sha256_recorded"].eq(
            manifest["trace_sha256_computed"]
        ).all(),
        "public D13 trace-hash record",
    )
    require((scenarios["unique_selected_zones"] == 15).all(), "public D13 zone coverage")
    require(
        int(parity.iloc[0]["index_mismatch_count"]) == 0
        and int(parity.iloc[0]["label_mismatch_count"]) == 0,
        "public D13 parity record",
    )
    overall = groups.loc[groups["group_value"].eq("all_24_scenarios")].iloc[0]
    require(int(overall["occupied_rows"]) == 400_896, "public D13 occupied total")
    require(int(overall["selected_zone_switches"]) == 237_794, "public D13 switches")
    require(int(overall["switch_opportunities"]) == 394_632, "public D13 opportunities")
    require(int(overall["warm_protection_rows"]) == 43_295, "public D13 warm rows")
    require(
        np.isclose(
            overall["warm_protection_top3_warmest_share"],
            0.76796396812565,
            rtol=0.0,
            atol=1e-14,
        ),
        "public D13 warm top-three concordance",
    )
    require(
        not any(Path(value).is_absolute() for value in manifest["trace_path"].astype(str)),
        "public D13 manifest contains an absolute path",
    )
    return {
        "status": "PASS",
        "scenarios": int(len(scenarios)),
        "occupied_rows": int(overall["occupied_rows"]),
        "published_outputs": int(len(provenance["published_outputs"])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-package",
        action="store_true",
        help="validate compact public D13 tables without external traces",
    )
    parser.add_argument(
        "--review-root",
        type=Path,
        help="path to the separately retained paperB_ENB_Rebuild review archive",
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="hash-pinned 24-trace input manifest",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="new directory for rebuilt compact outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_package:
        result = verify_package()
    else:
        require(args.review_root is not None, "--review-root is required to rebuild")
        require(args.output_dir is not None, "--output-dir is required to rebuild")
        result = rebuild(args.review_root, args.input_manifest, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
