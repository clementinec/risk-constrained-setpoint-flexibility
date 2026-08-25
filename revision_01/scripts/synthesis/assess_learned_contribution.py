#!/usr/bin/env python3
"""Closed assessment of learned-signal and learned-spatial contributions.

This is a derived analysis of the passing Rev01 v2 synthesis.  It does not
launch EnergyPlus or mutate any accepted/submitted artifact.  For M4 and M5 it
reorients the paired endpoints as accepted learned-p90 minus its matched
alternative and aligns the annual traces to count exact feedback-coupled
branch and actuator-outcome transitions.

Interpretive boundary:

* M4 isolates p90-zone versus building-mean aggregation of the same learned
  probability fields inside the same supervisory architecture.
* M5 isolates dependence on learned probability variation by replacing every
  probability vector with the frozen neutral signal.  The neutral policy is
  an always-relax structural null, not a contributor-disjoint refit, degraded
  predictor, deployment validation, or estimate of occupant transfer.
* Transition counts compare two closed-loop trajectories at the same annual
  timesteps.  They are not a no-feedback replay at identical thermal states.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA_VERSION = "paperb_enb_rev01_learned_contribution_assessment_v1"
EXPECTED_SYNTHESIS_SCHEMA = "paperb_enb_rev01_reviewer_synthesis_v2"
EXPECTED_SYNTHESIS_CLOSURE_SHA256 = (
    "703da705aa821642a91b2ed6cd4fcf3ad45127b09c3a62eee5f1609c2253c0d1"
)
EXPECTED_METRICS_SHA256 = (
    "2063a2c85f7ee55e8bd6394fc7d96f592b1095338f07514dcd3cd4ddf15115b4"
)
EXPECTED_MANIFEST_SHA256 = (
    "c10e79d341c9b8a0f4adc7f70dcbe6953023601ba8f49dd67514428a768b8c7e"
)
EXPECTED_EXPERIMENTS = {
    "M4": {
        "variant_id": "rev01_learned_mean_tail_full24",
        "alternative": "learned_mean_tail",
        "scope": "learned p90-zone aggregation minus learned building-mean aggregation",
    },
    "M5": {
        "variant_id": "rev01_p90_signal_a000_full24",
        "alternative": "neutral_always_relax_null",
        "scope": "learned p90 signal minus neutral always-relax p90 structural null",
    },
}
EXPECTED_ROWS_PER_EXPERIMENT = 24
EXPECTED_TRACE_ROWS = 35_040
EXPECTED_OCCUPIED_ROWS = 16_704
EXPECTED_TOTAL_OCCUPIED_PER_EXPERIMENT = 400_896
EXPECTED_M5_NULL_BRANCH = "relax"
ALIGN_COLUMNS = [
    "environment_num",
    "formal_weather_step",
    "month",
    "day",
    "hour",
    "current_time",
    "occupied",
]
TRACE_COLUMNS = ALIGN_COLUMNS + [
    "paperb_request_branch",
    "paperb_request_outcome",
]
BRANCH_ORDER = ["relax", "warm_protection", "cold_protection", "risk_hold"]
OUTCOME_ORDER = ["applied", "dwell_blocked", "bound_saturated", "hold"]
ENDPOINTS = {
    "total_delivered_site_electricity_plus_facility_gas_kwh": "delivered site energy (kWh)",
    "occupied_worst_zone_degree_hours_above_26c": "worst-zone DH >26 C (C h)",
    "occupied_worst_zone_degree_hours_above_28c": "worst-zone DH >28 C (C h)",
    "occupied_worst_zone_degree_hours_above_30c": "worst-zone DH >30 C (C h)",
    "occupied_worst_zone_degree_hours_below_18c": "worst-zone DH <18 C (C h)",
    "occupied_worst_zone_degree_hours_below_16c": "worst-zone DH <16 C (C h)",
    "occupied_mean_heating_setpoint_c": "occupied mean heating setpoint (C)",
    "occupied_mean_cooling_setpoint_c": "occupied mean cooling setpoint (C)",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_table(frame: pd.DataFrame, stem: Path) -> list[dict]:
    csv_path = stem.with_suffix(".csv")
    parquet_path = stem.with_suffix(".parquet")
    frame.to_csv(csv_path, index=False, lineterminator="\n", float_format="%.17g")
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False),
        parquet_path,
        compression="zstd",
        compression_level=9,
    )
    round_trip = pd.read_parquet(parquet_path)
    require(
        list(round_trip.columns) == list(frame.columns) and len(round_trip) == len(frame),
        f"Parquet structural round trip failed for {stem.name}",
    )
    return [
        {
            "path": csv_path.name,
            "format": "csv",
            "rows": len(frame),
            "columns": len(frame.columns),
            "sha256": sha256_file(csv_path),
        },
        {
            "path": parquet_path.name,
            "format": "parquet",
            "rows": len(frame),
            "columns": len(frame.columns),
            "sha256": sha256_file(parquet_path),
        },
    ]


def classify_counts(values: pd.Series, tolerance: float = 1e-9) -> tuple[int, int, int]:
    arr = values.to_numpy(dtype=float)
    return (
        int(np.count_nonzero(arr < -tolerance)),
        int(np.count_nonzero(np.abs(arr) <= tolerance)),
        int(np.count_nonzero(arr > tolerance)),
    )


def describe(values: pd.Series, prefix: str) -> dict:
    lower, equal, higher = classify_counts(values)
    return {
        f"mean_{prefix}": float(values.mean()),
        f"median_{prefix}": float(values.median()),
        f"min_{prefix}": float(values.min()),
        f"max_{prefix}": float(values.max()),
        f"learned_p90_lower_count_{prefix}": lower,
        f"equal_within_1e_9_count_{prefix}": equal,
        f"learned_p90_higher_count_{prefix}": higher,
    }


def transition_rows(
    experiment: str,
    cell_id: str,
    stress_grid_class: str,
    kind: str,
    accepted: pd.Series,
    alternative: pd.Series,
) -> list[dict]:
    table = pd.crosstab(accepted, alternative, dropna=False)
    rows: list[dict] = []
    for accepted_value in table.index:
        for alternative_value in table.columns:
            count = int(table.loc[accepted_value, alternative_value])
            if count:
                rows.append(
                    {
                        "experiment": experiment,
                        "cell_id": cell_id,
                        "stress_grid_class": stress_grid_class,
                        "transition_kind": kind,
                        "learned_p90_value": str(accepted_value),
                        "alternative_value": str(alternative_value),
                        "count": count,
                    }
                )
    return rows


def build_assessment(synthesis_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    closure_path = synthesis_dir / "SYNTHESIS_CLOSURE.json"
    metrics_path = synthesis_dir / "matched_cell_metrics.parquet"
    manifest_path = synthesis_dir / "input_artifact_manifest.csv"
    require(closure_path.is_file(), f"Missing {closure_path}")
    require(metrics_path.is_file(), f"Missing {metrics_path}")
    require(manifest_path.is_file(), f"Missing {manifest_path}")
    require(
        sha256_file(closure_path) == EXPECTED_SYNTHESIS_CLOSURE_SHA256,
        "Synthesis closure hash does not match the authorized v2 synthesis",
    )
    require(sha256_file(metrics_path) == EXPECTED_METRICS_SHA256, "Metrics hash mismatch")
    require(sha256_file(manifest_path) == EXPECTED_MANIFEST_SHA256, "Manifest hash mismatch")

    closure = load_json(closure_path)
    require(closure.get("schema_version") == EXPECTED_SYNTHESIS_SCHEMA, "Wrong synthesis schema")
    require(closure.get("status") == "PASS", "Synthesis closure did not pass")
    require(closure.get("all_gates_passed") is True, "Synthesis gates did not all pass")
    require(closure.get("matched_cells") == 136, "Expected the closed 136-cell synthesis")

    metrics = pd.read_parquet(metrics_path)
    learned = metrics.loc[metrics["experiment"].isin(EXPECTED_EXPERIMENTS)].copy()
    require(len(learned) == 48, "Expected exactly 48 M4/M5 cells")
    for experiment, contract in EXPECTED_EXPERIMENTS.items():
        subset = learned.loc[learned["experiment"] == experiment]
        require(len(subset) == EXPECTED_ROWS_PER_EXPERIMENT, f"Wrong {experiment} cell count")
        require(
            set(subset["variant_id"]) == {contract["variant_id"]},
            f"Wrong {experiment} variant",
        )
        require(subset["anchor_id"].nunique() == 24, f"Wrong {experiment} anchor coverage")

    manifest = pd.read_csv(manifest_path)
    cell_rows: list[dict] = []
    transitions: list[dict] = []
    trace_hash_checks = 0

    for row in learned.sort_values(["experiment", "cell_id"]).itertuples(index=False):
        artifact_rows = manifest.loc[manifest["match_cell_id"] == row.cell_id]
        require(
            set(artifact_rows["artifact_role"]) == {"accepted", "revision"}
            and len(artifact_rows) == 2,
            f"Wrong artifact role coverage for {row.cell_id}",
        )
        artifacts = artifact_rows.set_index("artifact_role")
        traces: dict[str, pd.DataFrame] = {}
        for role in ("accepted", "revision"):
            trace_path = Path(artifacts.loc[role, "trace_path"])
            expected_hash = str(artifacts.loc[role, "trace_sha256"])
            require(trace_path.is_file(), f"Missing trace {trace_path}")
            require(sha256_file(trace_path) == expected_hash, f"Trace hash mismatch: {trace_path}")
            trace_hash_checks += 1
            traces[role] = pd.read_parquet(trace_path, columns=TRACE_COLUMNS)
            require(len(traces[role]) == EXPECTED_TRACE_ROWS, f"Wrong trace rows for {row.cell_id}")

        accepted = traces["accepted"]
        alternative = traces["revision"]
        require(
            accepted[ALIGN_COLUMNS].equals(alternative[ALIGN_COLUMNS]),
            f"Trace calendars/occupancy do not align for {row.cell_id}",
        )
        occupied = accepted["occupied"].astype(bool)
        require(int(occupied.sum()) == EXPECTED_OCCUPIED_ROWS, f"Wrong occupied rows for {row.cell_id}")
        accepted_occ = accepted.loc[occupied]
        alternative_occ = alternative.loc[occupied]

        if row.experiment == "M5":
            require(
                set(alternative_occ["paperb_request_branch"]) == {EXPECTED_M5_NULL_BRANCH},
                f"M5 null was not always-relax for {row.cell_id}",
            )

        accepted_branch = accepted_occ["paperb_request_branch"].astype(str)
        alternative_branch = alternative_occ["paperb_request_branch"].astype(str)
        accepted_outcome = accepted_occ["paperb_request_outcome"].astype(str)
        alternative_outcome = alternative_occ["paperb_request_outcome"].astype(str)
        branch_changed = int((accepted_branch.to_numpy() != alternative_branch.to_numpy()).sum())
        outcome_changed = int((accepted_outcome.to_numpy() != alternative_outcome.to_numpy()).sum())

        transitions.extend(
            transition_rows(
                row.experiment,
                row.cell_id,
                row.stress_grid_class,
                "request_branch",
                accepted_branch,
                alternative_branch,
            )
        )
        transitions.extend(
            transition_rows(
                row.experiment,
                row.cell_id,
                row.stress_grid_class,
                "request_outcome",
                accepted_outcome,
                alternative_outcome,
            )
        )

        record = {
            "experiment": row.experiment,
            "variant_id": row.variant_id,
            "cell_id": row.cell_id,
            "accepted_comparator_cell_id": row.accepted_comparator_cell_id,
            "anchor_id": row.anchor_id,
            "stress_grid_class": row.stress_grid_class,
            "city": row.city,
            "scenario": row.scenario,
            "alternative": EXPECTED_EXPERIMENTS[row.experiment]["alternative"],
            "contrast_scope": EXPECTED_EXPERIMENTS[row.experiment]["scope"],
            "occupied_decisions": EXPECTED_OCCUPIED_ROWS,
            "branch_changed_count": branch_changed,
            "branch_changed_pct": 100.0 * branch_changed / EXPECTED_OCCUPIED_ROWS,
            "outcome_changed_count": outcome_changed,
            "outcome_changed_pct": 100.0 * outcome_changed / EXPECTED_OCCUPIED_ROWS,
        }
        for endpoint in ENDPOINTS:
            record[f"learned_p90_minus_alternative_{endpoint}"] = -float(
                getattr(row, f"delta_{endpoint}")
            )
        record["learned_p90_minus_alternative_energy_pct_of_learned_p90"] = -float(
            row.delta_total_delivered_site_energy_pct_of_accepted
        )
        cell_rows.append(record)

    cells = pd.DataFrame(cell_rows)
    transition_frame = pd.DataFrame(transitions)
    require(trace_hash_checks == 96, "Expected 96 M4/M5 trace hash checks")

    summary_rows: list[dict] = []
    for experiment in EXPECTED_EXPERIMENTS:
        experiment_cells = cells.loc[cells["experiment"] == experiment]
        for aggregation, subset in (
            ("all_stress_test_cells", experiment_cells),
            (
                "present_typical",
                experiment_cells.loc[experiment_cells["stress_grid_class"] == "present_typical"],
            ),
            (
                "late_century_hot_extreme",
                experiment_cells.loc[
                    experiment_cells["stress_grid_class"] == "late_century_hot_extreme"
                ],
            ),
        ):
            require(len(subset) in {12, 24}, f"Wrong aggregation size for {experiment}/{aggregation}")
            summary = {
                "experiment": experiment,
                "variant_id": EXPECTED_EXPERIMENTS[experiment]["variant_id"],
                "alternative": EXPECTED_EXPERIMENTS[experiment]["alternative"],
                "contrast_scope": EXPECTED_EXPERIMENTS[experiment]["scope"],
                "aggregation": aggregation,
                "n_matched_cells": len(subset),
                "occupied_decisions": int(subset["occupied_decisions"].sum()),
                "branch_changed_count": int(subset["branch_changed_count"].sum()),
                "branch_changed_pct": float(
                    100.0
                    * subset["branch_changed_count"].sum()
                    / subset["occupied_decisions"].sum()
                ),
                "outcome_changed_count": int(subset["outcome_changed_count"].sum()),
                "outcome_changed_pct": float(
                    100.0
                    * subset["outcome_changed_count"].sum()
                    / subset["occupied_decisions"].sum()
                ),
            }
            energy_pct = subset[
                "learned_p90_minus_alternative_energy_pct_of_learned_p90"
            ]
            summary.update(describe(energy_pct, "energy_pct_of_learned_p90"))
            for endpoint in ENDPOINTS:
                column = f"learned_p90_minus_alternative_{endpoint}"
                summary.update(describe(subset[column], endpoint))
            summary_rows.append(summary)

    summaries = pd.DataFrame(summary_rows)
    require(len(summaries) == 6, "Expected six M4/M5 summary rows")
    for experiment in EXPECTED_EXPERIMENTS:
        total = summaries.loc[
            (summaries["experiment"] == experiment)
            & (summaries["aggregation"] == "all_stress_test_cells"),
            "occupied_decisions",
        ].item()
        require(total == EXPECTED_TOTAL_OCCUPIED_PER_EXPERIMENT, f"Wrong {experiment} decision total")

    mechanism = {
        "trace_hash_checks": trace_hash_checks,
        "m4_occupied_decisions": EXPECTED_TOTAL_OCCUPIED_PER_EXPERIMENT,
        "m5_occupied_decisions": EXPECTED_TOTAL_OCCUPIED_PER_EXPERIMENT,
        "m5_null_all_relax": True,
    }
    return cells, summaries, transition_frame, mechanism


def format_count_triplet(row: pd.Series, metric: str) -> str:
    return (
        f"{int(row[f'learned_p90_lower_count_{metric}'])}/"
        f"{int(row[f'equal_within_1e_9_count_{metric}'])}/"
        f"{int(row[f'learned_p90_higher_count_{metric}'])}"
    )


def render_digest(summaries: pd.DataFrame, transitions: pd.DataFrame) -> str:
    all_rows = summaries.loc[summaries["aggregation"] == "all_stress_test_cells"].set_index(
        "experiment"
    )
    m4 = all_rows.loc["M4"]
    m5 = all_rows.loc["M5"]

    m5_branch = transitions.loc[
        (transitions["experiment"] == "M5")
        & (transitions["transition_kind"] == "request_branch")
    ]
    m5_branch = (
        m5_branch.groupby(["learned_p90_value", "alternative_value"], as_index=False)["count"]
        .sum()
        .sort_values(["learned_p90_value", "alternative_value"])
    )
    m5_nonrelax = int(
        m5_branch.loc[m5_branch["learned_p90_value"] != "relax", "count"].sum()
    )

    lines = [
        "# Learned-predictor contribution assessment",
        "",
        "This assessment separates two questions across all 24 frozen scenario anchors:",
        "",
        "1. **M4 spatial aggregation:** accepted learned p90-zone selection versus a learned building-mean tail.",
        "2. **M5 learned signal:** accepted learned p90 probabilities versus a neutral, always-relax structural null.",
        "",
        "All signs below are **accepted learned p90 minus its alternative**. Negative degree-hours mean the learned p90 trajectory had less exposure on that diagnostic; negative energy means it used less delivered site energy.",
        "",
        "## Full-grid results",
        "",
        "| contrast | branch decisions changed | energy (%) | DH >26 C | DH >28 C | DH >30 C | DH <18 C |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| M4: learned p90 minus learned mean | {int(m4['branch_changed_count']):,} "
            f"({m4['branch_changed_pct']:.3f}%) | {m4['mean_energy_pct_of_learned_p90']:+.3f} | "
            f"{m4['mean_occupied_worst_zone_degree_hours_above_26c']:+.3f} | "
            f"{m4['mean_occupied_worst_zone_degree_hours_above_28c']:+.3f} | "
            f"{m4['mean_occupied_worst_zone_degree_hours_above_30c']:+.3f} | "
            f"{m4['mean_occupied_worst_zone_degree_hours_below_18c']:+.3f} |"
        ),
        (
            f"| M5: learned p90 minus neutral null | {int(m5['branch_changed_count']):,} "
            f"({m5['branch_changed_pct']:.3f}%) | {m5['mean_energy_pct_of_learned_p90']:+.3f} | "
            f"{m5['mean_occupied_worst_zone_degree_hours_above_26c']:+.3f} | "
            f"{m5['mean_occupied_worst_zone_degree_hours_above_28c']:+.3f} | "
            f"{m5['mean_occupied_worst_zone_degree_hours_above_30c']:+.3f} | "
            f"{m5['mean_occupied_worst_zone_degree_hours_below_18c']:+.3f} |"
        ),
        "",
        "Directional counts (learned p90 lower / equal / higher) show that the mean can conceal heterogeneous signs:",
        "",
        "| contrast | energy | DH >26 C | DH >28 C | DH >30 C | DH <18 C |",
        "|---|---:|---:|---:|---:|---:|",
        (
            "| M4 | "
            f"{format_count_triplet(m4, 'energy_pct_of_learned_p90')} | "
            f"{format_count_triplet(m4, 'occupied_worst_zone_degree_hours_above_26c')} | "
            f"{format_count_triplet(m4, 'occupied_worst_zone_degree_hours_above_28c')} | "
            f"{format_count_triplet(m4, 'occupied_worst_zone_degree_hours_above_30c')} | "
            f"{format_count_triplet(m4, 'occupied_worst_zone_degree_hours_below_18c')} |"
        ),
        (
            "| M5 | "
            f"{format_count_triplet(m5, 'energy_pct_of_learned_p90')} | "
            f"{format_count_triplet(m5, 'occupied_worst_zone_degree_hours_above_26c')} | "
            f"{format_count_triplet(m5, 'occupied_worst_zone_degree_hours_above_28c')} | "
            f"{format_count_triplet(m5, 'occupied_worst_zone_degree_hours_above_30c')} | "
            f"{format_count_triplet(m5, 'occupied_worst_zone_degree_hours_below_18c')} |"
        ),
        "",
        "## What the runs reveal",
        "",
        (
            f"- The learned probabilities had a **non-negligible decision-level effect**. "
            f"Neutralization changed {int(m5['branch_changed_count']):,} of "
            f"{int(m5['occupied_decisions']):,} occupied branches ({m5['branch_changed_pct']:.3f}%). "
            f"Exactly {m5_nonrelax:,} accepted non-relax decisions became relax decisions under the null."
        ),
        (
            f"- Their **annual endpoint effect was modest and metric-dependent**. Relative to the null, "
            f"learned p90 used {m5['mean_energy_pct_of_learned_p90']:+.3f}% energy on the frozen p90-denominator convention, "
            f"reduced DH >26 C by {-m5['mean_occupied_worst_zone_degree_hours_above_26c']:.3f} C h and DH >28 C by "
            f"{-m5['mean_occupied_worst_zone_degree_hours_above_28c']:.3f} C h on average, but increased DH >30 C by "
            f"{m5['mean_occupied_worst_zone_degree_hours_above_30c']:.3f} C h."
        ),
        (
            f"- Learned p90 spatial selection also changed decisions ({int(m4['branch_changed_count']):,}; "
            f"{m4['branch_changed_pct']:.3f}%), but its incremental full-grid effect on DH >28 C was only "
            f"{m4['mean_occupied_worst_zone_degree_hours_above_28c']:+.3f} C h. It reduced DH >26 C by "
            f"{-m4['mean_occupied_worst_zone_degree_hours_above_26c']:.3f} C h while using "
            f"{m4['mean_energy_pct_of_learned_p90']:+.3f}% more energy than learned mean-tail."
        ),
        "- Therefore the learned signal is best described as a protection modifier inside a relaxation-dominated supervisor, not as the isolated source of the reported energy savings. The spatial signal is not useless, but its benefit is threshold-specific and does not provide a robust improvement in the submitted >28 C endpoint.",
        "",
        "## Boundaries",
        "",
        "- M5 is an always-relax structural null. It does not test contributor-disjoint refits, calibration degradation, random noise, personalization, or unseen-occupant transfer.",
        "- M4 changes learned-probability aggregation but retains synchronous building-level actuation; it is not a zone-level actuation test.",
        "- The present-typical and late-century-hot-extreme strata use different year selectors. Their contrast remains descriptive, not a causal future-climate effect.",
        "- Degree-hours are operative-temperature exposure diagnostics, not observed comfort, satisfaction, health outcomes, or standards compliance.",
        "- Branch transitions align separate feedback-coupled annual trajectories at common timesteps; they are not a no-feedback identical-state replay.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthesis-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    synthesis_dir = args.synthesis_dir.resolve()
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), f"Output directory already exists: {output_dir}")

    cells, summaries, transitions, mechanism = build_assessment(synthesis_dir)
    temporary_dir = output_dir.parent / f".{output_dir.name}.tmp.{uuid.uuid4().hex}"
    require(not temporary_dir.exists(), f"Temporary directory already exists: {temporary_dir}")
    temporary_dir.mkdir(parents=True)

    outputs: list[dict] = []
    try:
        outputs.extend(write_table(cells, temporary_dir / "learned_contribution_cell_metrics"))
        outputs.extend(write_table(summaries, temporary_dir / "learned_contribution_summary"))
        outputs.extend(write_table(transitions, temporary_dir / "feedback_branch_outcome_transitions"))

        digest_path = temporary_dir / "LEARNED_CONTRIBUTION_ASSESSMENT.md"
        digest_path.write_text(render_digest(summaries, transitions), encoding="utf-8")
        outputs.append(
            {
                "path": digest_path.name,
                "format": "markdown",
                "sha256": sha256_file(digest_path),
            }
        )

        closure = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "PASS",
            "all_gates_passed": True,
            "source_synthesis_dir": os.fspath(synthesis_dir),
            "source_synthesis_closure_sha256": EXPECTED_SYNTHESIS_CLOSURE_SHA256,
            "source_metrics_sha256": EXPECTED_METRICS_SHA256,
            "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "experiments": {key: value["variant_id"] for key, value in EXPECTED_EXPERIMENTS.items()},
            "matched_cells": len(cells),
            "summary_rows": len(summaries),
            "transition_rows": len(transitions),
            "mechanism_checks": mechanism,
            "closure_gates": {
                "authorized_v2_synthesis_hash_exact": True,
                "exact_m4_m5_48_cell_coverage": True,
                "all_96_trace_hashes_exact": True,
                "annual_timesteps_and_occupancy_aligned": True,
                "m5_null_always_relax": True,
                "signs_oriented_as_learned_p90_minus_alternative": True,
                "feedback_transition_interpretation_bounded": True,
            },
            "outputs": outputs,
            "script_sha256": sha256_file(Path(__file__).resolve()),
        }
        closure_path = temporary_dir / "ASSESSMENT_CLOSURE.json"
        closure_path.write_text(
            json.dumps(closure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_dir, output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    print(f"PASS: learned-contribution assessment written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
