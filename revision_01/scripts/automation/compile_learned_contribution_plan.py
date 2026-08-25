#!/usr/bin/env python3
"""Compile the full-grid learned-contribution extension for Paper B Rev01.

This additive compiler creates exactly 48 fresh scientific cells over the
accepted 24-anchor grid:

* M4: learned mean-tail aggregation versus the accepted learned p90 policy;
* M5: a neutral learned-signal null versus the accepted learned p90 policy.

The accepted Fresh96 archive and the previously frozen Rev01 matrices remain
read-only.  Both contrasts reuse the accepted 24 learned-p90 cells only as
comparison references; no accepted output is copied, spliced, or rerun.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from compile_rev01_plan import (
    DEFAULT_ACCEPTED_MATRIX,
    EXPECTED_ACCEPTED_MATRIX_SHA256,
    EXPECTED_ACCEPTED_RUNNER_SHA256,
    MATRIX_SCHEMA_VERSION,
    VARIANT_SCHEMA_VERSION,
    PlanContractError,
    make_cell_row,
    manifest_entry,
    sha256_file,
    validate_accepted_matrix,
    variant,
    write_parquet_exact,
)


SCHEMA_VERSION = "paperb_enb_rev01_compiled_plan_v1"
BATCH_ID = "rev01_learned48"
MATRIX_NAME = "learned48"
EXPECTED_ANCHORS = 24
EXPECTED_CELLS = 48


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def learned_contribution_variants() -> tuple[dict[str, Any], ...]:
    """Return the two full-grid, one-factor-at-a-time policy variants."""

    return (
        variant(
            "rev01_learned_mean_tail_full24",
            "paperb_gate_tail_asym_relax",
            "M4",
            (
                "Full-grid learned spatial-aggregation ablation: average the "
                "15 zone warm/cold probability tails before applying the same "
                "tail/asymmetry guard, actuator bounds, and dwell rules."
            ),
        ),
        variant(
            "rev01_p90_signal_a000_full24",
            "paperb_p90_tail_asym_relax",
            "M5",
            (
                "Full-grid learned-signal null: replace every frozen-model "
                "seven-class probability vector with neutral TSV while "
                "retaining the p90 selector and all policy mechanics."
            ),
            paperb_signal_alpha=0.0,
        ),
    )


def compile_learned_contribution_matrix(
    accepted: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    anchors = (
        accepted.loc[accepted["strategy"].astype(str).eq("diagnostic_reference")]
        .sort_values("anchor_order")
        .reset_index(drop=True)
    )
    if len(anchors) != EXPECTED_ANCHORS or anchors["anchor_id"].nunique() != EXPECTED_ANCHORS:
        raise PlanContractError("Learned-contribution plan requires all 24 accepted anchors")

    variants = learned_contribution_variants()
    rows: list[dict[str, Any]] = []
    sequence = 0
    for selected_variant in variants:
        for _index, source in anchors.iterrows():
            sequence += 1
            rows.append(
                make_cell_row(
                    source,
                    selected_variant,
                    batch_id=BATCH_ID,
                    cell_id=f"REV01-L048-{sequence:03d}",
                    accepted_matrix=accepted,
                )
            )

    matrix = pd.DataFrame.from_records(rows)
    if len(matrix) != EXPECTED_CELLS or matrix["cell_id"].nunique() != EXPECTED_CELLS:
        raise PlanContractError("Learned-contribution compilation did not close at 48 cells")
    if matrix.groupby("experiment").size().to_dict() != {"M4": 24, "M5": 24}:
        raise PlanContractError("Learned-contribution experiment coverage differs")
    if set(matrix["accepted_comparator_strategy"].astype(str)) != {
        "paperb_p90_tail_asym_relax"
    }:
        raise PlanContractError("Every learned-contribution cell must compare to accepted p90")
    if matrix.groupby("experiment")["anchor_id"].nunique().to_dict() != {"M4": 24, "M5": 24}:
        raise PlanContractError("Each learned-contribution experiment must span all 24 anchors")
    if matrix.groupby("experiment")["accepted_comparator_cell_id"].nunique().to_dict() != {
        "M4": 24,
        "M5": 24,
    }:
        raise PlanContractError("Accepted p90 comparator mapping is incomplete")

    manifest = {
        "schema_version": VARIANT_SCHEMA_VERSION,
        "parent_runner_sha256": EXPECTED_ACCEPTED_RUNNER_SHA256,
        "variants": {
            item["variant_id"]: manifest_entry(item)
            for item in variants
        },
    }
    return matrix, manifest


def freeze_plan(
    output_dir: Path,
    *,
    accepted_matrix_path: Path,
    matrix: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Plan output must be absent: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    validate_accepted_matrix(accepted_matrix_path)

    manifest_path = output_dir / "VARIANT_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / f"{MATRIX_NAME}.csv"
    parquet_path = output_dir / f"{MATRIX_NAME}.parquet"
    matrix.to_csv(csv_path, index=False)
    write_parquet_exact(matrix, parquet_path)

    matrices = {
        MATRIX_NAME: {
            "rows": int(len(matrix)),
            "csv_path": str(csv_path.resolve()),
            "csv_sha256": sha256_file(csv_path),
            "parquet_path": str(parquet_path.resolve()),
            "parquet_sha256": sha256_file(parquet_path),
            "unique_cell_ids": int(matrix["cell_id"].nunique()),
        }
    }
    closure = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "all_gates_passed": True,
        "created_utc": utc_now(),
        "scope": "full_grid_learned_predictor_contribution",
        "accepted_matrix_path": str(accepted_matrix_path.resolve()),
        "accepted_matrix_sha256": sha256_file(accepted_matrix_path),
        "learned_spatial_ablation_cells": 24,
        "learned_signal_null_cells": 24,
        "total_scientific_cells": EXPECTED_CELLS,
        "total_execution_cells": EXPECTED_CELLS,
        "total_cells": EXPECTED_CELLS,
        "accepted_comparator_cells_reused_read_only": 24,
        "conditional_batches_inferred_at_runtime": False,
        "city_ddy_improvised": False,
        "variant_manifest_path": str(manifest_path.resolve()),
        "variant_manifest_sha256": sha256_file(manifest_path),
        "matrices": matrices,
    }
    closure_path = output_dir / "PLAN_CLOSURE.json"
    closure_path.write_text(
        json.dumps(closure, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return closure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-matrix", type=Path, default=DEFAULT_ACCEPTED_MATRIX)
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--freeze", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    accepted = validate_accepted_matrix(args.accepted_matrix)
    matrix, manifest = compile_learned_contribution_matrix(accepted)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "accepted_matrix_sha256": sha256_file(args.accepted_matrix),
        "expected_accepted_matrix_sha256": EXPECTED_ACCEPTED_MATRIX_SHA256,
        "learned_spatial_ablation_cells": 24,
        "learned_signal_null_cells": 24,
        "total_cells": len(matrix),
        "variant_count": len(manifest["variants"]),
        "filesystem_mutated": False,
    }
    if args.preflight_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.output_dir is None:
        raise PlanContractError("--freeze requires --output-dir")
    closure = freeze_plan(
        args.output_dir,
        accepted_matrix_path=args.accepted_matrix,
        matrix=matrix,
        manifest=manifest,
    )
    print(json.dumps(closure, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
