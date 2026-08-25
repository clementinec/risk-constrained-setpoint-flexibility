#!/usr/bin/env python3
"""Compile the bounded Paper B ENB Rev01 scientific matrices.

The accepted Fresh96 matrix is read-only anchor/comparator authority.  This
compiler creates an explicit 64-cell core matrix and, only when requested with
trigger evidence, a separate 24-cell adaptive-clamp matrix.  It never infers a
conditional batch during execution and it has no sizing-DDY improvisation
path.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA_VERSION = "paperb_enb_rev01_compiled_plan_v1"
MATRIX_SCHEMA_VERSION = "paperb_enb_rev01_cell_matrix_v1"
VARIANT_SCHEMA_VERSION = "paperb_enb_rev01_variant_manifest_v1"
EXPECTED_ACCEPTED_MATRIX_SHA256 = (
    "fa03767f0c3d581e65e00ff6149546c44fe3e936608da73ad2f5485bf19895e0"
)
EXPECTED_ACCEPTED_RUNNER_SHA256 = (
    "a9693042883ed5c20edad6fcbc757c62c7216d5abc120ad18de1a003932848a4"
)
EXPECTED_IDF_SHA256 = (
    "1144b58b848992d1730e49ef9c252569e3a515d82a2f99b2c0233352f625a7e4"
)
EXPECTED_MODEL_SHA256 = (
    "6fbeb06644a36b226b17824a1fca7526bd518e0a9b76a41f878fb1c27efcd619"
)
EXPECTED_ROWS = 35_040
EXPECTED_OCCUPIED_ROWS = 16_704

REBUILD_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ACCEPTED_MATRIX = (
    REBUILD_ROOT
    / "06_runs/scaled/20260803_routeS96_v2_envscoped/freeze/authorization/execution"
    / "scaled_matrix_fresh96_v2_envscoped.csv"
)

SENTINELS = (
    "beijing_ssp585_present_typical",
    "guangzhou_ssp585_present_typical",
    "kolkata_ssp585_future_extreme",
    "phoenix_ssp585_future_extreme",
)

ACCEPTED_DEFAULTS: dict[str, Any] = {
    "paperb_spatial_quantile": 0.90,
    "paperb_tail_threshold": 0.20,
    "paperb_asym_threshold": 0.10,
    "paperb_save_heat_c": 20.0,
    "paperb_save_cool_c": 26.0,
    "paperb_warm_protect_cool_c": 23.25,
    "paperb_cold_protect_heat_c": 23.25,
    "paperb_tighten_dwell_steps": 4,
    "paperb_relax_dwell_steps": 1,
    "paperb_pmv_threshold": 0.50,
    "paperb_signal_alpha": 1.0,
    "paperb_adaptive_rm_clamp_low_c": 10.0,
    "paperb_adaptive_rm_clamp_high_c": 33.5,
    "paperb_met": 0.854333,
    "paperb_people_activity_w": 93.2,
    "controller_semantics": "corrected",
    "inference_preprocessing": "training_contract",
    "sizing_contract": "accepted_denver",
}


class PlanContractError(RuntimeError):
    """A source, scope, or output contract failed closed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def variant(
    variant_id: str,
    strategy: str,
    experiment: str,
    rationale: str,
    **changes: Any,
) -> dict[str, Any]:
    result = dict(ACCEPTED_DEFAULTS)
    result.update(
        {
            "variant_id": variant_id,
            "strategy": strategy,
            "experiment": experiment,
            "rationale": rationale,
        }
    )
    result.update(changes)
    return result


def core_variants() -> tuple[dict[str, Any], ...]:
    return (
        variant(
            "rev01_p90_abs_pmv_relax",
            "paperb_p90_abs_pmv_relax",
            "M1",
            "Matched PMV spatial-aggregation ablation over all 24 anchors.",
        ),
        variant(
            "rev01_p90_q075",
            "paperb_p90_tail_asym_relax",
            "M2",
            "One-at-a-time spatial-quantile sensitivity.",
            paperb_spatial_quantile=0.75,
        ),
        variant(
            "rev01_p90_q100",
            "paperb_p90_tail_asym_relax",
            "M2",
            "One-at-a-time spatial-quantile sensitivity.",
            paperb_spatial_quantile=1.00,
        ),
        variant(
            "rev01_p90_ptail015",
            "paperb_p90_tail_asym_relax",
            "M2",
            "One-at-a-time tail-threshold sensitivity.",
            paperb_tail_threshold=0.15,
        ),
        variant(
            "rev01_p90_ptail025",
            "paperb_p90_tail_asym_relax",
            "M2",
            "One-at-a-time tail-threshold sensitivity.",
            paperb_tail_threshold=0.25,
        ),
        variant(
            "rev01_p90_dtail005",
            "paperb_p90_tail_asym_relax",
            "M2",
            "One-at-a-time directional-threshold sensitivity.",
            paperb_asym_threshold=0.05,
        ),
        variant(
            "rev01_p90_dtail015",
            "paperb_p90_tail_asym_relax",
            "M2",
            "One-at-a-time directional-threshold sensitivity.",
            paperb_asym_threshold=0.15,
        ),
        variant(
            "rev01_p90_bounds21_25",
            "paperb_p90_tail_asym_relax",
            "M2",
            "Conservative relaxation-bound sensitivity.",
            paperb_save_heat_c=21.0,
            paperb_save_cool_c=25.0,
        ),
        variant(
            "rev01_p90_bounds19_27",
            "paperb_p90_tail_asym_relax",
            "M2",
            "Permissive relaxation-bound sensitivity.",
            paperb_save_heat_c=19.0,
            paperb_save_cool_c=27.0,
        ),
        variant(
            "rev01_p90_signal_a050",
            "paperb_p90_tail_asym_relax",
            "M3",
            "Frozen learned probability signal shrunk halfway to neutral.",
            paperb_signal_alpha=0.50,
        ),
        variant(
            "rev01_p90_signal_a000",
            "paperb_p90_tail_asym_relax",
            "M3",
            "Neutral policy-only null with no learned probability variation.",
            paperb_signal_alpha=0.00,
        ),
    )


def adaptive_variant() -> dict[str, Any]:
    return variant(
        "rev01_adaptive_rm_clamp_10_33p5",
        "paperb_adaptive_band_clamped_relax",
        "C1",
        "Adaptive-band sensitivity using running mean clipped to 10--33.5 C.",
    )


def parity_variant() -> dict[str, Any]:
    return variant(
        "rev01_parity_submitted_p90",
        "paperb_p90_tail_asym_relax",
        "T1",
        "Technical annual parity sentinel at every accepted submitted setting.",
    )


def manifest_entry(item: dict[str, Any]) -> dict[str, Any]:
    """Translate compiler names into the runner's exact canonical manifest."""

    effective_config = {
        "strategy": item["strategy"],
        "spatial_quantile": item["paperb_spatial_quantile"],
        "signal_alpha": item["paperb_signal_alpha"],
        "tail_threshold": item["paperb_tail_threshold"],
        "asym_threshold": item["paperb_asym_threshold"],
        "save_heat_c": item["paperb_save_heat_c"],
        "save_cool_c": item["paperb_save_cool_c"],
        "warm_protect_cool_c": item["paperb_warm_protect_cool_c"],
        "cold_protect_heat_c": item["paperb_cold_protect_heat_c"],
        "tighten_dwell_steps": item["paperb_tighten_dwell_steps"],
        "relax_dwell_steps": item["paperb_relax_dwell_steps"],
        "pmv_threshold": item["paperb_pmv_threshold"],
        "adaptive_rm_clamp_low_c": item["paperb_adaptive_rm_clamp_low_c"],
        "adaptive_rm_clamp_high_c": item["paperb_adaptive_rm_clamp_high_c"],
        "paperb_met": item["paperb_met"],
        "people_activity_w_per_person": item["paperb_people_activity_w"],
        "controller_semantics": item["controller_semantics"],
        "inference_preprocessing": item["inference_preprocessing"],
        "sizing_contract": item["sizing_contract"],
    }
    return {
        "strategy": item["strategy"],
        "effective_config": effective_config,
        "sizing": {
            "mode": "accepted_denver",
            "source_idf_sha256": EXPECTED_IDF_SHA256,
        },
    }


def validate_accepted_matrix(path: Path) -> pd.DataFrame:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_hash = sha256_file(path)
    if observed_hash != EXPECTED_ACCEPTED_MATRIX_SHA256:
        raise PlanContractError(
            f"Accepted matrix hash mismatch: expected {EXPECTED_ACCEPTED_MATRIX_SHA256}, "
            f"observed {observed_hash}"
        )
    frame = pd.read_csv(path)
    required = {
        "scaled_run_id",
        "anchor_id",
        "anchor_order",
        "anchor_role",
        "city",
        "climate_role",
        "scenario",
        "time_slice",
        "selector_role",
        "recomputed_weather_year",
        "selector_label_id",
        "physical_condition_id",
        "strategy",
        "strategy_order",
        "execution_epw_path",
        "epw_sha256",
        "epw_data_sha256",
        "runner_sha256",
        "model_sha256",
        "idf_sha256",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PlanContractError(f"Accepted matrix missing columns: {missing}")
    if len(frame) != 96 or frame["anchor_id"].nunique() != 24:
        raise PlanContractError("Accepted matrix must contain 96 rows and 24 anchors")
    expected_strategies = {
        "diagnostic_reference",
        "paperb_pmv_relax",
        "paperb_adaptive_band_relax",
        "paperb_p90_tail_asym_relax",
    }
    if set(frame["strategy"].astype(str)) != expected_strategies:
        raise PlanContractError("Accepted matrix strategy coverage differs")
    counts = frame.groupby("anchor_id")["strategy"].nunique()
    if not counts.eq(4).all():
        raise PlanContractError("Every accepted anchor must contain four strategies")
    if set(frame["runner_sha256"].astype(str)) != {EXPECTED_ACCEPTED_RUNNER_SHA256}:
        raise PlanContractError("Accepted runner identity differs")
    if set(frame["model_sha256"].astype(str)) != {EXPECTED_MODEL_SHA256}:
        raise PlanContractError("Accepted model identity differs")
    if set(frame["idf_sha256"].astype(str)) != {EXPECTED_IDF_SHA256}:
        raise PlanContractError("Accepted IDF identity differs")
    for row in frame.drop_duplicates("anchor_id").itertuples(index=False):
        epw = Path(str(row.execution_epw_path))
        if not epw.is_file() or sha256_file(epw) != str(row.epw_sha256):
            raise PlanContractError(f"Accepted EPW missing or hash mismatch: {epw}")
    return frame


def comparator_id(frame: pd.DataFrame, anchor_id: str, strategy: str) -> str:
    rows = frame.loc[
        frame["anchor_id"].astype(str).eq(anchor_id)
        & frame["strategy"].astype(str).eq(strategy),
        "scaled_run_id",
    ]
    if len(rows) != 1:
        raise PlanContractError(f"Comparator lookup is not unique: {anchor_id}, {strategy}")
    return str(rows.iloc[0])


def make_cell_row(
    source: pd.Series,
    selected_variant: dict[str, Any],
    *,
    batch_id: str,
    cell_id: str,
    accepted_matrix: pd.DataFrame,
) -> dict[str, Any]:
    comparator_strategy = (
        "paperb_pmv_relax"
        if selected_variant["experiment"] == "M1"
        else "paperb_adaptive_band_relax"
        if selected_variant["experiment"] == "C1"
        else "paperb_p90_tail_asym_relax"
    )
    row = {
        "matrix_schema_version": MATRIX_SCHEMA_VERSION,
        "batch_id": batch_id,
        "cell_id": cell_id,
        "experiment": selected_variant["experiment"],
        "variant_id": selected_variant["variant_id"],
        "strategy": selected_variant["strategy"],
        "anchor_id": str(source["anchor_id"]),
        "anchor_order": int(source["anchor_order"]),
        "anchor_role": str(source["anchor_role"]),
        "city": str(source["city"]),
        "climate_role": str(source["climate_role"]),
        "scenario": str(source["scenario"]),
        "time_slice": str(source["time_slice"]),
        "selector_role": str(source["selector_role"]),
        "weather_year": int(source["recomputed_weather_year"]),
        "physical_condition_id": str(source["physical_condition_id"]),
        "selector_label_id": str(source["selector_label_id"]),
        "weather_path": str(Path(str(source["execution_epw_path"])).resolve()),
        "weather_sha256": str(source["epw_sha256"]),
        "weather_data_sha256": str(source["epw_data_sha256"]),
        "weather_stem": Path(str(source["execution_epw_path"])).stem,
        "accepted_comparator_strategy": comparator_strategy,
        "accepted_comparator_cell_id": comparator_id(
            accepted_matrix, str(source["anchor_id"]), comparator_strategy
        ),
        "expected_trace_rows": EXPECTED_ROWS,
        "expected_occupied_rows": EXPECTED_OCCUPIED_ROWS,
        "accepted_matrix_sha256": EXPECTED_ACCEPTED_MATRIX_SHA256,
        "accepted_idf_sha256": EXPECTED_IDF_SHA256,
        "accepted_model_sha256": EXPECTED_MODEL_SHA256,
        "resume_permitted": False,
        "reuse_or_splice_permitted": False,
        "purge_permitted": False,
    }
    row.update(
        {
            key: value
            for key, value in selected_variant.items()
            if key not in {"variant_id", "strategy", "experiment", "rationale"}
        }
    )
    return row


def compile_matrices(
    accepted: pd.DataFrame,
    *,
    include_adaptive: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    anchors = (
        accepted.loc[accepted["strategy"].astype(str).eq("diagnostic_reference")]
        .sort_values("anchor_order")
        .reset_index(drop=True)
    )
    if len(anchors) != 24 or set(SENTINELS) - set(anchors["anchor_id"].astype(str)):
        raise PlanContractError("Anchor/sentinel coverage is incomplete")

    variants = core_variants()
    core_rows: list[dict[str, Any]] = []
    sequence = 0
    for selected_variant in variants:
        subset = anchors if selected_variant["experiment"] == "M1" else anchors.loc[
            anchors["anchor_id"].astype(str).isin(SENTINELS)
        ]
        for _index, source in subset.iterrows():
            sequence += 1
            core_rows.append(
                make_cell_row(
                    source,
                    selected_variant,
                    batch_id="rev01_core64",
                    cell_id=f"REV01-C064-{sequence:03d}",
                    accepted_matrix=accepted,
                )
            )
    core = pd.DataFrame.from_records(core_rows)
    if len(core) != 64 or core["cell_id"].nunique() != 64:
        raise PlanContractError("Core compilation did not close at exactly 64 cells")
    if core.groupby("experiment").size().to_dict() != {"M1": 24, "M2": 32, "M3": 8}:
        raise PlanContractError("Core experiment counts differ from the predeclared design")

    adaptive: pd.DataFrame | None = None
    manifest_variants = [parity_variant(), *variants]
    if include_adaptive:
        selected_variant = adaptive_variant()
        manifest_variants.append(selected_variant)
        adaptive_rows = [
            make_cell_row(
                source,
                selected_variant,
                batch_id="rev01_adaptive24",
                cell_id=f"REV01-A024-{index:03d}",
                accepted_matrix=accepted,
            )
            for index, (_row_index, source) in enumerate(anchors.iterrows(), start=1)
        ]
        adaptive = pd.DataFrame.from_records(adaptive_rows)
        if len(adaptive) != 24 or adaptive["cell_id"].nunique() != 24:
            raise PlanContractError("Adaptive compilation did not close at exactly 24 cells")

    manifest = {
        "schema_version": VARIANT_SCHEMA_VERSION,
        "parent_runner_sha256": EXPECTED_ACCEPTED_RUNNER_SHA256,
        "variants": {
            item["variant_id"]: manifest_entry(item)
            for item in manifest_variants
        },
    }
    return core, adaptive, manifest


def compile_parity_matrix(accepted: pd.DataFrame) -> pd.DataFrame:
    source_rows = accepted.loc[
        accepted["anchor_id"].astype(str).eq("beijing_ssp585_present_typical")
        & accepted["strategy"].astype(str).eq("diagnostic_reference")
    ]
    if len(source_rows) != 1:
        raise PlanContractError("Parity anchor lookup is not unique")
    row = make_cell_row(
        source_rows.iloc[0],
        parity_variant(),
        batch_id="rev01_parity1",
        cell_id="REV01-PARITY-001",
        accepted_matrix=accepted,
    )
    parity = pd.DataFrame.from_records([row])
    if row["accepted_comparator_cell_id"] != "SCALED-96-028":
        raise PlanContractError("Parity comparator is not accepted SCALED-96-028")
    return parity


def write_parquet_exact(frame: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(
        table,
        path,
        version="2.6",
        compression="zstd",
        compression_level=3,
        row_group_size=65_536,
        write_page_checksum=True,
    )
    restored = pq.read_table(path, page_checksum_verification=True).to_pandas()
    pd.testing.assert_frame_equal(frame, restored, check_dtype=True, check_exact=True)


def freeze_plan(
    output_dir: Path,
    *,
    accepted_matrix_path: Path,
    core: pd.DataFrame,
    adaptive: pd.DataFrame | None,
    manifest: dict[str, Any],
    adaptive_trigger_evidence: Path | None,
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Plan output must be absent: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    accepted = validate_accepted_matrix(accepted_matrix_path)
    parity = compile_parity_matrix(accepted)
    manifest_path = output_dir / "VARIANT_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    matrices: dict[str, dict[str, Any]] = {}
    for name, frame in (("parity1", parity), ("core64", core), ("adaptive24", adaptive)):
        if frame is None:
            continue
        csv_path = output_dir / f"{name}.csv"
        parquet_path = output_dir / f"{name}.parquet"
        frame.to_csv(csv_path, index=False)
        write_parquet_exact(frame, parquet_path)
        matrices[name] = {
            "rows": int(len(frame)),
            "csv_path": str(csv_path.resolve()),
            "csv_sha256": sha256_file(csv_path),
            "parquet_path": str(parquet_path.resolve()),
            "parquet_sha256": sha256_file(parquet_path),
            "unique_cell_ids": int(frame["cell_id"].nunique()),
        }
    closure = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "all_gates_passed": True,
        "created_utc": utc_now(),
        "accepted_matrix_path": str(accepted_matrix_path.resolve()),
        "accepted_matrix_sha256": sha256_file(accepted_matrix_path),
        "core_cells": int(len(core)),
        "adaptive_cells": int(0 if adaptive is None else len(adaptive)),
        "sizing_cells": 0,
        "technical_parity_cells": 1,
        "total_scientific_cells": int(len(core) + (0 if adaptive is None else len(adaptive))),
        "total_execution_cells": int(1 + len(core) + (0 if adaptive is None else len(adaptive))),
        "total_cells": int(len(core) + (0 if adaptive is None else len(adaptive))),
        "conditional_batches_inferred_at_runtime": False,
        "city_ddy_improvised": False,
        "variant_manifest_path": str(manifest_path.resolve()),
        "variant_manifest_sha256": sha256_file(manifest_path),
        "adaptive_trigger_evidence_path": (
            None
            if adaptive_trigger_evidence is None
            else str(adaptive_trigger_evidence.resolve())
        ),
        "adaptive_trigger_evidence_sha256": (
            None
            if adaptive_trigger_evidence is None
            else sha256_file(adaptive_trigger_evidence)
        ),
        "matrices": matrices,
    }
    closure_path = output_dir / "PLAN_CLOSURE.json"
    closure_path.write_text(
        json.dumps(closure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return closure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-matrix", type=Path, default=DEFAULT_ACCEPTED_MATRIX)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--include-adaptive-clamp", action="store_true")
    parser.add_argument("--adaptive-trigger-evidence", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--freeze", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.include_adaptive_clamp and args.adaptive_trigger_evidence is None:
        raise PlanContractError("Adaptive inclusion requires explicit trigger evidence")
    if args.adaptive_trigger_evidence is not None and not args.adaptive_trigger_evidence.is_file():
        raise FileNotFoundError(args.adaptive_trigger_evidence)
    accepted = validate_accepted_matrix(args.accepted_matrix)
    core, adaptive, manifest = compile_matrices(
        accepted,
        include_adaptive=args.include_adaptive_clamp,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "accepted_matrix_sha256": sha256_file(args.accepted_matrix),
        "core_cells": len(core),
        "adaptive_cells": 0 if adaptive is None else len(adaptive),
        "total_cells": len(core) + (0 if adaptive is None else len(adaptive)),
        "technical_parity_cells": 1,
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
        core=core,
        adaptive=adaptive,
        manifest=manifest,
        adaptive_trigger_evidence=args.adaptive_trigger_evidence,
    )
    print(json.dumps(closure, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
