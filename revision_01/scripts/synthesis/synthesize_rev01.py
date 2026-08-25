#!/usr/bin/env python3
"""Fail-closed reviewer-facing synthesis for the Paper B ENB Rev01 reruns.

The program accepts only the frozen 64-cell core, 24-cell adaptive, and
48-cell full-grid learned-contribution batches.  It validates their closures,
frozen matrices, exact cell coverage,
storage closures, per-cell hashes, and accepted-comparator lineage before it
creates an output directory.  Physical temperature endpoints are descriptive
occupied operative-temperature exposure diagnostics; they are not observed
comfort, satisfaction, health outcomes, or standards-compliance tests.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
from typing import Any, Iterable
import uuid

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA_VERSION = "paperb_enb_rev01_reviewer_synthesis_v2"
EXPECTED_BATCH_SCHEMA = "paperb_enb_rev01_batch_closure_v1"
EXPECTED_FREEZE_SCHEMA = "paperb_enb_rev01_batch_freeze_v1"
EXPECTED_MATRIX_SCHEMA = "paperb_enb_rev01_cell_matrix_v1"
EXPECTED_STORAGE_SCHEMA = "paperb_enb_rev01_storage_final_closure_v1"
EXPECTED_ACCEPTED_CLOSURE_SCHEMA = "paperb_enb_fresh96_closure_v2_envscoped"
EXPECTED_ACCEPTED_MATRIX_SHA256 = (
    "fa03767f0c3d581e65e00ff6149546c44fe3e936608da73ad2f5485bf19895e0"
)
EXPECTED_ACCEPTED_RUNNER_SHA256 = (
    "a9693042883ed5c20edad6fcbc757c62c7216d5abc120ad18de1a003932848a4"
)
EXPECTED_REV01_RUNNER_SHA256 = (
    "4c061fc3b25f7ee6fe66df00cfacfab42f1d3ab171185306fe1ce161a84a0baf"
)
EXPECTED_REV01_HELPER_SHA256 = (
    "b941c0bfc4beb3dc44557ed341bb4e018992620b0d3ad4f3bcb6973558d78bf7"
)
EXPECTED_MODEL_SHA256 = (
    "6fbeb06644a36b226b17824a1fca7526bd518e0a9b76a41f878fb1c27efcd619"
)
EXPECTED_MODEL_METRICS_SHA256 = (
    "b5922472a72727d71d263e5968dd474ab13ee8ec630570578d199d2275494713"
)
EXPECTED_IDF_SHA256 = (
    "1144b58b848992d1730e49ef9c252569e3a515d82a2f99b2c0233352f625a7e4"
)
EXPECTED_BASE_VARIANT_MANIFEST_SHA256 = (
    "8daf357be8683346aec533b2a9b5489d32c7a3b39a658c34bbcf383ce703cadf"
)
EXPECTED_BASE_PLAN_CLOSURE_SHA256 = (
    "02f86cc8138d6fd24fe0a80de28ce7a550522df6d3d07e3c7173a49240be754c"
)
EXPECTED_LEARNED_VARIANT_MANIFEST_SHA256 = (
    "2ba95c3fd5a3d46c2bb17e0286868913528526b36f5b2175ce2b37880b4478e2"
)
EXPECTED_LEARNED_PLAN_CLOSURE_SHA256 = (
    "b99a7efa7ab70747d02304698f9a2b86e99103831edd53991c30c774ad4c6a51"
)
EXPECTED_RUNNER_CLOSURE_SHA256 = (
    "3b74c078a8496e618e4bddd66f3b2916715a65a210d5f2d495df0498ca1b198b"
)
EXPECTED_STORAGE_MODULE_SHA256 = (
    "b30e96238e8c5c443350c2250fb379a37292ac06caaff98073a4b2bf92b74448"
)
EXPECTED_STORAGE_MAPPING_SHA256 = (
    "40d8b59cd2f2085bef8aafe16634fe1364c2aa5f317e0e62986c83aca5b65238"
)
EXPECTED_BATCH_GATES = frozenset(
    {
        "accepted_comparator_evidence_unchanged",
        "all_cells_final_validated",
        "all_cells_fresh_preliminary_validated",
        "authorized_sources_unchanged_at_close",
        "exact_cell_id_coverage",
        "exact_frozen_runner_per_subprocess",
        "frozen_inputs_unchanged_at_close",
        "maximum_four_workers",
        "one_strategy_per_subprocess",
        "parity_authorization_present_when_required",
        "shared_model_once_per_batch",
        "storage_finalize_passed",
        "storage_stage_passed",
        "zero_resume_reuse_splice_purge",
    }
)
EXPECTED_ACCEPTED_GATES = frozenset(
    {
        "all_96_cells_complete_validated",
        "all_cell_result_hashes_exact",
        "cell_status_exact_key_coverage",
        "cell_status_schema_exact",
        "exact_physical_strategy_key_coverage",
        "exact_scaled_run_id_coverage",
        "exactly_24_physical_anchors",
        "exactly_96_distinct_trace_paths",
        "exactly_96_fresh_cell_directories",
        "exactly_96_jsonl_records",
        "exactly_96_planned_cells",
        "exactly_96_result_records",
        "four_strategies_per_anchor",
        "max_four_concurrent_cells",
        "one_fresh_subprocess_per_passing_cell",
        "runner_hash_exact",
        "zero_purge",
        "zero_resume",
        "zero_reuse_or_splice",
    }
)
EXPECTED_RUNNER_GATES = frozenset(
    {
        "accepted_parent_hash_exact",
        "alpha_one_returns_original_probability_object",
        "atomic_exclusive_trace_placement",
        "derived_runner_compiles",
        "dry_manifest_contract_passed",
        "helper_is_pure_and_compiles",
        "manifest_effective_config_exact",
        "matched_pmv_quantile_retains_sign",
        "runner_and_helper_are_frozen_together",
        "trace_full_frame_exact_roundtrip",
        "unit_tests_passed",
    }
)
EXPECTED_BASE_VARIANT_IDS = frozenset(
    {
        "rev01_adaptive_rm_clamp_10_33p5",
        "rev01_p90_abs_pmv_relax",
        "rev01_p90_bounds19_27",
        "rev01_p90_bounds21_25",
        "rev01_p90_dtail005",
        "rev01_p90_dtail015",
        "rev01_p90_ptail015",
        "rev01_p90_ptail025",
        "rev01_p90_q075",
        "rev01_p90_q100",
        "rev01_p90_signal_a000",
        "rev01_p90_signal_a050",
        "rev01_parity_submitted_p90",
    }
)
EXPECTED_LEARNED_VARIANT_IDS = frozenset(
    {
        "rev01_learned_mean_tail_full24",
        "rev01_p90_signal_a000_full24",
    }
)
EXPECTED_SCIENTIFIC_VARIANT_COUNTS = {
    ("M1", "rev01_p90_abs_pmv_relax"): 24,
    ("M2", "rev01_p90_q075"): 4,
    ("M2", "rev01_p90_q100"): 4,
    ("M2", "rev01_p90_ptail015"): 4,
    ("M2", "rev01_p90_ptail025"): 4,
    ("M2", "rev01_p90_dtail005"): 4,
    ("M2", "rev01_p90_dtail015"): 4,
    ("M2", "rev01_p90_bounds21_25"): 4,
    ("M2", "rev01_p90_bounds19_27"): 4,
    ("M3", "rev01_p90_signal_a050"): 4,
    ("M3", "rev01_p90_signal_a000"): 4,
    ("C1", "rev01_adaptive_rm_clamp_10_33p5"): 24,
    ("M4", "rev01_learned_mean_tail_full24"): 24,
    ("M5", "rev01_p90_signal_a000_full24"): 24,
}
EXPECTED_SENTINEL_IDS = frozenset(
    {
        "beijing_ssp585_present_typical",
        "guangzhou_ssp585_present_typical",
        "kolkata_ssp585_future_extreme",
        "phoenix_ssp585_future_extreme",
    }
)
EXPECTED_REVIEWER_IDS = frozenset(
    {
        "R1-1", "R1-2", "R1-3", "R1-4", "R1-5", "R1-6", "R1-7",
        "R2-1", "R2-2", "R2-3", "R2-4", "R2-5", "R2-6",
        "R2-M1", "R2-M2", "R2-M3", "R2-M4",
        "R3-1", "R3-2", "R3-3", "R3-4", "R3-5", "R3-6", "R3-7",
        "R3-M1", "R3-M2", "R3-M3", "R3-M4", "R3-M5", "R3-M6",
        "R3-M7", "R3-M8", "R3-M9", "R3-DATA",
    }
)
EXPECTED_ROWS = 35_040
EXPECTED_OCCUPIED_ROWS = 16_704
EXPECTED_ZONE_COUNT = 15
JOULES_PER_KWH = 3_600_000.0
STEP_HOURS = 0.25
WARM_THRESHOLDS_C = (26.0, 28.0, 30.0)
COLD_THRESHOLDS_C = (18.0, 16.0)
ADAPTIVE_RUNNING_MEAN_LOW_C = 10.0
ADAPTIVE_RUNNING_MEAN_HIGH_C = 33.5
ZONE_AIR_RE = re.compile(r"^zone_(?P<zone>.+)_ta_c$")

REBUILD_ROOT = Path(__file__).resolve().parents[2]
ACCEPTED_ROOT = REBUILD_ROOT / "06_runs/scaled/20260803_routeS96_v2_envscoped"


@dataclass(frozen=True)
class BatchSpec:
    name: str
    batch_id: str
    matrix_filename: str
    matrix_sha256: str
    rows: int
    plan_key: str
    plan_closure_sha256: str
    variant_manifest_sha256: str
    variant_ids: frozenset[str]


CORE_SPEC = BatchSpec(
    name="core",
    batch_id="rev01_core64",
    matrix_filename="core64.csv",
    matrix_sha256="1853c5ac0658a905cb80280396575f6ac4777fd3d2eee260f174f1f410f1f662",
    rows=64,
    plan_key="core64",
    plan_closure_sha256=EXPECTED_BASE_PLAN_CLOSURE_SHA256,
    variant_manifest_sha256=EXPECTED_BASE_VARIANT_MANIFEST_SHA256,
    variant_ids=EXPECTED_BASE_VARIANT_IDS,
)
ADAPTIVE_SPEC = BatchSpec(
    name="adaptive",
    batch_id="rev01_adaptive24",
    matrix_filename="adaptive24.csv",
    matrix_sha256="14d3075d83d7e688f39677c2f02ed21b083b8a30c08df02b90425f976d524ac9",
    rows=24,
    plan_key="adaptive24",
    plan_closure_sha256=EXPECTED_BASE_PLAN_CLOSURE_SHA256,
    variant_manifest_sha256=EXPECTED_BASE_VARIANT_MANIFEST_SHA256,
    variant_ids=EXPECTED_BASE_VARIANT_IDS,
)
LEARNED_SPEC = BatchSpec(
    name="learned",
    batch_id="rev01_learned48",
    matrix_filename="learned48.csv",
    matrix_sha256="c38576e1a0dd70a4df2dd09e033a9bdd70c04f0446929570fd01e9bee14776dd",
    rows=48,
    plan_key="learned48",
    plan_closure_sha256=EXPECTED_LEARNED_PLAN_CLOSURE_SHA256,
    variant_manifest_sha256=EXPECTED_LEARNED_VARIANT_MANIFEST_SHA256,
    variant_ids=EXPECTED_LEARNED_VARIANT_IDS,
)


class SynthesisContractError(RuntimeError):
    """An input provenance, coverage, metric, or output gate failed."""


@dataclass(frozen=True)
class AcceptedContext:
    root: Path
    closure_path: Path
    closure_sha256: str
    matrix_path: Path
    matrix: pd.DataFrame
    status_by_cell_id: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class Artifact:
    artifact_cell_id: str
    trace: Path
    summary: Path
    sql: Path
    trace_sha256: str
    summary_sha256: str
    sql_sha256: str


@dataclass(frozen=True)
class MatchedArtifact:
    batch_name: str
    plan: dict[str, Any]
    revision: Artifact
    accepted: Artifact


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SynthesisContractError(message)


def require_exact_true_gates(
    gates: Any,
    expected: frozenset[str],
    label: str,
) -> None:
    require(isinstance(gates, dict), f"{label} must be a JSON object")
    require(bool(gates), f"{label} must be nonempty")
    observed = frozenset(str(key) for key in gates)
    require(observed == expected, f"{label} keys differ: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}")
    require(all(gates[key] is True for key in expected), f"At least one {label} value is not exactly true")


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SynthesisContractError(f"{label} is not numeric") from exc
    require(np.isfinite(result), f"{label} is not finite")
    return result


def validate_output_location(
    output_dir: Path,
    core_batch: Path,
    adaptive_batch: Path,
    learned_batch: Path,
    accepted_root: Path = ACCEPTED_ROOT,
) -> Path:
    output = output_dir.resolve()
    protected = {
        "accepted root": accepted_root.resolve(),
        "core batch": core_batch.resolve(),
        "adaptive batch": adaptive_batch.resolve(),
        "learned batch": learned_batch.resolve(),
    }
    for label, root in protected.items():
        require(not path_within(output, root), f"Output directory is inside protected {label}: {output}")
        require(not path_within(root, output), f"Output directory would contain protected {label}: {output}")
    return output


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SynthesisContractError(f"Cannot read {label} {path}: {exc}") from exc
    require(isinstance(value, dict), f"{label} is not a JSON object: {path}")
    return value


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    require(path.is_file(), f"Missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                require(isinstance(value, dict), f"{label} row is not an object: {line_number}")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SynthesisContractError(f"Cannot read {label} {path}: {exc}") from exc
    return rows


def one_path(root: Path, pattern: str, label: str) -> Path:
    paths = sorted(root.glob(pattern))
    require(len(paths) == 1, f"Expected exactly one {label} under {root}: {paths}")
    return paths[0]


def path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def verify_file(path: Path, expected_hash: str, label: str) -> str:
    require(path.is_file(), f"Missing {label}: {path}")
    observed = sha256_file(path)
    require(observed == expected_hash, f"{label} hash differs: {path}")
    return observed


def validate_accepted_context(root: Path = ACCEPTED_ROOT) -> AcceptedContext:
    root = root.resolve()
    closure_path = root / "FRESH96_CLOSURE.json"
    closure = read_json(closure_path, "accepted Fresh96 closure")
    require(
        closure.get("schema_version") == EXPECTED_ACCEPTED_CLOSURE_SCHEMA,
        "Accepted Fresh96 closure schema differs",
    )
    for key in ("all_gates_passed", "overall_pass", "all_cells_fresh"):
        require(closure.get(key) is True, f"Accepted Fresh96 closure gate failed: {key}")
    require(closure.get("planned_cells") == 96, "Accepted plan is not 96 cells")
    require(closure.get("passed_cells") == 96, "Accepted batch is not 96/96 passing")
    require(closure.get("failed_cells") == 0, "Accepted batch records failed cells")
    require(
        closure.get("runner_sha256") == EXPECTED_ACCEPTED_RUNNER_SHA256,
        "Accepted runner identity differs",
    )
    require_exact_true_gates(
        closure.get("closure_gates"),
        EXPECTED_ACCEPTED_GATES,
        "accepted Fresh96 closure_gates",
    )
    matrix_path = root / "freeze/authorization/execution/scaled_matrix_fresh96_v2_envscoped.csv"
    verify_file(matrix_path, EXPECTED_ACCEPTED_MATRIX_SHA256, "accepted execution matrix")
    matrix = pd.read_csv(matrix_path, dtype=str, keep_default_na=False)
    require(len(matrix) == 96, "Accepted execution matrix does not contain 96 rows")
    require(matrix["scaled_run_id"].nunique() == 96, "Accepted cell IDs are not unique")
    require(
        set(matrix["strategy"])
        == {
            "diagnostic_reference",
            "paperb_pmv_relax",
            "paperb_adaptive_band_relax",
            "paperb_p90_tail_asym_relax",
        },
        "Accepted strategy coverage differs",
    )
    status_path = root / "CELL_STATUS.jsonl"
    verify_file(status_path, str(closure.get("cell_status_sha256", "")), "accepted CELL_STATUS journal")
    statuses = read_jsonl(status_path, "accepted CELL_STATUS journal")
    require(len(statuses) == 96, "Accepted CELL_STATUS does not contain exactly 96 records")
    status_ids = [str(row.get("scaled_run_id", "")) for row in statuses]
    require(len(set(status_ids)) == 96, "Accepted CELL_STATUS cell IDs are not unique")
    require(set(status_ids) == set(matrix["scaled_run_id"]), "Accepted CELL_STATUS coverage differs from matrix")
    require(all(row.get("all_gates_passed") is True for row in statuses), "Accepted CELL_STATUS contains a failed gate")
    require(all(row.get("status") == "complete_validated" for row in statuses), "Accepted CELL_STATUS status differs")
    return AcceptedContext(
        root=root,
        closure_path=closure_path,
        closure_sha256=sha256_file(closure_path),
        matrix_path=matrix_path,
        matrix=matrix,
        status_by_cell_id={str(row["scaled_run_id"]): row for row in statuses},
    )


def validate_batch_closure_fields(closure: dict[str, Any], spec: BatchSpec) -> None:
    require(closure.get("schema_version") == EXPECTED_BATCH_SCHEMA, f"{spec.name} closure schema differs")
    require(closure.get("batch_id") == spec.batch_id, f"{spec.name} batch ID differs")
    require(closure.get("status") == "PASS", f"{spec.name} batch status is not PASS")
    require(closure.get("all_gates_passed") is True, f"{spec.name} batch gates did not pass")
    require(closure.get("planned_cells") == spec.rows, f"{spec.name} planned-cell count differs")
    require(closure.get("passed_cells") == spec.rows, f"{spec.name} passed-cell count differs")
    require(closure.get("failed_cells") == 0, f"{spec.name} records failed cells")
    require(closure.get("accepted_outputs_modified") is False, f"{spec.name} records accepted-output modification")
    require(closure.get("final_validation_failures") == [], f"{spec.name} records final validation failures")
    require(1 <= int(closure.get("workers", 0)) <= 4, f"{spec.name} worker count is invalid")
    require(closure.get("matrix_sha256") == spec.matrix_sha256, f"{spec.name} matrix identity differs")
    require(closure.get("runner_sha256") == EXPECTED_REV01_RUNNER_SHA256, f"{spec.name} runner differs")
    require(closure.get("runner_helper_sha256") == EXPECTED_REV01_HELPER_SHA256, f"{spec.name} runner helper differs")
    require(closure.get("runner_closure_sha256") == EXPECTED_RUNNER_CLOSURE_SHA256, f"{spec.name} runner closure differs")
    require(closure.get("storage_module_sha256") == EXPECTED_STORAGE_MODULE_SHA256, f"{spec.name} storage module differs")
    require(closure.get("storage_mapping_sha256") == EXPECTED_STORAGE_MAPPING_SHA256, f"{spec.name} storage mapping differs")
    require(
        closure.get("variant_manifest_sha256") == spec.variant_manifest_sha256,
        f"{spec.name} variant manifest differs",
    )
    require_exact_true_gates(
        closure.get("closure_gates"), EXPECTED_BATCH_GATES, f"{spec.name} closure_gates"
    )
    for storage_key in ("storage_stage", "storage_final"):
        storage = closure.get(storage_key, {})
        require(storage.get("status") == "PASS", f"{spec.name} {storage_key} is not PASS")
        require(storage.get("all_gates_passed") is True, f"{spec.name} {storage_key} failed")


def validate_frozen_files(
    batch: Path,
    freeze: dict[str, Any],
    closure: dict[str, Any],
    spec: BatchSpec,
) -> None:
    require(freeze.get("schema_version") == EXPECTED_FREEZE_SCHEMA, f"{spec.name} freeze schema differs")
    require(freeze.get("batch_id") == spec.batch_id, f"{spec.name} freeze batch ID differs")
    require(freeze.get("planned_cells") == spec.rows, f"{spec.name} freeze cell count differs")
    require(freeze.get("matrix_sha256") == spec.matrix_sha256, f"{spec.name} freeze matrix differs")
    top_level_expected = {
        "runner_sha256": EXPECTED_REV01_RUNNER_SHA256,
        "runner_helper_sha256": EXPECTED_REV01_HELPER_SHA256,
        "runner_closure_sha256": EXPECTED_RUNNER_CLOSURE_SHA256,
        "model_sha256": EXPECTED_MODEL_SHA256,
        "model_metrics_sha256": EXPECTED_MODEL_METRICS_SHA256,
        "idf_sha256": EXPECTED_IDF_SHA256,
        "variant_manifest_sha256": spec.variant_manifest_sha256,
        "storage_module_sha256": EXPECTED_STORAGE_MODULE_SHA256,
        "storage_mapping_sha256": EXPECTED_STORAGE_MAPPING_SHA256,
        "parent_runner_sha256": EXPECTED_ACCEPTED_RUNNER_SHA256,
    }
    for key, expected in top_level_expected.items():
        require(freeze.get(key) == expected, f"{spec.name} freeze top-level {key} differs")
        if key in closure:
            require(closure.get(key) == expected, f"{spec.name} closure/freeze {key} binding differs")
    require(freeze.get("resume_permitted") is False, f"{spec.name} freeze permits resume")
    require(freeze.get("reuse_or_splice_permitted") is False, f"{spec.name} freeze permits reuse/splice")
    require(freeze.get("purge_permitted") is False, f"{spec.name} freeze permits purge")
    frozen = freeze.get("frozen_file_sha256")
    require(isinstance(frozen, dict) and frozen, f"{spec.name} frozen-file map is absent")
    for relative, expected in frozen.items():
        path = (batch / str(relative)).resolve()
        require(path_within(path, batch), f"{spec.name} frozen path escapes batch: {relative}")
        verify_file(path, str(expected), f"{spec.name} frozen file")
    require(
        frozen.get(f"freeze/plan/{spec.matrix_filename}") == spec.matrix_sha256,
        f"{spec.name} frozen matrix is not bound in FREEZE.json",
    )
    require(
        frozen.get("freeze/contracts/PLAN_CLOSURE.json") == spec.plan_closure_sha256,
        f"{spec.name} frozen plan closure differs",
    )
    exact_frozen_authorities = {
        "freeze/code/run_medium_office_enb_rev01.py": EXPECTED_REV01_RUNNER_SHA256,
        "freeze/code/rev01_controller_variants.py": EXPECTED_REV01_HELPER_SHA256,
        "freeze/model/control_predictors.joblib": EXPECTED_MODEL_SHA256,
        "freeze/model/control_predictor_metrics.json": EXPECTED_MODEL_METRICS_SHA256,
        "freeze/building/ASHRAE901_OfficeMedium_STD2019_Denver.idf": EXPECTED_IDF_SHA256,
        "freeze/plan/VARIANT_MANIFEST.json": spec.variant_manifest_sha256,
        "freeze/contracts/RUNNER_CLOSURE.json": EXPECTED_RUNNER_CLOSURE_SHA256,
        "freeze/contracts/PLAN_CLOSURE.json": spec.plan_closure_sha256,
        "freeze/code/rev01_storage.py": EXPECTED_STORAGE_MODULE_SHA256,
        "freeze/contracts/storage_mapping_v1.json": EXPECTED_STORAGE_MAPPING_SHA256,
    }
    for relative, expected in exact_frozen_authorities.items():
        require(frozen.get(relative) == expected, f"{spec.name} frozen authority map differs: {relative}")
        verify_file(batch / relative, expected, f"{spec.name} independently pinned frozen authority")

    runner_closure = read_json(batch / "freeze/contracts/RUNNER_CLOSURE.json", f"{spec.name} runner closure")
    require(runner_closure.get("schema_version") == "paperb_enb_rev01_runner_closure_v1", f"{spec.name} runner-closure schema differs")
    require(runner_closure.get("all_gates_passed") is True, f"{spec.name} runner closure did not pass")
    require_exact_true_gates(runner_closure.get("closure_gates"), EXPECTED_RUNNER_GATES, f"{spec.name} runner closure_gates")
    require(runner_closure.get("derived_runner_sha256") == EXPECTED_REV01_RUNNER_SHA256, f"{spec.name} runner closure runner differs")
    require(runner_closure.get("helper_sha256") == EXPECTED_REV01_HELPER_SHA256, f"{spec.name} runner closure helper differs")
    require(runner_closure.get("parent_runner_sha256") == EXPECTED_ACCEPTED_RUNNER_SHA256, f"{spec.name} runner closure parent differs")


def plan_effective_config(row: dict[str, Any]) -> dict[str, Any]:
    integer_fields = {
        "tighten_dwell_steps": "paperb_tighten_dwell_steps",
        "relax_dwell_steps": "paperb_relax_dwell_steps",
    }
    float_fields = {
        "spatial_quantile": "paperb_spatial_quantile",
        "tail_threshold": "paperb_tail_threshold",
        "asym_threshold": "paperb_asym_threshold",
        "save_heat_c": "paperb_save_heat_c",
        "save_cool_c": "paperb_save_cool_c",
        "warm_protect_cool_c": "paperb_warm_protect_cool_c",
        "cold_protect_heat_c": "paperb_cold_protect_heat_c",
        "pmv_threshold": "paperb_pmv_threshold",
        "signal_alpha": "paperb_signal_alpha",
        "adaptive_rm_clamp_low_c": "paperb_adaptive_rm_clamp_low_c",
        "adaptive_rm_clamp_high_c": "paperb_adaptive_rm_clamp_high_c",
        "paperb_met": "paperb_met",
        "people_activity_w_per_person": "paperb_people_activity_w",
    }
    config: dict[str, Any] = {
        "strategy": str(row["strategy"]),
        "controller_semantics": str(row["controller_semantics"]),
        "inference_preprocessing": str(row["inference_preprocessing"]),
        "sizing_contract": str(row["sizing_contract"]),
    }
    for target, source in float_fields.items():
        config[target] = finite_float(row[source], f"matrix {source}")
    for target, source in integer_fields.items():
        numeric = finite_float(row[source], f"matrix {source}")
        require(numeric.is_integer(), f"matrix {source} is not an integer")
        config[target] = int(numeric)
    return config


def validate_frozen_plan_authority(
    batch: Path,
    freeze: dict[str, Any],
    matrix: pd.DataFrame,
    spec: BatchSpec,
) -> dict[str, Any]:
    plan_closure = read_json(batch / "freeze/contracts/PLAN_CLOSURE.json", f"{spec.name} plan closure")
    require(plan_closure.get("schema_version") == "paperb_enb_rev01_compiled_plan_v1", f"{spec.name} plan-closure schema differs")
    require(plan_closure.get("all_gates_passed") is True and plan_closure.get("status") == "PASS", f"{spec.name} plan closure did not pass")
    require(plan_closure.get("accepted_matrix_sha256") == EXPECTED_ACCEPTED_MATRIX_SHA256, f"{spec.name} plan closure accepted matrix differs")
    require(plan_closure.get("variant_manifest_sha256") == spec.variant_manifest_sha256, f"{spec.name} plan closure manifest differs")
    record = plan_closure.get("matrices", {}).get(spec.plan_key, {})
    require(record.get("csv_sha256") == spec.matrix_sha256, f"{spec.name} plan closure matrix hash differs")
    require(record.get("rows") == spec.rows and record.get("unique_cell_ids") == spec.rows, f"{spec.name} plan closure matrix counts differ")

    manifest_path = batch / "freeze/plan/VARIANT_MANIFEST.json"
    manifest = read_json(manifest_path, f"{spec.name} variant manifest")
    require(manifest.get("schema_version") == "paperb_enb_rev01_variant_manifest_v1", f"{spec.name} variant-manifest schema differs")
    require(manifest.get("parent_runner_sha256") == EXPECTED_ACCEPTED_RUNNER_SHA256, f"{spec.name} variant manifest parent differs")
    variants = manifest.get("variants")
    require(isinstance(variants, dict), f"{spec.name} variant manifest variants is not an object")
    require(frozenset(variants) == spec.variant_ids, f"{spec.name} variant manifest ID coverage differs")
    for variant_id, entry in variants.items():
        require(isinstance(entry, dict), f"{spec.name} manifest entry is not an object: {variant_id}")
        config = entry.get("effective_config")
        require(isinstance(config, dict), f"{spec.name} manifest config absent: {variant_id}")
        require(entry.get("strategy") == config.get("strategy"), f"{spec.name} manifest strategy/config differs: {variant_id}")
        sizing = entry.get("sizing")
        require(isinstance(sizing, dict), f"{spec.name} sizing entry absent: {variant_id}")
        require(sizing.get("mode") == "accepted_denver", f"{spec.name} unexpected sizing mode: {variant_id}")
        require(sizing.get("source_idf_sha256") == EXPECTED_IDF_SHA256, f"{spec.name} sizing IDF differs: {variant_id}")
        for key, value in config.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                require(np.isfinite(float(value)), f"{spec.name} nonfinite manifest config: {variant_id}/{key}")
    for row in matrix.to_dict(orient="records"):
        variant_id = str(row["variant_id"])
        require(variant_id in variants, f"{spec.name} matrix variant not in manifest: {variant_id}")
        effective = plan_effective_config(row)
        require(effective == variants[variant_id]["effective_config"], f"{spec.name} matrix/manifest effective config differs: {row['cell_id']}")
        require(str(row["accepted_idf_sha256"]) == EXPECTED_IDF_SHA256, f"{spec.name} matrix IDF differs: {row['cell_id']}")
        require(str(row["accepted_model_sha256"]) == EXPECTED_MODEL_SHA256, f"{spec.name} matrix model differs: {row['cell_id']}")

    unique_weather = matrix[["weather_stem", "weather_sha256"]].drop_duplicates()
    require(len(unique_weather) == int(freeze.get("weather_files", -1)), f"{spec.name} frozen weather count differs")
    frozen_map = freeze["frozen_file_sha256"]
    for row in unique_weather.to_dict(orient="records"):
        stem = str(row["weather_stem"])
        expected = str(row["weather_sha256"])
        candidates = sorted((batch / "freeze/weather").glob(f"**/{stem}.epw"))
        require(len(candidates) == 1, f"{spec.name} frozen weather coverage differs: {stem}")
        weather = candidates[0]
        verify_file(weather, expected, f"{spec.name} frozen weather")
        relative = str(weather.relative_to(batch))
        require(frozen_map.get(relative) == expected, f"{spec.name} weather is not cross-bound in FREEZE.json: {stem}")
    return manifest


def require_constant_string(series: pd.Series, expected: str, label: str) -> None:
    require(not series.isna().any(), f"{label} contains nulls")
    values = set(series.astype(str))
    require(values == {expected}, f"{label} differs: {sorted(values)} != {expected}")


def require_constant_numeric(series: pd.Series, expected: float, label: str) -> None:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(float)
    require(np.isfinite(numeric).all(), f"{label} contains nonfinite values")
    require(np.allclose(numeric, float(expected), rtol=0.0, atol=1.0e-12), f"{label} differs from {expected}")


def command_flag_value(command: Any, flag: str, label: str) -> str:
    require(isinstance(command, list) and all(isinstance(item, str) for item in command), f"{label} command is not a string list")
    positions = [index for index, item in enumerate(command) if item == flag]
    require(len(positions) == 1, f"{label} command flag coverage differs: {flag}")
    position = positions[0]
    require(position + 1 < len(command), f"{label} command flag lacks value: {flag}")
    return command[position + 1]


def validate_revision_trace_summary_metadata(
    trace: Path,
    summary: Path,
    plan: dict[str, Any],
    effective_config_sha256: str,
    variant_manifest_sha256: str,
) -> None:
    trace_expected_strings = {
        "strategy": str(plan["strategy"]),
        "weather": str(plan["weather_stem"]),
        "rev01_parent_runner_sha256": EXPECTED_ACCEPTED_RUNNER_SHA256,
        "rev01_variant_id": str(plan["variant_id"]),
        "rev01_variant_manifest_sha256": variant_manifest_sha256,
        "rev01_effective_config_sha256": effective_config_sha256,
        "sizing_contract": str(plan["sizing_contract"]),
        "controller_semantics": str(plan["controller_semantics"]),
        "inference_preprocessing": str(plan["inference_preprocessing"]),
    }
    trace_numeric_map = {
        "paperb_spatial_quantile": "paperb_spatial_quantile",
        "paperb_signal_alpha": "paperb_signal_alpha",
        "paperb_tail_threshold": "paperb_tail_threshold",
        "paperb_asym_threshold": "paperb_asym_threshold",
        "paperb_save_heat_c": "paperb_save_heat_c",
        "paperb_save_cool_c": "paperb_save_cool_c",
        "paperb_warm_protect_cool_c": "paperb_warm_protect_cool_c",
        "paperb_cold_protect_heat_c": "paperb_cold_protect_heat_c",
        "paperb_tighten_dwell_steps": "paperb_tighten_dwell_steps",
        "paperb_relax_dwell_steps": "paperb_relax_dwell_steps",
        "paperb_pmv_threshold": "paperb_pmv_threshold",
        "paperb_adaptive_rm_clamp_low_c": "paperb_adaptive_rm_clamp_low_c",
        "paperb_adaptive_rm_clamp_high_c": "paperb_adaptive_rm_clamp_high_c",
        "paperb_met": "paperb_met",
        "paperb_people_activity_w_per_person": "paperb_people_activity_w",
    }
    required_columns = list(trace_expected_strings) + list(trace_numeric_map)
    schema = pq.read_schema(trace)
    require(set(required_columns).issubset(schema.names), f"Revision trace metadata columns are incomplete: {trace}")
    require(pq.ParquetFile(trace).metadata.num_rows == int(plan["expected_trace_rows"]), f"Revision trace metadata row count differs: {plan['cell_id']}")
    metadata = pd.read_parquet(trace, columns=required_columns)
    for column, expected in trace_expected_strings.items():
        require_constant_string(metadata[column], expected, f"revision trace {column}")
    for column, plan_column in trace_numeric_map.items():
        require_constant_numeric(metadata[column], finite_float(plan[plan_column], f"matrix {plan_column}"), f"revision trace {column}")

    summary_frame = pd.read_csv(summary, keep_default_na=False)
    require(len(summary_frame) == 1, f"Revision summary is not one row: {summary}")
    row = summary_frame.iloc[0]
    summary_strings = {
        "weather": str(plan["weather_stem"]),
        "strategy": str(plan["strategy"]),
        "rev01_variant_id": str(plan["variant_id"]),
        "rev01_variant_manifest_sha256": variant_manifest_sha256,
        "rev01_effective_config_sha256": effective_config_sha256,
        "sizing_contract": str(plan["sizing_contract"]),
        "controller_semantics": str(plan["controller_semantics"]),
        "inference_preprocessing": str(plan["inference_preprocessing"]),
    }
    for column, expected in summary_strings.items():
        require(column in row.index and str(row[column]) == expected, f"Revision summary {column} differs: {plan['cell_id']}")
    summary_numeric = {
        "n_steps": float(plan["expected_trace_rows"]),
        "occupied_steps": float(plan["expected_occupied_rows"]),
        "paperb_met": float(plan["paperb_met"]),
        "paperb_people_activity_w_per_person": float(plan["paperb_people_activity_w"]),
        "paperb_spatial_quantile": float(plan["paperb_spatial_quantile"]),
        "paperb_signal_alpha": float(plan["paperb_signal_alpha"]),
        "paperb_tail_threshold": float(plan["paperb_tail_threshold"]),
        "paperb_asym_threshold": float(plan["paperb_asym_threshold"]),
        "paperb_save_heat_c": float(plan["paperb_save_heat_c"]),
        "paperb_save_cool_c": float(plan["paperb_save_cool_c"]),
    }
    for column, expected in summary_numeric.items():
        require(column in row.index, f"Revision summary lacks {column}: {plan['cell_id']}")
        observed = finite_float(row[column], f"revision summary {column}")
        require(math.isclose(observed, expected, rel_tol=0.0, abs_tol=1.0e-12), f"Revision summary {column} differs: {plan['cell_id']}")


def local_revision_artifact(
    batch: Path,
    plan: dict[str, Any],
    index_row: pd.Series,
    status_row: dict[str, Any],
    variant_manifest: dict[str, Any],
    spec: BatchSpec,
) -> Artifact:
    cell_id = str(plan["cell_id"])
    cell = (batch / "cells" / cell_id).resolve()
    require(cell.parent == (batch / "cells").resolve(), f"Revision cell path escapes batch: {cell_id}")
    require(cell.is_dir(), f"Missing revision cell directory: {cell}")
    trace = one_path(cell / "traces", "*.parquet", "revision trace")
    summary = cell / "summary/medium_office_trace_summary.csv"
    sql = one_path(cell / "energyplus", "**/eplusout.sql", "revision EnergyPlus SQL")
    trace_hash = verify_file(trace, str(index_row["trace_sha256"]), "revision trace")
    summary_hash = verify_file(summary, str(index_row["summary_sha256"]), "revision summary")
    require(Path(str(index_row["trace_path"])).resolve() == trace.resolve(), "CELL_INDEX trace path differs")
    require(Path(str(index_row["summary_path"])).resolve() == summary.resolve(), "CELL_INDEX summary path differs")
    for key in ("cell_id", "batch_id", "variant_id", "strategy", "anchor_id"):
        require(str(index_row[key]) == str(plan[key]), f"CELL_INDEX {key} differs: {cell_id}")
    require(str(index_row["status"]) == "complete_final_validated", f"CELL_INDEX status differs: {cell_id}")
    require(finite_float(index_row["elapsed_seconds"], f"CELL_INDEX elapsed_seconds {cell_id}") > 0, f"CELL_INDEX elapsed time is not positive: {cell_id}")
    result = read_json(cell / "CELL_RESULT.json", "revision cell result")
    status_copy = dict(status_row)
    require(status_copy.pop("event", None) == "cell_finished", f"Revision CELL_STATUS event differs: {cell_id}")
    require(status_copy == result, f"Revision CELL_RESULT differs from hashed CELL_STATUS journal: {cell_id}")
    expected_result = {
        "cell_id": cell_id,
        "batch_id": str(plan["batch_id"]),
        "variant_id": str(plan["variant_id"]),
        "strategy": str(plan["strategy"]),
        "anchor_id": str(plan["anchor_id"]),
        "accepted_comparator_cell_id": str(plan["accepted_comparator_cell_id"]),
        "runner_sha256": EXPECTED_REV01_RUNNER_SHA256,
    }
    for key, expected in expected_result.items():
        require(result.get(key) == expected, f"Revision CELL_RESULT {key} differs: {cell_id}")
    require(result.get("all_preliminary_gates_passed") is True, f"Revision preliminary gate failed: {cell_id}")
    require(result.get("status") == "complete_preliminary_validated", f"Revision result status differs: {cell_id}")
    require(result.get("schema_version") == "paperb_enb_rev01_cell_result_v1", f"Revision result schema differs: {cell_id}")
    outputs = result.get("outputs", {})
    require(outputs.get("all_cell_gates_passed") is True, f"Revision output gates failed: {cell_id}")
    retained = result.get("outputs", {}).get("retained_energyplus_sha256", {})
    sql_hash = verify_file(sql, str(retained.get("eplusout.sql", "")), "revision SQL")
    trace_record = outputs.get("trace", {})
    summary_record = outputs.get("summary", {})
    require(trace_record.get("sha256") == trace_hash, "Revision result trace hash differs")
    require(summary_record.get("sha256") == summary_hash, "Revision result summary hash differs")
    require(Path(str(trace_record.get("path", ""))).resolve() == trace.resolve(), f"Revision result trace path differs: {cell_id}")
    require(Path(str(summary_record.get("path", ""))).resolve() == summary.resolve(), f"Revision result summary path differs: {cell_id}")
    require(trace_record.get("rows") == int(plan["expected_trace_rows"]), f"Revision result trace rows differ: {cell_id}")
    require(trace_record.get("occupied_rows") == int(plan["expected_occupied_rows"]), f"Revision result occupied rows differ: {cell_id}")
    require(trace_record.get("strategy") == str(plan["strategy"]), f"Revision result trace strategy differs: {cell_id}")
    require(trace_record.get("weather") == str(plan["weather_stem"]), f"Revision result trace weather differs: {cell_id}")
    require(Path(str(outputs.get("energyplus_output_dir", ""))).resolve() == sql.parent.resolve(), f"Revision SQL/output-directory binding differs: {cell_id}")

    contract = read_json(cell / "RUN_CONTRACT.json", "revision run contract")
    contract_expected = {
        "schema_version": "paperb_enb_rev01_cell_contract_v1",
        "cell_id": cell_id,
        "batch_id": str(plan["batch_id"]),
        "variant_id": str(plan["variant_id"]),
        "strategy": str(plan["strategy"]),
        "anchor_id": str(plan["anchor_id"]),
        "accepted_comparator_cell_id": str(plan["accepted_comparator_cell_id"]),
        "runner_sha256": EXPECTED_REV01_RUNNER_SHA256,
        "runner_helper_sha256": EXPECTED_REV01_HELPER_SHA256,
        "model_sha256": EXPECTED_MODEL_SHA256,
        "weather_sha256": str(plan["weather_sha256"]),
        "resume_permitted": False,
        "reuse_or_splice_permitted": False,
        "purge_permitted": False,
    }
    for key, expected in contract_expected.items():
        require(contract.get(key) == expected, f"Revision RUN_CONTRACT {key} differs: {cell_id}")
    require(contract.get("command") == result.get("command"), f"Revision command differs between contract/result: {cell_id}")
    command = result.get("command")
    require(isinstance(command, list) and len(command) > 1 and all(isinstance(item, str) for item in command), f"Revision command is invalid: {cell_id}")
    require(Path(str(command[1])).resolve() == (batch / "freeze/code/run_medium_office_enb_rev01.py").resolve(), f"Revision command runner path differs: {cell_id}")
    command_paths = {
        "--output-dir": cell,
        "--idf": batch / "freeze/building/ASHRAE901_OfficeMedium_STD2019_Denver.idf",
        "--model-path": batch / "freeze/model/control_predictors.joblib",
        "--model-metrics-path": batch / "freeze/model/control_predictor_metrics.json",
        "--rev01-variant-manifest": batch / "freeze/plan/VARIANT_MANIFEST.json",
    }
    for flag, expected_path in command_paths.items():
        observed_path = Path(command_flag_value(command, flag, f"revision {cell_id}")).resolve()
        require(observed_path == expected_path.resolve(), f"Revision command path differs: {cell_id}/{flag}")
    weather_path = Path(command_flag_value(command, "--weather", f"revision {cell_id}")).resolve()
    require(path_within(weather_path, batch / "freeze/weather"), f"Revision command weather is outside frozen weather tree: {cell_id}")
    require(weather_path.name == f"{plan['weather_stem']}.epw", f"Revision command weather stem differs: {cell_id}")
    verify_file(weather_path, str(plan["weather_sha256"]), f"revision command weather {cell_id}")
    require(command_flag_value(command, "--strategies", f"revision {cell_id}") == str(plan["strategy"]), f"Revision command strategy differs: {cell_id}")
    require(command_flag_value(command, "--rev01-variant-id", f"revision {cell_id}") == str(plan["variant_id"]), f"Revision command variant differs: {cell_id}")

    variants = variant_manifest["variants"]
    effective_config = plan_effective_config(plan)
    require(effective_config == variants[str(plan["variant_id"])]["effective_config"], f"Revision effective config differs from manifest: {cell_id}")
    effective_hash = canonical_json_sha256(effective_config)
    variant_audit = read_json(cell / "REV01_VARIANT_AUDIT.json", "revision variant audit")
    audit_expected = {
        "schema_version": "paperb_enb_rev01_variant_audit_v1",
        "variant_id": str(plan["variant_id"]),
        "manifest_sha256": spec.variant_manifest_sha256,
        "parent_runner_sha256": EXPECTED_ACCEPTED_RUNNER_SHA256,
        "effective_config_sha256": effective_hash,
        "all_gates_passed": True,
    }
    for key, expected in audit_expected.items():
        require(variant_audit.get(key) == expected, f"Revision variant audit {key} differs: {cell_id}")
    require(variant_audit.get("effective_config") == effective_config, f"Revision variant-audit config differs: {cell_id}")
    require(variant_audit.get("sizing_audit") == {"mode": "accepted_denver", "source_idf_sha256": EXPECTED_IDF_SHA256}, f"Revision sizing audit differs: {cell_id}")
    require(Path(str(variant_audit.get("manifest_path", ""))).resolve() == (batch / "freeze/plan/VARIANT_MANIFEST.json").resolve(), f"Revision variant manifest path differs: {cell_id}")
    validate_revision_trace_summary_metadata(
        trace,
        summary,
        plan,
        effective_hash,
        spec.variant_manifest_sha256,
    )
    return Artifact(cell_id, trace, summary, sql, trace_hash, summary_hash, sql_hash)


def local_accepted_artifact(
    context: AcceptedContext,
    matrix_row: dict[str, Any],
    cell_id: str,
    snapshot: dict[str, Any],
) -> Artifact:
    require(re.fullmatch(r"SCALED-96-\d{3}", cell_id) is not None, f"Invalid accepted cell ID: {cell_id}")
    cell = (context.root / "cells" / cell_id).resolve()
    require(cell.parent == (context.root / "cells").resolve(), f"Accepted cell path escapes root: {cell_id}")
    trace = one_path(cell / "traces", "*.parquet", "accepted trace")
    summary = cell / "summary/medium_office_trace_summary.csv"
    sql = one_path(cell / "energyplus", "**/eplusout.sql", "accepted EnergyPlus SQL")
    result_path = one_path(cell, "CELL_RESULT.*.json", "accepted cell result")
    result = read_json(result_path, "accepted cell result")
    status = dict(context.status_by_cell_id[cell_id])
    status_result_hash = str(status.pop("cell_result_sha256", ""))
    verify_file(result_path, status_result_hash, f"accepted CELL_RESULT journal binding {cell_id}")
    require(status.pop("schema_version", None) == "paperb_enb_fresh96_cell_status_v2_envscoped", f"Accepted CELL_STATUS schema differs: {cell_id}")
    status["schema_version"] = "paperb_enb_fresh96_cell_result_v2_envscoped"
    require(status == result, f"Accepted CELL_RESULT differs from hashed CELL_STATUS journal: {cell_id}")
    require(result.get("all_gates_passed") is True, f"Accepted cell gate failed: {cell_id}")
    require(result.get("status") == "complete_validated", f"Accepted cell status differs: {cell_id}")
    require(result.get("scaled_run_id") == cell_id, f"Accepted result cell ID differs: {cell_id}")
    accepted_expected = {
        "runner_sha256": EXPECTED_ACCEPTED_RUNNER_SHA256,
        "anchor_id": str(matrix_row["anchor_id"]),
        "strategy": str(matrix_row["strategy"]),
        "physical_condition_id": str(matrix_row["physical_condition_id"]),
        "selector_label_id": str(matrix_row["selector_label_id"]),
    }
    for key, expected in accepted_expected.items():
        require(result.get(key) == expected, f"Accepted CELL_RESULT {key} differs: {cell_id}")
    for key, path in (("trace", trace), ("summary", summary), ("cell_result", result_path)):
        entry = snapshot.get(key, {})
        require(Path(str(entry.get("path", ""))).resolve() == path.resolve(), f"Accepted snapshot path differs: {cell_id}/{key}")
        verify_file(path, str(entry.get("sha256", "")), f"accepted snapshot {key}")
    trace_hash = verify_file(trace, str(result["outputs"]["trace"]["sha256"]), "accepted trace")
    summary_hash = verify_file(summary, str(result["outputs"]["summary"]["sha256"]), "accepted summary")
    sql_hash = verify_file(sql, str(result["outputs"]["retained_energyplus_sha256"]["eplusout.sql"]), "accepted SQL")
    trace_record = result["outputs"]["trace"]
    require(Path(str(trace_record.get("path", ""))).resolve() == trace.resolve(), f"Accepted result trace path differs: {cell_id}")
    require(trace_record.get("rows") == EXPECTED_ROWS and trace_record.get("occupied_rows") == EXPECTED_OCCUPIED_ROWS, f"Accepted result trace counts differ: {cell_id}")
    require(trace_record.get("strategy") == str(matrix_row["strategy"]), f"Accepted trace strategy metadata differs: {cell_id}")
    require(trace_record.get("weather") == str(matrix_row["selector_label_id"]), f"Accepted trace weather metadata differs: {cell_id}")
    require(Path(str(result["outputs"].get("energyplus_output_dir", ""))).resolve() == sql.parent.resolve(), f"Accepted SQL/output-directory binding differs: {cell_id}")
    trace_metadata = pd.read_parquet(trace, columns=["strategy", "weather"])
    require_constant_string(trace_metadata["strategy"], str(matrix_row["strategy"]), f"accepted trace strategy {cell_id}")
    require_constant_string(trace_metadata["weather"], str(matrix_row["selector_label_id"]), f"accepted trace weather {cell_id}")
    summary_frame = pd.read_csv(summary, keep_default_na=False)
    require(len(summary_frame) == 1, f"Accepted summary is not one row: {cell_id}")
    require(str(summary_frame.iloc[0]["strategy"]) == str(matrix_row["strategy"]), f"Accepted summary strategy differs: {cell_id}")
    require(str(summary_frame.iloc[0]["weather"]) == str(matrix_row["selector_label_id"]), f"Accepted summary weather differs: {cell_id}")
    return Artifact(cell_id, trace, summary, sql, trace_hash, summary_hash, sql_hash)


def validate_batch(
    batch: Path,
    spec: BatchSpec,
    accepted: AcceptedContext,
) -> tuple[list[MatchedArtifact], dict[str, Any]]:
    batch = batch.resolve()
    require(batch.is_dir(), f"Missing {spec.name} batch directory: {batch}")
    closure_path = batch / "BATCH_CLOSURE.json"
    closure = read_json(closure_path, f"{spec.name} batch closure")
    validate_batch_closure_fields(closure, spec)
    freeze_path = batch / "FREEZE.json"
    verify_file(freeze_path, str(closure.get("freeze_sha256", "")), f"{spec.name} freeze")
    freeze = read_json(freeze_path, f"{spec.name} freeze")
    validate_frozen_files(batch, freeze, closure, spec)

    matrix_path = batch / "freeze/plan" / spec.matrix_filename
    verify_file(matrix_path, spec.matrix_sha256, f"{spec.name} matrix")
    matrix = pd.read_csv(matrix_path, keep_default_na=False)
    require(len(matrix) == spec.rows, f"{spec.name} matrix row count differs")
    require(matrix["cell_id"].nunique() == spec.rows, f"{spec.name} cell IDs are not unique")
    require(set(matrix["matrix_schema_version"]) == {EXPECTED_MATRIX_SCHEMA}, f"{spec.name} matrix schema differs")
    require(set(matrix["batch_id"]) == {spec.batch_id}, f"{spec.name} matrix batch ID differs")
    require(
        set(matrix["accepted_matrix_sha256"]) == {EXPECTED_ACCEPTED_MATRIX_SHA256},
        f"{spec.name} does not pin the accepted matrix",
    )
    variant_manifest = validate_frozen_plan_authority(batch, freeze, matrix, spec)

    index_path = batch / "CELL_INDEX.parquet"
    index_info = closure.get("cell_index", {})
    require(Path(str(index_info.get("path", ""))).resolve() == index_path.resolve(), f"{spec.name} cell-index path differs")
    require(index_info.get("rows") == spec.rows, f"{spec.name} closure cell-index row count differs")
    verify_file(index_path, str(index_info.get("sha256", "")), f"{spec.name} cell index")
    index = pd.read_parquet(index_path)
    require(list(index.columns) == [
        "cell_id", "batch_id", "variant_id", "strategy", "anchor_id", "status",
        "elapsed_seconds", "trace_path", "trace_sha256", "summary_path", "summary_sha256",
    ], f"{spec.name} CELL_INDEX exact schema differs")
    require(len(index) == spec.rows, f"{spec.name} CELL_INDEX row count differs")
    require(index["cell_id"].nunique() == spec.rows, f"{spec.name} CELL_INDEX IDs are not unique")
    require(set(index["cell_id"]) == set(matrix["cell_id"]), f"{spec.name} exact cell coverage differs")
    require(set(index["batch_id"]) == {spec.batch_id}, f"{spec.name} CELL_INDEX batch ID differs")
    require(set(index["status"]) == {"complete_final_validated"}, f"{spec.name} final cell status differs")
    merged = matrix.merge(
        index[["cell_id", "variant_id", "strategy", "anchor_id"]],
        on="cell_id",
        suffixes=("", "_index"),
        validate="one_to_one",
    )
    for column in ("variant_id", "strategy", "anchor_id"):
        require((merged[column] == merged[f"{column}_index"]).all(), f"{spec.name} index {column} differs")

    attempts_path = batch / "ATTEMPTS.jsonl"
    status_path = batch / "CELL_STATUS.jsonl"
    verify_file(attempts_path, str(closure.get("attempts_sha256", "")), f"{spec.name} attempts journal")
    verify_file(status_path, str(closure.get("status_sha256", "")), f"{spec.name} cell-status journal")
    attempts = read_jsonl(attempts_path, f"{spec.name} attempts journal")
    statuses = read_jsonl(status_path, f"{spec.name} cell-status journal")
    require(len(attempts) == 2, f"{spec.name} attempts journal does not have start/finish records")
    require(attempts[0].get("event") == "batch_started", f"{spec.name} attempts start record differs")
    require(attempts[-1].get("event") == "batch_finished", f"{spec.name} attempts finish record differs")
    finish_record = dict(attempts[-1])
    finish_record.pop("event", None)
    # The final closure adds the hash of ATTEMPTS.jsonl after its terminal
    # record is written.  That self-referential hash cannot be embedded in the
    # record it hashes; every other field must remain exactly identical.
    closure_without_attempts_hash = dict(closure)
    attempts_hash = closure_without_attempts_hash.pop("attempts_sha256", None)
    require(isinstance(attempts_hash, str) and len(attempts_hash) == 64, f"{spec.name} attempts hash declaration is absent")
    require(finish_record == closure_without_attempts_hash, f"{spec.name} BATCH_CLOSURE differs from hashed attempts finish record")
    require(len(statuses) == spec.rows, f"{spec.name} status journal row count differs")
    require({str(row.get("cell_id")) for row in statuses} == set(matrix["cell_id"]), f"{spec.name} status exact cell coverage differs")
    require(all(row.get("all_preliminary_gates_passed") is True for row in statuses), f"{spec.name} preliminary status gate failed")
    require(all(row.get("status") == "complete_preliminary_validated" for row in statuses), f"{spec.name} preliminary status differs")
    status_by_id = {str(row["cell_id"]): row for row in statuses}

    final_path = batch / "storage_migration/FINAL_CLOSURE.json"
    final = read_json(final_path, f"{spec.name} storage final closure")
    require(final.get("schema_version") == EXPECTED_STORAGE_SCHEMA, f"{spec.name} storage schema differs")
    require(final.get("status") == "PASS" and final.get("all_gates_passed") is True, f"{spec.name} storage did not pass")
    require(final.get("remaining_source_csvs") == [], f"{spec.name} retained allowlisted source CSVs")
    require(final == closure.get("storage_final"), f"{spec.name} embedded/final storage closure differs")
    stage_path = batch / "storage_migration/STAGE_CLOSURE.json"
    verify_file(stage_path, str(final.get("stage_closure_sha256", "")), f"{spec.name} storage stage closure")
    stage = read_json(stage_path, f"{spec.name} storage stage closure")
    require(stage == closure.get("storage_stage"), f"{spec.name} embedded/stage storage closure differs")
    require(stage.get("all_gates_passed") is True and stage.get("status") == "PASS", f"{spec.name} storage stage did not pass")
    stage_journal = batch / "storage_migration/STAGE_JOURNAL.jsonl"
    finalize_journal = batch / "storage_migration/FINALIZE_JOURNAL.jsonl"
    verify_file(stage_journal, str(stage.get("journal_sha256", "")), f"{spec.name} storage stage journal")
    verify_file(finalize_journal, str(final.get("finalize_journal_sha256", "")), f"{spec.name} storage finalize journal")
    manifest_path = batch / "storage_migration/STORAGE_MANIFEST.parquet"
    require(Path(str(final.get("manifest_path", ""))).resolve() == manifest_path.resolve(), f"{spec.name} storage-manifest path differs")
    verify_file(manifest_path, str(final.get("manifest_sha256", "")), f"{spec.name} storage manifest")

    actual_cell_dirs = {path.name for path in (batch / "cells").iterdir() if path.is_dir()}
    require(actual_cell_dirs == set(matrix["cell_id"]), f"{spec.name} cell directories differ from plan")
    index_by_id = index.set_index("cell_id", drop=False)
    accepted_matrix = accepted.matrix.set_index("scaled_run_id", drop=False)
    accepted_cache: dict[str, Artifact] = {}
    snapshot = freeze.get("accepted_comparator_snapshot", {})
    require(isinstance(snapshot, dict), f"{spec.name} accepted-comparator snapshot is not an object")
    require(set(snapshot) == set(matrix["accepted_comparator_cell_id"]), f"{spec.name} accepted-comparator snapshot exact coverage differs")
    matches: list[MatchedArtifact] = []
    for plan in matrix.to_dict(orient="records"):
        cell_id = str(plan["cell_id"])
        comparator_id = str(plan["accepted_comparator_cell_id"])
        require(comparator_id in snapshot, f"{spec.name} freeze lacks comparator snapshot: {comparator_id}")
        require(comparator_id in accepted_matrix.index, f"Comparator absent from accepted matrix: {comparator_id}")
        accepted_row = accepted_matrix.loc[comparator_id]
        require(str(accepted_row["strategy"]) == str(plan["accepted_comparator_strategy"]), f"Comparator strategy differs: {cell_id}")
        require(str(accepted_row["anchor_id"]) == str(plan["anchor_id"]), f"Comparator anchor differs: {cell_id}")
        require(str(accepted_row["physical_condition_id"]) == str(plan["physical_condition_id"]), f"Comparator physical condition differs: {cell_id}")
        revision_artifact = local_revision_artifact(
            batch,
            plan,
            index_by_id.loc[cell_id],
            status_by_id[cell_id],
            variant_manifest,
            spec,
        )
        if comparator_id not in accepted_cache:
            accepted_cache[comparator_id] = local_accepted_artifact(
                accepted,
                accepted_row.to_dict(),
                comparator_id,
                snapshot[comparator_id],
            )
        matches.append(MatchedArtifact(spec.name, plan, revision_artifact, accepted_cache[comparator_id]))
    require(len(matches) == spec.rows, f"{spec.name} matched-artifact coverage differs")
    audit = {
        "batch_name": spec.name,
        "batch_root": str(batch),
        "batch_closure_path": str(closure_path),
        "batch_closure_sha256": sha256_file(closure_path),
        "freeze_sha256": sha256_file(freeze_path),
        "matrix_path": str(matrix_path),
        "matrix_sha256": spec.matrix_sha256,
        "cell_index_sha256": sha256_file(index_path),
        "storage_final_closure_sha256": sha256_file(final_path),
        "planned_cells": spec.rows,
        "validated_cells": len(matches),
        "accepted_comparator_cells": len(accepted_cache),
    }
    return matches, audit


def discover_zone_pairs(columns: Iterable[str]) -> list[tuple[str, str, str]]:
    names = set(columns)
    pairs: list[tuple[str, str, str]] = []
    for column in sorted(names):
        match = ZONE_AIR_RE.match(column)
        if match is None:
            continue
        zone = match.group("zone")
        radiant = f"zone_{zone}_tr_c"
        if radiant in names:
            pairs.append((zone, column, radiant))
    return pairs


def coerce_occupied(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series.dtype):
        require(not series.isna().any(), "Occupancy contains nulls")
        return series.to_numpy(dtype=bool)
    numeric = pd.to_numeric(series, errors="coerce")
    require(numeric.notna().all(), "Occupancy is not Boolean/0/1")
    require(numeric.isin([0, 1]).all(), "Occupancy contains values other than 0/1")
    return numeric.to_numpy(dtype=int).astype(bool)


def strict_bool_array(series: pd.Series, label: str) -> np.ndarray:
    require(not series.isna().any(), f"{label} contains nulls")
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(series.dtype):
        numeric = pd.to_numeric(series, errors="coerce")
        require(numeric.notna().all() and numeric.isin([0, 1]).all(), f"{label} must contain only Boolean/0/1")
        return numeric.to_numpy(dtype=int).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    require(normalized.isin(["true", "false"]).all(), f"{label} contains an invalid Boolean token")
    return normalized.eq("true").to_numpy(dtype=bool)


def finite_numeric_array(series: pd.Series, label: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    require(values.size > 0, f"{label} is empty")
    require(np.isfinite(values).all(), f"{label} contains nonfinite values")
    return values


def infer_step_hours(frame: pd.DataFrame) -> float:
    require("sim_time_hours" in frame, "Trace lacks sim_time_hours")
    elapsed = pd.to_numeric(frame["sim_time_hours"], errors="coerce").to_numpy(float)
    require(np.isfinite(elapsed).all(), "sim_time_hours contains non-finite values")
    differences = np.diff(elapsed)
    require((differences > 0).all(), "sim_time_hours is not strictly increasing")
    step = float(np.median(differences))
    require(math.isclose(step, STEP_HOURS, abs_tol=1.0e-9), f"Trace cadence differs: {step}")
    require(np.allclose(differences, step, rtol=0.0, atol=1.0e-9), "Trace cadence is irregular")
    return step


def read_sql_facility_meters(sql_path: Path) -> dict[str, dict[str, Any]]:
    uri = f"{sql_path.resolve().as_uri()}?mode=ro"
    query = """
        SELECT d.VariableName, d.ReportingFrequency, d.VariableUnits,
               COUNT(*) AS n, SUM(r.VariableValue) AS joules
        FROM ReportMeterData AS r
        JOIN ReportMeterDataDictionary AS d
          USING (ReportMeterDataDictionaryIndex)
        WHERE d.VariableName IN ('Electricity:Facility', 'NaturalGas:Facility')
          AND d.ReportingFrequency = 'Hourly'
        GROUP BY d.VariableName, d.ReportingFrequency, d.VariableUnits
    """
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            rows = connection.execute(query).fetchall()
    except sqlite3.Error as exc:
        raise SynthesisContractError(f"Cannot read facility meters from {sql_path}: {exc}") from exc
    result = {
        str(name): {"frequency": str(frequency), "units": str(units), "rows": int(n), "joules": float(joules)}
        for name, frequency, units, n, joules in rows
    }
    require(set(result) == {"Electricity:Facility", "NaturalGas:Facility"}, f"SQL facility-meter coverage differs: {sql_path}")
    for name, item in result.items():
        require(item["frequency"] == "Hourly", f"SQL meter frequency differs: {name}")
        require(item["units"] == "J", f"SQL meter units are not joules: {name}")
        require(item["rows"] == 8760, f"SQL meter does not have 8760 hourly rows: {name}")
        require(np.isfinite(item["joules"]) and item["joules"] > 0, f"SQL meter total is not finite and positive: {name}")
    return result


def tag(value: float) -> str:
    return f"{value:g}c".replace(".", "p")


def count_value(frame: pd.DataFrame, field: str, value: Any, mask: np.ndarray | None = None) -> int:
    if field not in frame:
        return 0
    series = frame[field] if mask is None else frame.loc[mask, field]
    if isinstance(value, str):
        return int(series.fillna("").astype(str).eq(value).sum())
    numeric = pd.to_numeric(series, errors="coerce")
    return int(numeric.eq(value).sum())


def validate_action_domain(frame: pd.DataFrame) -> None:
    """Reject action tokens outside the frozen controller trace contract."""
    text_domains = {
        "paperb_request_branch": {
            "not_recorded", "relax", "warm_protection", "cold_protection", "risk_hold",
        },
        "paperb_request_outcome": {
            "not_k_recorded", "not_recorded", "applied", "dwell_blocked",
            "bound_saturated", "hold",
        },
    }
    # ``not_k_recorded`` never appears in the frozen contract; keeping the
    # accepted set construction explicit avoids permissive string coercion.
    text_domains["paperb_request_outcome"].remove("not_k_recorded")
    for field, allowed in text_domains.items():
        require(field in frame, f"Trace lacks action field: {field}")
        require(not frame[field].isna().any(), f"Action field contains nulls: {field}")
        observed = set(frame[field].astype(str))
        require(
            observed <= allowed,
            f"Action field contains out-of-contract values: {field}/{sorted(observed - allowed)}",
        )
    require("paperb_request_source" in frame, "Trace lacks action field: paperb_request_source")
    require(not frame["paperb_request_source"].isna().any(), "Action field contains nulls: paperb_request_source")
    numeric_domains = {
        "paperb_requested_direction": {-1, 0, 1},
        "paperb_setpoint_moved": {0, 1},
        "paperb_hold_guard": {0, 1},
    }
    for field, allowed in numeric_domains.items():
        require(field in frame, f"Trace lacks action field: {field}")
        numeric = pd.to_numeric(frame[field], errors="coerce")
        require(numeric.notna().all(), f"Action field is not finite numeric: {field}")
        require(np.isfinite(numeric.to_numpy(float)).all(), f"Action field is nonfinite: {field}")
        integral = numeric.astype(int)
        require(np.equal(numeric, integral).all(), f"Action field is not integral: {field}")
        require(set(integral) <= allowed, f"Action field contains out-of-contract values: {field}")


def compute_frame_metrics(
    frame: pd.DataFrame,
    summary: pd.Series,
    sql_meters: dict[str, dict[str, Any]],
    *,
    expected_rows: int,
    expected_occupied_rows: int,
    expected_zone_count: int = EXPECTED_ZONE_COUNT,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(len(frame) == expected_rows, f"Trace row count differs: {len(frame)} != {expected_rows}")
    occupied = coerce_occupied(frame["occupied"])
    require(int(occupied.sum()) == expected_occupied_rows, "Occupied trace-row count differs")
    require(expected_occupied_rows > 0, "Occupied-row denominator is zero")
    step = infer_step_hours(frame)
    zones = discover_zone_pairs(frame.columns)
    require(len(zones) == expected_zone_count, f"Zone operative-temperature coverage differs: {len(zones)}")
    validate_action_domain(frame)

    electricity_j = finite_numeric_array(frame["electricity_facility_j"], "trace facility electricity")
    gas_j = finite_numeric_array(frame["natural_gas_facility_j"], "trace facility natural gas")
    require(np.isfinite(electricity_j).all() and (electricity_j >= 0).all(), "Trace facility electricity is invalid")
    require(np.isfinite(gas_j).all() and (gas_j >= 0).all(), "Trace facility natural gas is invalid")
    trace_electricity_j = float(electricity_j.sum())
    trace_gas_j = float(gas_j.sum())
    require(trace_electricity_j > 0 and trace_gas_j > 0, "Trace facility-energy totals must be positive")
    sql_electricity_j = finite_float(sql_meters["Electricity:Facility"]["joules"], "SQL facility electricity")
    sql_gas_j = finite_float(sql_meters["NaturalGas:Facility"]["joules"], "SQL facility natural gas")
    require(math.isclose(trace_electricity_j, sql_electricity_j, rel_tol=1.0e-12, abs_tol=0.1), "Trace/SQL facility electricity totals differ")
    require(math.isclose(trace_gas_j, sql_gas_j, rel_tol=1.0e-12, abs_tol=0.1), "Trace/SQL facility natural-gas totals differ")
    electricity_kwh = trace_electricity_j / JOULES_PER_KWH
    gas_kwh = trace_gas_j / JOULES_PER_KWH
    summary_electricity = finite_float(summary["electricity_kwh"], "summary electricity_kwh")
    summary_gas = finite_float(summary["natural_gas_kwh"], "summary natural_gas_kwh")
    require(summary_electricity > 0 and summary_gas > 0, "Summary facility-energy totals must be positive")
    require(math.isclose(electricity_kwh, summary_electricity, rel_tol=1.0e-10, abs_tol=1.0e-5), "Trace/summary electricity differs")
    require(math.isclose(gas_kwh, summary_gas, rel_tol=1.0e-10, abs_tol=1.0e-5), "Trace/summary natural gas differs")

    heating_setpoint = finite_numeric_array(frame.loc[occupied, "heating_setpoint_c"], "occupied heating setpoint")
    cooling_setpoint = finite_numeric_array(frame.loc[occupied, "cooling_setpoint_c"], "occupied cooling setpoint")

    metrics: dict[str, Any] = {
        "trace_rows": int(len(frame)),
        "occupied_rows": int(occupied.sum()),
        "occupied_hours": float(occupied.sum() * step),
        "zone_count": int(len(zones)),
        "timestep_hours": step,
        "facility_electricity_j": trace_electricity_j,
        "facility_natural_gas_including_service_water_j": trace_gas_j,
        "facility_electricity_kwh_delivered": electricity_kwh,
        "facility_natural_gas_including_service_water_kwh_delivered": gas_kwh,
        "total_delivered_site_electricity_plus_facility_gas_kwh": electricity_kwh + gas_kwh,
        "occupied_mean_heating_setpoint_c": float(heating_setpoint.mean()),
        "occupied_mean_cooling_setpoint_c": float(cooling_setpoint.mean()),
    }
    operative_by_zone: dict[str, np.ndarray] = {}
    for zone, air_column, radiant_column in zones:
        air = pd.to_numeric(frame[air_column], errors="coerce").to_numpy(float)
        radiant = pd.to_numeric(frame[radiant_column], errors="coerce").to_numpy(float)
        operative = (air + radiant) / 2.0
        require(np.isfinite(operative[occupied]).all(), f"Occupied operative temperature is invalid: {zone}")
        operative_by_zone[zone] = operative
    for threshold in WARM_THRESHOLDS_C:
        values = {
            zone: float(np.maximum(operative[occupied] - threshold, 0.0).sum() * step)
            for zone, operative in operative_by_zone.items()
        }
        worst_zone, worst_value = sorted(values.items(), key=lambda item: (-item[1], item[0]))[0]
        metrics[f"occupied_worst_zone_degree_hours_above_{tag(threshold)}"] = worst_value
        metrics[f"occupied_worst_zone_degree_hours_above_{tag(threshold)}_zone"] = worst_zone
    for threshold in COLD_THRESHOLDS_C:
        values = {
            zone: float(np.maximum(threshold - operative[occupied], 0.0).sum() * step)
            for zone, operative in operative_by_zone.items()
        }
        worst_zone, worst_value = sorted(values.items(), key=lambda item: (-item[1], item[0]))[0]
        metrics[f"occupied_worst_zone_degree_hours_below_{tag(threshold)}"] = worst_value
        metrics[f"occupied_worst_zone_degree_hours_below_{tag(threshold)}_zone"] = worst_zone

    fixed_counts = {
        "controller_evaluated_rows": int(pd.Series(frame["paperb_request_branch"]).fillna("").astype(str).ne("not_recorded").sum()),
        "branch_relax_count": count_value(frame, "paperb_request_branch", "relax"),
        "branch_warm_protection_count": count_value(frame, "paperb_request_branch", "warm_protection"),
        "branch_cold_protection_count": count_value(frame, "paperb_request_branch", "cold_protection"),
        "branch_risk_hold_count": count_value(frame, "paperb_request_branch", "risk_hold"),
        "outcome_applied_count": count_value(frame, "paperb_request_outcome", "applied"),
        "outcome_dwell_blocked_count": count_value(frame, "paperb_request_outcome", "dwell_blocked"),
        "outcome_bound_saturated_count": count_value(frame, "paperb_request_outcome", "bound_saturated"),
        "outcome_hold_count": count_value(frame, "paperb_request_outcome", "hold"),
        "requested_warm_direction_count": count_value(frame, "paperb_requested_direction", 1),
        "requested_cold_direction_count": count_value(frame, "paperb_requested_direction", -1),
        "setpoint_moved_count": count_value(frame, "paperb_setpoint_moved", 1),
        "hold_guard_count": count_value(frame, "paperb_hold_guard", 1),
    }
    metrics.update(fixed_counts)
    running_mean = pd.to_numeric(frame.loc[occupied, "running_mean_outdoor_c"], errors="coerce").to_numpy(float)
    require(np.isfinite(running_mean).all(), "Occupied running-mean outdoor temperature is invalid")
    outside = (running_mean < ADAPTIVE_RUNNING_MEAN_LOW_C) | (running_mean > ADAPTIVE_RUNNING_MEAN_HIGH_C)
    metrics["adaptive_running_mean_outside_10_to_33p5c_rows_occ"] = int(outside.sum())
    metrics["adaptive_running_mean_outside_10_to_33p5c_pct_occ"] = float(100.0 * outside.mean())
    if "adaptive_running_mean_was_clamped" in frame:
        clamped = strict_bool_array(
            frame.loc[occupied, "adaptive_running_mean_was_clamped"],
            "adaptive_running_mean_was_clamped",
        )
        metrics["adaptive_running_mean_clamped_rows_occ"] = int(clamped.sum())
        metrics["adaptive_running_mean_clamped_pct_occ"] = float(100.0 * clamped.mean())
        metrics["adaptive_running_mean_clamp_flag_present"] = True
    else:
        # Accepted Fresh96 traces predate the clamp implementation.  Zero is the
        # executed-clamp count (not an imputation of the running-mean signal),
        # while the explicit presence flag preserves that schema distinction.
        metrics["adaptive_running_mean_clamped_rows_occ"] = 0
        metrics["adaptive_running_mean_clamped_pct_occ"] = 0.0
        metrics["adaptive_running_mean_clamp_flag_present"] = False

    action_rows: list[dict[str, Any]] = []
    for field in (
        "paperb_request_branch",
        "paperb_request_outcome",
        "paperb_request_source",
        "paperb_requested_direction",
        "paperb_setpoint_moved",
        "paperb_hold_guard",
    ):
        require(field in frame, f"Trace lacks action field: {field}")
        require(not frame[field].isna().any(), f"Action field contains nulls: {field}")
        values = frame[field].astype(str)
        occupied_values = values[occupied]
        for value in sorted(set(values)):
            action_rows.append(
                {
                    "field": field,
                    "value": value,
                    "count_all_rows": int(values.eq(value).sum()),
                    "count_occupied_rows": int(occupied_values.eq(value).sum()),
                }
            )
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            require(np.isfinite(float(value)), f"Computed metric is nonfinite: {key}")
    return metrics, action_rows


def read_artifact_metrics(
    artifact: Artifact,
    expected_rows: int,
    expected_occupied_rows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    schema = pq.read_schema(artifact.trace)
    zones = discover_zone_pairs(schema.names)
    required = {
        "occupied",
        "sim_time_hours",
        "running_mean_outdoor_c",
        "heating_setpoint_c",
        "cooling_setpoint_c",
        "electricity_facility_j",
        "natural_gas_facility_j",
        "paperb_request_branch",
        "paperb_request_outcome",
        "paperb_request_source",
        "paperb_requested_direction",
        "paperb_setpoint_moved",
        "paperb_hold_guard",
    }
    for _, air, radiant in zones:
        required.update((air, radiant))
    if "adaptive_running_mean_was_clamped" in schema.names:
        required.add("adaptive_running_mean_was_clamped")
    missing = sorted(required - set(schema.names))
    require(not missing, f"Trace lacks required columns {missing}: {artifact.trace}")
    frame = pd.read_parquet(artifact.trace, columns=sorted(required))
    summary_frame = pd.read_csv(artifact.summary)
    require(len(summary_frame) == 1, f"Summary is not one row: {artifact.summary}")
    sql_meters = read_sql_facility_meters(artifact.sql)
    return compute_frame_metrics(
        frame,
        summary_frame.iloc[0],
        sql_meters,
        expected_rows=expected_rows,
        expected_occupied_rows=expected_occupied_rows,
    )


def stress_grid_class(anchor_role: str) -> str:
    mapping = {
        "present_typical": "present_typical",
        "future_extreme": "late_century_hot_extreme",
    }
    require(anchor_role in mapping, f"Unknown stress-grid anchor role: {anchor_role}")
    return mapping[anchor_role]


DELTA_METRICS = (
    "facility_electricity_kwh_delivered",
    "facility_natural_gas_including_service_water_kwh_delivered",
    "total_delivered_site_electricity_plus_facility_gas_kwh",
    "occupied_worst_zone_degree_hours_above_26c",
    "occupied_worst_zone_degree_hours_above_28c",
    "occupied_worst_zone_degree_hours_above_30c",
    "occupied_worst_zone_degree_hours_below_18c",
    "occupied_worst_zone_degree_hours_below_16c",
    "occupied_mean_heating_setpoint_c",
    "occupied_mean_cooling_setpoint_c",
    "branch_relax_count",
    "branch_warm_protection_count",
    "branch_cold_protection_count",
    "branch_risk_hold_count",
    "outcome_applied_count",
    "outcome_dwell_blocked_count",
    "outcome_bound_saturated_count",
    "outcome_hold_count",
    "setpoint_moved_count",
    "hold_guard_count",
    "adaptive_running_mean_outside_10_to_33p5c_rows_occ",
    "adaptive_running_mean_outside_10_to_33p5c_pct_occ",
    "adaptive_running_mean_clamped_rows_occ",
    "adaptive_running_mean_clamped_pct_occ",
)
ROLE_METRIC_KEYS = (
    "trace_rows",
    "occupied_rows",
    "occupied_hours",
    "zone_count",
    "timestep_hours",
    "facility_electricity_j",
    "facility_natural_gas_including_service_water_j",
    "facility_electricity_kwh_delivered",
    "facility_natural_gas_including_service_water_kwh_delivered",
    "total_delivered_site_electricity_plus_facility_gas_kwh",
    "occupied_mean_heating_setpoint_c",
    "occupied_mean_cooling_setpoint_c",
    "occupied_worst_zone_degree_hours_above_26c",
    "occupied_worst_zone_degree_hours_above_26c_zone",
    "occupied_worst_zone_degree_hours_above_28c",
    "occupied_worst_zone_degree_hours_above_28c_zone",
    "occupied_worst_zone_degree_hours_above_30c",
    "occupied_worst_zone_degree_hours_above_30c_zone",
    "occupied_worst_zone_degree_hours_below_18c",
    "occupied_worst_zone_degree_hours_below_18c_zone",
    "occupied_worst_zone_degree_hours_below_16c",
    "occupied_worst_zone_degree_hours_below_16c_zone",
    "controller_evaluated_rows",
    "branch_relax_count",
    "branch_warm_protection_count",
    "branch_cold_protection_count",
    "branch_risk_hold_count",
    "outcome_applied_count",
    "outcome_dwell_blocked_count",
    "outcome_bound_saturated_count",
    "outcome_hold_count",
    "requested_warm_direction_count",
    "requested_cold_direction_count",
    "setpoint_moved_count",
    "hold_guard_count",
    "adaptive_running_mean_outside_10_to_33p5c_rows_occ",
    "adaptive_running_mean_outside_10_to_33p5c_pct_occ",
    "adaptive_running_mean_clamped_rows_occ",
    "adaptive_running_mean_clamped_pct_occ",
    "adaptive_running_mean_clamp_flag_present",
)
MATCHED_BASE_COLUMNS = (
    "batch_name",
    "batch_id",
    "cell_id",
    "experiment",
    "variant_id",
    "revision_strategy",
    "accepted_comparator_strategy",
    "accepted_comparator_cell_id",
    "anchor_id",
    "stress_grid_class",
    "city",
    "scenario",
    "time_slice",
    "selector_role",
    "weather_year",
    "physical_condition_id",
    "sentinel_id",
    "sampling_frame",
    "interpretation_scope",
)
CONFIG_COLUMNS = (
    "paperb_spatial_quantile",
    "paperb_tail_threshold",
    "paperb_asym_threshold",
    "paperb_save_heat_c",
    "paperb_save_cool_c",
    "paperb_warm_protect_cool_c",
    "paperb_cold_protect_heat_c",
    "paperb_tighten_dwell_steps",
    "paperb_relax_dwell_steps",
    "paperb_pmv_threshold",
    "paperb_signal_alpha",
    "paperb_adaptive_rm_clamp_low_c",
    "paperb_adaptive_rm_clamp_high_c",
    "paperb_met",
    "paperb_people_activity_w",
    "controller_semantics",
    "inference_preprocessing",
    "sizing_contract",
)
LOWER_COUNT_METRICS = (
    "total_delivered_site_electricity_plus_facility_gas_kwh",
    "occupied_worst_zone_degree_hours_above_26c",
    "occupied_worst_zone_degree_hours_above_28c",
    "occupied_worst_zone_degree_hours_above_30c",
    "occupied_worst_zone_degree_hours_below_18c",
    "occupied_worst_zone_degree_hours_below_16c",
)


def build_matched_tables(
    matches: list[MatchedArtifact],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_cache: dict[tuple[str, str], tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    matched_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    for match in matches:
        plan = match.plan
        expected_rows = int(plan["expected_trace_rows"])
        expected_occ = int(plan["expected_occupied_rows"])
        require(expected_rows == EXPECTED_ROWS, f"Plan trace-row contract differs: {plan['cell_id']}")
        require(expected_occ == EXPECTED_OCCUPIED_ROWS, f"Plan occupied-row contract differs: {plan['cell_id']}")
        role_metrics: dict[str, dict[str, Any]] = {}
        for role, artifact in (("revision", match.revision), ("accepted", match.accepted)):
            cache_key = (role, artifact.artifact_cell_id)
            if cache_key not in metric_cache:
                metric_cache[cache_key] = read_artifact_metrics(artifact, expected_rows, expected_occ)
            metrics, counts = metric_cache[cache_key]
            require(tuple(metrics) == ROLE_METRIC_KEYS, f"Metric schema differs: {artifact.artifact_cell_id}")
            role_metrics[role] = metrics
            for count in counts:
                action_rows.append(
                    {
                        "match_cell_id": str(plan["cell_id"]),
                        "experiment": str(plan["experiment"]),
                        "variant_id": str(plan["variant_id"]),
                        "anchor_id": str(plan["anchor_id"]),
                        "stress_grid_class": stress_grid_class(str(plan["anchor_role"])),
                        "artifact_role": role,
                        "artifact_cell_id": artifact.artifact_cell_id,
                        **count,
                    }
                )
            input_rows.append(
                {
                    "match_cell_id": str(plan["cell_id"]),
                    "artifact_role": role,
                    "artifact_cell_id": artifact.artifact_cell_id,
                    "trace_path": str(artifact.trace),
                    "trace_sha256": artifact.trace_sha256,
                    "summary_path": str(artifact.summary),
                    "summary_sha256": artifact.summary_sha256,
                    "sql_path": str(artifact.sql),
                    "sql_sha256": artifact.sql_sha256,
                }
            )
        row: dict[str, Any] = {
            "batch_name": match.batch_name,
            "batch_id": str(plan["batch_id"]),
            "cell_id": str(plan["cell_id"]),
            "experiment": str(plan["experiment"]),
            "variant_id": str(plan["variant_id"]),
            "revision_strategy": str(plan["strategy"]),
            "accepted_comparator_strategy": str(plan["accepted_comparator_strategy"]),
            "accepted_comparator_cell_id": str(plan["accepted_comparator_cell_id"]),
            "anchor_id": str(plan["anchor_id"]),
            "stress_grid_class": stress_grid_class(str(plan["anchor_role"])),
            "city": str(plan["city"]),
            "scenario": str(plan["scenario"]),
            "time_slice": str(plan["time_slice"]),
            "selector_role": str(plan["selector_role"]),
            "weather_year": int(plan["weather_year"]),
            "physical_condition_id": str(plan["physical_condition_id"]),
            "sentinel_id": str(plan["anchor_id"]) if str(plan["experiment"]) in {"M2", "M3"} else "",
            "sampling_frame": "predeclared_4_anchor_sentinel_panel" if str(plan["experiment"]) in {"M2", "M3"} else "full_24_anchor_stress_test_grid",
            "interpretation_scope": "stress-test grid; time horizon and weather-year selection rule are confounded; not a causal future-climate contrast",
        }
        for config_column in CONFIG_COLUMNS:
            row[config_column] = plan[config_column]
        for role in ("accepted", "revision"):
            for metric, value in role_metrics[role].items():
                row[f"{role}_{metric}"] = value
        for metric in DELTA_METRICS:
            accepted_value = role_metrics["accepted"].get(metric, np.nan)
            revision_value = role_metrics["revision"].get(metric, np.nan)
            if pd.isna(accepted_value) or pd.isna(revision_value):
                row[f"delta_{metric}"] = np.nan
            else:
                row[f"delta_{metric}"] = float(revision_value) - float(accepted_value)
        denominator = float(role_metrics["accepted"]["total_delivered_site_electricity_plus_facility_gas_kwh"])
        require(np.isfinite(denominator) and denominator > 0, f"Accepted delivered-site-energy denominator is not finite and positive: {plan['cell_id']}")
        row["delta_total_delivered_site_energy_pct_of_accepted"] = (
            100.0 * row["delta_total_delivered_site_electricity_plus_facility_gas_kwh"] / denominator
        )
        matched_rows.append(row)
    matched = pd.DataFrame(matched_rows).sort_values(["experiment", "variant_id", "anchor_id", "cell_id"]).reset_index(drop=True)
    actions = pd.DataFrame(action_rows).sort_values(["match_cell_id", "artifact_role", "field", "value"]).reset_index(drop=True)
    inputs = pd.DataFrame(input_rows).drop_duplicates().sort_values(["match_cell_id", "artifact_role"]).reset_index(drop=True)
    validate_action_count_mass(matched, actions)
    summaries = aggregate_matched(matched)
    return matched, summaries, actions, inputs


def validate_action_count_mass(matched: pd.DataFrame, actions: pd.DataFrame) -> None:
    required_fields = {
        "paperb_request_branch",
        "paperb_request_outcome",
        "paperb_request_source",
        "paperb_requested_direction",
        "paperb_setpoint_moved",
        "paperb_hold_guard",
    }
    require(not actions.empty, "Action-count table is empty")
    require(not actions.duplicated(["match_cell_id", "artifact_role", "field", "value"]).any(), "Action-count keys are not unique")
    for (cell_id, role), group in actions.groupby(["match_cell_id", "artifact_role"], sort=False):
        require(set(group["field"]) == required_fields, f"Action-count field coverage differs: {cell_id}/{role}")
        for field, field_group in group.groupby("field", sort=False):
            require(int(field_group["count_all_rows"].sum()) == EXPECTED_ROWS, f"Action-count all-row mass differs: {cell_id}/{role}/{field}")
            require(int(field_group["count_occupied_rows"].sum()) == EXPECTED_OCCUPIED_ROWS, f"Action-count occupied-row mass differs: {cell_id}/{role}/{field}")
    require(actions.groupby(["match_cell_id", "artifact_role"]).ngroups == len(matched) * 2, "Action-count artifact coverage differs")

    lookup = actions.set_index(["match_cell_id", "artifact_role", "field", "value"])["count_all_rows"]
    for row in matched.to_dict(orient="records"):
        cell_id = str(row["cell_id"])
        for role in ("accepted", "revision"):
            checks = {
                ("paperb_request_branch", "risk_hold"): f"{role}_branch_risk_hold_count",
                ("paperb_request_outcome", "hold"): f"{role}_outcome_hold_count",
                ("paperb_hold_guard", "1"): f"{role}_hold_guard_count",
            }
            for (field, value), metric in checks.items():
                observed = int(lookup.get((cell_id, role, field, value), 0))
                require(observed == int(row[metric]), f"Action-count/fixed-metric binding differs: {cell_id}/{role}/{metric}")


def validate_learned_contribution_mechanisms(
    matched: pd.DataFrame,
    actions: pd.DataFrame,
) -> dict[str, Any]:
    """Close the full-grid null mechanism and four-sentinel reproducibility gates."""

    null_rows = matched.loc[
        matched["experiment"].eq("M5")
        & matched["variant_id"].eq("rev01_p90_signal_a000_full24")
    ].copy()
    require(len(null_rows) == 24 and null_rows["anchor_id"].nunique() == 24, "M5 full-grid null coverage differs")
    null_actions = actions.loc[
        actions["experiment"].eq("M5")
        & actions["variant_id"].eq("rev01_p90_signal_a000_full24")
        & actions["artifact_role"].eq("revision")
        & actions["field"].eq("paperb_request_branch")
    ]
    occupied_by_branch = (
        null_actions.groupby("value", sort=True)["count_occupied_rows"].sum().astype(int).to_dict()
    )
    expected_occupied = 24 * EXPECTED_OCCUPIED_ROWS
    require(occupied_by_branch.get("relax", 0) == expected_occupied, "M5 null does not request relax at every occupied step")
    for branch in ("warm_protection", "cold_protection", "risk_hold"):
        require(occupied_by_branch.get(branch, 0) == 0, f"M5 null unexpectedly enters {branch}")

    prior = matched.loc[
        matched["experiment"].eq("M3")
        & matched["variant_id"].eq("rev01_p90_signal_a000")
    ].sort_values("anchor_id")
    overlap = null_rows.loc[null_rows["anchor_id"].isin(prior["anchor_id"])].sort_values("anchor_id")
    require(len(prior) == 4 and len(overlap) == 4, "M3/M5 null overlap coverage differs")
    require(list(prior["anchor_id"]) == list(overlap["anchor_id"]), "M3/M5 null overlap anchors differ")
    for metric in DELTA_METRICS:
        old = pd.to_numeric(prior[f"revision_{metric}"], errors="raise").to_numpy(float)
        new = pd.to_numeric(overlap[f"revision_{metric}"], errors="raise").to_numpy(float)
        require(np.array_equal(old, new), f"M3/M5 null endpoint reproducibility differs: {metric}")

    prior_action = actions.loc[
        actions["experiment"].eq("M3")
        & actions["variant_id"].eq("rev01_p90_signal_a000")
        & actions["artifact_role"].eq("revision"),
        ["anchor_id", "field", "value", "count_all_rows", "count_occupied_rows"],
    ].sort_values(["anchor_id", "field", "value"]).reset_index(drop=True)
    overlap_action = actions.loc[
        actions["experiment"].eq("M5")
        & actions["variant_id"].eq("rev01_p90_signal_a000_full24")
        & actions["artifact_role"].eq("revision")
        & actions["anchor_id"].isin(prior["anchor_id"]),
        ["anchor_id", "field", "value", "count_all_rows", "count_occupied_rows"],
    ].sort_values(["anchor_id", "field", "value"]).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(prior_action, overlap_action, check_dtype=True, check_exact=True)
    except AssertionError as exc:
        raise SynthesisContractError("M3/M5 null action-count reproducibility differs") from exc

    return {
        "m5_full_grid_cells": 24,
        "m5_occupied_decisions": expected_occupied,
        "m5_occupied_relax_decisions": occupied_by_branch.get("relax", 0),
        "m5_occupied_warm_protection_decisions": occupied_by_branch.get("warm_protection", 0),
        "m5_occupied_cold_protection_decisions": occupied_by_branch.get("cold_protection", 0),
        "m5_occupied_risk_hold_decisions": occupied_by_branch.get("risk_hold", 0),
        "m3_m5_overlap_cells_exact": 4,
        "m3_m5_endpoints_exact": True,
        "m3_m5_action_counts_exact": True,
    }


def aggregate_matched(matched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["experiment", "variant_id", "revision_strategy", "accepted_comparator_strategy", "sampling_frame"]
    for keys, group in matched.groupby(group_columns, sort=True, dropna=False):
        partitions = [("all_stress_test_cells", group)]
        partitions.extend((name, part) for name, part in group.groupby("stress_grid_class", sort=True))
        for aggregation, part in partitions:
            row: dict[str, Any] = {column: value for column, value in zip(group_columns, keys)}
            row.update(
                {
                    "aggregation": aggregation,
                    "n_matched_cells": int(len(part)),
                    "interpretation_scope": "descriptive matched stress-test cells; present-typical and late-century-hot-extreme strata use different year-selection rules",
                }
            )
            for metric in DELTA_METRICS:
                accepted_column = f"accepted_{metric}"
                revision_column = f"revision_{metric}"
                delta_column = f"delta_{metric}"
                if accepted_column not in part or delta_column not in part:
                    continue
                accepted_values = pd.to_numeric(part[accepted_column], errors="coerce")
                revision_values = pd.to_numeric(part[revision_column], errors="coerce")
                delta_values = pd.to_numeric(part[delta_column], errors="coerce")
                require(len(accepted_values) == len(part), f"Aggregate accepted metric coverage differs: {metric}")
                require(np.isfinite(accepted_values.to_numpy(float)).all(), f"Aggregate accepted metric is incomplete/nonfinite: {metric}")
                require(np.isfinite(revision_values.to_numpy(float)).all(), f"Aggregate revision metric is incomplete/nonfinite: {metric}")
                require(np.isfinite(delta_values.to_numpy(float)).all(), f"Aggregate delta metric is incomplete/nonfinite: {metric}")
                row[f"accepted_mean_{metric}"] = float(accepted_values.mean())
                row[f"revision_mean_{metric}"] = float(revision_values.mean())
                row[f"mean_delta_{metric}"] = float(delta_values.mean())
                row[f"median_delta_{metric}"] = float(delta_values.median())
                row[f"min_delta_{metric}"] = float(delta_values.min())
                row[f"max_delta_{metric}"] = float(delta_values.max())
                if metric in LOWER_COUNT_METRICS:
                    finite = delta_values.dropna()
                    row[f"revision_lower_count_{metric}"] = int((finite < -1.0e-9).sum())
                    row[f"equal_within_1e_9_count_{metric}"] = int((finite.abs() <= 1.0e-9).sum())
                    row[f"revision_higher_count_{metric}"] = int((finite > 1.0e-9).sum())
            energy_pct = pd.to_numeric(part["delta_total_delivered_site_energy_pct_of_accepted"], errors="coerce")
            require(np.isfinite(energy_pct.to_numpy(float)).all(), "Aggregate delivered-energy percentage is incomplete/nonfinite")
            row["mean_delta_total_delivered_site_energy_pct_of_accepted"] = float(energy_pct.mean())
            row["min_delta_total_delivered_site_energy_pct_of_accepted"] = float(energy_pct.min())
            row["max_delta_total_delivered_site_energy_pct_of_accepted"] = float(energy_pct.max())
            rows.append(row)
    result = pd.DataFrame(rows).sort_values(["experiment", "variant_id", "aggregation"]).reset_index(drop=True)
    require(not result.isna().any().any(), "Aggregate summary contains missing values")
    return result


def experiment_variant_sentinel_view(matched: pd.DataFrame) -> pd.DataFrame:
    metadata = [
        "experiment",
        "variant_id",
        "cell_id",
        "sentinel_id",
        "sampling_frame",
        "anchor_id",
        "stress_grid_class",
        "city",
        "scenario",
        "weather_year",
        "revision_strategy",
        "accepted_comparator_strategy",
        "accepted_comparator_cell_id",
        "paperb_spatial_quantile",
        "paperb_tail_threshold",
        "paperb_asym_threshold",
        "paperb_save_heat_c",
        "paperb_save_cool_c",
        "paperb_signal_alpha",
        "paperb_adaptive_rm_clamp_low_c",
        "paperb_adaptive_rm_clamp_high_c",
        "interpretation_scope",
    ]
    metrics = []
    for metric in (
        "total_delivered_site_electricity_plus_facility_gas_kwh",
        "occupied_worst_zone_degree_hours_above_28c",
        "occupied_worst_zone_degree_hours_below_18c",
        "occupied_mean_heating_setpoint_c",
        "occupied_mean_cooling_setpoint_c",
        "adaptive_running_mean_outside_10_to_33p5c_pct_occ",
        "adaptive_running_mean_clamped_pct_occ",
    ):
        metrics.extend((f"accepted_{metric}", f"revision_{metric}", f"delta_{metric}"))
    metrics.append("delta_total_delivered_site_energy_pct_of_accepted")
    return matched[metadata + metrics].copy()


def expected_sentinel_columns() -> list[str]:
    metadata = [
        "experiment", "variant_id", "cell_id", "sentinel_id", "sampling_frame",
        "anchor_id", "stress_grid_class", "city", "scenario", "weather_year",
        "revision_strategy", "accepted_comparator_strategy", "accepted_comparator_cell_id",
        "paperb_spatial_quantile", "paperb_tail_threshold", "paperb_asym_threshold",
        "paperb_save_heat_c", "paperb_save_cool_c", "paperb_signal_alpha",
        "paperb_adaptive_rm_clamp_low_c", "paperb_adaptive_rm_clamp_high_c",
        "interpretation_scope",
    ]
    metrics: list[str] = []
    for metric in (
        "total_delivered_site_electricity_plus_facility_gas_kwh",
        "occupied_worst_zone_degree_hours_above_28c",
        "occupied_worst_zone_degree_hours_below_18c",
        "occupied_mean_heating_setpoint_c",
        "occupied_mean_cooling_setpoint_c",
        "adaptive_running_mean_outside_10_to_33p5c_pct_occ",
        "adaptive_running_mean_clamped_pct_occ",
    ):
        metrics.extend((f"accepted_{metric}", f"revision_{metric}", f"delta_{metric}"))
    metrics.append("delta_total_delivered_site_energy_pct_of_accepted")
    return metadata + metrics


def metric_dictionary() -> pd.DataFrame:
    rows = [
        {
            "metric_family": "energy",
            "metric_or_pattern": "facility_electricity_kwh_delivered",
            "definition": "Sum of retained Electricity:Facility joules divided explicitly by 3,600,000 J/kWh; reconciled to the hourly SQL meter.",
            "interpretation_limit": "Delivered site electricity only; not primary energy, carbon emissions, cost, or source energy.",
        },
        {
            "metric_family": "energy",
            "metric_or_pattern": "facility_natural_gas_including_service_water_kwh_delivered",
            "definition": "Sum of retained NaturalGas:Facility joules divided explicitly by 3,600,000 J/kWh; reconciled to the hourly SQL meter.",
            "interpretation_limit": "Facility gas includes supply heating and service-water heating; not HVAC-only gas.",
        },
        {
            "metric_family": "energy",
            "metric_or_pattern": "total_delivered_site_electricity_plus_facility_gas_kwh",
            "definition": "Arithmetic sum of delivered facility electricity and facility natural-gas kWh after explicit joule conversion.",
            "interpretation_limit": "Delivered site energy; not primary energy, carbon emissions, cost, or source energy.",
        },
        {
            "metric_family": "temperature exposure",
            "metric_or_pattern": "occupied_worst_zone_degree_hours_above_{26c|28c|30c}",
            "definition": "Maximum across 15 zones of sum[0.25 h * max((zone air + zone MRT)/2 - threshold, 0)] during occupied timesteps.",
            "interpretation_limit": "Study-defined warm-side operative-temperature diagnostic; not observed comfort, satisfaction, health, or standards compliance.",
        },
        {
            "metric_family": "temperature exposure",
            "metric_or_pattern": "occupied_worst_zone_degree_hours_below_{18c|16c}",
            "definition": "Maximum across 15 zones of sum[0.25 h * max(threshold - (zone air + zone MRT)/2, 0)] during occupied timesteps.",
            "interpretation_limit": "Predeclared descriptive cold-side screens; not observed comfort, health, or standards compliance.",
        },
        {
            "metric_family": "setpoint",
            "metric_or_pattern": "occupied_mean_{heating|cooling}_setpoint_c",
            "definition": "Arithmetic mean of the building-level commanded setpoint during occupied timesteps only.",
            "interpretation_limit": "Does not include unoccupied setback/setup periods.",
        },
        {
            "metric_family": "adaptive context",
            "metric_or_pattern": "adaptive_running_mean_outside_10_to_33p5c_{rows|pct}_occ",
            "definition": "Occupied timesteps with retained running-mean outdoor temperature below 10 C or above 33.5 C.",
            "interpretation_limit": "Applicability-context diagnostic for the engineering adaptive-band benchmark; not a comfort-compliance result.",
        },
        {
            "metric_family": "adaptive context",
            "metric_or_pattern": "adaptive_running_mean_clamped_{rows|pct}_occ",
            "definition": "Occupied timesteps on which the Rev01 clamped benchmark actually bounded running-mean outdoor temperature to 10--33.5 C; accepted pre-clamp traces have executed-clamp count zero and clamp_flag_present=false.",
            "interpretation_limit": "Controller-mechanism count, not evidence that the adaptive model is valid outside its usual applicability context.",
        },
    ]
    return pd.DataFrame(rows)


def reviewer_evidence_map() -> pd.DataFrame:
    """Map every received comment cluster to evidence or an explicit scope boundary."""
    rows = [
        ("R1-1", "predictor transfer and frozen production model", "partly_empirical_plus_reframing", "M5 full-grid learned-signal null; existing contributor-disjoint validation", "Do not imply occupant-general transfer; describe M5 as controller dependence, not a substitute for contributor-disjoint model validation."),
        ("R1-2", "PMV/adaptive information dependence", "empirical_plus_clarification", "M1 PMV aggregation, M4 learned aggregation, and M5 signal-null matched results", "Call the method AI-assisted supervision and acknowledge PMV/adaptive-derived inputs; do not claim information-independent comparison."),
        ("R1-3", "single selected zone and synchronous building actuation", "empirical_plus_limitation", "M4 full-grid learned mean-tail versus learned-p90 ablation", "M4 tests aggregation, not zone-level actuation; retain zonal actuation as untested architecture and limitation."),
        ("R1-4", "period and weather-selection confounding", "reframing", "stress_grid_class and interpretation_scope in every matched row", "Report a stress-test grid only; make no causal future-climate contrast."),
        ("R1-5", "single building/HVAC/city sizing/GCM generalizability", "scoped_limitation", "D01 sizing audit; no additional building or GCM rerun", "Limit all quantitative findings to this prototype/configuration and identify multi-building/multi-GCM validation as future work."),
        ("R1-6", "temperature degree-hours versus comfort/health", "metric_clarification", "metric_dictionary; warm and cold absolute matched endpoints", "Describe study-defined operative-temperature exposure diagnostics; do not infer satisfaction, health, safety, or standards compliance."),
        ("R1-7", "two suggested 2026 papers", "literature_action_outside_pipeline", "citation review required separately", "Read, position, and cite if substantively relevant; no simulation result resolves this comment."),
        ("R2-1", "controller threshold justification/sensitivity", "empirical", "M2 eight-variant four-anchor sentinel sensitivity", "Report local robustness without calling the sentinel screen optimization or global sensitivity."),
        ("R2-2", "AI role is supervisory/rule based", "clarification", "M5 full-grid learned-signal null matched results", "Use AI-assisted supervisory controller consistently; final actions remain prespecified rule logic."),
        ("R2-3", "occupant-level model generalization", "discussion_plus_sensitivity", "M5 full-grid learned-signal null; existing contributor-disjoint metrics", "State deployment risk and discuss personalization/online adaptation as untested remedies."),
        ("R2-4", "single future climate pathway/model uncertainty", "reframing", "stress-test labels and construct limits", "Do not generalize over climate-model uncertainty or attribute stratum differences solely to climate change."),
        ("R2-5", "predicted sensation versus overheating metrics", "construct_separation", "metric_dictionary and RESULTS_DIGEST", "Explain that sensation probabilities drive actions while independently reconstructed operative temperature audits physical consequences."),
        ("R2-6", "case-study generalizability", "scoped_limitation", "full 24-anchor weather grid within one prototype", "Restrict transfer claims across building types, HVAC systems, and zoning configurations."),
        ("R2-M1", "site energy meaning", "metric_clarification", "delivered energy columns and explicit J-to-kWh dictionary", "State delivered site electricity/gas; not primary/source energy, carbon, cost, or emissions."),
        ("R2-M2", "actuator-constraint justification", "empirical_plus_methods", "M2 bounds sensitivities; frozen actuator contract", "Justify prespecification and report limited bounds sensitivity without claiming optimality."),
        ("R2-M3", "conventional signals are model inputs", "clarification", "M1 matched PMV aggregation plus M4/M5 learned-policy decomposition", "Acknowledge dependence on conventional comfort-derived covariates."),
        ("R2-M4", "cold-side impacts", "empirical", "matched occupied worst-zone degree-hours below 18 C and 16 C", "Present as descriptive cold-side screens, not overall comfort compliance."),
        ("R3-1", "learned model/spatial aggregation/logic are confounded", "empirical", "M1 PMV spatial ablation, M4 full-grid learned spatial ablation, and M5 full-grid learned-signal null", "Attribute only what each matched ablation isolates; avoid assigning residual differences generically to AI."),
        ("R3-2", "weak contributor transfer and production predictor dependence", "empirical_plus_limitation", "M5 full-grid learned-signal null; existing contributor-disjoint validation", "Test controller dependence on the learned signal while retaining the separate occupant-transfer limitation."),
        ("R3-3", "Denver design-day sizing and saturation", "diagnostic_scope", "09_rev01/04_analysis/trigger_diagnostics/d01_sizing_trigger_20260824", "Report retained capacities/unmet diagnostics; C2 city-DDY reruns were not authorized because authoritative DDYs/component saturation evidence were absent."),
        ("R3-4", "weather selection confound and provenance", "reframing_plus_provenance", "matched stress-grid fields; hashed frozen weather lineage in batch freezes", "Use stress-test wording and provide traceable weather identifiers/hashes; do not infer a clean future effect."),
        ("R3-5", "adaptive comparator out-of-domain use", "empirical", "C1 24-cell running-mean clamp sensitivity and applicability counts", "Call it the adaptive-band engineering benchmark and report outside-range/clamp behavior."),
        ("R3-6", "controller parameter robustness", "empirical", "M2 q/tail/directional-threshold/bounds sensitivities", "Treat the four-anchor panel as local sensitivity, not optimization or universal robustness."),
        ("R3-7", "one-sided thermal evaluation", "empirical", "matched cold-side 18 C and 16 C degree-hour screens", "Report warm and cold screens separately and keep overall-comfort claims out of scope."),
        ("R3-M1", "absolute baseline degree-hours", "empirical", "accepted_* absolute columns in matched_cell_metrics", "Report absolute accepted comparator and revision values alongside deltas."),
        ("R3-M2", "annual setpoint mean includes unoccupied periods", "empirical", "occupied_mean_heating_setpoint_c and occupied_mean_cooling_setpoint_c", "State submitted annual means included unoccupied schedules; add occupied-period means."),
        ("R3-M3", "contributor-disjoint split variability", "existing_analysis_action", "contributor-disjoint split-level results required separately", "Add split dispersion/individual results; the building reruns do not answer this point."),
        ("R3-M4", "contributor identity canonicalization", "methods_action", "training-data identity audit required separately", "Document merge/ambiguity rules and checks; no EnergyPlus rerun is relevant."),
        ("R3-M5", "336-hour extreme-period duration sensitivity", "scoped_analysis", "annual outputs here; duration sensitivity requires separate retained-trace window analysis", "Do not claim event-duration robustness unless 168/504-hour analyses are added."),
        ("R3-M6", "specific metabolic-rate value", "methods_action", "frozen met/activity fields in batch matrices", "Explain derivation and avoid false physical precision; no controller rerun isolates this assumption."),
        ("R3-M7", "facility gas includes service water", "metric_clarification", "facility_natural_gas_including_service_water_* columns", "Use the explicit facility-gas label and note dilution by non-HVAC service-water use."),
        ("R3-M8", "figure scale comparability", "presentation_action", "normalized delta columns and absolute baseline columns", "Use common-scale or normalized supplemental presentation where it improves comparison."),
        ("R3-M9", "adaptive control terminology", "terminology_action", "C1 experiment and evidence text use adaptive-band engineering benchmark", "Replace ambiguous 'adaptive control' wording consistently."),
        ("R3-DATA", "data/code availability", "repository_action", "CSV/Parquet synthesis tables, input hashes, code, and closures", "Deposit processed evidence/controller/configuration artifacts subject to third-party weather redistribution limits."),
    ]
    return pd.DataFrame(
        rows,
        columns=(
            "reviewer_comment_id",
            "issue_cluster",
            "resolution_mode",
            "evidence_or_artifact",
            "response_boundary",
        ),
    )


def expected_matched_columns() -> list[str]:
    columns = list(MATCHED_BASE_COLUMNS) + list(CONFIG_COLUMNS)
    for role in ("accepted", "revision"):
        columns.extend(f"{role}_{metric}" for metric in ROLE_METRIC_KEYS)
    columns.extend(f"delta_{metric}" for metric in DELTA_METRICS)
    columns.append("delta_total_delivered_site_energy_pct_of_accepted")
    return columns


def expected_summary_columns() -> list[str]:
    columns = [
        "experiment",
        "variant_id",
        "revision_strategy",
        "accepted_comparator_strategy",
        "sampling_frame",
        "aggregation",
        "n_matched_cells",
        "interpretation_scope",
    ]
    for metric in DELTA_METRICS:
        columns.extend(
            [
                f"accepted_mean_{metric}",
                f"revision_mean_{metric}",
                f"mean_delta_{metric}",
                f"median_delta_{metric}",
                f"min_delta_{metric}",
                f"max_delta_{metric}",
            ]
        )
        if metric in LOWER_COUNT_METRICS:
            columns.extend(
                [
                    f"revision_lower_count_{metric}",
                    f"equal_within_1e_9_count_{metric}",
                    f"revision_higher_count_{metric}",
                ]
            )
    columns.extend(
        [
            "mean_delta_total_delivered_site_energy_pct_of_accepted",
            "min_delta_total_delivered_site_energy_pct_of_accepted",
            "max_delta_total_delivered_site_energy_pct_of_accepted",
        ]
    )
    return columns


def require_no_missing_or_nonfinite(frame: pd.DataFrame, label: str) -> None:
    require(not frame.isna().any().any(), f"{label} contains missing values")
    for column in frame.select_dtypes(include=[np.number]).columns:
        values = frame[column].to_numpy(float)
        require(np.isfinite(values).all(), f"{label} numeric column is nonfinite: {column}")


def validate_scientific_output_frames(
    matched: pd.DataFrame,
    summary: pd.DataFrame,
    sentinel: pd.DataFrame,
    actions: pd.DataFrame,
    inputs: pd.DataFrame,
    dictionary: pd.DataFrame,
    reviewer_map: pd.DataFrame,
) -> None:
    require(list(matched.columns) == expected_matched_columns(), "matched_cell_metrics exact schema differs")
    require(len(matched) == 136 and matched["cell_id"].nunique() == 136, "matched_cell_metrics exact row/key count differs")
    require_no_missing_or_nonfinite(matched, "matched_cell_metrics")
    observed_counts = matched.groupby(["experiment", "variant_id"]).size().to_dict()
    require(observed_counts == EXPECTED_SCIENTIFIC_VARIANT_COUNTS, "Scientific experiment/variant cell counts differ")
    for (experiment, variant), expected_count in EXPECTED_SCIENTIFIC_VARIANT_COUNTS.items():
        group = matched[(matched["experiment"] == experiment) & (matched["variant_id"] == variant)]
        stress_counts = group["stress_grid_class"].value_counts().to_dict()
        require(stress_counts == {"present_typical": expected_count // 2, "late_century_hot_extreme": expected_count // 2}, f"Stress-grid stratum count differs: {experiment}/{variant}")
        if experiment in {"M2", "M3"}:
            require(set(group["sentinel_id"]) == EXPECTED_SENTINEL_IDS, f"Sentinel identity coverage differs: {experiment}/{variant}")
        else:
            require(set(group["sentinel_id"]) == {""}, f"Non-sentinel experiment has sentinel IDs: {experiment}/{variant}")
    require((matched["accepted_trace_rows"] == EXPECTED_ROWS).all() and (matched["revision_trace_rows"] == EXPECTED_ROWS).all(), "Matched trace-row values differ")
    require((matched["accepted_occupied_rows"] == EXPECTED_OCCUPIED_ROWS).all() and (matched["revision_occupied_rows"] == EXPECTED_OCCUPIED_ROWS).all(), "Matched occupied-row values differ")
    require((matched["accepted_zone_count"] == EXPECTED_ZONE_COUNT).all() and (matched["revision_zone_count"] == EXPECTED_ZONE_COUNT).all(), "Matched zone counts differ")
    require(np.allclose(matched["accepted_timestep_hours"], STEP_HOURS, rtol=0.0, atol=1.0e-12), "Accepted timestep values differ")
    require(np.allclose(matched["revision_timestep_hours"], STEP_HOURS, rtol=0.0, atol=1.0e-12), "Revision timestep values differ")
    require((matched["accepted_adaptive_running_mean_clamp_flag_present"] == False).all(), "Accepted pre-clamp presence flags differ")  # noqa: E712
    require((matched["revision_adaptive_running_mean_clamp_flag_present"] == True).all(), "Revision clamp presence flags differ")  # noqa: E712
    for metric in DELTA_METRICS:
        expected_delta = pd.to_numeric(matched[f"revision_{metric}"], errors="raise") - pd.to_numeric(matched[f"accepted_{metric}"], errors="raise")
        require(np.allclose(matched[f"delta_{metric}"], expected_delta, rtol=0.0, atol=1.0e-12), f"Matched delta identity differs: {metric}")

    require(list(summary.columns) == expected_summary_columns(), "matched_experiment_variant_summary exact schema differs")
    require(len(summary) == 42, "matched_experiment_variant_summary row count differs")
    require_no_missing_or_nonfinite(summary, "matched_experiment_variant_summary")
    summary_keys = summary.groupby(["experiment", "variant_id"])["aggregation"].apply(set)
    require(all(value == {"all_stress_test_cells", "present_typical", "late_century_hot_extreme"} for value in summary_keys), "Aggregate stratum coverage differs")
    for metric in LOWER_COUNT_METRICS:
        total = (
            summary[f"revision_lower_count_{metric}"]
            + summary[f"equal_within_1e_9_count_{metric}"]
            + summary[f"revision_higher_count_{metric}"]
        )
        require((total == summary["n_matched_cells"]).all(), f"Aggregate direction-count mass differs: {metric}")

    require(list(sentinel.columns) == expected_sentinel_columns(), "matched_experiment_variant_sentinel exact schema differs")
    require(len(sentinel) == 136 and sentinel["cell_id"].nunique() == 136, "matched_experiment_variant_sentinel row/key count differs")
    require_no_missing_or_nonfinite(sentinel, "matched_experiment_variant_sentinel")
    require(list(actions.columns) == [
        "match_cell_id", "experiment", "variant_id", "anchor_id", "stress_grid_class",
        "artifact_role", "artifact_cell_id", "field", "value", "count_all_rows", "count_occupied_rows",
    ], "action_branch_counts exact schema differs")
    require_no_missing_or_nonfinite(actions, "action_branch_counts")
    require((actions[["count_all_rows", "count_occupied_rows"]].to_numpy(int) >= 0).all(), "Action counts contain negative values")

    require(list(inputs.columns) == [
        "match_cell_id", "artifact_role", "artifact_cell_id", "trace_path", "trace_sha256",
        "summary_path", "summary_sha256", "sql_path", "sql_sha256",
    ], "input_artifact_manifest exact schema differs")
    require(len(inputs) == 272 and not inputs.duplicated(["match_cell_id", "artifact_role"]).any(), "input_artifact_manifest row/key count differs")
    require_no_missing_or_nonfinite(inputs, "input_artifact_manifest")
    for hash_column in ("trace_sha256", "summary_sha256", "sql_sha256"):
        require(inputs[hash_column].astype(str).str.fullmatch(r"[0-9a-f]{64}").all(), f"Input manifest hash format differs: {hash_column}")

    require(list(dictionary.columns) == ["metric_family", "metric_or_pattern", "definition", "interpretation_limit"], "metric_dictionary exact schema differs")
    require(len(dictionary) == 8 and dictionary["metric_or_pattern"].nunique() == 8, "metric_dictionary row/key count differs")
    require_no_missing_or_nonfinite(dictionary, "metric_dictionary")
    require(list(reviewer_map.columns) == ["reviewer_comment_id", "issue_cluster", "resolution_mode", "evidence_or_artifact", "response_boundary"], "reviewer_evidence_map exact schema differs")
    require(len(reviewer_map) == len(EXPECTED_REVIEWER_IDS), "reviewer_evidence_map row count differs")
    require(set(reviewer_map["reviewer_comment_id"]) == EXPECTED_REVIEWER_IDS, "reviewer_evidence_map exact ID coverage differs")
    require(reviewer_map["reviewer_comment_id"].is_unique, "reviewer_evidence_map IDs are not unique")
    require_no_missing_or_nonfinite(reviewer_map, "reviewer_evidence_map")


def evidence_payload(
    matched: pd.DataFrame,
    summary: pd.DataFrame,
    batch_audits: list[dict[str, Any]],
    accepted: AcceptedContext,
    mechanism_checks: dict[str, Any],
) -> dict[str, Any]:
    overall = summary[summary["aggregation"] == "all_stress_test_cells"]
    records = []
    for row in overall.to_dict(orient="records"):
        records.append(
            {
                "experiment": row["experiment"],
                "variant_id": row["variant_id"],
                "revision_strategy": row["revision_strategy"],
                "accepted_comparator_strategy": row["accepted_comparator_strategy"],
                "sampling_frame": row["sampling_frame"],
                "n_matched_cells": row["n_matched_cells"],
                "mean_delta_total_delivered_site_energy_pct_of_accepted": row["mean_delta_total_delivered_site_energy_pct_of_accepted"],
                "mean_delta_worst_zone_degree_hours_above_28c": row["mean_delta_occupied_worst_zone_degree_hours_above_28c"],
                "revision_lower_count_worst_zone_degree_hours_above_28c": row["revision_lower_count_occupied_worst_zone_degree_hours_above_28c"],
                "mean_delta_worst_zone_degree_hours_below_18c": row["mean_delta_occupied_worst_zone_degree_hours_below_18c"],
                "revision_lower_count_worst_zone_degree_hours_below_18c": row["revision_lower_count_occupied_worst_zone_degree_hours_below_18c"],
            }
        )
    return {
        "schema_version": "paperb_enb_rev01_response_evidence_v2",
        "created_utc": utc_now(),
        "coverage": {
            "matched_revision_cells": int(len(matched)),
            "core_cells": int((matched["batch_name"] == "core").sum()),
            "adaptive_cells": int((matched["batch_name"] == "adaptive").sum()),
            "learned_contribution_cells": int((matched["batch_name"] == "learned").sum()),
            "batch_audits": batch_audits,
            "accepted_fresh96_closure_path": str(accepted.closure_path),
            "accepted_fresh96_closure_sha256": accepted.closure_sha256,
            "accepted_matrix_path": str(accepted.matrix_path),
            "accepted_matrix_sha256": EXPECTED_ACCEPTED_MATRIX_SHA256,
        },
        "experiment_map": {
            "M1": "Spatially matched absolute-PMV p90 guard versus accepted building-mean PMV guard across 24 stress-test cells; isolates spatial aggregation while retaining the PMV signal.",
            "M2": "Local controller-parameter sensitivities versus the accepted p90 learned-signal controller on a predeclared four-anchor sentinel panel.",
            "M3": "Learned-signal attenuation/null sensitivities versus the accepted p90 learned-signal controller on the same four-anchor sentinel panel.",
            "M4": "Full-grid learned spatial-aggregation ablation: building-mean learned probability tails versus the accepted learned-probability p90 selector across 24 stress-test cells.",
            "M5": "Full-grid learned-signal contribution assessment: neutral probability null versus the accepted frozen learned-probability p90 policy across 24 stress-test cells.",
            "C1": "Running-mean clamp sensitivity versus the accepted adaptive-band engineering benchmark across 24 stress-test cells.",
        },
        "mechanism_checks": mechanism_checks,
        "construct_limits": [
            "The weather cases form a stress-test grid. Present-typical and late-century-hot-extreme cases use different year-selection rules, so stratum differences are not causal future-climate effects.",
            "Operative-temperature degree-hours are descriptive physical exposure diagnostics, not observed comfort, satisfaction, health outcomes, or validated safety thresholds.",
            "The energy quantities are delivered site electricity and facility natural gas after explicit J-to-kWh conversion; they are not primary energy, carbon emissions, cost, or source energy.",
            "NaturalGas:Facility includes supply heating and service-water heating and is not an HVAC-only gas meter.",
            "Lower warm- or cold-side diagnostic degree-hours mean less exposure on that specific screen, not globally better comfort.",
        ],
        "overall_matched_results": records,
        "reviewer_evidence_map": reviewer_evidence_map().to_dict(orient="records"),
    }


def format_number(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):,.{digits}f}"


def results_digest(
    matched: pd.DataFrame,
    summary: pd.DataFrame,
    batch_audits: list[dict[str, Any]],
    mechanism_checks: dict[str, Any],
) -> str:
    overall = summary[summary["aggregation"] == "all_stress_test_cells"]
    lines = [
        "# Rev01 reviewer-evidence digest",
        "",
        f"Validated coverage: **{len(matched)} matched revision cells** (64 core + 24 adaptive + 48 full-grid learned-contribution), each joined to its frozen `accepted_comparator_cell_id`.",
        "",
        "The two weather strata are reported strictly as a **stress-test grid**: present-typical years and late-century hot-extreme years were chosen by different rules. Their contrast is not interpreted as a causal future-climate effect. Operative-temperature degree-hours are descriptive diagnostics, not observed comfort, satisfaction, health outcomes, or validated safety/comfort thresholds.",
        "",
        "## Overall matched results",
        "",
        "| experiment | variant | comparator | n | delivered site energy delta (%) | worst-zone DH >28 C delta (C h) | lower / higher DH >28 C | worst-zone DH <18 C delta (C h) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in overall.to_dict(orient="records"):
        lower = int(row["revision_lower_count_occupied_worst_zone_degree_hours_above_28c"])
        higher = int(row["revision_higher_count_occupied_worst_zone_degree_hours_above_28c"])
        lines.append(
            "| {experiment} | `{variant}` | `{comparator}` | {n} | {energy} | {dh28} | {lower} / {higher} | {cold18} |".format(
                experiment=row["experiment"],
                variant=row["variant_id"],
                comparator=row["accepted_comparator_strategy"],
                n=int(row["n_matched_cells"]),
                energy=format_number(row["mean_delta_total_delivered_site_energy_pct_of_accepted"]),
                dh28=format_number(row["mean_delta_occupied_worst_zone_degree_hours_above_28c"]),
                lower=lower,
                higher=higher,
                cold18=format_number(row["mean_delta_occupied_worst_zone_degree_hours_below_18c"]),
            )
        )
    lines.extend(
        [
            "",
            "Negative energy deltas indicate less delivered site energy than the matched accepted comparator. Negative degree-hour deltas indicate less exposure on that one diagnostic screen; they do not establish overall comfort improvement.",
            "",
            "## Measurement and interpretation boundaries",
            "",
            "- Facility electricity and `NaturalGas:Facility` are reconciled between retained trace joules and EnergyPlus hourly SQL meters, then divided by exactly 3,600,000 J/kWh. The total is delivered site electricity plus facility gas—not primary/source energy, carbon, cost, or emissions.",
            "- Facility gas includes both supply heating and service-water heating; the gas comparison is not HVAC-only.",
            "- Warm screens are occupied worst-zone operative-temperature degree-hours above 26, 28, and 30 C. Cold screens are the corresponding descriptive degree-hours below 18 and 16 C.",
            "- Setpoint means use occupied periods only, avoiding the submitted annual means' mixing with unoccupied setbacks/setup.",
            "- M1 tests spatial aggregation of the same PMV signal. M2 tests local parameter sensitivity. M3 is the earlier four-sentinel learned-signal screen. M4 tests learned mean-tail versus learned p90 on all 24 anchors. M5 tests the learned-signal null versus learned p90 on all 24 anchors. C1 tests adaptive running-mean clamping.",
            f"- M5 mechanism closure: {mechanism_checks['m5_occupied_relax_decisions']:,}/{mechanism_checks['m5_occupied_decisions']:,} occupied decisions requested relaxation under the neutral null, with zero warm-protection, cold-protection, or risk-hold decisions; the four overlapping M3 null cells reproduced exactly.",
            "",
            "## Provenance",
            "",
        ]
    )
    for audit in batch_audits:
        lines.append(
            f"- `{audit['batch_name']}`: {audit['validated_cells']}/{audit['planned_cells']} cells; matrix `{audit['matrix_sha256']}`; batch closure `{audit['batch_closure_sha256']}`."
        )
    return "\n".join(lines) + "\n"


def validate_response_products(evidence: dict[str, Any], digest: str) -> None:
    require(set(evidence) == {
        "schema_version", "created_utc", "coverage", "experiment_map",
        "mechanism_checks", "construct_limits", "overall_matched_results", "reviewer_evidence_map",
    }, "response_evidence top-level schema differs")
    require(evidence.get("schema_version") == "paperb_enb_rev01_response_evidence_v2", "response_evidence schema version differs")
    coverage = evidence.get("coverage", {})
    require(coverage.get("matched_revision_cells") == 136, "response_evidence matched coverage differs")
    require(
        coverage.get("core_cells") == 64
        and coverage.get("adaptive_cells") == 24
        and coverage.get("learned_contribution_cells") == 48,
        "response_evidence batch coverage differs",
    )
    require(set(evidence.get("experiment_map", {})) == {"M1", "M2", "M3", "M4", "M5", "C1"}, "response_evidence experiment map differs")
    mechanism = evidence.get("mechanism_checks", {})
    require(mechanism.get("m5_occupied_decisions") == 400_896, "response_evidence M5 mechanism mass differs")
    require(mechanism.get("m5_occupied_relax_decisions") == 400_896, "response_evidence M5 relax mass differs")
    require(len(evidence.get("overall_matched_results", [])) == len(EXPECTED_SCIENTIFIC_VARIANT_COUNTS), "response_evidence overall result count differs")
    reviewer_ids = {row.get("reviewer_comment_id") for row in evidence.get("reviewer_evidence_map", [])}
    require(reviewer_ids == EXPECTED_REVIEWER_IDS, "response_evidence reviewer IDs differ")
    try:
        json.dumps(evidence, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SynthesisContractError("response_evidence is not finite strict JSON") from exc
    require(digest.startswith("# Rev01 reviewer-evidence digest\n"), "RESULTS_DIGEST heading differs")
    for required_phrase in (
        "not interpreted as a causal future-climate effect",
        "not observed comfort, satisfaction, health outcomes",
        "not primary/source energy, carbon, cost, or emissions",
        "Facility gas includes both supply heating and service-water heating",
    ):
        require(required_phrase in digest, f"RESULTS_DIGEST required boundary is absent: {required_phrase}")


def verify_table_round_trip(
    original: pd.DataFrame,
    csv_path: Path,
    parquet_path: Path,
) -> None:
    csv_frame = pd.read_csv(
        csv_path,
        keep_default_na=False,
        float_precision="round_trip",
    )
    parquet_frame = pd.read_parquet(parquet_path)
    require(list(csv_frame.columns) == list(original.columns), f"CSV round-trip schema differs: {csv_path.name}")
    require(list(parquet_frame.columns) == list(original.columns), f"Parquet round-trip schema differs: {parquet_path.name}")
    require(len(csv_frame) == len(original) == len(parquet_frame), f"Table round-trip row count differs: {csv_path.stem}")
    try:
        pd.testing.assert_frame_equal(
            parquet_frame,
            original,
            check_dtype=True,
            check_exact=True,
            check_like=False,
        )
    except AssertionError as exc:
        raise SynthesisContractError(f"Parquet round-trip values differ: {parquet_path.name}") from exc
    for column in original.columns:
        expected = original[column]
        observed = csv_frame[column]
        if pd.api.types.is_bool_dtype(expected.dtype):
            parsed = strict_bool_array(observed, f"CSV Boolean column {column}")
            require(np.array_equal(parsed, expected.to_numpy(bool)), f"CSV Boolean round-trip differs: {column}")
        elif pd.api.types.is_numeric_dtype(expected.dtype):
            parsed = pd.to_numeric(observed, errors="coerce").to_numpy(float)
            values = expected.to_numpy(float)
            require(np.isfinite(parsed).all(), f"CSV numeric round-trip is nonfinite: {column}")
            require(np.array_equal(parsed, values), f"CSV numeric round-trip differs: {column}")
        else:
            require(np.array_equal(observed.astype(str).to_numpy(), expected.astype(str).to_numpy()), f"CSV string round-trip differs: {column}")


def write_table(frame: pd.DataFrame, stem: Path) -> list[dict[str, Any]]:
    csv_path = stem.with_suffix(".csv")
    parquet_path = stem.with_suffix(".parquet")
    frame.to_csv(csv_path, index=False, lineterminator="\n", float_format="%.17g")
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(
        table,
        parquet_path,
        compression="zstd",
        compression_level=7,
        use_dictionary=True,
        write_statistics=True,
        data_page_version="2.0",
    )
    verify_table_round_trip(frame, csv_path, parquet_path)
    return [
        {"path": csv_path.name, "sha256": sha256_file(csv_path), "rows": int(len(frame)), "columns": int(len(frame.columns)), "format": "csv", "exact_round_trip": True},
        {"path": parquet_path.name, "sha256": sha256_file(parquet_path), "rows": int(len(frame)), "columns": int(len(frame.columns)), "format": "parquet", "exact_round_trip": True},
    ]


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    return value


def synthesize(
    core_batch: Path,
    adaptive_batch: Path,
    learned_batch: Path,
    output_dir: Path,
) -> Path:
    output_dir = validate_output_location(
        output_dir,
        core_batch,
        adaptive_batch,
        learned_batch,
    )
    require(not output_dir.exists(), f"Output directory already exists; refusing overwrite: {output_dir}")
    accepted = validate_accepted_context()
    core_matches, core_audit = validate_batch(core_batch, CORE_SPEC, accepted)
    adaptive_matches, adaptive_audit = validate_batch(adaptive_batch, ADAPTIVE_SPEC, accepted)
    learned_matches, learned_audit = validate_batch(learned_batch, LEARNED_SPEC, accepted)
    matches = core_matches + adaptive_matches + learned_matches
    require(len(matches) == 136, "Combined scientific cell coverage is not exactly 136")
    require(len({match.plan["cell_id"] for match in matches}) == 136, "Combined revision cell IDs are not unique")
    matched, summary, actions, inputs = build_matched_tables(matches)
    require(len(matched) == 136, "Matched metric table does not contain 136 rows")
    require(set(matched["experiment"]) == {"M1", "M2", "M3", "M4", "M5", "C1"}, "Experiment coverage differs")
    mechanism_checks = validate_learned_contribution_mechanisms(matched, actions)
    dictionary = metric_dictionary()
    reviewer_map = reviewer_evidence_map()
    sentinel_view = experiment_variant_sentinel_view(matched)
    validate_scientific_output_frames(
        matched,
        summary,
        sentinel_view,
        actions,
        inputs,
        dictionary,
        reviewer_map,
    )
    audits = [core_audit, adaptive_audit, learned_audit]
    evidence = json_safe(
        evidence_payload(
            matched,
            summary,
            audits,
            accepted,
            mechanism_checks,
        )
    )
    digest = results_digest(matched, summary, audits, mechanism_checks)
    validate_response_products(evidence, digest)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp_{uuid.uuid4().hex}"
    require(not temporary.exists(), f"Temporary output collision: {temporary}")
    temporary.mkdir()
    try:
        output_files: list[dict[str, Any]] = []
        for frame, name in (
            (matched, "matched_cell_metrics"),
            (summary, "matched_experiment_variant_summary"),
            (sentinel_view, "matched_experiment_variant_sentinel"),
            (actions, "action_branch_counts"),
            (inputs, "input_artifact_manifest"),
            (dictionary, "metric_dictionary"),
            (reviewer_map, "reviewer_evidence_map"),
        ):
            output_files.extend(write_table(frame, temporary / name))
        evidence_path = temporary / "response_evidence.json"
        write_json(evidence_path, evidence)
        output_files.append({"path": evidence_path.name, "sha256": sha256_file(evidence_path), "format": "json"})
        digest_path = temporary / "RESULTS_DIGEST.md"
        digest_path.write_text(digest, encoding="utf-8")
        output_files.append({"path": digest_path.name, "sha256": sha256_file(digest_path), "format": "markdown"})
        expected_output_paths = {
            f"{stem}.{suffix}"
            for stem in (
                "matched_cell_metrics",
                "matched_experiment_variant_summary",
                "matched_experiment_variant_sentinel",
                "action_branch_counts",
                "input_artifact_manifest",
                "metric_dictionary",
                "reviewer_evidence_map",
            )
            for suffix in ("csv", "parquet")
        } | {"response_evidence.json", "RESULTS_DIGEST.md"}
        require({record["path"] for record in output_files} == expected_output_paths, "Synthesis output-file coverage differs")
        require(len(output_files) == 16, "Synthesis output-file count differs")
        closure = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": utc_now(),
            "status": "PASS",
            "all_gates_passed": True,
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "matched_cells": 136,
            "core_cells": 64,
            "adaptive_cells": 24,
            "learned_contribution_cells": 48,
            "learned_contribution_mechanism_checks": mechanism_checks,
            "batch_audits": audits,
            "accepted_fresh96_closure_path": str(accepted.closure_path),
            "accepted_fresh96_closure_sha256": accepted.closure_sha256,
            "accepted_matrix_path": str(accepted.matrix_path),
            "accepted_matrix_sha256": EXPECTED_ACCEPTED_MATRIX_SHA256,
            "metric_contract": {
                "joules_per_kwh": JOULES_PER_KWH,
                "warm_thresholds_c": list(WARM_THRESHOLDS_C),
                "cold_thresholds_c": list(COLD_THRESHOLDS_C),
                "operative_temperature": "equal-weight arithmetic mean of zone air and mean-radiant temperature",
                "energy_scope": "delivered site electricity plus NaturalGas:Facility; facility gas includes service-water heating",
                "excluded_interpretations": ["primary energy", "source energy", "carbon emissions", "cost", "observed comfort", "health outcome", "standards compliance", "causal future-climate effect"],
            },
            "closure_gates": {
                "exact_core64_batch_closure_and_matrix": True,
                "exact_adaptive24_batch_closure_and_matrix": True,
                "exact_learned48_batch_closure_and_matrix": True,
                "exact_136_cell_coverage": True,
                "m5_full_grid_null_mechanism_closed": True,
                "m3_m5_overlap_reproduced_exactly": True,
                "accepted_comparators_pinned_and_unchanged": True,
                "trace_summary_sql_hashes_validated": True,
                "trace_sql_facility_joules_reconciled": True,
                "explicit_joule_to_kwh_conversion": True,
                "stress_test_not_causal_future_contrast": True,
                "temperature_diagnostics_not_comfort_or_health": True,
            },
            "outputs": output_files,
        }
        write_json(temporary / "SYNTHESIS_CLOSURE.json", closure)
        temporary.rename(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-batch", required=True, type=Path)
    parser.add_argument("--adaptive-batch", required=True, type=Path)
    parser.add_argument("--learned-batch", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output = synthesize(
            args.core_batch,
            args.adaptive_batch,
            args.learned_batch,
            args.output_dir,
        )
    except SynthesisContractError as exc:
        print(f"SYNTHESIS CONTRACT FAILURE: {exc}", file=sys.stderr)
        return 2
    print(f"PASS: reviewer-facing synthesis written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
