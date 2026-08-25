#!/usr/bin/env python3
"""Fail-closed launcher for one frozen Paper B ENB Rev01 batch.

Each matrix row runs in one fresh subprocess and one fresh cell directory.
The launcher centralizes frozen inputs, validates outputs with the accepted
Fresh96 validator, applies the allowlisted two-phase storage finalizer, then
validates every cell again before writing the sole batch closure.  There is no
resume, overwrite, reuse, splice, purge, or conditional-cell path.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from types import ModuleType
from typing import Any
import uuid

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


sys.dont_write_bytecode = True

BATCH_SCHEMA = "paperb_enb_rev01_batch_closure_v1"
FREEZE_SCHEMA = "paperb_enb_rev01_batch_freeze_v1"
CELL_SCHEMA = "paperb_enb_rev01_cell_result_v1"
STATUS_SCHEMA = "paperb_enb_rev01_cell_status_v1"
EXPECTED_MATRIX_SCHEMA = "paperb_enb_rev01_cell_matrix_v1"
EXPECTED_PLAN_SCHEMA = "paperb_enb_rev01_compiled_plan_v1"
EXPECTED_C2_PLAN_SCHEMA = "paperb_enb_rev01_c2_compiled_plan_v1"
EXPECTED_VARIANT_SCHEMA = "paperb_enb_rev01_variant_manifest_v1"
EXPECTED_PARITY_SCHEMA = "paperb_enb_rev01_annual_parity_closure_v1"
EXPECTED_PARENT_RUNNER_SHA256 = (
    "a9693042883ed5c20edad6fcbc757c62c7216d5abc120ad18de1a003932848a4"
)
EXPECTED_ACCEPTED_VALIDATOR_SHA256 = (
    "ab7968640f22d91d4ba44c8b0ca874cbdbbfc1c0f2d9700ba2fd95d7ebfe62d0"
)
EXPECTED_MODEL_SHA256 = (
    "6fbeb06644a36b226b17824a1fca7526bd518e0a9b76a41f878fb1c27efcd619"
)
EXPECTED_MODEL_METRICS_SHA256 = (
    "b5922472a72727d71d263e5968dd474ab13ee8ec630570578d199d2275494713"
)
EXPECTED_TRAINING_DATA_SHA256 = (
    "16156602c5d64e1324359ef9f45ad25d839df0353d1b7fda18170a5ff16ad514"
)
EXPECTED_IDF_SHA256 = (
    "1144b58b848992d1730e49ef9c252569e3a515d82a2f99b2c0233352f625a7e4"
)
EXPECTED_PLAN_CLOSURE_SHA256_BY_BATCH = {
    "rev01_parity1": "02f86cc8138d6fd24fe0a80de28ce7a550522df6d3d07e3c7173a49240be754c",
    "rev01_core64": "02f86cc8138d6fd24fe0a80de28ce7a550522df6d3d07e3c7173a49240be754c",
    "rev01_adaptive24": "02f86cc8138d6fd24fe0a80de28ce7a550522df6d3d07e3c7173a49240be754c",
    "rev01_learned48": "b99a7efa7ab70747d02304698f9a2b86e99103831edd53991c30c774ad4c6a51",
    "REV01-C2-012-20260825-v1": "1ec43157d5596ee1d883d90c6bc104c00e24be236dd73e2e411244c14856f8c2",
    "REV01-C2-012-20260825-v2": "73f402563cdfe8784e3c02012bf6f95f8bd2f642b958aa5b7887a95c91727149",
}
EXPECTED_VARIANT_MANIFEST_SHA256_BY_BATCH = {
    "rev01_parity1": "8daf357be8683346aec533b2a9b5489d32c7a3b39a658c34bbcf383ce703cadf",
    "rev01_core64": "8daf357be8683346aec533b2a9b5489d32c7a3b39a658c34bbcf383ce703cadf",
    "rev01_adaptive24": "8daf357be8683346aec533b2a9b5489d32c7a3b39a658c34bbcf383ce703cadf",
    "rev01_learned48": "2ba95c3fd5a3d46c2bb17e0286868913528526b36f5b2175ce2b37880b4478e2",
    "REV01-C2-012-20260825-v1": "f1afa95852f50da35ca7249af03dbd3bfa3cfddb7d6d81e3174e09108c0e7579",
    "REV01-C2-012-20260825-v2": "b01b8857958a3a29c82557225de393ed5ebb3066e62056c021e845d78d658ae6",
}
EXPECTED_RUNNER_SHA256 = (
    "4c061fc3b25f7ee6fe66df00cfacfab42f1d3ab171185306fe1ce161a84a0baf"
)
EXPECTED_RUNNER_HELPER_SHA256 = (
    "b941c0bfc4beb3dc44557ed341bb4e018992620b0d3ad4f3bcb6973558d78bf7"
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
EXPECTED_MATRIX_SHA256 = {
    "rev01_parity1": "a02f998a8bdce6694f4dd78aaa8dd7548faf0ee05b2d728819e2e0cca5fafba6",
    "rev01_core64": "1853c5ac0658a905cb80280396575f6ac4777fd3d2eee260f174f1f410f1f662",
    "rev01_adaptive24": "14d3075d83d7e688f39677c2f02ed21b083b8a30c08df02b90425f976d524ac9",
    "rev01_learned48": "c38576e1a0dd70a4df2dd09e033a9bdd70c04f0446929570fd01e9bee14776dd",
    "REV01-C2-012-20260825-v1": "87d9108cec89acb81a447bbeaf635c27bc017255d269c2a33900092f273e0cf2",
    "REV01-C2-012-20260825-v2": "2f578dc9335b31d2ad7bef9f0f20e6536c1aaedfff2d667abdce0b620250fa97",
}
EXPECTED_BATCH_COUNTS = {
    "rev01_parity1": 1,
    "rev01_core64": 64,
    "rev01_adaptive24": 24,
    "rev01_learned48": 48,
    "REV01-C2-012-20260825-v1": 12,
    "REV01-C2-012-20260825-v2": 12,
}
EXPECTED_ROWS = 35_040
EXPECTED_OCCUPIED_ROWS = 16_704
MAX_WORKERS = 4
BANNED_FLAGS = frozenset(
    {"--resume", "--purge-energyplus-after-trace", "--purge-case-traces-after-summary"}
)

REBUILD_ROOT = Path(__file__).resolve().parents[3]
REV01_RERUN_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PYTHON = REBUILD_ROOT / ".venv/bin/python"
DEFAULT_VALIDATOR = (
    REBUILD_ROOT
    / "06_runs/scaled/20260803_routeS96_v2_envscoped/freeze/code"
    / "run_enb_scaled_matrix_v2_envscoped.py"
)
DEFAULT_MODEL = (
    REBUILD_ROOT
    / "06_runs/scaled/20260803_routeS96_v2_envscoped/freeze/model"
    / "control_predictors.joblib"
)
DEFAULT_MODEL_METRICS = (
    REBUILD_ROOT
    / "06_runs/scaled/20260803_routeS96_v2_envscoped/freeze/model"
    / "control_predictor_metrics.json"
)
DEFAULT_TRAINING_DATA = (
    REBUILD_ROOT.parent.parent / "TCN/newin_with_bmr.csv"
)
DEFAULT_IDF = (
    REBUILD_ROOT
    / "06_runs/scaled/20260803_routeS96_v2_envscoped/freeze/building"
    / "ASHRAE901_OfficeMedium_STD2019_Denver.idf"
)
DEFAULT_STORAGE_MODULE = Path(__file__).resolve().parent / "rev01_storage.py"
DEFAULT_STORAGE_MAPPING = Path(__file__).resolve().parent / "storage_mapping_v1.json"
DEFAULT_EPLUS_ROOT = Path("/Applications/EnergyPlus-25-1-0")
ACCEPTED_CELLS_ROOT = (
    REBUILD_ROOT / "06_runs/scaled/20260803_routeS96_v2_envscoped/cells"
)


class BatchContractError(RuntimeError):
    """A batch preflight, execution, validation, or closure gate failed."""


@dataclass(frozen=True)
class Preflight:
    matrix: pd.DataFrame
    plan: dict[str, Any]
    variant_manifest: dict[str, Any]
    runner_closure: dict[str, Any]
    runner_sha256: str
    runner_helper_sha256: str
    storage_sha256: str
    storage_mapping_sha256: str
    parity_closure: dict[str, Any] | None
    parity_closure_sha256: str | None
    accepted_comparator_snapshot: dict[str, Any]


@dataclass(frozen=True)
class FrozenInputs:
    runner: Path
    runner_helper: Path
    storage_module: Path
    storage_mapping: Path
    accepted_validator: Path
    model: Path
    model_metrics: Path
    idf: Path
    idf_by_source: dict[str, Path]
    variant_manifest: Path
    parity_closure: Path | None
    weather_by_source: dict[str, Path]
    freeze_record: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BatchContractError(f"Cannot read {name} ({path}): {exc}") from exc
    if not isinstance(value, dict):
        raise BatchContractError(f"{name} must be a JSON object: {path}")
    return value


def validate_parity_authorization(path: Path) -> dict[str, Any]:
    closure = read_json(path, name="annual parity closure")
    required_true = (
        "all_gates_passed",
        "accepted_cell_pinned",
        "canonical_intersection_exact",
        "generated_idf_exact",
        "full_revision_trace_validated",
        "warmup_audit_exact",
        "environment_control_contract_exact",
        "sql_sizing_tables_exact",
    )
    if closure.get("schema_version") != EXPECTED_PARITY_SCHEMA:
        raise BatchContractError("Annual parity closure schema differs")
    failed = [key for key in required_true if closure.get(key) is not True]
    if failed:
        raise BatchContractError(f"Annual parity authorization failed gates: {failed}")
    if closure.get("accepted_cell_id") != "SCALED-96-028":
        raise BatchContractError("Annual parity accepted-cell identity differs")
    batch_path = Path(str(closure.get("batch_closure_path", ""))).resolve()
    if not batch_path.is_file() or sha256_file(batch_path) != closure.get(
        "batch_closure_sha256"
    ):
        raise BatchContractError("Annual parity batch closure is missing or changed")
    batch = read_json(batch_path, name="annual parity batch closure")
    if not (
        batch.get("all_gates_passed") is True
        and batch.get("batch_id") == "rev01_parity1"
        and batch.get("planned_cells") == 1
        and batch.get("passed_cells") == 1
        and batch.get("runner_sha256") == EXPECTED_RUNNER_SHA256
        and batch.get("runner_helper_sha256") == EXPECTED_RUNNER_HELPER_SHA256
    ):
        raise BatchContractError("Annual parity batch does not authorize this runner")
    return closure


def one_path(root: Path, pattern: str, *, name: str) -> Path:
    paths = sorted(root.glob(pattern))
    if len(paths) != 1:
        raise BatchContractError(f"Expected exactly one {name} under {root}: {paths}")
    return paths[0]


def snapshot_accepted_comparators(frame: pd.DataFrame) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    accepted_root = ACCEPTED_CELLS_ROOT.resolve()
    for cell_id in sorted(set(frame["accepted_comparator_cell_id"].astype(str))):
        if not re.fullmatch(r"SCALED-96-\d{3}", cell_id):
            raise BatchContractError(f"Invalid accepted comparator cell ID: {cell_id}")
        cell = (accepted_root / cell_id).resolve()
        if cell.parent != accepted_root or not cell.is_dir():
            raise BatchContractError(f"Accepted comparator cell is missing: {cell}")
        files = {
            "trace": one_path(cell / "traces", "*.parquet", name="accepted trace"),
            "summary": cell / "summary/medium_office_trace_summary.csv",
            "generated_idf": one_path(cell / "model", "*.idf", name="accepted IDF"),
            "eio": one_path(cell / "energyplus", "**/eplusout.eio", name="accepted eio"),
            "sql": one_path(cell / "energyplus", "**/eplusout.sql", name="accepted sql"),
            "cell_result": one_path(
                cell, "CELL_RESULT.*.json", name="accepted cell result"
            ),
        }
        missing = [path for path in files.values() if not path.is_file()]
        if missing:
            raise BatchContractError(f"Accepted comparator evidence is missing: {missing}")
        snapshot[cell_id] = {
            key: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for key, path in files.items()
        }
    return snapshot


def write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path: Path, value: Any, lock: threading.Lock | None = None) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    context = lock if lock is not None else _NullLock()
    with context:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())


class _NullLock:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: Any) -> None:
        return None


def copy_new(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=8 * 1024 * 1024)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    shutil.copystat(source, destination, follow_symlinks=True)
    if sha256_file(destination) != sha256_file(source):
        raise BatchContractError(f"Frozen copy hash mismatch: {destination}")


def load_module(path: Path, name: str, *, expected_sha256: str | None = None) -> ModuleType:
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise BatchContractError(f"Module checksum mismatch: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise BatchContractError(f"Cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def exact_number(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return left == right


def expected_manifest_config(row: pd.Series) -> dict[str, Any]:
    return {
        "strategy": str(row["strategy"]),
        "spatial_quantile": float(row["paperb_spatial_quantile"]),
        "signal_alpha": float(row["paperb_signal_alpha"]),
        "tail_threshold": float(row["paperb_tail_threshold"]),
        "asym_threshold": float(row["paperb_asym_threshold"]),
        "save_heat_c": float(row["paperb_save_heat_c"]),
        "save_cool_c": float(row["paperb_save_cool_c"]),
        "warm_protect_cool_c": float(row["paperb_warm_protect_cool_c"]),
        "cold_protect_heat_c": float(row["paperb_cold_protect_heat_c"]),
        "tighten_dwell_steps": int(row["paperb_tighten_dwell_steps"]),
        "relax_dwell_steps": int(row["paperb_relax_dwell_steps"]),
        "pmv_threshold": float(row["paperb_pmv_threshold"]),
        "adaptive_rm_clamp_low_c": float(row["paperb_adaptive_rm_clamp_low_c"]),
        "adaptive_rm_clamp_high_c": float(row["paperb_adaptive_rm_clamp_high_c"]),
        "paperb_met": float(row["paperb_met"]),
        "people_activity_w_per_person": float(row["paperb_people_activity_w"]),
        "controller_semantics": str(row["controller_semantics"]),
        "inference_preprocessing": str(row["inference_preprocessing"]),
        "sizing_contract": str(row["sizing_contract"]),
    }


def validate_output_root(path: Path) -> None:
    base = REV01_RERUN_ROOT.resolve()
    lexical = path.absolute()
    try:
        parts = lexical.relative_to(base).parts
    except ValueError as exc:
        raise BatchContractError(f"Output root must be below {base}: {path}") from exc
    current = base
    for part in parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise BatchContractError(f"Symlink output component forbidden: {current}")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Fresh batch output root must be absent: {path}")
    if not parts:
        raise BatchContractError("The rerun root itself cannot be a batch output")


def validate_python(path: Path) -> None:
    invocation = Path(os.path.abspath(os.fspath(path)))
    if invocation != EXPECTED_PYTHON:
        raise BatchContractError(f"Expected project venv invocation {EXPECTED_PYTHON}, got {path}")
    if not invocation.is_symlink() or not os.access(invocation, os.X_OK):
        raise BatchContractError(f"Project venv Python is not an executable symlink: {path}")


def validate_runner_closure(
    path: Path,
    runner_source: Path,
    runner_helper: Path,
) -> tuple[dict[str, Any], str, str]:
    if sha256_file(path) != EXPECTED_RUNNER_CLOSURE_SHA256:
        raise BatchContractError("Rev01 runner closure is not the authorized closure")
    closure = read_json(path, name="Rev01 runner closure")
    if closure.get("all_gates_passed") is not True:
        raise BatchContractError("Rev01 runner closure is not passing")
    if closure.get("parent_runner_sha256") != EXPECTED_PARENT_RUNNER_SHA256:
        raise BatchContractError("Rev01 runner parent identity differs")
    runner_hash = sha256_file(runner_source)
    if runner_hash != EXPECTED_RUNNER_SHA256:
        raise BatchContractError("Rev01 derived runner is not the authorized source")
    declared = closure.get("derived_runner_sha256", closure.get("runner_sha256"))
    if declared != runner_hash:
        raise BatchContractError("Rev01 runner closure does not bind the derived runner")
    declared_path = closure.get("derived_runner_path", closure.get("runner_path"))
    if declared_path is not None and Path(str(declared_path)).resolve() != runner_source.resolve():
        raise BatchContractError("Rev01 runner closure path differs")
    helper_hash = sha256_file(runner_helper)
    if helper_hash != EXPECTED_RUNNER_HELPER_SHA256:
        raise BatchContractError("Rev01 helper is not the authorized source")
    if closure.get("helper_sha256") != helper_hash:
        raise BatchContractError("Rev01 runner closure does not bind the helper module")
    declared_helper_path = closure.get("helper_path")
    if (
        declared_helper_path is not None
        and Path(str(declared_helper_path)).resolve() != runner_helper.resolve()
    ):
        raise BatchContractError("Rev01 helper closure path differs")
    return closure, runner_hash, helper_hash


def validate_matrix_and_manifest(
    matrix_path: Path,
    plan: dict[str, Any],
    manifest: dict[str, Any],
) -> pd.DataFrame:
    if plan.get("schema_version") not in {EXPECTED_PLAN_SCHEMA, EXPECTED_C2_PLAN_SCHEMA} or plan.get("all_gates_passed") is not True:
        raise BatchContractError("Compiled plan closure is not passing")
    if manifest.get("schema_version") != EXPECTED_VARIANT_SCHEMA:
        raise BatchContractError("Variant manifest schema differs")
    if manifest.get("parent_runner_sha256") != EXPECTED_PARENT_RUNNER_SHA256:
        raise BatchContractError("Variant manifest parent runner differs")
    variants = manifest.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise BatchContractError("Variant manifest has no variants")
    frame = pd.read_csv(matrix_path)
    matrix_key = matrix_path.stem
    if plan.get("schema_version") == EXPECTED_C2_PLAN_SCHEMA:
        if Path(str(plan.get("matrix_path", ""))).resolve() != matrix_path.resolve():
            raise BatchContractError("C2 plan matrix path differs")
        if plan.get("matrix_sha256") != sha256_file(matrix_path):
            raise BatchContractError("C2 plan matrix hash differs")
    else:
        binding = (plan.get("matrices") or {}).get(matrix_key)
        if not isinstance(binding, dict):
            raise BatchContractError(f"Plan does not bind matrix {matrix_key}")
        if Path(str(binding.get("csv_path", ""))).resolve() != matrix_path.resolve():
            raise BatchContractError("Plan matrix path differs")
        if binding.get("csv_sha256") != sha256_file(matrix_path):
            raise BatchContractError("Plan matrix hash differs")
    required = {
        "matrix_schema_version", "batch_id", "cell_id", "experiment", "variant_id",
        "strategy", "anchor_id", "anchor_order", "physical_condition_id",
        "selector_label_id", "scenario", "city", "weather_path", "weather_sha256",
        "weather_data_sha256", "weather_stem", "expected_trace_rows",
        "expected_occupied_rows", "resume_permitted", "reuse_or_splice_permitted",
        "purge_permitted", "sizing_contract", "accepted_comparator_cell_id",
        "accepted_matrix_sha256", "accepted_idf_sha256", "accepted_model_sha256",
    } | {
        "paperb_spatial_quantile", "paperb_signal_alpha", "paperb_tail_threshold",
        "paperb_asym_threshold", "paperb_save_heat_c", "paperb_save_cool_c",
        "paperb_warm_protect_cool_c", "paperb_cold_protect_heat_c",
        "paperb_tighten_dwell_steps", "paperb_relax_dwell_steps",
        "paperb_pmv_threshold", "paperb_adaptive_rm_clamp_low_c",
        "paperb_adaptive_rm_clamp_high_c", "paperb_met",
        "paperb_people_activity_w", "controller_semantics",
        "inference_preprocessing",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise BatchContractError(f"Matrix missing columns: {missing}")
    if frame.empty or frame["cell_id"].nunique() != len(frame):
        raise BatchContractError("Matrix must contain unique, nonempty cells")
    if set(frame["matrix_schema_version"].astype(str)) != {EXPECTED_MATRIX_SCHEMA}:
        raise BatchContractError("Cell matrix schema differs")
    batch_ids = set(frame["batch_id"].astype(str))
    expected_count = (
        EXPECTED_BATCH_COUNTS.get(next(iter(batch_ids)))
        if len(batch_ids) == 1
        else None
    )
    if expected_count is None or len(frame) != expected_count:
        raise BatchContractError(f"Unexpected batch/count: {batch_ids}, {len(frame)}")
    batch_id = next(iter(batch_ids))
    if sha256_file(matrix_path) != EXPECTED_MATRIX_SHA256[batch_id]:
        raise BatchContractError(f"Matrix is not the authorized {batch_id} cell set")
    is_c2 = batch_id.startswith("REV01-C2-012-")
    exact_sets = {
        "paperb_met": {0.854333},
        "paperb_people_activity_w": {93.2},
        "controller_semantics": {"corrected"},
        "inference_preprocessing": {"training_contract"},
        "sizing_contract": {"city_ddy" if is_c2 else "accepted_denver"},
        "accepted_matrix_sha256": {
            "fa03767f0c3d581e65e00ff6149546c44fe3e936608da73ad2f5485bf19895e0"
        },
        "accepted_idf_sha256": {EXPECTED_IDF_SHA256},
        "accepted_model_sha256": {EXPECTED_MODEL_SHA256},
    }
    for field, expected_values in exact_sets.items():
        observed_values = set(frame[field])
        if observed_values != expected_values:
            raise BatchContractError(
                f"Authorized matrix contract differs for {field}: {observed_values}"
            )
    for field in ("resume_permitted", "reuse_or_splice_permitted", "purge_permitted"):
        normalized = frame[field].map(lambda x: str(x).strip().casefold())
        if set(normalized) != {"false"}:
            raise BatchContractError(f"Matrix enables forbidden behavior: {field}")
    if set(pd.to_numeric(frame["expected_trace_rows"], errors="raise").astype(int)) != {EXPECTED_ROWS}:
        raise BatchContractError("Matrix trace-row contract differs")
    if set(pd.to_numeric(frame["expected_occupied_rows"], errors="raise").astype(int)) != {EXPECTED_OCCUPIED_ROWS}:
        raise BatchContractError("Matrix occupied-row contract differs")
    for _index, row in frame.iterrows():
        variant_id = str(row["variant_id"])
        entry = variants.get(variant_id)
        if not isinstance(entry, dict) or entry.get("strategy") != str(row["strategy"]):
            raise BatchContractError(f"Variant/strategy not bound: {variant_id}")
        config = entry.get("effective_config")
        expected_config = expected_manifest_config(row)
        if not isinstance(config, dict) or set(config) != set(expected_config):
            raise BatchContractError(f"Variant effective-config shape differs: {variant_id}")
        mismatched = [key for key in expected_config if not exact_number(config[key], expected_config[key])]
        if mismatched:
            raise BatchContractError(f"Variant config differs for {variant_id}: {mismatched}")
        sizing = entry.get("sizing")
        if is_c2:
            c2_required = {
                "source_idf_path", "source_idf_sha256", "source_ddy_sha256",
                "heating_design_day_name", "cooling_design_day_name",
                "companion_site_location_name", "companion_latitude_deg",
                "companion_longitude_deg", "companion_time_zone_hours",
                "companion_elevation_m", "runtime_epw_location_name",
                "runtime_epw_latitude_deg", "runtime_epw_longitude_deg",
                "runtime_epw_time_zone_hours", "runtime_epw_elevation_m",
                "runtime_epw_location_header_sha256",
            }
            if c2_required - set(frame.columns):
                raise BatchContractError("C2 matrix lacks city-DDY source bindings")
            source_idf = Path(str(row["source_idf_path"]))
            if not source_idf.is_file() or sha256_file(source_idf) != str(row["source_idf_sha256"]):
                raise BatchContractError(f"C2 source IDF missing/hash mismatch: {source_idf}")
            expected_names = [
                str(row["heating_design_day_name"]),
                str(row["cooling_design_day_name"]),
            ]
            if not isinstance(sizing, dict) or not (
                sizing.get("mode") == "city_ddy"
                and sizing.get("city") == str(row["city"])
                and sizing.get("source_idf_sha256") == str(row["source_idf_sha256"])
                and sizing.get("design_day_names") == expected_names
                and sizing.get("non_sizing_semantic_sha256")
                == sizing.get("accepted_base_non_sizing_semantic_sha256")
            ):
                raise BatchContractError(f"Unsupported C2 sizing contract in {variant_id}")
        elif sizing != {"mode": "accepted_denver", "source_idf_sha256": EXPECTED_IDF_SHA256}:
            raise BatchContractError(f"Unsupported sizing contract in {variant_id}")
        weather = Path(str(row["weather_path"]))
        if not weather.is_file() or sha256_file(weather) != str(row["weather_sha256"]):
            raise BatchContractError(f"Weather missing/hash mismatch: {weather}")
    return frame


def validate_preflight(args: argparse.Namespace) -> Preflight:
    validate_output_root(args.output_root)
    validate_python(args.python)
    required_files = (
        args.plan_closure, args.matrix, args.variant_manifest, args.runner_source,
        args.runner_helper, args.runner_closure, args.storage_module,
        args.storage_mapping, args.accepted_validator, args.model,
        args.model_metrics, args.training_data, args.idf,
    )
    missing = [path for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(map(str, missing)))
    if not args.eplus_root.is_dir():
        raise FileNotFoundError(args.eplus_root)
    matrix_probe = pd.read_csv(args.matrix, usecols=["batch_id"])
    probe_batch_ids = set(matrix_probe["batch_id"].astype(str))
    if len(probe_batch_ids) != 1:
        raise BatchContractError(f"Matrix must contain one authorized batch ID: {probe_batch_ids}")
    probed_batch_id = next(iter(probe_batch_ids))
    if probed_batch_id not in EXPECTED_BATCH_COUNTS:
        raise BatchContractError(f"Unauthorized batch ID: {probed_batch_id}")
    if probed_batch_id.startswith("REV01-C2-012-"):
        c2_probe = pd.read_csv(args.matrix, usecols=["source_idf_path", "source_idf_sha256"])
        for source_path, expected_hash in c2_probe.drop_duplicates().itertuples(index=False):
            source = Path(str(source_path))
            if not source.is_file() or sha256_file(source) != str(expected_hash):
                raise BatchContractError(f"C2 source IDF missing/hash mismatch: {source}")
    exact_authorization_hashes = {
        args.plan_closure: EXPECTED_PLAN_CLOSURE_SHA256_BY_BATCH[probed_batch_id],
        args.variant_manifest: EXPECTED_VARIANT_MANIFEST_SHA256_BY_BATCH[probed_batch_id],
        args.runner_source: EXPECTED_RUNNER_SHA256,
        args.runner_helper: EXPECTED_RUNNER_HELPER_SHA256,
        args.runner_closure: EXPECTED_RUNNER_CLOSURE_SHA256,
        args.storage_module: EXPECTED_STORAGE_MODULE_SHA256,
        args.storage_mapping: EXPECTED_STORAGE_MAPPING_SHA256,
    }
    for path, expected in exact_authorization_hashes.items():
        observed = sha256_file(path)
        if observed != expected:
            raise BatchContractError(f"Execution authorization hash mismatch: {path}: {observed}")
    hashes = {
        args.accepted_validator: EXPECTED_ACCEPTED_VALIDATOR_SHA256,
        args.model: EXPECTED_MODEL_SHA256,
        args.model_metrics: EXPECTED_MODEL_METRICS_SHA256,
        args.training_data: EXPECTED_TRAINING_DATA_SHA256,
        args.idf: EXPECTED_IDF_SHA256,
    }
    for path, expected in hashes.items():
        observed = sha256_file(path)
        if observed != expected:
            raise BatchContractError(f"Pinned input hash mismatch: {path}: {observed}")
    plan = read_json(args.plan_closure, name="compiled plan closure")
    manifest = read_json(args.variant_manifest, name="variant manifest")
    storage_mapping = read_json(args.storage_mapping, name="storage mapping")
    if storage_mapping.get("schema_version") != "paperb_enb_rev01_storage_mapping_v1":
        raise BatchContractError("Storage mapping schema differs")
    if plan.get("variant_manifest_sha256") != sha256_file(args.variant_manifest):
        raise BatchContractError("Plan does not bind the variant manifest")
    matrix = validate_matrix_and_manifest(args.matrix, plan, manifest)
    batch_id = str(matrix["batch_id"].iloc[0])
    parity_closure: dict[str, Any] | None = None
    parity_hash: str | None = None
    if batch_id == "rev01_parity1":
        if args.parity_closure is not None:
            raise BatchContractError("The parity sentinel cannot authorize itself")
    else:
        if args.parity_closure is None or not args.parity_closure.is_file():
            raise BatchContractError(
                "Core/adaptive execution requires a passing annual parity closure"
            )
        parity_closure = validate_parity_authorization(args.parity_closure)
        parity_hash = sha256_file(args.parity_closure)
    runner_closure, runner_hash, helper_hash = validate_runner_closure(
        args.runner_closure,
        args.runner_source,
        args.runner_helper,
    )
    storage_hash = sha256_file(args.storage_module)
    if storage_hash != EXPECTED_STORAGE_MODULE_SHA256:
        raise BatchContractError("Storage module is not the authorized source")
    accepted_snapshot = snapshot_accepted_comparators(matrix)
    return Preflight(
        matrix=matrix,
        plan=plan,
        variant_manifest=manifest,
        runner_closure=runner_closure,
        runner_sha256=runner_hash,
        runner_helper_sha256=helper_hash,
        storage_sha256=storage_hash,
        storage_mapping_sha256=sha256_file(args.storage_mapping),
        parity_closure=parity_closure,
        parity_closure_sha256=parity_hash,
        accepted_comparator_snapshot=accepted_snapshot,
    )


def freeze_inputs(args: argparse.Namespace, preflight: Preflight) -> FrozenInputs:
    root = args.output_root
    root.mkdir(parents=True, exist_ok=False)
    freeze = root / "freeze"
    files = {
        args.runner_source: freeze / "code" / args.runner_source.name,
        args.runner_helper: freeze / "code" / args.runner_helper.name,
        Path(__file__).resolve(): freeze / "code" / Path(__file__).name,
        args.storage_module: freeze / "code" / args.storage_module.name,
        args.storage_mapping: freeze / "contracts" / args.storage_mapping.name,
        args.accepted_validator: freeze / "code" / "accepted_validator.py",
        args.runner_closure: freeze / "contracts" / args.runner_closure.name,
        args.plan_closure: freeze / "contracts" / args.plan_closure.name,
        args.matrix: freeze / "plan" / args.matrix.name,
        args.variant_manifest: freeze / "plan" / args.variant_manifest.name,
        args.model: freeze / "model" / "control_predictors.joblib",
        args.model_metrics: freeze / "model" / "control_predictor_metrics.json",
        args.idf: freeze / "building" / args.idf.name,
    }
    is_c2 = str(preflight.matrix["batch_id"].iloc[0]).startswith("REV01-C2-012-")
    if is_c2:
        for _index, row in preflight.matrix.drop_duplicates("source_idf_path").iterrows():
            source_idf = Path(str(row["source_idf_path"])).resolve()
            files[source_idf] = freeze / "building" / "city_ddy" / str(row["city"]) / source_idf.name
    if args.parity_closure is not None:
        files[args.parity_closure] = freeze / "contracts" / args.parity_closure.name
    for source, destination in files.items():
        copy_new(source, destination)
    matrix_binding_sha = (
        preflight.plan["matrix_sha256"]
        if preflight.plan.get("schema_version") == EXPECTED_C2_PLAN_SCHEMA
        else preflight.plan["matrices"][args.matrix.stem]["csv_sha256"]
    )
    frozen_contracts = {
        files[args.runner_source]: preflight.runner_sha256,
        files[args.runner_helper]: preflight.runner_helper_sha256,
        files[args.storage_module]: preflight.storage_sha256,
        files[args.storage_mapping]: preflight.storage_mapping_sha256,
        files[args.accepted_validator]: EXPECTED_ACCEPTED_VALIDATOR_SHA256,
        files[args.matrix]: matrix_binding_sha,
        files[args.variant_manifest]: preflight.plan["variant_manifest_sha256"],
        files[args.model]: EXPECTED_MODEL_SHA256,
        files[args.model_metrics]: EXPECTED_MODEL_METRICS_SHA256,
        files[args.idf]: EXPECTED_IDF_SHA256,
    }
    if is_c2:
        for _index, row in preflight.matrix.drop_duplicates("source_idf_path").iterrows():
            source_idf = Path(str(row["source_idf_path"])).resolve()
            frozen_contracts[files[source_idf]] = str(row["source_idf_sha256"])
    if args.parity_closure is not None:
        frozen_contracts[files[args.parity_closure]] = str(
            preflight.parity_closure_sha256
        )
    for path, expected in frozen_contracts.items():
        if sha256_file(path) != expected:
            raise BatchContractError(f"Frozen pinned-input hash mismatch: {path}")
    weather_by_source: dict[str, Path] = {}
    for _index, row in preflight.matrix.drop_duplicates("weather_path").iterrows():
        source = Path(str(row["weather_path"])).resolve()
        destination = freeze / "weather" / str(row["scenario"]) / str(row["city"]) / source.name
        copy_new(source, destination)
        if sha256_file(destination) != str(row["weather_sha256"]):
            raise BatchContractError(f"Frozen weather hash mismatch: {destination}")
        weather_by_source[str(source)] = destination.resolve()
    idf_by_source = {
        str(Path(str(row["source_idf_path"])).resolve()): files[
            Path(str(row["source_idf_path"])).resolve()
        ].resolve()
        for _index, row in preflight.matrix.drop_duplicates("source_idf_path").iterrows()
    } if is_c2 else {str(args.idf.resolve()): files[args.idf].resolve()}
    frozen_hashes = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(freeze.rglob("*"))
        if path.is_file()
    }
    record = {
        "schema_version": FREEZE_SCHEMA,
        "created_utc": utc_now(),
        "batch_id": str(preflight.matrix["batch_id"].iloc[0]),
        "planned_cells": int(len(preflight.matrix)),
        "runner_sha256": preflight.runner_sha256,
        "runner_helper_sha256": preflight.runner_helper_sha256,
        "runner_closure_sha256": sha256_file(args.runner_closure),
        "launcher_sha256": sha256_file(Path(__file__).resolve()),
        "storage_module_sha256": preflight.storage_sha256,
        "parent_runner_sha256": EXPECTED_PARENT_RUNNER_SHA256,
        "model_sha256": EXPECTED_MODEL_SHA256,
        "model_metrics_sha256": EXPECTED_MODEL_METRICS_SHA256,
        "training_data_path": str(args.training_data.resolve()),
        "training_data_sha256": EXPECTED_TRAINING_DATA_SHA256,
        "idf_sha256": EXPECTED_IDF_SHA256,
        "source_idf_files": {
            source: {"frozen_path": str(path), "sha256": sha256_file(path)}
            for source, path in sorted(idf_by_source.items())
        },
        "storage_mapping_sha256": preflight.storage_mapping_sha256,
        "parity_closure_sha256": preflight.parity_closure_sha256,
        "accepted_comparator_snapshot": preflight.accepted_comparator_snapshot,
        "matrix_sha256": sha256_file(args.matrix),
        "variant_manifest_sha256": sha256_file(args.variant_manifest),
        "weather_files": len(weather_by_source),
        "frozen_file_sha256": frozen_hashes,
        "launch_performed_at_freeze_time": False,
        "resume_permitted": False,
        "reuse_or_splice_permitted": False,
        "purge_permitted": False,
    }
    write_json_new(root / "FREEZE.json", record)
    return FrozenInputs(
        runner=files[args.runner_source].resolve(),
        runner_helper=files[args.runner_helper].resolve(),
        storage_module=files[args.storage_module].resolve(),
        storage_mapping=files[args.storage_mapping].resolve(),
        accepted_validator=files[args.accepted_validator].resolve(),
        model=files[args.model].resolve(),
        model_metrics=files[args.model_metrics].resolve(),
        idf=files[args.idf].resolve(),
        idf_by_source=idf_by_source,
        variant_manifest=files[args.variant_manifest].resolve(),
        parity_closure=(
            files[args.parity_closure].resolve()
            if args.parity_closure is not None
            else None
        ),
        weather_by_source=weather_by_source,
        freeze_record=record,
    )


def command_for_cell(
    row: pd.Series,
    *,
    cell_dir: Path,
    frozen: FrozenInputs,
    args: argparse.Namespace,
) -> list[str]:
    weather = frozen.weather_by_source[str(Path(str(row["weather_path"])).resolve())]
    source_idf = Path(str(row.get("source_idf_path", args.idf))).resolve()
    if str(source_idf) not in frozen.idf_by_source:
        raise BatchContractError(f"Row-resolved source IDF is not frozen: {source_idf}")
    frozen_idf = frozen.idf_by_source[str(source_idf)]
    command = [
        str(args.python), str(frozen.runner),
        "--data", str(args.training_data.resolve()),
        "--output-dir", str(cell_dir.resolve()),
        "--idf", str(frozen_idf),
        "--eplus-root", str(args.eplus_root.resolve()),
        "--weather", str(weather),
        "--begin-month", "1", "--begin-day", "1", "--end-month", "12", "--end-day", "31",
        "--strategies", str(row["strategy"]),
        "--paperb-met", str(row["paperb_met"]),
        "--paperb-people-activity-w", str(row["paperb_people_activity_w"]),
        "--paperb-save-heat-c", str(row["paperb_save_heat_c"]),
        "--paperb-save-cool-c", str(row["paperb_save_cool_c"]),
        "--paperb-warm-protect-cool-c", str(row["paperb_warm_protect_cool_c"]),
        "--paperb-cold-protect-heat-c", str(row["paperb_cold_protect_heat_c"]),
        "--paperb-tighten-dwell-steps", str(int(row["paperb_tighten_dwell_steps"])),
        "--paperb-relax-dwell-steps", str(int(row["paperb_relax_dwell_steps"])),
        "--paperb-tail-threshold", str(row["paperb_tail_threshold"]),
        "--paperb-asym-threshold", str(row["paperb_asym_threshold"]),
        "--paperb-pmv-threshold", str(row["paperb_pmv_threshold"]),
        "--paperb-spatial-quantile", str(row["paperb_spatial_quantile"]),
        "--paperb-signal-alpha", str(row["paperb_signal_alpha"]),
        "--paperb-adaptive-rm-clamp-low-c", str(row["paperb_adaptive_rm_clamp_low_c"]),
        "--paperb-adaptive-rm-clamp-high-c", str(row["paperb_adaptive_rm_clamp_high_c"]),
        "--controller-semantics", str(row["controller_semantics"]),
        "--inference-preprocessing", str(row["inference_preprocessing"]),
        "--model-path", str(frozen.model),
        "--model-metrics-path", str(frozen.model_metrics),
        "--rev01-variant-manifest", str(frozen.variant_manifest),
        "--rev01-variant-id", str(row["variant_id"]),
        "--sizing-contract", str(row["sizing_contract"]),
        "--skip-train", "--skip-plot", "--skip-combined-trace", "--trace-format", "parquet",
    ]
    if BANNED_FLAGS.intersection(command):
        raise BatchContractError("Cell command contains a forbidden flag")
    if command.count("--strategies") != 1 or command[command.index("--strategies") + 1] != str(row["strategy"]):
        raise BatchContractError("Cell command is not strategy-isolated")
    return command


def validator_job(validator: ModuleType, row: pd.Series, weather: Path) -> Any:
    return validator.CellJob(
        scaled_run_id=str(row["cell_id"]),
        anchor_id=str(row["anchor_id"]),
        anchor_order=int(row["anchor_order"]),
        physical_condition_id=str(row["physical_condition_id"]),
        selector_label_id=str(row["selector_label_id"]),
        scenario=str(row["scenario"]),
        city=str(row["city"]),
        strategy=str(row["strategy"]),
        strategy_order=1,
        weather_source=weather,
        weather_sha256=str(row["weather_sha256"]),
        weather_data_sha256=str(row["weather_data_sha256"]),
        weather_stem=str(row["weather_stem"]),
    )


def parse_idf_semantic_objects(path: Path) -> list[list[str]]:
    text = path.read_text(encoding="utf-8", errors="strict")
    uncommented = "\n".join(line.split("!", 1)[0] for line in text.splitlines())
    objects: list[list[str]] = []
    for raw in uncommented.split(";"):
        fields = [field.strip() for field in raw.split(",")]
        if fields and fields[0]:
            objects.append(fields)
    return objects


def validate_c2_runtime_contract(cell_dir: Path, row: pd.Series) -> dict[str, Any]:
    """Verify companion source objects and the EPW-overridden runtime site."""

    if not str(row["batch_id"]).startswith("REV01-C2-012-"):
        return {"applicable": False, "all_gates_passed": True}
    generated_idf = one_path(cell_dir / "model", "*.idf", name="generated C2 IDF")
    objects = parse_idf_semantic_objects(generated_idf)
    locations = [x for x in objects if x[0].casefold() == "site:location"]
    design_days = [x for x in objects if x[0].casefold() == "sizingperiod:designday"]
    expected_location = [
        "Site:Location",
        str(row["companion_site_location_name"]),
        str(row["companion_latitude_deg"]),
        str(row["companion_longitude_deg"]),
        str(row["companion_time_zone_hours"]),
        str(row["companion_elevation_m"]),
    ]
    if len(locations) != 1 or len(locations[0]) != 6:
        raise BatchContractError("Generated C2 IDF does not contain one six-field Site:Location")
    observed_location = locations[0]
    if observed_location[1] != expected_location[1] or any(
        float(observed_location[index]) != float(expected_location[index])
        for index in range(2, 6)
    ):
        raise BatchContractError("Generated C2 IDF companion Site:Location differs")
    expected_names = {
        str(row["heating_design_day_name"]), str(row["cooling_design_day_name"])
    }
    if len(design_days) != 2 or {fields[1] for fields in design_days} != expected_names:
        raise BatchContractError("Generated C2 IDF design-day names differ")

    eio = one_path(cell_dir / "energyplus", "**/eplusout.eio", name="C2 eio")
    sql = one_path(cell_dir / "energyplus", "**/eplusout.sql", name="C2 sql")
    err = one_path(cell_dir / "energyplus", "**/eplusout.err", name="C2 err")
    site_lines = [
        line for line in eio.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.startswith("Site:Location,")
    ]
    if len(site_lines) != 1:
        raise BatchContractError("C2 eio does not report exactly one runtime Site:Location")
    runtime_fields = [field.strip() for field in site_lines[0].split(",")]
    if len(runtime_fields) < 6:
        raise BatchContractError("C2 eio runtime Site:Location is malformed")
    observed_runtime = tuple(float(runtime_fields[index]) for index in range(2, 6))
    expected_runtime = (
        float(row["runtime_epw_latitude_deg"]),
        float(row["runtime_epw_longitude_deg"]),
        float(row["runtime_epw_time_zone_hours"]),
        float(row["runtime_epw_elevation_m"]),
    )
    # EIO formats latitude/longitude to two decimals; timezone/elevation are
    # also formatted, so compare at the precision actually retained there.
    if not (
        abs(observed_runtime[0] - expected_runtime[0]) <= 0.0051
        and abs(observed_runtime[1] - expected_runtime[1]) <= 0.0051
        and observed_runtime[2] == expected_runtime[2]
        and observed_runtime[3] == expected_runtime[3]
    ):
        raise BatchContractError(
            f"C2 runtime site is not the frozen EPW site: {observed_runtime} vs {expected_runtime}"
        )
    err_text = err.read_text(encoding="utf-8", errors="replace")
    override_warning_present = (
        "Weather file location will be used rather than entered (IDF) Location object." in err_text
    )
    if "** Severe **" in err_text or "**  Fatal  **" in err_text or "** Fatal **" in err_text:
        raise BatchContractError("C2 EnergyPlus output contains a severe/fatal error")
    with sqlite3.connect(sql) as connection:
        zone_names = {
            str(value[0])
            for value in connection.execute("SELECT DISTINCT DesDayName FROM ZoneSizes")
            if value[0] is not None
        }
        system_names = {
            str(value[0])
            for value in connection.execute("SELECT DISTINCT DesDayName FROM SystemSizes")
            if value[0] is not None
        }
    expected_upper = {name.upper() for name in expected_names}
    if not (expected_upper <= zone_names and expected_upper <= system_names):
        raise BatchContractError("C2 SQL sizing records do not bind both selected design days")
    return {
        "applicable": True,
        "all_gates_passed": True,
        "generated_idf": {"path": str(generated_idf.resolve()), "sha256": sha256_file(generated_idf)},
        "companion_site_location": {
            "name": observed_location[1],
            "latitude_deg": float(observed_location[2]),
            "longitude_deg": float(observed_location[3]),
            "time_zone_hours": float(observed_location[4]),
            "elevation_m": float(observed_location[5]),
        },
        "runtime_epw_site_location": {
            "reported_name": runtime_fields[1],
            "latitude_deg": observed_runtime[0],
            "longitude_deg": observed_runtime[1],
            "time_zone_hours": observed_runtime[2],
            "elevation_m": observed_runtime[3],
            "matrix_header_sha256": str(row["runtime_epw_location_header_sha256"]),
        },
        "design_day_names": sorted(expected_names),
        "eio": {"path": str(eio.resolve()), "sha256": sha256_file(eio)},
        "sql": {"path": str(sql.resolve()), "sha256": sha256_file(sql)},
        "err": {"path": str(err.resolve()), "sha256": sha256_file(err)},
        "epw_override_warning_present_diagnostic": override_warning_present,
        "zero_severe_fatal": True,
    }


def run_cell(
    row: pd.Series,
    *,
    args: argparse.Namespace,
    frozen: FrozenInputs,
    validator: ModuleType,
    attempt_id: str,
    status_path: Path,
    status_lock: threading.Lock,
) -> dict[str, Any]:
    cell_id = str(row["cell_id"])
    cell_dir = args.output_root / "cells" / cell_id
    started = time.time()
    runner_pid: int | None = None
    command: list[str] = []
    try:
        cell_dir.mkdir(parents=True, exist_ok=False)
        (cell_dir / ".mplconfig").mkdir(exist_ok=False)
        command = command_for_cell(row, cell_dir=cell_dir, frozen=frozen, args=args)
        contract = {
            "schema_version": "paperb_enb_rev01_cell_contract_v1",
            "attempt_id": attempt_id,
            "cell_id": cell_id,
            "batch_id": str(row["batch_id"]),
            "variant_id": str(row["variant_id"]),
            "strategy": str(row["strategy"]),
            "anchor_id": str(row["anchor_id"]),
            "accepted_comparator_cell_id": str(row["accepted_comparator_cell_id"]),
            "runner_sha256": sha256_file(frozen.runner),
            "runner_helper_sha256": sha256_file(frozen.runner_helper),
            "model_sha256": sha256_file(frozen.model),
            "weather_sha256": str(row["weather_sha256"]),
            "command": command,
            "resume_permitted": False,
            "reuse_or_splice_permitted": False,
            "purge_permitted": False,
        }
        write_json_new(cell_dir / "RUN_CONTRACT.json", contract)
        pd.DataFrame([row.to_dict()]).to_csv(cell_dir / "AUTHORIZED_ROW.csv", index=False)
        log_path = cell_dir / "run.log"
        environment = os.environ.copy()
        environment.update(
            {
                "MPLCONFIGDIR": str((cell_dir / ".mplconfig").resolve()),
                "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                "XDG_CACHE_HOME": str((cell_dir / ".xdg_cache").resolve()),
                "FC_CACHEDIR": str((cell_dir / ".fontconfig_cache").resolve()),
            }
        )
        with log_path.open("x", encoding="utf-8") as log:
            log.write(json.dumps({"attempt_id": attempt_id, "command": command}) + "\n")
            log.flush()
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
            runner_pid = process.pid
            returncode = process.wait()
        if returncode != 0:
            raise BatchContractError(f"Runner exited {returncode}; see {log_path}")
        weather = frozen.weather_by_source[str(Path(str(row["weather_path"])).resolve())]
        job = validator_job(validator, row, weather)
        outputs = validator.validate_cell_outputs(cell_dir, job)
        c2_runtime = validate_c2_runtime_contract(cell_dir, row)
        result = {
            "schema_version": CELL_SCHEMA,
            "status": "complete_preliminary_validated",
            "all_preliminary_gates_passed": True,
            "attempt_id": attempt_id,
            "cell_id": cell_id,
            "batch_id": str(row["batch_id"]),
            "variant_id": str(row["variant_id"]),
            "strategy": str(row["strategy"]),
            "anchor_id": str(row["anchor_id"]),
            "accepted_comparator_cell_id": str(row["accepted_comparator_cell_id"]),
            "runner_pid": runner_pid,
            "elapsed_seconds": time.time() - started,
            "runner_sha256": sha256_file(frozen.runner),
            "outputs": outputs,
            "c2_runtime_contract": c2_runtime,
            "command": command,
        }
        write_json_new(cell_dir / "CELL_RESULT.json", result)
        append_jsonl(
            status_path,
            {"schema_version": STATUS_SCHEMA, "event": "cell_finished", **result},
            status_lock,
        )
        return result
    except Exception as exc:
        failure = {
            "schema_version": CELL_SCHEMA,
            "status": "failed",
            "all_preliminary_gates_passed": False,
            "attempt_id": attempt_id,
            "cell_id": cell_id,
            "batch_id": str(row["batch_id"]),
            "variant_id": str(row["variant_id"]),
            "strategy": str(row["strategy"]),
            "anchor_id": str(row["anchor_id"]),
            "runner_pid": runner_pid,
            "elapsed_seconds": time.time() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "command": command,
        }
        if cell_dir.is_dir():
            try:
                write_json_new(cell_dir / "CELL_FAILURE.json", failure)
            except FileExistsError:
                pass
        append_jsonl(status_path, {"schema_version": STATUS_SCHEMA, "event": "cell_failed", **failure}, status_lock)
        return failure


def assert_execution_exclusive(runner_name: str) -> None:
    completed = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,command="], check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise BatchContractError("Cannot establish execution exclusivity")
    conflicts: list[str] = []
    for line in completed.stdout.splitlines():
        match = re.match(r"\s*(\d+)\s+(\d+)\s+(.*)", line)
        if not match or int(match.group(1)) in {os.getpid(), os.getppid()}:
            continue
        command = match.group(3)
        if (
            ("run_rev01_batch.py" in command and "--execute" in command)
            or runner_name in command
            or "/EnergyPlus-25-1-0/energyplus" in command
        ):
            conflicts.append(command)
    if conflicts:
        raise BatchContractError("Another relevant execution is active: " + " | ".join(conflicts))


def write_cell_index(records: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to replace cell index: {path}")
    columns = [
        "cell_id", "batch_id", "variant_id", "strategy", "anchor_id", "status",
        "elapsed_seconds", "trace_path", "trace_sha256", "summary_path", "summary_sha256",
    ]
    rows = [
        {
            "cell_id": record["cell_id"], "batch_id": record["batch_id"],
            "variant_id": record["variant_id"], "strategy": record["strategy"],
            "anchor_id": record["anchor_id"], "status": record["status"],
            "elapsed_seconds": record["elapsed_seconds"],
            "trace_path": record.get("outputs", {}).get("trace", {}).get("path"),
            "trace_sha256": record.get("outputs", {}).get("trace", {}).get("sha256"),
            "summary_path": record.get("outputs", {}).get("summary", {}).get("path"),
            "summary_sha256": record.get("outputs", {}).get("summary", {}).get("sha256"),
        }
        for record in records
    ]
    frame = pd.DataFrame.from_records(rows, columns=columns).sort_values("cell_id").reset_index(drop=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    pq.write_table(
        table, temporary, version="2.6", compression="zstd", compression_level=3,
        row_group_size=65_536, write_page_checksum=True,
    )
    restored = pq.read_table(temporary, page_checksum_verification=True).to_pandas()
    pd.testing.assert_frame_equal(frame, restored, check_dtype=True, check_exact=True)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "rows": len(frame)}


def verify_frozen_integrity(root: Path, record: dict[str, Any]) -> bool:
    declared = record.get("frozen_file_sha256")
    if not isinstance(declared, dict) or not declared:
        return False
    observed_paths = {
        path.relative_to(root).as_posix()
        for path in (root / "freeze").rglob("*")
        if path.is_file()
    }
    if observed_paths != set(declared):
        return False
    return all(
        sha256_file(root / relative) == expected
        for relative, expected in declared.items()
    )


def verify_accepted_snapshot(snapshot: dict[str, Any]) -> bool:
    if not snapshot:
        return False
    for evidence in snapshot.values():
        if not isinstance(evidence, dict):
            return False
        for identity in evidence.values():
            path = Path(str(identity.get("path", "")))
            if not path.is_file():
                return False
            if path.stat().st_size != int(identity.get("bytes", -1)):
                return False
            if sha256_file(path) != identity.get("sha256"):
                return False
    return True


def verify_command_contracts(
    records: list[dict[str, Any]], frozen: FrozenInputs
) -> dict[str, bool]:
    one_strategy = True
    zero_forbidden = True
    shared_model = True
    exact_runner = True
    for record in records:
        command = list(record.get("command") or [])
        if command.count("--strategies") != 1:
            one_strategy = False
        else:
            position = command.index("--strategies")
            one_strategy = one_strategy and position + 1 < len(command)
        zero_forbidden = zero_forbidden and not BANNED_FLAGS.intersection(command)
        shared_model = shared_model and command.count("--model-path") == 1
        if "--model-path" in command:
            position = command.index("--model-path")
            shared_model = shared_model and position + 1 < len(command)
            if position + 1 < len(command):
                shared_model = shared_model and Path(command[position + 1]).resolve() == frozen.model
        exact_runner = exact_runner and len(command) >= 2 and Path(command[1]).resolve() == frozen.runner
    return {
        "one_strategy_per_subprocess": one_strategy,
        "zero_resume_reuse_splice_purge": zero_forbidden,
        "shared_model_once_per_batch": shared_model,
        "exact_frozen_runner_per_subprocess": exact_runner,
    }


def execute(args: argparse.Namespace, preflight: Preflight) -> int:
    assert_execution_exclusive(args.runner_source.name)
    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    try:
        frozen = freeze_inputs(args, preflight)
    except Exception as exc:
        if args.output_root.is_dir():
            write_json_new(
                args.output_root / "BATCH_FAILURE.json",
                {"schema_version": BATCH_SCHEMA, "status": "FAIL", "stage": "freeze", "error": str(exc)},
            )
        raise
    status_path = args.output_root / "CELL_STATUS.jsonl"
    attempt_path = args.output_root / "ATTEMPTS.jsonl"
    append_jsonl(
        attempt_path,
        {
            "attempt_id": attempt_id, "created_utc": utc_now(), "event": "batch_started",
            "batch_id": str(preflight.matrix["batch_id"].iloc[0]),
            "planned_cells": len(preflight.matrix), "workers": args.workers,
        },
    )
    validator = load_module(
        frozen.accepted_validator, "paperb_enb_rev01_accepted_validator",
        expected_sha256=EXPECTED_ACCEPTED_VALIDATOR_SHA256,
    )
    status_lock = threading.Lock()
    records: list[dict[str, Any]] = []
    rows = [row for _index, row in preflight.matrix.sort_values("cell_id").iterrows()]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_cell, row, args=args, frozen=frozen, validator=validator,
                attempt_id=attempt_id, status_path=status_path, status_lock=status_lock,
            ): str(row["cell_id"])
            for row in rows
        }
        for future in as_completed(futures):
            records.append(future.result())
    passed = [record for record in records if record.get("all_preliminary_gates_passed") is True]
    failed = [record for record in records if record.get("all_preliminary_gates_passed") is not True]
    if failed or len(passed) != len(preflight.matrix):
        failure = {
            "schema_version": BATCH_SCHEMA, "status": "FAIL", "stage": "preliminary_cells",
            "attempt_id": attempt_id, "planned_cells": len(preflight.matrix),
            "passed_cells": len(passed), "failed_cells": len(failed),
            "failed_cell_ids": sorted(record["cell_id"] for record in failed),
            "source_csvs_removed": 0,
        }
        write_json_new(args.output_root / "BATCH_FAILURE.json", failure)
        return 1

    storage = load_module(
        frozen.storage_module, "paperb_enb_rev01_storage_runtime",
        expected_sha256=preflight.storage_sha256,
    )
    storage_dir = args.output_root / "storage_migration"
    try:
        stage = storage.stage_conversions(args.output_root, storage_dir)
        final_storage = storage.finalize_conversions(
            args.output_root, storage_dir, remove_source_csvs=True
        )
    except Exception as exc:
        write_json_new(
            args.output_root / "BATCH_FAILURE.json",
            {
                "schema_version": BATCH_SCHEMA, "status": "FAIL", "stage": "storage",
                "attempt_id": attempt_id, "planned_cells": len(preflight.matrix),
                "passed_preliminary_cells": len(passed), "error_type": type(exc).__name__,
                "error": str(exc), "source_csv_removal_may_be_partial": True,
            },
        )
        return 1
    final_failures: list[str] = []
    final_records: list[dict[str, Any]] = []
    by_cell = {record["cell_id"]: record for record in passed}
    for _index, row in preflight.matrix.sort_values("cell_id").iterrows():
        cell_id = str(row["cell_id"])
        cell_dir = args.output_root / "cells" / cell_id
        weather = frozen.weather_by_source[str(Path(str(row["weather_path"])).resolve())]
        try:
            outputs = validator.validate_cell_outputs(cell_dir, validator_job(validator, row, weather))
            c2_runtime = validate_c2_runtime_contract(cell_dir, row)
            record = dict(by_cell[cell_id])
            record["status"] = "complete_final_validated"
            record["all_final_gates_passed"] = True
            record["outputs"] = outputs
            record["c2_runtime_contract"] = c2_runtime
            final_records.append(record)
        except Exception as exc:
            final_failures.append(f"{cell_id}:{type(exc).__name__}:{exc}")
    index = write_cell_index(final_records, args.output_root / "CELL_INDEX.parquet")
    exact_ids = set(preflight.matrix["cell_id"].astype(str)) == {
        record["cell_id"] for record in final_records
    }
    command_gates = verify_command_contracts(final_records, frozen)
    frozen_integrity = verify_frozen_integrity(
        args.output_root, frozen.freeze_record
    )
    accepted_unchanged = verify_accepted_snapshot(
        preflight.accepted_comparator_snapshot
    )
    batch_id = str(preflight.matrix["batch_id"].iloc[0])
    source_pin_map = {
            args.plan_closure: EXPECTED_PLAN_CLOSURE_SHA256_BY_BATCH[batch_id],
            args.matrix: EXPECTED_MATRIX_SHA256[batch_id],
            args.variant_manifest: EXPECTED_VARIANT_MANIFEST_SHA256_BY_BATCH[batch_id],
            args.runner_source: EXPECTED_RUNNER_SHA256,
            args.runner_helper: EXPECTED_RUNNER_HELPER_SHA256,
            args.runner_closure: EXPECTED_RUNNER_CLOSURE_SHA256,
            args.storage_module: EXPECTED_STORAGE_MODULE_SHA256,
            args.storage_mapping: EXPECTED_STORAGE_MAPPING_SHA256,
            args.accepted_validator: EXPECTED_ACCEPTED_VALIDATOR_SHA256,
            args.model: EXPECTED_MODEL_SHA256,
            args.model_metrics: EXPECTED_MODEL_METRICS_SHA256,
            args.training_data: EXPECTED_TRAINING_DATA_SHA256,
            args.idf: EXPECTED_IDF_SHA256,
    }
    if batch_id.startswith("REV01-C2-012-"):
        for _index, row in preflight.matrix.drop_duplicates("source_idf_path").iterrows():
            source_pin_map[Path(str(row["source_idf_path"])).resolve()] = str(row["source_idf_sha256"])
    source_pins_unchanged = all(
        path.is_file() and sha256_file(path) == expected
        for path, expected in source_pin_map.items()
    )
    gates = {
        "all_cells_fresh_preliminary_validated": len(passed) == len(preflight.matrix),
        "all_cells_final_validated": not final_failures and len(final_records) == len(preflight.matrix),
        "exact_cell_id_coverage": exact_ids,
        "maximum_four_workers": args.workers <= MAX_WORKERS,
        "storage_stage_passed": stage.get("all_gates_passed") is True,
        "storage_finalize_passed": final_storage.get("all_gates_passed") is True,
        "frozen_inputs_unchanged_at_close": frozen_integrity,
        "authorized_sources_unchanged_at_close": source_pins_unchanged,
        "accepted_comparator_evidence_unchanged": accepted_unchanged,
        "c2_runtime_contracts_validated": (
            not batch_id.startswith("REV01-C2-012-")
            or all(
                record.get("c2_runtime_contract", {}).get("all_gates_passed") is True
                for record in final_records
            )
        ),
        "parity_authorization_present_when_required": (
            str(preflight.matrix["batch_id"].iloc[0]) == "rev01_parity1"
            or preflight.parity_closure is not None
        ),
        **command_gates,
    }
    closure = {
        "schema_version": BATCH_SCHEMA,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "all_gates_passed": all(gates.values()),
        "created_utc": utc_now(), "attempt_id": attempt_id,
        "batch_id": str(preflight.matrix["batch_id"].iloc[0]),
        "planned_cells": len(preflight.matrix), "passed_cells": len(final_records),
        "failed_cells": len(final_failures), "workers": args.workers,
        "runner_sha256": preflight.runner_sha256,
        "runner_helper_sha256": preflight.runner_helper_sha256,
        "runner_closure_sha256": sha256_file(args.runner_closure),
        "launcher_sha256": sha256_file(Path(__file__).resolve()),
        "storage_module_sha256": preflight.storage_sha256,
        "matrix_sha256": sha256_file(args.matrix),
        "variant_manifest_sha256": sha256_file(args.variant_manifest),
        "freeze_sha256": sha256_file(args.output_root / "FREEZE.json"),
        "status_sha256": sha256_file(status_path),
        "cell_index": index,
        "storage_stage": stage,
        "storage_final": final_storage,
        "storage_mapping_sha256": preflight.storage_mapping_sha256,
        "parity_closure_sha256": preflight.parity_closure_sha256,
        "closure_gates": gates,
        "final_validation_failures": final_failures,
        "accepted_outputs_modified": not accepted_unchanged,
    }
    append_jsonl(
        attempt_path,
        {"attempt_id": attempt_id, "created_utc": utc_now(), "event": "batch_finished", **closure},
    )
    closure["attempts_sha256"] = sha256_file(attempt_path)
    write_json_new(args.output_root / "BATCH_CLOSURE.json", closure)
    return 0 if closure["all_gates_passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-closure", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--variant-manifest", type=Path, required=True)
    parser.add_argument("--runner-source", type=Path, required=True)
    parser.add_argument("--runner-helper", type=Path, required=True)
    parser.add_argument("--runner-closure", type=Path, required=True)
    parser.add_argument("--storage-module", type=Path, default=DEFAULT_STORAGE_MODULE)
    parser.add_argument("--storage-mapping", type=Path, default=DEFAULT_STORAGE_MAPPING)
    parser.add_argument("--parity-closure", type=Path, default=None)
    parser.add_argument("--accepted-validator", type=Path, default=DEFAULT_VALIDATOR)
    parser.add_argument("--python", type=Path, default=EXPECTED_PYTHON)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-metrics", type=Path, default=DEFAULT_MODEL_METRICS)
    parser.add_argument("--training-data", type=Path, default=DEFAULT_TRAINING_DATA)
    parser.add_argument("--idf", type=Path, default=DEFAULT_IDF)
    parser.add_argument("--eplus-root", type=Path, default=DEFAULT_EPLUS_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.workers <= MAX_WORKERS:
        raise BatchContractError(f"workers must be in [1,{MAX_WORKERS}]")
    preflight = validate_preflight(args)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "schema_version": BATCH_SCHEMA,
                    "status": "PASS", "filesystem_mutated": False,
                    "subprocesses_started": 0,
                    "batch_id": str(preflight.matrix["batch_id"].iloc[0]),
                    "planned_cells": len(preflight.matrix),
                    "runner_sha256": preflight.runner_sha256,
                    "runner_helper_sha256": preflight.runner_helper_sha256,
                    "storage_module_sha256": preflight.storage_sha256,
                    "storage_mapping_sha256": preflight.storage_mapping_sha256,
                    "parity_closure_sha256": preflight.parity_closure_sha256,
                    "matrix_sha256": sha256_file(args.matrix),
                    "variant_manifest_sha256": sha256_file(args.variant_manifest),
                    "weather_files": preflight.matrix["weather_path"].nunique(),
                },
                indent=2, sort_keys=True,
            )
        )
        return 0
    return execute(args, preflight)


if __name__ == "__main__":
    raise SystemExit(main())
