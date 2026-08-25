#!/usr/bin/env python3
"""Launch the fresh, environment-scoped Paper B ENB 24 x 4 production matrix.

This launcher deliberately treats the existing 96-row matrix as authorization
evidence, not as an instruction to reuse pilot or partial-production outputs.
It revalidates the original authorization through the frozen v1 launcher, then
creates an execution copy in which all 96 cells are fresh.  Every cell is run
by a separate Python subprocess with exactly one weather file and one strategy.

``--preflight-only`` is fully non-mutating and never starts a subprocess.
``--execute`` requires the one predeclared, absent output root.  No resume,
splice, reuse, purge, overwrite, or cleanup path exists in this program.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from types import ModuleType
from typing import Any, Iterable
import uuid

# The read-only preflight dynamically imports the pinned v1 validator.  Never
# let that import create or refresh a source-tree bytecode cache.
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd


# These constants are intentionally centralized.  The smoke and affected-24
# producers must use these exact schemas/field names or this launcher must be
# deliberately patched and retested before production.
EXPECTED_SMOKE_CLOSURE_SCHEMA_VERSION = (
    "paperb_enb_production_warmup_python_smoke_v4_closure_v1"
)
SMOKE_CLOSURE_PASS_FIELD = "all_gates_passed"
SMOKE_CLOSURE_STATUS_FIELD = "closure_status"
SMOKE_CLOSURE_STATUS_VALUE = "complete"
EXPECTED_AFFECTED24_CLOSURE_SCHEMA_VERSION = (
    "paperb_enb_production_warmup_affected24_closure_v2"
)
AFFECTED24_CLOSURE_PASS_FIELD = "all_gates_passed"
AFFECTED24_CLOSURE_STATUS_FIELD = "closure_status"
AFFECTED24_CLOSURE_STATUS_VALUE = "PASS"
AFFECTED24_REQUIRED_FIELDS = {
    "base_runner_sha256": (
        "f8d544e01fc3e997fd75234504b4077ca023ef026ef31513910e6d299056eebc"
    ),
    "execution_contract": "production_warmup_affected24_v2_fresh_process_exact24",
    "target_anchors": 6,
    "target_cells": 24,
    "cells_recorded": 24,
    "cells_passed": 24,
    "cells_failed": 0,
    "exact_unique_key_coverage": True,
    "one_strategy_per_fresh_subprocess": True,
    "workers": 4,
    "max_workers": 4,
    "reuse_cells": 0,
    "resume_used": False,
    "purge_used": False,
    "annual_rows_each": 35_040,
    "occupied_rows_each": 16_704,
    "generated_idf_contract_identical": True,
    "frozen_hashes_passed": True,
    "all_environment_audits_passed": True,
    "all_strict_energyplus_gates_passed": True,
    "all_annual_trace_contracts_passed": True,
    "all_summary_contracts_passed": True,
    "all_generated_idf_contracts_passed": True,
    "source_hashes_passed": True,
}

EXPECTED_PRODUCTION_RUNNER_SHA256 = (
    "a9693042883ed5c20edad6fcbc757c62c7216d5abc120ad18de1a003932848a4"
)
EXPECTED_AUTHORIZATION_RUNNER_SHA256 = (
    "f8d544e01fc3e997fd75234504b4077ca023ef026ef31513910e6d299056eebc"
)
EXPECTED_BASE_LAUNCHER_SHA256 = (
    "1a575e5a8703c149fc92d677bbf83f53bd7917b6ba1e5057aa588d455a3e691c"
)
EXPECTED_SMOKE_PRODUCER_SHA256 = (
    "7de650e22ba20c6f2e2f3ae7ebce878488e08db38f78e5f40e10166c7846e3bb"
)
EXPECTED_AFFECTED24_PRODUCER_SHA256 = (
    "b0240463cc62f5603356999e94db780ddc1afd5674ce4aa241171e5b66d03b12"
)
EXPECTED_SMOKE_CLOSURE_SHA256 = (
    "8a8562cd5ab98b201f62b6303d00ba439433936d157f433bfb1d3102ae3315be"
)
EXPECTED_AFFECTED24_CLOSURE_SHA256 = (
    "1bb437737e9b555f945815af81c4c7152b6711dfd523558d50569126f2e2ae04"
)
EXPECTED_MATRIX_SHA256 = (
    "d673fd008a219d3ce19d9e6d350b34f44da4b88f919ce6b12831270abfdde960"
)
EXPECTED_MATRIX_MANIFEST_SHA256 = (
    "1ee6d9e7aeb3f778990c283b6a6f32b617d5c05dccf73a6dd1e211a441bcce78"
)
EXPECTED_ANCHOR_CATALOG_SHA256 = (
    "6e65c668c40ff3e9aa2fec4ef087a63d3d0bc8c1035c188d119a06fb9b15b96b"
)
EXPECTED_WEATHER_MAP_SHA256 = (
    "7d92c4ace313afe8a620e217e2dcec0b23fd8e9510f3f5e33a8aeebb7b7fe655"
)
EXPECTED_PHYSICAL_MAP_SHA256 = (
    "ca873bd0158c9ef28363f770c428b918748a43fa7f63a082fcbe6688a1e16f06"
)
EXPECTED_MECHANISM_REPORT_SHA256 = (
    "6d622436eb8272be5247420607e7ed2c46e80dcba1f1bed96d83f93c6553cf61"
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
EXPECTED_SOURCE_IDF_SHA256 = (
    "1144b58b848992d1730e49ef9c252569e3a515d82a2f99b2c0233352f625a7e4"
)
EXPECTED_ENVIRONMENT_RECORD_SHA256 = (
    "c535d4f21aa29d0da44980861e3b20a6c2317e4fe87f6c8b9e876d8eb6140bd7"
)
EXPECTED_PILOT_DECISION_SHA256 = (
    "294f1f747944cff0c495dadc5c7639ebd44bd664c7d5b9d5199ac6dd7a4365c2"
)
EXPECTED_COMPLETION_INDEX_SHA256 = (
    "0ceeec0915891982271b32e6578f47709cee5d59c75d56fa7ff9cf6b2a471ada"
)
SMOKE_REQUIRED_FIELDS = {
    "base_runner_sha256": EXPECTED_AUTHORIZATION_RUNNER_SHA256,
    "execution_contract": "production_warmup_python_smoke_v4_fresh_process_13",
    "configured_workers": 4,
    "maximum_concurrent_subprocesses": 4,
    "planned_jobs": 13,
    "job_records": 13,
    "passed_jobs": 13,
    "failed_jobs": 0,
    "exact_job_key_coverage": True,
    "one_strategy_per_subprocess": True,
    "all_environment_audits_passed": True,
    "all_strict_energyplus_gates_passed": True,
    "all_generated_idf_contracts_passed": True,
    "full_year_trace_rows": 35_040,
    "short_trace_count": 12,
    "short_trace_rows_each": 672,
    "canonical_strategy_comparisons_expected": 4,
    "canonical_strategy_comparisons_passed": 4,
    "sql_sizing_fingerprint_tables": 3,
    "sql_sizing_weather_groups_expected": 2,
    "sql_sizing_weather_groups_observed": 2,
    "sql_sizing_weather_group_counts": {"houston": 1, "guangzhou": 12},
    "sql_sizing_weather_groups_exact": True,
    "sql_sizing_within_weather_exact": True,
    "guangzhou_sql_sizing_fingerprints_exact_across_12": True,
    "houston_sql_sizing_fingerprints_independently_valid": True,
    "cross_weather_sql_sizing_difference_expected": True,
    "cross_weather_sql_sizing_difference_observed": True,
    "short_generated_idf_sha256_global_exact": True,
}

EXPECTED_CELLS = 96
EXPECTED_ANCHORS = 24
EXPECTED_RAW_CMIP_CSVS = 12
EXPECTED_SELECTED_EPWS = 24
EXPECTED_TRACE_ROWS = 35_040
EXPECTED_OCCUPIED_ROWS = 16_704
MAX_CONCURRENT_CELLS = 4
WEATHER_FILE_RUN_PERIOD_KIND = 3
MAX_WARMUP_DAYS = 50
MIN_WARMUP_DAYS = 6
WARMUP_SETPOINT_CONTRACT = "production_occupancy_baseline_22_24_else_12_30"
ENVIRONMENT_AUDIT_SCHEMA = "paperb_enb_environment_scoped_control_audit_v1"
EXECUTION_MATRIX_SCHEMA = "paperb_enb_fresh96_execution_matrix_v2_envscoped"
FREEZE_SCHEMA = "paperb_enb_fresh96_execution_freeze_v2_envscoped"
CLOSURE_SCHEMA = "paperb_enb_fresh96_closure_v2_envscoped"
CELL_STATUS_SCHEMA = "paperb_enb_fresh96_cell_status_v2_envscoped"
CELL_RESULT_SCHEMA = "paperb_enb_fresh96_cell_result_v2_envscoped"

CORE_STRATEGIES = (
    "diagnostic_reference",
    "paperb_pmv_relax",
    "paperb_adaptive_band_relax",
    "paperb_p90_tail_asym_relax",
)
BANNED_RUNNER_FLAGS = frozenset(
    {
        "--resume",
        "--purge-energyplus-after-trace",
        "--purge-case-traces-after-summary",
    }
)

_WORKSPACE = Path(__file__).resolve().parents[4]
EXPECTED_PYTHON_INVOCATION = _WORKSPACE / "paperB_ENB_Rebuild" / ".venv" / "bin" / "python"
SMOKE_PRODUCER_PATH = (
    _WORKSPACE
    / "paperB_ENB_Rebuild"
    / "04_validation"
    / "scaled_design"
    / "warmup_diagnostic"
    / "run_production_warmup_python_smoke_v4.py"
)
AFFECTED24_PRODUCER_PATH = (
    _WORKSPACE
    / "paperB_ENB_Rebuild"
    / "04_validation"
    / "scaled_design"
    / "warmup_diagnostic"
    / "run_production_warmup_affected24_v2.py"
)
EXPECTED_SMOKE_CLOSURE_PATH = (
    _WORKSPACE
    / "paperB_ENB_Rebuild"
    / "06_runs"
    / "diagnostics"
    / "20260803_production_warmup_python_smoke_v4"
    / "SMOKE_CLOSURE.json"
)
EXPECTED_AFFECTED24_CLOSURE_PATH = (
    _WORKSPACE
    / "paperB_ENB_Rebuild"
    / "06_runs"
    / "diagnostics"
    / "20260803_production_warmup_affected24_v2"
    / "AFFECTED24_CLOSURE.json"
)
EXPECTED_OUTPUT_ROOT = (
    _WORKSPACE
    / "paperB_ENB_Rebuild"
    / "06_runs"
    / "scaled"
    / "20260803_routeS96_v2_envscoped"
)
PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "PRODUCTION_ENVSCOPED_FRESH96_PROTOCOL_20260803_v1.md"
)


class ContractError(RuntimeError):
    """A prerequisite, execution, or output contract failed closed."""


@dataclass(frozen=True)
class CellJob:
    scaled_run_id: str
    anchor_id: str
    anchor_order: int
    physical_condition_id: str
    selector_label_id: str
    scenario: str
    city: str
    strategy: str
    strategy_order: int
    weather_source: Path
    weather_sha256: str
    weather_data_sha256: str
    weather_stem: str

    @property
    def key(self) -> tuple[str, str]:
        return self.physical_condition_id, self.strategy


@dataclass(frozen=True)
class PreflightResult:
    source_matrix: pd.DataFrame
    execution_matrix: pd.DataFrame
    jobs: tuple[CellJob, ...]
    authorization_artifacts: Any
    production_artifacts: Any
    matrix_manifest: dict[str, Any]
    weather_audit: dict[str, Any]
    smoke_closure: dict[str, Any]
    affected24_closure: dict[str, Any]
    input_hashes: dict[str, str]


@dataclass(frozen=True)
class FrozenArtifacts:
    runner: Path
    model: Path
    model_metrics: Path
    source_idf: Path
    weather_by_source: dict[str, Path]
    freeze_record: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_epw_payload(path: Path) -> str:
    """Hash EPW data rows after the eight role-specific header lines."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for _ in range(8):
            if handle.readline() == b"":
                raise ContractError(f"EPW has fewer than eight header rows: {path}")
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"Cannot read {name} as JSON ({path}): {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be a JSON object: {path}")
    return value


def require_columns(frame: pd.DataFrame, columns: Iterable[str], *, name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ContractError(f"{name} lacks required columns: {missing}")


def csv_boolean(value: Any, *, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ContractError(f"{field} is not an explicit CSV boolean: {value!r}")


def validate_workers(workers: int) -> int:
    if not 1 <= workers <= MAX_CONCURRENT_CELLS:
        raise ContractError(
            f"workers must be in [1, {MAX_CONCURRENT_CELLS}], got {workers}."
        )
    return workers


def absolute_without_resolving(path: Path) -> Path:
    """Make a path absolute while preserving its final symlink invocation."""

    return Path(os.path.abspath(os.fspath(path)))


def validate_python_invocation(path: Path) -> Path:
    invocation = absolute_without_resolving(path)
    if invocation != EXPECTED_PYTHON_INVOCATION:
        raise ContractError(
            "Production must be invoked through the project virtual environment symlink: "
            f"expected {EXPECTED_PYTHON_INVOCATION}, got {invocation}."
        )
    if not invocation.is_symlink():
        raise ContractError(f"Expected venv Python invocation is not a symlink: {invocation}")
    return invocation


def validate_output_root(path: Path) -> None:
    if path.resolve() != EXPECTED_OUTPUT_ROOT.resolve():
        raise ContractError(
            "The v2 production root is predeclared and immutable: "
            f"expected {EXPECTED_OUTPUT_ROOT}, got {path.resolve()}."
        )
    if path.exists() or path.is_symlink():
        raise FileExistsError(
            f"Fresh96 output root already exists; no overwrite/resume is allowed: {path}"
        )


def validate_upstream_closure(
    path: Path,
    *,
    name: str,
    expected_schema: str,
    pass_field: str,
    status_field: str | None = None,
    status_value: Any = None,
    required_fields: dict[str, Any] | None = None,
    expected_path: Path | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Require semantic closure, not mere file presence or truthiness."""

    if expected_path is not None and path.resolve() != expected_path.resolve():
        raise ContractError(
            f"{name} must come from {expected_path}, observed {path.resolve()}."
        )
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ContractError(
            f"{name} checksum mismatch: expected {expected_sha256}, observed {digest}."
        )
    value = read_json(path, name=name)
    if value.get("schema_version") != expected_schema:
        raise ContractError(
            f"{name} schema mismatch: expected {expected_schema!r}, "
            f"observed {value.get('schema_version')!r}."
        )
    if pass_field not in value or type(value[pass_field]) is not bool:
        raise ContractError(f"{name}.{pass_field} must be an explicit JSON boolean.")
    if value[pass_field] is not True:
        raise ContractError(f"{name} is not passing ({pass_field}=false).")
    if status_field is not None and value.get(status_field) != status_value:
        raise ContractError(
            f"{name}.{status_field} mismatch: expected {status_value!r}, "
            f"observed {value.get(status_field)!r}."
        )
    if value.get("runner_sha256") != EXPECTED_PRODUCTION_RUNNER_SHA256:
        raise ContractError(
            f"{name} runner_sha256 is not the production runner hash."
        )
    for field, expected in (required_fields or {}).items():
        observed = value.get(field)
        if isinstance(expected, bool):
            matches = type(observed) is bool and observed is expected
        elif isinstance(expected, int):
            matches = type(observed) is int and observed == expected
        else:
            matches = observed == expected
        if not matches:
            raise ContractError(
                f"{name}.{field} mismatch: expected {expected!r}, observed {observed!r}."
            )
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "schema_version": expected_schema,
        "pass_field": pass_field,
        "pass_value": True,
        "status_field": status_field,
        "status_value": status_value,
        "runner_sha256": EXPECTED_PRODUCTION_RUNNER_SHA256,
        "document": value,
    }


def load_base_launcher(path: Path) -> ModuleType:
    if sha256_file(path) != EXPECTED_BASE_LAUNCHER_SHA256:
        raise ContractError(f"Base launcher checksum mismatch: {path}")
    module_name = "paperb_enb_frozen_scaled_launcher_v1"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ContractError(f"Cannot import base launcher: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def validate_required_inputs(args: argparse.Namespace) -> None:
    files = (
        args.matrix,
        args.matrix_manifest,
        args.pilot_decision,
        args.completion_index,
        args.anchor_catalog,
        args.weather_map,
        args.physical_map,
        args.base_launcher,
        args.authorization_runner,
        args.production_runner,
        args.python,
        args.model,
        args.model_metrics,
        args.training_data,
        args.idf,
        args.environment_record,
        args.mechanism_report,
        args.smoke_closure,
        args.affected24_closure,
        PROTOCOL_PATH,
        SMOKE_PRODUCER_PATH,
        AFFECTED24_PRODUCER_PATH,
    )
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + ", ".join(map(str, missing)))
    if not args.cmip_root.is_dir():
        raise FileNotFoundError(args.cmip_root)
    if not args.eplus_root.is_dir():
        raise FileNotFoundError(args.eplus_root)
    if not os.access(args.python, os.X_OK):
        raise ContractError(f"Python interpreter is not executable: {args.python}")
    validate_python_invocation(args.python)


def validate_pinned_inputs(args: argparse.Namespace) -> dict[str, str]:
    contracts = {
        "matrix_sha256": (args.matrix, EXPECTED_MATRIX_SHA256),
        "matrix_manifest_sha256": (
            args.matrix_manifest,
            EXPECTED_MATRIX_MANIFEST_SHA256,
        ),
        "pilot_decision_sha256": (
            args.pilot_decision,
            EXPECTED_PILOT_DECISION_SHA256,
        ),
        "completion_index_sha256": (
            args.completion_index,
            EXPECTED_COMPLETION_INDEX_SHA256,
        ),
        "anchor_catalog_sha256": (
            args.anchor_catalog,
            EXPECTED_ANCHOR_CATALOG_SHA256,
        ),
        "weather_map_sha256": (args.weather_map, EXPECTED_WEATHER_MAP_SHA256),
        "physical_map_sha256": (args.physical_map, EXPECTED_PHYSICAL_MAP_SHA256),
        "base_launcher_sha256": (
            args.base_launcher,
            EXPECTED_BASE_LAUNCHER_SHA256,
        ),
        "authorization_runner_sha256": (
            args.authorization_runner,
            EXPECTED_AUTHORIZATION_RUNNER_SHA256,
        ),
        "production_runner_sha256": (
            args.production_runner,
            EXPECTED_PRODUCTION_RUNNER_SHA256,
        ),
        "model_sha256": (args.model, EXPECTED_MODEL_SHA256),
        "model_metrics_sha256": (
            args.model_metrics,
            EXPECTED_MODEL_METRICS_SHA256,
        ),
        "training_data_sha256": (
            args.training_data,
            EXPECTED_TRAINING_DATA_SHA256,
        ),
        "idf_sha256": (args.idf, EXPECTED_SOURCE_IDF_SHA256),
        "environment_record_sha256": (
            args.environment_record,
            EXPECTED_ENVIRONMENT_RECORD_SHA256,
        ),
        "mechanism_report_sha256": (
            args.mechanism_report,
            EXPECTED_MECHANISM_REPORT_SHA256,
        ),
        "smoke_producer_sha256": (
            SMOKE_PRODUCER_PATH,
            EXPECTED_SMOKE_PRODUCER_SHA256,
        ),
        "affected24_producer_sha256": (
            AFFECTED24_PRODUCER_PATH,
            EXPECTED_AFFECTED24_PRODUCER_SHA256,
        ),
    }
    observed: dict[str, str] = {}
    for field, (path, expected) in contracts.items():
        digest = sha256_file(path)
        observed[field] = digest
        if digest != expected:
            raise ContractError(
                f"Pinned input checksum mismatch for {path}: expected {expected}, got {digest}."
            )
    return observed


def validate_authorization(
    args: argparse.Namespace, base: ModuleType
) -> tuple[pd.DataFrame, dict[str, Any], Any, Any]:
    """Use the checksum-pinned v1 launcher to validate immutable authorization."""

    auth_args = argparse.Namespace(**vars(args))
    auth_args.runner = args.authorization_runner
    authorization_artifacts = base.validate_runtime_artifacts(auth_args)
    authorization_artifacts = replace(
        authorization_artifacts,
        python=validate_python_invocation(args.python),
    )
    manifest, matrix_hash = base.validate_matrix_manifest(
        args.matrix,
        args.matrix_manifest,
        pilot_decision=args.pilot_decision,
        completion_index=args.completion_index,
        anchor_catalog=args.anchor_catalog,
        weather_map=args.weather_map,
        physical_map=args.physical_map,
    )
    matrix, recommendation = base.validate_matrix(
        args.matrix,
        manifest,
        authorization_artifacts,
        pilot_decision=args.pilot_decision,
        completion_index=args.completion_index,
    )
    if matrix_hash != EXPECTED_MATRIX_SHA256 or recommendation != EXPECTED_CELLS:
        raise ContractError("Only the pinned authorized96 source matrix is accepted.")

    production_args = argparse.Namespace(**vars(args))
    production_args.runner = args.production_runner
    production_artifacts = base.validate_runtime_artifacts(production_args)
    production_artifacts = replace(
        production_artifacts,
        python=validate_python_invocation(args.python),
    )
    if (
        production_artifacts.hashes["runner_sha256"]
        != EXPECTED_PRODUCTION_RUNNER_SHA256
    ):
        raise ContractError("Production runtime resolved to the wrong runner.")
    return matrix, manifest, authorization_artifacts, production_artifacts


def _single_row(frame: pd.DataFrame, mask: pd.Series, *, name: str) -> pd.Series:
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise ContractError(f"Expected exactly one {name} row, observed {len(selected)}.")
    return selected.iloc[0]


def audit_weather_sources(
    *,
    matrix: pd.DataFrame,
    catalog_path: Path,
    weather_map_path: Path,
    physical_map_path: Path,
    cmip_root: Path,
) -> dict[str, Any]:
    """Rehash all 12 raw CMIP CSVs and all 24 selected EPWs before mutation."""

    catalog = pd.read_csv(catalog_path)
    weather_map = pd.read_csv(weather_map_path)
    physical_map = pd.read_csv(physical_map_path)
    required_catalog = {
        "anchor_id",
        "anchor_order",
        "selector_label_id",
        "physical_condition_id",
        "scenario",
        "city",
        "source_csv_path",
        "source_csv_sha256",
        "epw_path",
        "epw_sha256",
        "epw_data_sha256",
    }
    require_columns(catalog, required_catalog, name="anchor catalog")
    if len(catalog) != EXPECTED_ANCHORS or catalog["anchor_id"].nunique() != EXPECTED_ANCHORS:
        raise ContractError("Anchor catalog must contain exactly 24 unique anchors.")

    raw_contract = (
        catalog[["source_csv_path", "source_csv_sha256"]]
        .drop_duplicates()
        .sort_values("source_csv_path", kind="mergesort")
    )
    if len(raw_contract) != EXPECTED_RAW_CMIP_CSVS:
        raise ContractError("Anchor catalog must reference exactly 12 raw CMIP CSVs.")
    discovered = sorted(path.resolve() for path in cmip_root.rglob("*.csv") if path.is_file())
    contracted = sorted(Path(path).resolve() for path in raw_contract["source_csv_path"])
    if len(discovered) != EXPECTED_RAW_CMIP_CSVS or discovered != contracted:
        raise ContractError(
            "The raw CMIP root must contain exactly the same 12 CSVs as the catalog."
        )
    raw_records: list[dict[str, Any]] = []
    for row in raw_contract.itertuples(index=False):
        path = Path(str(row.source_csv_path)).resolve()
        digest = sha256_file(path)
        if digest != str(row.source_csv_sha256):
            raise ContractError(f"Raw CMIP checksum mismatch: {path}")
        raw_records.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(cmip_root.resolve())),
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )

    selected = catalog[
        ["anchor_id", "epw_path", "epw_sha256", "epw_data_sha256"]
    ].drop_duplicates()
    if len(selected) != EXPECTED_SELECTED_EPWS or selected["epw_path"].nunique() != EXPECTED_SELECTED_EPWS:
        raise ContractError("Anchor catalog must select exactly 24 distinct EPWs.")
    epw_records: list[dict[str, Any]] = []
    for row in selected.sort_values("anchor_id", kind="mergesort").itertuples(index=False):
        path = Path(str(row.epw_path)).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        full_digest = sha256_file(path)
        data_digest = sha256_epw_payload(path)
        if full_digest != str(row.epw_sha256) or data_digest != str(row.epw_data_sha256):
            raise ContractError(f"Selected EPW checksum mismatch: {path}")
        epw_records.append(
            {
                "anchor_id": str(row.anchor_id),
                "path": str(path),
                "sha256": full_digest,
                "data_sha256": data_digest,
                "size_bytes": path.stat().st_size,
            }
        )

    require_columns(
        weather_map,
        {
            "selector_label_id",
            "physical_condition_id",
            "source_csv_path",
            "source_csv_sha256",
            "epw_path",
            "epw_sha256",
            "epw_data_sha256",
            "mapping_pass",
            "epw_overall_pass",
            "epw_header_identity_pass",
            "epw_timestamp_pass",
            "epw_weather_fields_pass",
        },
        name="weather selector map",
    )
    require_columns(
        physical_map,
        {
            "physical_condition_id",
            "source_csv_path",
            "source_csv_sha256",
            "canonical_epw_path",
            "canonical_epw_sha256",
            "all_epw_paths",
            "all_epw_full_sha256",
            "epw_data_sha256",
            "payload_identity_within_condition_pass",
            "all_selector_mappings_pass",
        },
        name="physical weather map",
    )

    matrix_anchors = matrix.sort_values("strategy_order").drop_duplicates("anchor_id")
    if set(matrix_anchors["anchor_id"].astype(str)) != set(catalog["anchor_id"].astype(str)):
        raise ContractError("Matrix/catalog anchor identities differ.")
    cross_checks: list[dict[str, str]] = []
    for catalog_row in catalog.itertuples(index=False):
        anchor_id = str(catalog_row.anchor_id)
        matrix_row = _single_row(
            matrix_anchors,
            matrix_anchors["anchor_id"].astype(str).eq(anchor_id),
            name=f"matrix anchor {anchor_id}",
        )
        selector = str(catalog_row.selector_label_id)
        condition = str(catalog_row.physical_condition_id)
        selector_row = _single_row(
            weather_map,
            weather_map["selector_label_id"].astype(str).eq(selector),
            name=f"selector {selector}",
        )
        physical_row = _single_row(
            physical_map,
            physical_map["physical_condition_id"].astype(str).eq(condition),
            name=f"physical condition {condition}",
        )
        for frame_name, row, fields in (
            (
                "selector",
                selector_row,
                (
                    "mapping_pass",
                    "epw_overall_pass",
                    "epw_header_identity_pass",
                    "epw_timestamp_pass",
                    "epw_weather_fields_pass",
                ),
            ),
            (
                "physical",
                physical_row,
                ("payload_identity_within_condition_pass", "all_selector_mappings_pass"),
            ),
        ):
            for field in fields:
                if not csv_boolean(row[field], field=f"{frame_name}.{field}"):
                    raise ContractError(f"{frame_name}.{field} is false for {anchor_id}.")

        common_expected = {
            "physical_condition_id": condition,
            "source_csv_path": str(Path(str(catalog_row.source_csv_path)).resolve()),
            "source_csv_sha256": str(catalog_row.source_csv_sha256),
            "epw_path": str(Path(str(catalog_row.epw_path)).resolve()),
            "epw_sha256": str(catalog_row.epw_sha256),
            "epw_data_sha256": str(catalog_row.epw_data_sha256),
        }
        matrix_observed = {
            key: str(Path(str(matrix_row[key])).resolve()) if key.endswith("_path") else str(matrix_row[key])
            for key in common_expected
        }
        if matrix_observed != common_expected:
            raise ContractError(f"Matrix/catalog weather contract differs for {anchor_id}.")
        selector_observed = {
            key: str(Path(str(selector_row[key])).resolve()) if key.endswith("_path") else str(selector_row[key])
            for key in common_expected
        }
        if selector_observed != common_expected:
            raise ContractError(f"Selector/catalog weather contract differs for {anchor_id}.")
        physical_observed = {
            "physical_condition_id": str(physical_row["physical_condition_id"]),
            "source_csv_path": str(Path(str(physical_row["source_csv_path"])).resolve()),
            "source_csv_sha256": str(physical_row["source_csv_sha256"]),
            "epw_data_sha256": str(physical_row["epw_data_sha256"]),
        }
        physical_expected = {
            key: common_expected[key]
            for key in (
                "physical_condition_id",
                "source_csv_path",
                "source_csv_sha256",
                "epw_data_sha256",
            )
        }
        if physical_observed != physical_expected:
            raise ContractError(f"Physical-map/catalog weather contract differs for {anchor_id}.")
        condition_paths = [
            str(Path(value).resolve())
            for value in str(physical_row["all_epw_paths"]).split(";")
            if value
        ]
        condition_hashes = [
            value
            for value in str(physical_row["all_epw_full_sha256"]).split(";")
            if value
        ]
        if len(condition_paths) != len(condition_hashes) or (
            common_expected["epw_path"], common_expected["epw_sha256"]
        ) not in set(zip(condition_paths, condition_hashes)):
            raise ContractError(
                f"Selected catalog EPW is not a declared member of physical condition {condition}."
            )
        cross_checks.append(
            {
                "anchor_id": anchor_id,
                "selector_label_id": selector,
                "physical_condition_id": condition,
            }
        )

    return {
        "schema_version": "paperb_enb_fresh96_weather_source_audit_v1",
        "raw_cmip_root": str(cmip_root.resolve()),
        "raw_cmip_csv_count": len(raw_records),
        "raw_cmip_csvs": raw_records,
        "selected_epw_count": len(epw_records),
        "selected_epws": epw_records,
        "anchor_cross_check_count": len(cross_checks),
        "anchor_cross_checks": cross_checks,
        "all_weather_gates_passed": True,
    }


def transform_fresh96_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    frame = matrix.copy(deep=True)
    if len(frame) != EXPECTED_CELLS or set(frame["strategy"].astype(str)) != set(CORE_STRATEGIES):
        raise ContractError("Fresh execution requires the balanced 24 x 4 core matrix.")
    for column in (
        "execution_action",
        "run_authorized",
        "runner_path",
        "runner_sha256",
        "reused_summary_path",
        "reused_summary_sha256",
        "reused_trace_path",
        "reused_trace_sha256",
    ):
        if column in frame:
            frame[f"source_authorization_{column}"] = frame[column]
    frame["execution_matrix_schema_version"] = EXECUTION_MATRIX_SCHEMA
    frame["source_authorization_matrix_sha256"] = EXPECTED_MATRIX_SHA256
    frame["execution_action"] = "run_new_fresh"
    frame["run_authorized"] = True
    frame["runner_path"] = "FROZEN_AT_EXECUTION"
    frame["runner_sha256"] = EXPECTED_PRODUCTION_RUNNER_SHA256
    frame["reuse_or_splice_permitted"] = False
    frame["resume_permitted"] = False
    frame["purge_permitted"] = False
    for column in (
        "reused_summary_path",
        "reused_summary_sha256",
        "reused_trace_path",
        "reused_trace_sha256",
    ):
        if column in frame:
            frame[column] = ""
    if frame["scaled_run_id"].duplicated().any() or frame.duplicated(
        ["physical_condition_id", "strategy"]
    ).any():
        raise ContractError("Fresh execution keys are not unique.")
    if frame["physical_condition_id"].nunique() != EXPECTED_ANCHORS:
        raise ContractError("Fresh matrix does not contain 24 physical anchors.")
    if not frame.groupby("physical_condition_id").size().eq(4).all():
        raise ContractError("Fresh matrix is not balanced at four strategies per anchor.")
    return frame.sort_values(
        ["anchor_order", "strategy_order"], kind="mergesort"
    ).reset_index(drop=True)


def build_cell_jobs(frame: pd.DataFrame) -> tuple[CellJob, ...]:
    jobs: list[CellJob] = []
    for row in frame.itertuples(index=False):
        weather = Path(str(row.epw_path)).resolve()
        jobs.append(
            CellJob(
                scaled_run_id=str(row.scaled_run_id),
                anchor_id=str(row.anchor_id),
                anchor_order=int(row.anchor_order),
                physical_condition_id=str(row.physical_condition_id),
                selector_label_id=str(row.selector_label_id),
                scenario=str(row.scenario),
                city=str(row.city),
                strategy=str(row.strategy),
                strategy_order=int(row.strategy_order),
                weather_source=weather,
                weather_sha256=str(row.epw_sha256),
                weather_data_sha256=str(row.epw_data_sha256),
                weather_stem=weather.stem,
            )
        )
    expected_keys = set(
        frame[["physical_condition_id", "strategy"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    if (
        len(jobs) != EXPECTED_CELLS
        or len({job.scaled_run_id for job in jobs}) != EXPECTED_CELLS
        or {job.key for job in jobs} != expected_keys
        or len({job.key for job in jobs}) != EXPECTED_CELLS
    ):
        raise ContractError("Cell plan does not exactly cover the fresh 96-cell matrix.")
    if any(job.strategy not in CORE_STRATEGIES for job in jobs):
        raise ContractError("Cell plan contains a non-core strategy.")
    return tuple(jobs)


def preflight(args: argparse.Namespace) -> PreflightResult:
    """Perform every expensive/read-only check without touching the output root."""

    validate_workers(args.workers)
    validate_output_root(args.output_root)
    validate_required_inputs(args)
    input_hashes = validate_pinned_inputs(args)
    base = load_base_launcher(args.base_launcher)
    matrix, manifest, auth_artifacts, prod_artifacts = validate_authorization(args, base)
    smoke = validate_upstream_closure(
        args.smoke_closure,
        name="production smoke closure",
        expected_schema=EXPECTED_SMOKE_CLOSURE_SCHEMA_VERSION,
        pass_field=SMOKE_CLOSURE_PASS_FIELD,
        status_field=SMOKE_CLOSURE_STATUS_FIELD,
        status_value=SMOKE_CLOSURE_STATUS_VALUE,
        required_fields=SMOKE_REQUIRED_FIELDS,
        expected_path=EXPECTED_SMOKE_CLOSURE_PATH,
        expected_sha256=EXPECTED_SMOKE_CLOSURE_SHA256,
    )
    affected = validate_upstream_closure(
        args.affected24_closure,
        name="affected24 closure",
        expected_schema=EXPECTED_AFFECTED24_CLOSURE_SCHEMA_VERSION,
        pass_field=AFFECTED24_CLOSURE_PASS_FIELD,
        status_field=AFFECTED24_CLOSURE_STATUS_FIELD,
        status_value=AFFECTED24_CLOSURE_STATUS_VALUE,
        required_fields=AFFECTED24_REQUIRED_FIELDS,
        expected_path=EXPECTED_AFFECTED24_CLOSURE_PATH,
        expected_sha256=EXPECTED_AFFECTED24_CLOSURE_SHA256,
    )
    weather_audit = audit_weather_sources(
        matrix=matrix,
        catalog_path=args.anchor_catalog,
        weather_map_path=args.weather_map,
        physical_map_path=args.physical_map,
        cmip_root=args.cmip_root,
    )
    execution_matrix = transform_fresh96_matrix(matrix)
    jobs = build_cell_jobs(execution_matrix)
    return PreflightResult(
        source_matrix=matrix,
        execution_matrix=execution_matrix,
        jobs=jobs,
        authorization_artifacts=auth_artifacts,
        production_artifacts=prod_artifacts,
        matrix_manifest=manifest,
        weather_audit=weather_audit,
        smoke_closure=smoke,
        affected24_closure=affected,
        input_hashes=input_hashes,
    )


def preflight_summary(args: argparse.Namespace, result: PreflightResult) -> dict[str, Any]:
    return {
        "schema_version": "paperb_enb_fresh96_preflight_v2_envscoped",
        "preflight_passed": True,
        "output_root": str(args.output_root.resolve()),
        "output_root_touched": False,
        "subprocess_started": False,
        "cells": len(result.jobs),
        "anchors": len({job.physical_condition_id for job in result.jobs}),
        "strategies": list(CORE_STRATEGIES),
        "fresh_subprocesses_planned": len(result.jobs),
        "max_concurrent_cells": args.workers,
        "reuse_cells": 0,
        "resume_permitted": False,
        "purge_permitted": False,
        "production_runner_sha256": EXPECTED_PRODUCTION_RUNNER_SHA256,
        "python_invocation_path": str(result.production_artifacts.python),
        "python_resolved_target": str(result.production_artifacts.python.resolve()),
        "raw_cmip_csvs_rehashed": result.weather_audit["raw_cmip_csv_count"],
        "selected_epws_rehashed": result.weather_audit["selected_epw_count"],
        "smoke_closure_sha256": result.smoke_closure["sha256"],
        "affected24_closure_sha256": result.affected24_closure["sha256"],
    }


def copy_new(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to replace frozen input: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv_new(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)


def append_jsonl(path: Path, value: Any, lock: threading.Lock) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())


def freeze_execution_inputs(
    output_root: Path,
    args: argparse.Namespace,
    result: PreflightResult,
) -> FrozenArtifacts:
    freeze = output_root / "freeze"
    frozen_runner = freeze / "code" / args.production_runner.name
    frozen_auth_runner = freeze / "code" / "authorization" / args.authorization_runner.name
    frozen_base_launcher = freeze / "code" / "authorization" / args.base_launcher.name
    frozen_model = freeze / "model" / "control_predictors.joblib"
    frozen_metrics = freeze / "model" / "control_predictor_metrics.json"
    frozen_idf = freeze / "building" / args.idf.name

    copies: list[tuple[Path, Path]] = [
        (Path(__file__).resolve(), freeze / "code" / Path(__file__).name),
        (args.production_runner.resolve(), frozen_runner),
        (args.authorization_runner.resolve(), frozen_auth_runner),
        (args.base_launcher.resolve(), frozen_base_launcher),
        (args.model.resolve(), frozen_model),
        (args.model_metrics.resolve(), frozen_metrics),
        (args.idf.resolve(), frozen_idf),
        (
            args.environment_record.resolve(),
            freeze / "provenance" / args.environment_record.name,
        ),
        (
            args.matrix.resolve(),
            freeze / "authorization" / "source" / args.matrix.name,
        ),
        (
            args.matrix_manifest.resolve(),
            freeze / "authorization" / "source" / args.matrix_manifest.name,
        ),
        (
            args.pilot_decision.resolve(),
            freeze / "authorization" / "source" / args.pilot_decision.name,
        ),
        (
            args.completion_index.resolve(),
            freeze / "authorization" / "source" / args.completion_index.name,
        ),
        (
            args.anchor_catalog.resolve(),
            freeze / "weather" / "catalog" / args.anchor_catalog.name,
        ),
        (
            args.weather_map.resolve(),
            freeze / "weather" / "maps" / args.weather_map.name,
        ),
        (
            args.physical_map.resolve(),
            freeze / "weather" / "maps" / args.physical_map.name,
        ),
        (
            args.mechanism_report.resolve(),
            freeze / "mechanism" / args.mechanism_report.name,
        ),
        (
            args.smoke_closure.resolve(),
            freeze / "prerequisites" / "smoke" / args.smoke_closure.name,
        ),
        (
            SMOKE_PRODUCER_PATH,
            freeze / "prerequisites" / "smoke" / SMOKE_PRODUCER_PATH.name,
        ),
        (
            args.affected24_closure.resolve(),
            freeze / "prerequisites" / "affected24" / args.affected24_closure.name,
        ),
        (
            AFFECTED24_PRODUCER_PATH,
            freeze
            / "prerequisites"
            / "affected24"
            / AFFECTED24_PRODUCER_PATH.name,
        ),
    ]
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError(f"Production protocol must exist before execution: {PROTOCOL_PATH}")
    copies.append((PROTOCOL_PATH, freeze / "protocol" / PROTOCOL_PATH.name))
    for source, destination in copies:
        copy_new(source, destination)
        if sha256_file(destination) != sha256_file(source):
            raise ContractError(f"Frozen copy checksum differs from source: {destination}")

    job_by_weather: dict[str, CellJob] = {}
    for job in result.jobs:
        job_by_weather.setdefault(str(job.weather_source), job)
    weather_by_source: dict[str, Path] = {}
    for record in result.weather_audit["selected_epws"]:
        source = Path(record["path"]).resolve()
        job = job_by_weather.get(str(source))
        if job is None:
            raise ContractError(f"No cell maps to selected EPW during freeze: {source}")
        destination = (
            freeze
            / "weather"
            / "epw"
            / job.scenario
            / job.city
            / source.name
        )
        copy_new(source, destination)
        if (
            sha256_file(destination) != record["sha256"]
            or sha256_epw_payload(destination) != record["data_sha256"]
        ):
            raise ContractError(f"Frozen EPW failed post-copy checksum: {destination}")
        weather_by_source[str(source)] = destination.resolve()
    frozen_epws = sorted((freeze / "weather" / "epw").rglob("*.epw"))
    if len(frozen_epws) != EXPECTED_SELECTED_EPWS or len(weather_by_source) != EXPECTED_SELECTED_EPWS:
        raise ContractError("Freeze must contain exactly the 24 selected EPWs.")

    execution_matrix = result.execution_matrix.copy()
    execution_matrix["runner_path"] = str(frozen_runner.resolve())
    execution_matrix["execution_epw_path"] = execution_matrix["epw_path"].map(
        lambda value: str(weather_by_source[str(Path(str(value)).resolve())])
    )
    execution_matrix_path = (
        freeze / "authorization" / "execution" / "scaled_matrix_fresh96_v2_envscoped.csv"
    )
    write_csv_new(execution_matrix_path, execution_matrix)

    training_record = {
        "schema_version": "paperb_enb_training_data_checksum_v1",
        "path": str(args.training_data.resolve()),
        "sha256": EXPECTED_TRAINING_DATA_SHA256,
        "size_bytes": args.training_data.stat().st_size,
        "copied_into_freeze": False,
        "used_with_skip_train": True,
    }
    write_json_new(freeze / "model" / "TRAINING_DATA_CHECKSUM.json", training_record)
    write_json_new(
        freeze / "weather" / "RAW_CMIP_SOURCE_CHECKSUMS.json",
        result.weather_audit,
    )
    runtime_record = {
        "schema_version": "paperb_enb_runtime_checksums_v2_envscoped",
        "python_invocation_path": str(validate_python_invocation(args.python)),
        "python_resolved_target": str(args.python.resolve()),
        "python_sha256": result.production_artifacts.hashes["python_sha256"],
        "energyplus_root": str(args.eplus_root.resolve()),
        "energyplus_invoked_path": str(result.production_artifacts.energyplus_invoked),
        "energyplus_resolved_executable": str(result.production_artifacts.energyplus_target),
        "energyplus_executable_sha256": result.production_artifacts.hashes[
            "energyplus_executable_sha256"
        ],
        "energyplus_api_library": str(result.production_artifacts.energyplus_api_library),
        "energyplus_api_library_sha256": result.production_artifacts.hashes[
            "energyplus_api_library_sha256"
        ],
        "energyplus_idd": str(result.production_artifacts.energyplus_idd),
        "energyplus_idd_sha256": result.production_artifacts.hashes[
            "energyplus_idd_sha256"
        ],
        "energyplus_python_api_sha256": result.production_artifacts.hashes[
            "energyplus_python_api_sha256"
        ],
    }
    write_json_new(freeze / "provenance" / "RUNTIME_CHECKSUMS.json", runtime_record)

    frozen_hashes = {
        str(path.relative_to(output_root)): sha256_file(path)
        for path in sorted(freeze.rglob("*"))
        if path.is_file()
    }
    freeze_record = {
        "schema_version": FREEZE_SCHEMA,
        "created_utc": utc_now(),
        "source_authorization_matrix_sha256": EXPECTED_MATRIX_SHA256,
        "source_authorization_runner_sha256": EXPECTED_AUTHORIZATION_RUNNER_SHA256,
        "production_runner_sha256": EXPECTED_PRODUCTION_RUNNER_SHA256,
        "base_launcher_sha256": EXPECTED_BASE_LAUNCHER_SHA256,
        "model_sha256": EXPECTED_MODEL_SHA256,
        "model_metrics_sha256": EXPECTED_MODEL_METRICS_SHA256,
        "training_data_sha256": EXPECTED_TRAINING_DATA_SHA256,
        "source_idf_sha256": EXPECTED_SOURCE_IDF_SHA256,
        "environment_record_sha256": EXPECTED_ENVIRONMENT_RECORD_SHA256,
        "mechanism_report_sha256": EXPECTED_MECHANISM_REPORT_SHA256,
        "smoke_closure_sha256": result.smoke_closure["sha256"],
        "smoke_producer_sha256": EXPECTED_SMOKE_PRODUCER_SHA256,
        "affected24_closure_sha256": result.affected24_closure["sha256"],
        "affected24_producer_sha256": EXPECTED_AFFECTED24_PRODUCER_SHA256,
        "execution_matrix_path": str(execution_matrix_path.resolve()),
        "execution_matrix_sha256": sha256_file(execution_matrix_path),
        "fresh_cells": len(result.jobs),
        "fresh_subprocesses": len(result.jobs),
        "physical_anchors": len({job.physical_condition_id for job in result.jobs}),
        "selected_epws_copied": len(frozen_epws),
        "raw_cmip_csvs_rehashed": result.weather_audit["raw_cmip_csv_count"],
        "raw_cmip_csvs_copied": 0,
        "reuse_cells": 0,
        "resume_permitted": False,
        "purge_permitted": False,
        "source_outputs_spliced": False,
        "pinned_input_sha256": result.input_hashes,
        "frozen_file_sha256": frozen_hashes,
        "launch_performed_at_freeze_time": False,
    }
    write_json_new(output_root / "FREEZE.json", freeze_record)
    return FrozenArtifacts(
        runner=frozen_runner.resolve(),
        model=frozen_model.resolve(),
        model_metrics=frozen_metrics.resolve(),
        source_idf=frozen_idf.resolve(),
        weather_by_source=weather_by_source,
        freeze_record=freeze_record,
    )


def command_for_cell(
    job: CellJob,
    *,
    cell_dir: Path,
    frozen: FrozenArtifacts,
    artifacts: Any,
) -> list[str]:
    weather = frozen.weather_by_source[str(job.weather_source)]
    command = [
        str(artifacts.python),
        str(frozen.runner),
        "--data",
        str(artifacts.training_data),
        "--output-dir",
        str(cell_dir.resolve()),
        "--idf",
        str(frozen.source_idf),
        "--eplus-root",
        str(artifacts.eplus_root),
        "--weather",
        str(weather),
        "--begin-month",
        "1",
        "--begin-day",
        "1",
        "--end-month",
        "12",
        "--end-day",
        "31",
        "--strategies",
        job.strategy,
        "--paperb-met",
        "0.854333",
        "--paperb-people-activity-w",
        "93.2",
        "--paperb-tail-threshold",
        "0.20",
        "--paperb-asym-threshold",
        "0.10",
        "--controller-semantics",
        "corrected",
        "--inference-preprocessing",
        "training_contract",
        "--skip-train",
        "--skip-plot",
        "--skip-combined-trace",
        "--trace-format",
        "parquet",
    ]
    validate_cell_command(command, job)
    return command


def validate_cell_command(command: list[str], job: CellJob) -> None:
    if command.count("--strategies") != 1:
        raise ContractError("Cell command must have exactly one --strategies flag.")
    strategy_index = command.index("--strategies")
    strategy_tokens: list[str] = []
    for token in command[strategy_index + 1 :]:
        if token.startswith("--"):
            break
        strategy_tokens.append(token)
    if strategy_tokens != [job.strategy]:
        raise ContractError(f"Cell command is not strategy-isolated: {strategy_tokens}")
    if command.count("--weather") != 1:
        raise ContractError("Cell command must have exactly one weather flag.")
    present_banned = sorted(BANNED_RUNNER_FLAGS.intersection(command))
    if present_banned:
        raise ContractError(f"Cell command contains destructive/reuse flags: {present_banned}")
    if "--skip-train" not in command or "--skip-combined-trace" not in command:
        raise ContractError("Cell command lacks the frozen-model/isolated-trace contract.")


def parse_idf_fields(obj: str) -> list[str]:
    uncommented = [line.split("!", 1)[0] for line in obj.splitlines()]
    raw = "\n".join(uncommented).replace("\n", " ")
    return [field.strip() for field in raw.split(",") if field.strip()]


def validate_generated_idf_contract(path: Path) -> dict[str, Any]:
    objects: dict[str, list[list[str]]] = {}
    for raw in path.read_text(encoding="utf-8", errors="strict").split(";"):
        fields = parse_idf_fields(raw)
        if fields:
            objects.setdefault(fields[0].casefold(), []).append(fields)

    def exactly_one(object_type: str) -> list[str]:
        found = objects.get(object_type.casefold(), [])
        if len(found) != 1:
            raise ContractError(
                f"Generated IDF requires one {object_type}; observed {len(found)} in {path}."
            )
        return found[0]

    timestep = exactly_one("Timestep")
    convergence = exactly_one("ConvergenceLimits")
    building = exactly_one("Building")
    runperiod = exactly_one("RunPeriod")
    nightcycles = objects.get("availabilitymanager:nightcycle", [])
    if timestep != ["Timestep", "4"]:
        raise ContractError(f"Unexpected Timestep contract: {timestep}")
    if convergence != ["ConvergenceLimits", "15", "5"]:
        raise ContractError(f"Unexpected ConvergenceLimits contract: {convergence}")
    if len(building) < 9 or (
        float(building[4]),
        float(building[5]),
        building[6].casefold(),
        int(building[7]),
        int(building[8]),
    ) != (0.04, 0.20, "minimalshadowing", 50, 6):
        raise ContractError(f"Unexpected Building contract: {building[:9]}")
    expected_runperiod = [
        "RunPeriod",
        "OTC_CONTROL_TRACE",
        "1",
        "1",
        "12",
        "31",
        "Monday",
        "No",
        "No",
        "No",
        "Yes",
        "Yes",
    ]
    if runperiod != expected_runperiod:
        raise ContractError(f"Unexpected annual RunPeriod contract: {runperiod}")
    if len(nightcycles) != 3:
        raise ContractError(f"Expected three NightCycle objects, observed {len(nightcycles)}.")
    expected_names = {
        "PACU_VAV_bot Availability Manager",
        "PACU_VAV_mid Availability Manager",
        "PACU_VAV_top Availability Manager",
    }
    observed_names: set[str] = set()
    for fields in nightcycles:
        if len(fields) != 8:
            raise ContractError(f"Malformed NightCycle object: {fields}")
        observed_names.add(fields[1])
        observed = (
            fields[2].casefold(),
            fields[3].casefold(),
            fields[4].casefold(),
            float(fields[5]),
            fields[6].casefold(),
            int(fields[7]),
        )
        if observed != (
            "always_on",
            "hvacoperationschd",
            "cycleonany",
            1.0,
            "fixedruntime",
            1800,
        ):
            raise ContractError(f"Unexpected NightCycle contract: {fields}")
    if observed_names != expected_names:
        raise ContractError(f"Unexpected NightCycle names: {sorted(observed_names)}")
    return {
        "idf_sha256": sha256_file(path),
        "timestep_per_hour": 4,
        "minimum_system_timestep_minutes": 15,
        "maximum_hvac_iterations": 5,
        "loads_convergence_tolerance": 0.04,
        "temperature_convergence_tolerance_c": 0.20,
        "solar_distribution": "MinimalShadowing",
        "maximum_warmup_days": 50,
        "minimum_warmup_days": 6,
        "nightcycle_object_count": 3,
        "nightcycle_names": sorted(observed_names),
        "nightcycle_contract": (
            "Always_On/HVACOperationSchd/CycleOnAny/1.0/FixedRunTime/1800"
        ),
        "runperiod_contract": "01-01_to_12-31_starting_Monday",
    }


def expected_annual_calendar() -> dict[str, np.ndarray]:
    dates = [date(2001, 1, 1) + timedelta(days=index) for index in range(365)]
    current_time = np.tile(np.arange(1, 97, dtype=float) / 4.0, 365)
    day_of_week_daily = np.array([((item.weekday() + 1) % 7) + 1 for item in dates])
    day_of_week = np.repeat(day_of_week_daily, 96)
    occupied = (
        (day_of_week >= 2)
        & (day_of_week <= 6)
        & (np.floor(current_time).astype(int) >= 6)
        & (np.floor(current_time).astype(int) < 22)
    )
    if int(occupied.sum()) != EXPECTED_OCCUPIED_ROWS:
        raise AssertionError("Internal occupied-calendar contract is inconsistent.")
    return {
        "month": np.repeat(np.array([item.month for item in dates]), 96),
        "day": np.repeat(np.array([item.day for item in dates]), 96),
        "day_of_week": day_of_week,
        "hour": np.tile(np.repeat(np.arange(24), 4), 365),
        "current_time": current_time,
        "sim_time_hours": np.arange(1, EXPECTED_TRACE_ROWS + 1, dtype=float) / 4.0,
        "occupied": occupied,
    }


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy()
    if not np.isfinite(values.astype(float)).all():
        raise ContractError(f"Trace column contains non-finite values: {column}")
    return values


def validate_trace_frame(frame: pd.DataFrame, job: CellJob) -> dict[str, Any]:
    required = {
        "strategy",
        "weather",
        "environment_num",
        "kind_of_sim",
        "formal_weather_step",
        "calendar_year",
        "month",
        "day",
        "day_of_week",
        "hour",
        "current_time",
        "sim_time_hours",
        "occupied",
        "heating_setpoint_c",
        "cooling_setpoint_c",
        "controller_semantics",
        "inference_preprocessing",
    }
    require_columns(frame, required, name=f"trace for {job.scaled_run_id}")
    if len(frame) != EXPECTED_TRACE_ROWS:
        raise ContractError(
            f"Trace row count mismatch for {job.scaled_run_id}: {len(frame)}."
        )
    exact_strings = {
        "strategy": job.strategy,
        "weather": job.weather_stem,
        "controller_semantics": "corrected",
        "inference_preprocessing": "training_contract",
    }
    for column, expected in exact_strings.items():
        if set(frame[column].astype(str)) != {expected}:
            raise ContractError(f"Trace {column} mismatch for {job.scaled_run_id}.")
    if len(set(_numeric(frame, "environment_num").astype(int))) != 1:
        raise ContractError("Trace must contain exactly one EnergyPlus environment number.")
    if set(_numeric(frame, "kind_of_sim").astype(int)) != {WEATHER_FILE_RUN_PERIOD_KIND}:
        raise ContractError("Trace includes a non-weather-file environment.")
    if not np.array_equal(
        _numeric(frame, "formal_weather_step").astype(int),
        np.arange(1, EXPECTED_TRACE_ROWS + 1),
    ):
        raise ContractError("formal_weather_step is not exactly 1..35040.")

    expected = expected_annual_calendar()
    for column in ("month", "day", "day_of_week", "hour"):
        if not np.array_equal(_numeric(frame, column).astype(int), expected[column]):
            raise ContractError(f"Annual trace calendar mismatch in {column}.")
    for column in ("current_time", "sim_time_hours"):
        if not np.allclose(
            _numeric(frame, column).astype(float), expected[column], atol=1e-9, rtol=0.0
        ):
            raise ContractError(f"Annual trace clock mismatch in {column}.")
    years = _numeric(frame, "calendar_year").astype(int)
    start_year = int(years[0])
    if not np.all(years[:-1] == start_year) or int(years[-1]) not in {
        start_year,
        start_year + 1,
    }:
        raise ContractError("calendar_year has an unexpected within-run transition.")
    if ((frame["month"].astype(int) == 2) & (frame["day"].astype(int) == 29)).any():
        raise ContractError("Annual trace unexpectedly contains February 29.")
    occupancy = frame["occupied"].map(lambda value: csv_boolean(value, field="occupied")).to_numpy()
    if not np.array_equal(occupancy, expected["occupied"]):
        raise ContractError("Trace occupancy is not the Monday-start 06:00-22:00 contract.")
    if int(occupancy.sum()) != EXPECTED_OCCUPIED_ROWS:
        raise ContractError("Trace occupied-step count is not exactly 16704.")

    heating = _numeric(frame, "heating_setpoint_c").astype(float)
    cooling = _numeric(frame, "cooling_setpoint_c").astype(float)
    unoccupied = ~occupancy
    if not (
        np.allclose(heating[unoccupied], 12.0, atol=1e-9, rtol=0.0)
        and np.allclose(cooling[unoccupied], 30.0, atol=1e-9, rtol=0.0)
    ):
        raise ContractError("Unoccupied formal setpoints are not exactly 12/30 C.")
    if job.strategy == "diagnostic_reference":
        if not (
            np.allclose(heating[occupancy], 22.0, atol=1e-9, rtol=0.0)
            and np.allclose(cooling[occupancy], 24.0, atol=1e-9, rtol=0.0)
        ):
            raise ContractError("Diagnostic occupied setpoints are not exactly 22/24 C.")
    elif not (
        np.all((heating[occupancy] >= 12.0) & (heating[occupancy] <= 23.25))
        and np.all((cooling[occupancy] >= 23.25) & (cooling[occupancy] <= 30.0))
        and np.all(cooling[occupancy] - heating[occupancy] >= 2.0 - 1e-9)
    ):
        raise ContractError("Occupied controlled setpoints violate bounds/deadband.")
    return {
        "rows": len(frame),
        "occupied_rows": int(occupancy.sum()),
        "environment_num": int(_numeric(frame, "environment_num")[0]),
        "calendar_start": "01-01 00:15 Monday",
        "calendar_end": "12-31 24:00",
        "weather": job.weather_stem,
        "strategy": job.strategy,
    }


def validate_trace_file(path: Path, job: CellJob) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ContractError(f"Missing/empty Parquet trace: {path}")
    columns = [
        "strategy",
        "weather",
        "environment_num",
        "kind_of_sim",
        "formal_weather_step",
        "calendar_year",
        "month",
        "day",
        "day_of_week",
        "hour",
        "current_time",
        "sim_time_hours",
        "occupied",
        "heating_setpoint_c",
        "cooling_setpoint_c",
        "controller_semantics",
        "inference_preprocessing",
    ]
    try:
        frame = pd.read_parquet(path, columns=columns)
    except Exception as exc:
        raise ContractError(f"Cannot read required Parquet trace columns: {path}: {exc}") from exc
    result = validate_trace_frame(frame, job)
    result.update(
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    )
    return result


def parse_eio_warmup_days(path: Path) -> int:
    matches = re.findall(
        r"(?im)^\s*Environment:WarmupDays\s*,\s*(\d+)\s*$",
        path.read_text(encoding="utf-8", errors="strict"),
    )
    if len(matches) != 1:
        raise ContractError(f"Expected one Environment:WarmupDays row in {path}: {matches}")
    warmup_days = int(matches[0])
    if not MIN_WARMUP_DAYS <= warmup_days <= MAX_WARMUP_DAYS:
        raise ContractError(f"WarmupDays is outside 6..50 in {path}: {warmup_days}")
    return warmup_days


def validate_err_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="strict")
    severe = [line.strip() for line in text.splitlines() if "** Severe  **" in line]
    fatal = [line.strip() for line in text.splitlines() if "**  Fatal  **" in line]
    if "EnergyPlus Completed Successfully" not in text:
        raise ContractError(f"EnergyPlus success marker absent: {path}")
    if severe or fatal:
        raise ContractError(
            f"EnergyPlus strict error gate failed ({len(severe)} severe, {len(fatal)} fatal): {path}"
        )
    if "CheckWarmupConvergence" in text:
        raise ContractError(f"CheckWarmupConvergence is present: {path}")
    return {
        "completed_successfully": True,
        "severe_error_count": 0,
        "fatal_error_count": 0,
        "check_warmup_convergence_present": False,
        "sha256": sha256_file(path),
    }


def validate_environment_audit(
    path: Path,
    *,
    job: CellJob,
    warmup_days: int,
    generated_contract: dict[str, Any],
) -> dict[str, Any]:
    audit = read_json(path, name="environment-scoped control audit")
    if audit.get("schema_version") != ENVIRONMENT_AUDIT_SCHEMA:
        raise ContractError(f"Environment-audit schema mismatch: {path}")
    if audit.get("strategy") != job.strategy or audit.get("weather") != job.weather_stem:
        raise ContractError(f"Environment-audit strategy/weather mismatch: {path}")
    if audit.get("control_scope") != "kind_of_sim_3_only":
        raise ContractError(f"Environment audit does not declare kind-3-only control: {path}")
    if audit.get("weather_file_run_period_kind") != WEATHER_FILE_RUN_PERIOD_KIND:
        raise ContractError(f"Environment audit has the wrong weather kind: {path}")
    if audit.get("warmup_setpoint_contract") != WARMUP_SETPOINT_CONTRACT:
        raise ContractError(f"Environment audit has the wrong warmup setpoint contract: {path}")
    if audit.get("weather_runperiod_warmup_days") != warmup_days:
        raise ContractError(f"Environment-audit/EIO warmup counts differ: {path}")
    environments = audit.get("weather_environment_nums")
    if not isinstance(environments, list) or len(environments) != 1:
        raise ContractError(f"Environment audit must identify one weather environment: {path}")
    gates = audit.get("audit_gates")
    if not isinstance(gates, dict) or not gates:
        raise ContractError(f"Environment audit gates are missing: {path}")
    invalid_gates = sorted(
        name for name, value in gates.items() if type(value) is not bool or value is not True
    )
    if invalid_gates or audit.get("all_audit_gates_passed") is not True:
        raise ContractError(f"Environment audit failed gates {invalid_gates}: {path}")
    expected_callbacks = warmup_days * 24 * 4
    if audit.get("expected_weather_warmup_callbacks") != expected_callbacks:
        raise ContractError(f"Environment audit has an invalid warmup callback count: {path}")
    callback_counts = audit.get("callback_counts")
    if not isinstance(callback_counts, dict):
        raise ContractError(f"Environment callback counts are absent: {path}")
    if callback_counts.get("weather_warmup_control_callbacks") != expected_callbacks:
        raise ContractError(f"Not every warmup timestep was audited: {path}")
    if callback_counts.get("warmup_setpoint_mismatches") != 0:
        raise ContractError(f"Warmup setpoint mismatches were recorded: {path}")
    if callback_counts.get("warmup_controller_state_mutations") != 0:
        raise ContractError(f"Warmup controller-state mutations were recorded: {path}")
    runner_contract = {
        key: value
        for key, value in generated_contract.items()
        if key != "runperiod_contract"
    }
    if audit.get("generated_idf_contract") != runner_contract:
        raise ContractError(f"Environment audit carries a different generated-IDF contract: {path}")
    error_gate = audit.get("energyplus_error_gate")
    if not isinstance(error_gate, dict) or not (
        error_gate.get("completed_successfully") is True
        and error_gate.get("severe_error_count") == 0
        and error_gate.get("fatal_error_count") == 0
        and error_gate.get("check_warmup_convergence_present") is False
    ):
        raise ContractError(f"Environment audit carries a failed EnergyPlus error gate: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "schema_version": ENVIRONMENT_AUDIT_SCHEMA,
        "all_audit_gates_passed": True,
        "weather_environment_num": int(environments[0]),
        "warmup_days": warmup_days,
        "expected_warmup_callbacks": expected_callbacks,
    }


def validate_warmup_audit_csv(
    path: Path,
    *,
    warmup_days: int,
    environment_num: int,
) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ContractError(f"Missing/empty warmup setpoint audit: {path}")
    frame = pd.read_csv(path)
    required = {
        "environment_num",
        "kind_of_sim",
        "day_of_week",
        "current_time",
        "occupied",
        "requested_heating_setpoint_c",
        "requested_cooling_setpoint_c",
        "applied_heating_actuator_value_c",
        "applied_cooling_actuator_value_c",
        "setpoint_contract",
        "setpoint_mismatch",
        "controller_state_mutated",
    }
    require_columns(frame, required, name="warmup setpoint audit")
    expected_rows = warmup_days * 24 * 4
    if len(frame) != expected_rows:
        raise ContractError(
            f"Warmup setpoint audit has {len(frame)} rows, expected {expected_rows}: {path}"
        )
    if set(pd.to_numeric(frame["environment_num"], errors="raise").astype(int)) != {
        environment_num
    }:
        raise ContractError(f"Warmup audit environment identity mismatch: {path}")
    if set(pd.to_numeric(frame["kind_of_sim"], errors="raise").astype(int)) != {
        WEATHER_FILE_RUN_PERIOD_KIND
    }:
        raise ContractError(f"Warmup audit contains a non-weather environment: {path}")
    if set(frame["setpoint_contract"].astype(str)) != {WARMUP_SETPOINT_CONTRACT}:
        raise ContractError(f"Warmup audit setpoint-contract label mismatch: {path}")
    mismatch = frame["setpoint_mismatch"].map(
        lambda value: csv_boolean(value, field="setpoint_mismatch")
    )
    mutated = frame["controller_state_mutated"].map(
        lambda value: csv_boolean(value, field="controller_state_mutated")
    )
    if mismatch.any() or mutated.any():
        raise ContractError(f"Warmup audit records mismatch or controller-state mutation: {path}")
    occupied = frame["occupied"].map(
        lambda value: csv_boolean(value, field="warmup.occupied")
    ).to_numpy()
    day_of_week = pd.to_numeric(frame["day_of_week"], errors="raise").astype(int).to_numpy()
    current_time = pd.to_numeric(frame["current_time"], errors="raise").to_numpy(float)
    expected_occupied = (
        (day_of_week >= 2)
        & (day_of_week <= 6)
        & (np.floor(current_time).astype(int) >= 6)
        & (np.floor(current_time).astype(int) < 22)
    )
    if not np.array_equal(occupied, expected_occupied):
        raise ContractError(f"Warmup occupancy/setpoint schedule mismatch: {path}")
    expected_heat = np.where(occupied, 22.0, 12.0)
    expected_cool = np.where(occupied, 24.0, 30.0)
    for column, expected in (
        ("requested_heating_setpoint_c", expected_heat),
        ("applied_heating_actuator_value_c", expected_heat),
        ("requested_cooling_setpoint_c", expected_cool),
        ("applied_cooling_actuator_value_c", expected_cool),
    ):
        observed = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
        if not np.allclose(observed, expected, atol=1e-9, rtol=0.0):
            raise ContractError(f"Warmup setpoint values differ in {column}: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": len(frame),
        "setpoint_mismatches": 0,
        "controller_state_mutations": 0,
    }


def validate_summary(path: Path, job: CellJob) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ContractError(f"Missing/empty cell summary: {path}")
    frame = pd.read_csv(path)
    require_columns(
        frame,
        {
            "weather",
            "strategy",
            "controller_semantics",
            "inference_preprocessing",
            "n_steps",
            "occupied_steps",
        },
        name=f"summary for {job.scaled_run_id}",
    )
    if len(frame) != 1:
        raise ContractError(f"One-strategy cell must produce exactly one summary row: {path}")
    row = frame.iloc[0]
    expected = {
        "weather": job.weather_stem,
        "strategy": job.strategy,
        "controller_semantics": "corrected",
        "inference_preprocessing": "training_contract",
    }
    for field, value in expected.items():
        if str(row[field]) != value:
            raise ContractError(f"Summary {field} mismatch: {path}")
    if int(row["n_steps"]) != EXPECTED_TRACE_ROWS:
        raise ContractError(f"Summary n_steps is not 35040: {path}")
    if int(row["occupied_steps"]) != EXPECTED_OCCUPIED_ROWS:
        raise ContractError(f"Summary occupied_steps is not 16704: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def validate_cell_outputs(cell_dir: Path, job: CellJob) -> dict[str, Any]:
    generated_idfs = sorted((cell_dir / "model").glob("*.idf"))
    if len(generated_idfs) != 1:
        raise ContractError(
            f"Expected exactly one generated IDF for {job.scaled_run_id}, got {generated_idfs}."
        )
    idf_contract = validate_generated_idf_contract(generated_idfs[0])
    generated_contract_path = cell_dir / "model" / "PAPERB_GENERATED_IDF_CONTRACT.json"
    generated_contract_json = read_json(
        generated_contract_path, name="runner generated-IDF contract"
    )
    runner_contract = {
        key: value for key, value in idf_contract.items() if key != "runperiod_contract"
    }
    if generated_contract_json != runner_contract:
        raise ContractError(
            f"Runner generated-IDF contract does not match independent validation: {cell_dir}"
        )

    eplus_dir = (
        cell_dir
        / "energyplus"
        / job.weather_stem
        / "met0p854"
        / "act93p2W"
        / job.strategy
    )
    required_outputs = [
        eplus_dir / name
        for name in (
            "eplusout.err",
            "eplusout.eio",
            "eplusout.end",
            "eplusout.sql",
            "eplusout.mtr",
        )
    ]
    missing = [path for path in required_outputs if not path.is_file()]
    if missing:
        raise ContractError(f"Missing retained EnergyPlus outputs: {missing}")
    err_gate = validate_err_file(eplus_dir / "eplusout.err")
    warmup_days = parse_eio_warmup_days(eplus_dir / "eplusout.eio")
    environment_audit = validate_environment_audit(
        eplus_dir / "PAPERB_CONTROL_ENVIRONMENT_AUDIT.json",
        job=job,
        warmup_days=warmup_days,
        generated_contract=idf_contract,
    )
    warmup_audit = validate_warmup_audit_csv(
        eplus_dir / "paperb_weather_warmup_setpoint_audit.csv",
        warmup_days=warmup_days,
        environment_num=environment_audit["weather_environment_num"],
    )

    trace = (
        cell_dir
        / "traces"
        / f"{job.weather_stem}_met0p854_act93p2W_{job.strategy}.parquet"
    )
    all_traces = sorted((cell_dir / "traces").glob("*.parquet"))
    if all_traces != [trace]:
        raise ContractError(f"Cell must contain exactly its one declared Parquet trace: {cell_dir}")
    trace_result = validate_trace_file(trace, job)
    if trace_result["environment_num"] != environment_audit["weather_environment_num"]:
        raise ContractError(f"Trace/environment-audit environment identity mismatch: {cell_dir}")
    summary = validate_summary(
        cell_dir / "summary" / "medium_office_trace_summary.csv", job
    )
    return {
        "generated_idf_path": str(generated_idfs[0].resolve()),
        "generated_idf_sha256": idf_contract["idf_sha256"],
        "generated_idf_contract_path": str(generated_contract_path.resolve()),
        "generated_idf_contract_sha256": sha256_file(generated_contract_path),
        "energyplus_output_dir": str(eplus_dir.resolve()),
        "retained_energyplus_sha256": {
            path.name: sha256_file(path) for path in required_outputs
        },
        "eio_sha256": sha256_file(eplus_dir / "eplusout.eio"),
        "warmup_days": warmup_days,
        "err_gate": err_gate,
        "environment_audit": environment_audit,
        "warmup_setpoint_audit": warmup_audit,
        "trace": trace_result,
        "summary": summary,
        "all_cell_gates_passed": True,
    }


def assert_execution_exclusive(args: argparse.Namespace) -> None:
    """Refuse launch while another scaled launcher or production cell is active."""

    completed = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,command="],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ContractError("Cannot verify exclusive execution with ps; refusing launch.")
    process_rows: list[tuple[int, int, str]] = []
    for line in completed.stdout.splitlines():
        match = re.match(r"\s*(\d+)\s+(\d+)\s+(.*)", line)
        if match:
            process_rows.append((int(match.group(1)), int(match.group(2)), match.group(3)))
    parent_by_pid = {pid: parent for pid, parent, _command in process_rows}
    excluded_pids = {os.getpid()}
    ancestor = os.getppid()
    while ancestor > 0 and ancestor not in excluded_pids:
        excluded_pids.add(ancestor)
        ancestor = parent_by_pid.get(ancestor, 0)
    conflicts: list[str] = []
    for pid, _parent, command in process_rows:
        if pid in excluded_pids:
            continue
        scaled_execute = "run_enb_scaled_matrix" in command and "--execute" in command
        production_cell = args.production_runner.name in command
        standalone_energyplus = (
            "/EnergyPlus-25-1-0/energyplus" in command
            or "energyplus-25.1.0" in command
        )
        if scaled_execute or production_cell or standalone_energyplus:
            conflicts.append(f"pid={pid} command={command}")
    if conflicts:
        raise ContractError(
            "Another scaled execution or production cell is active; launch must be exclusive:\n"
            + "\n".join(conflicts)
        )


def run_cell(
    job: CellJob,
    *,
    attempt_id: str,
    output_root: Path,
    frozen: FrozenArtifacts,
    result: PreflightResult,
    status_path: Path,
    append_lock: threading.Lock,
) -> dict[str, Any]:
    cell_dir = output_root / "cells" / job.scaled_run_id
    started = time.time()
    runner_pid: int | None = None
    returncode: int | None = None
    log_path = cell_dir / f"run.{attempt_id}.log"
    try:
        cell_dir.mkdir(parents=True, exist_ok=False)
        copy_new(frozen.model, cell_dir / "models" / "control_predictors.joblib")
        copy_new(
            frozen.model_metrics,
            cell_dir / "models" / "control_predictor_metrics.json",
        )
        if sha256_file(cell_dir / "models" / "control_predictors.joblib") != EXPECTED_MODEL_SHA256:
            raise ContractError(f"Cell model copy checksum mismatch: {job.scaled_run_id}")
        if (
            sha256_file(cell_dir / "models" / "control_predictor_metrics.json")
            != EXPECTED_MODEL_METRICS_SHA256
        ):
            raise ContractError(f"Cell metrics copy checksum mismatch: {job.scaled_run_id}")
        (cell_dir / ".mplconfig").mkdir(exist_ok=False)
        command = command_for_cell(
            job,
            cell_dir=cell_dir,
            frozen=frozen,
            artifacts=result.production_artifacts,
        )
        contract = {
            "schema_version": "paperb_enb_fresh_cell_contract_v2_envscoped",
            "attempt_id": attempt_id,
            "scaled_run_id": job.scaled_run_id,
            "anchor_id": job.anchor_id,
            "physical_condition_id": job.physical_condition_id,
            "selector_label_id": job.selector_label_id,
            "strategy": job.strategy,
            "weather_source_path": str(job.weather_source),
            "weather_frozen_path": command[command.index("--weather") + 1],
            "weather_sha256": job.weather_sha256,
            "weather_data_sha256": job.weather_data_sha256,
            "runner_sha256": EXPECTED_PRODUCTION_RUNNER_SHA256,
            "command": command,
            "fresh_python_subprocesses": 1,
            "strategies_in_subprocess": 1,
            "reuse_or_splice_permitted": False,
            "resume_permitted": False,
            "purge_permitted": False,
            "retain_energyplus_outputs": True,
            "trace_format": "parquet_zstd",
        }
        run_contract_path = cell_dir / f"RUN_CONTRACT.{attempt_id}.json"
        write_json_new(run_contract_path, contract)
        row = result.execution_matrix.loc[
            result.execution_matrix["scaled_run_id"].astype(str).eq(job.scaled_run_id)
        ]
        if len(row) != 1:
            raise ContractError(f"Execution matrix lost cell {job.scaled_run_id}.")
        row = row.copy()
        row["runner_path"] = str(frozen.runner)
        row["execution_epw_path"] = str(
            frozen.weather_by_source[str(job.weather_source)]
        )
        authorized_row_path = cell_dir / f"AUTHORIZED_ROW.{attempt_id}.csv"
        write_csv_new(authorized_row_path, row)

        environment = os.environ.copy()
        environment.update(
            {
                "MPLCONFIGDIR": str((cell_dir / ".mplconfig").resolve()),
                "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "XDG_CACHE_HOME": str((cell_dir / ".xdg_cache").resolve()),
                "FC_CACHEDIR": str((cell_dir / ".fontconfig_cache").resolve()),
            }
        )
        with log_path.open("x", encoding="utf-8") as log:
            log.write(json.dumps({"attempt_id": attempt_id, "command": command}) + "\n")
            log.flush()
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            runner_pid = process.pid
            returncode = process.wait()
        if returncode != 0:
            raise ContractError(
                f"Production runner exited {returncode} for {job.scaled_run_id}; see {log_path}."
            )
        outputs = validate_cell_outputs(cell_dir, job)
        record = {
            "schema_version": CELL_STATUS_SCHEMA,
            "event": "cell_finished",
            "status": "complete_validated",
            "attempt_id": attempt_id,
            "timestamp_utc": utc_now(),
            "scaled_run_id": job.scaled_run_id,
            "anchor_id": job.anchor_id,
            "physical_condition_id": job.physical_condition_id,
            "selector_label_id": job.selector_label_id,
            "strategy": job.strategy,
            "weather": job.weather_stem,
            "weather_source_path": str(job.weather_source),
            "weather_frozen_path": str(
                frozen.weather_by_source[str(job.weather_source)]
            ),
            "weather_sha256": job.weather_sha256,
            "weather_data_sha256": job.weather_data_sha256,
            "runner_sha256": EXPECTED_PRODUCTION_RUNNER_SHA256,
            "model_sha256": EXPECTED_MODEL_SHA256,
            "model_metrics_sha256": EXPECTED_MODEL_METRICS_SHA256,
            "training_data_sha256": EXPECTED_TRAINING_DATA_SHA256,
            "source_idf_sha256": EXPECTED_SOURCE_IDF_SHA256,
            "controller_semantics": "corrected",
            "inference_preprocessing": "training_contract",
            "execution_action": "run_new_fresh",
            "all_gates_passed": True,
            "runner_pid": runner_pid,
            "returncode": returncode,
            "elapsed_minutes": (time.time() - started) / 60.0,
            "cell_dir": str(cell_dir.resolve()),
            "log_path": str(log_path.resolve()),
            "log_sha256": sha256_file(log_path),
            "run_contract_path": str(run_contract_path.resolve()),
            "run_contract_sha256": sha256_file(run_contract_path),
            "authorized_row_path": str(authorized_row_path.resolve()),
            "authorized_row_sha256": sha256_file(authorized_row_path),
            "fresh_subprocesses": 1,
            "reuse_cells": 0,
            "resume_used": False,
            "purge_used": False,
            "outputs": outputs,
        }
        cell_result_path = cell_dir / f"CELL_RESULT.{attempt_id}.json"
        record["cell_result_path"] = str(cell_result_path.resolve())
        write_json_new(
            cell_result_path,
            {**record, "schema_version": CELL_RESULT_SCHEMA},
        )
        record["cell_result_sha256"] = sha256_file(cell_result_path)
    except Exception as exc:
        record = {
            "schema_version": CELL_STATUS_SCHEMA,
            "event": "cell_finished",
            "status": "failed",
            "attempt_id": attempt_id,
            "timestamp_utc": utc_now(),
            "scaled_run_id": job.scaled_run_id,
            "anchor_id": job.anchor_id,
            "physical_condition_id": job.physical_condition_id,
            "selector_label_id": job.selector_label_id,
            "strategy": job.strategy,
            "weather": job.weather_stem,
            "weather_source_path": str(job.weather_source),
            "weather_frozen_path": str(
                frozen.weather_by_source[str(job.weather_source)]
            ),
            "weather_sha256": job.weather_sha256,
            "weather_data_sha256": job.weather_data_sha256,
            "runner_sha256": EXPECTED_PRODUCTION_RUNNER_SHA256,
            "model_sha256": EXPECTED_MODEL_SHA256,
            "model_metrics_sha256": EXPECTED_MODEL_METRICS_SHA256,
            "training_data_sha256": EXPECTED_TRAINING_DATA_SHA256,
            "source_idf_sha256": EXPECTED_SOURCE_IDF_SHA256,
            "controller_semantics": "corrected",
            "inference_preprocessing": "training_contract",
            "execution_action": "run_new_fresh",
            "all_gates_passed": False,
            "runner_pid": runner_pid,
            "returncode": returncode,
            "elapsed_minutes": (time.time() - started) / 60.0,
            "cell_dir": str(cell_dir.resolve()),
            "log_path": str(log_path.resolve()),
            "fresh_subprocesses": int(runner_pid is not None),
            "reuse_cells": 0,
            "resume_used": False,
            "purge_used": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if cell_dir.is_dir():
            failure_result_path = cell_dir / f"CELL_RESULT.{attempt_id}.json"
            record["cell_result_path"] = str(failure_result_path.resolve())
            if not failure_result_path.exists():
                try:
                    write_json_new(
                        failure_result_path,
                        {**record, "schema_version": CELL_RESULT_SCHEMA},
                    )
                except Exception as write_exc:
                    record["cell_result_write_error"] = str(write_exc)
    append_jsonl(status_path, record, append_lock)
    return record


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ContractError(f"JSONL record is not an object at {path}:{line_number}.")
        records.append(value)
    return records


def build_exact_closure(
    *,
    result: PreflightResult,
    records: list[dict[str, Any]],
    output_root: Path,
    status_path: Path,
    attempt_id: str,
    workers: int,
) -> dict[str, Any]:
    planned_ids = {job.scaled_run_id for job in result.jobs}
    planned_keys = {job.key for job in result.jobs}
    observed_ids = {str(record.get("scaled_run_id")) for record in records}
    observed_keys = {
        (str(record.get("physical_condition_id")), str(record.get("strategy")))
        for record in records
    }
    status_records = read_jsonl(status_path)
    cell_root = output_root / "cells"
    cell_directories = (
        {path.name for path in cell_root.iterdir() if path.is_dir()}
        if cell_root.is_dir()
        else set()
    )
    passing = [record for record in records if record.get("status") == "complete_validated"]
    trace_paths = [
        record.get("outputs", {}).get("trace", {}).get("path") for record in passing
    ]
    cell_result_pairs = [
        (record.get("cell_result_path"), record.get("cell_result_sha256"))
        for record in passing
    ]
    status_ids = {str(record.get("scaled_run_id")) for record in status_records}
    status_keys = {
        (str(record.get("physical_condition_id")), str(record.get("strategy")))
        for record in status_records
    }
    gates = {
        "exactly_96_planned_cells": len(result.jobs) == EXPECTED_CELLS,
        "exactly_96_result_records": len(records) == EXPECTED_CELLS,
        "exactly_96_jsonl_records": len(status_records) == EXPECTED_CELLS,
        "cell_status_schema_exact": all(
            record.get("schema_version") == CELL_STATUS_SCHEMA
            for record in status_records
        ),
        "cell_status_exact_key_coverage": (
            status_ids == planned_ids and status_keys == planned_keys
        ),
        "exact_scaled_run_id_coverage": observed_ids == planned_ids,
        "exact_physical_strategy_key_coverage": observed_keys == planned_keys,
        "all_96_cells_complete_validated": len(passing) == EXPECTED_CELLS,
        "exactly_96_fresh_cell_directories": cell_directories == planned_ids,
        "exactly_96_distinct_trace_paths": (
            len(trace_paths) == EXPECTED_CELLS
            and None not in trace_paths
            and len(set(trace_paths)) == EXPECTED_CELLS
        ),
        "all_cell_result_hashes_exact": (
            len(cell_result_pairs) == EXPECTED_CELLS
            and all(
                isinstance(path, str)
                and isinstance(digest, str)
                and Path(path).is_file()
                and sha256_file(Path(path)) == digest
                for path, digest in cell_result_pairs
            )
        ),
        "exactly_24_physical_anchors": (
            len({job.physical_condition_id for job in result.jobs}) == EXPECTED_ANCHORS
        ),
        "four_strategies_per_anchor": all(
            sum(job.physical_condition_id == condition for job in result.jobs) == 4
            for condition in {job.physical_condition_id for job in result.jobs}
        ),
        "one_fresh_subprocess_per_passing_cell": all(
            record.get("fresh_subprocesses") == 1 for record in passing
        ),
        "zero_reuse_or_splice": all(record.get("reuse_cells") == 0 for record in records),
        "zero_resume": all(record.get("resume_used") is False for record in records),
        "zero_purge": all(record.get("purge_used") is False for record in records),
        "max_four_concurrent_cells": 1 <= workers <= MAX_CONCURRENT_CELLS,
        "runner_hash_exact": all(
            record.get("runner_sha256") == EXPECTED_PRODUCTION_RUNNER_SHA256
            for record in records
        ),
    }
    failed_gates = sorted(name for name, passed in gates.items() if not passed)
    overall_pass = not failed_gates
    return {
        "schema_version": CLOSURE_SCHEMA,
        "attempt_id": attempt_id,
        "created_utc": utc_now(),
        "runner_sha256": EXPECTED_PRODUCTION_RUNNER_SHA256,
        "source_authorization_matrix_sha256": EXPECTED_MATRIX_SHA256,
        "fresh_cells_expected": EXPECTED_CELLS,
        "planned_cells": EXPECTED_CELLS,
        "completed_cells": len(passing),
        "passed_cells": len(passing),
        "fresh_cells_validated": len(passing),
        "failed_cells": len(records) - len(passing),
        "all_cells_fresh": True,
        "reuse_cells": 0,
        "resume_used": False,
        "purge_used": False,
        "one_strategy_per_fresh_subprocess": True,
        "exact_unique_key_coverage": gates["exact_physical_strategy_key_coverage"],
        "closure_status": "PASS" if overall_pass else "FAIL",
        "physical_anchors": len({job.physical_condition_id for job in result.jobs}),
        "strategies": list(CORE_STRATEGIES),
        "max_concurrent_cells": workers,
        "cell_status_path": str(status_path.resolve()),
        "cell_status_sha256": sha256_file(status_path),
        "freeze_path": str((output_root / "FREEZE.json").resolve()),
        "freeze_sha256": sha256_file(output_root / "FREEZE.json"),
        "closure_gates": gates,
        "failed_gates": failed_gates,
        "all_gates_passed": overall_pass,
        "overall_pass": overall_pass,
    }


def execute(args: argparse.Namespace, result: PreflightResult) -> int:
    assert_execution_exclusive(args)
    # Preflight established nonexistence; exist_ok=False closes the launch race.
    args.output_root.mkdir(parents=True, exist_ok=False)
    attempt_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    append_lock = threading.Lock()
    attempts_path = args.output_root / "ATTEMPTS.jsonl"
    status_path = args.output_root / "CELL_STATUS.jsonl"
    append_jsonl(
        attempts_path,
        {
            "event": "attempt_started",
            "attempt_id": attempt_id,
            "timestamp_utc": utc_now(),
            "workers": args.workers,
            "fresh_cells": len(result.jobs),
            "fresh_subprocesses_planned": len(result.jobs),
            "reuse_cells": 0,
            "resume_permitted": False,
            "purge_permitted": False,
            "runner_sha256": EXPECTED_PRODUCTION_RUNNER_SHA256,
        },
        append_lock,
    )
    try:
        frozen = freeze_execution_inputs(args.output_root, args, result)
    except Exception as exc:
        append_jsonl(
            attempts_path,
            {
                "event": "attempt_finished",
                "status": "freeze_failed",
                "attempt_id": attempt_id,
                "timestamp_utc": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "output_retained": True,
            },
            append_lock,
        )
        raise

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_cell,
                job,
                attempt_id=attempt_id,
                output_root=args.output_root,
                frozen=frozen,
                result=result,
                status_path=status_path,
                append_lock=append_lock,
            ): job
            for job in result.jobs
        }
        for future in as_completed(futures):
            records.append(future.result())

    closure = build_exact_closure(
        result=result,
        records=records,
        output_root=args.output_root,
        status_path=status_path,
        attempt_id=attempt_id,
        workers=args.workers,
    )
    closure_path = args.output_root / "FRESH96_CLOSURE.json"
    write_json_new(closure_path, closure)
    final = {
        "event": "attempt_finished",
        "status": "complete_validated" if closure["overall_pass"] else "failed",
        "attempt_id": attempt_id,
        "timestamp_utc": utc_now(),
        "fresh_cells_expected": EXPECTED_CELLS,
        "fresh_cells_validated": closure["fresh_cells_validated"],
        "failed_cells": closure["failed_cells"],
        "closure_path": str(closure_path.resolve()),
        "closure_sha256": sha256_file(closure_path),
        "cell_status_path": str(status_path.resolve()),
        "cell_status_sha256": sha256_file(status_path),
        "freeze_path": str((args.output_root / "FREEZE.json").resolve()),
        "freeze_sha256": sha256_file(args.output_root / "FREEZE.json"),
        "output_retained": True,
    }
    append_jsonl(attempts_path, final, append_lock)
    write_json_new(args.output_root / f"ATTEMPT_SUMMARY.{attempt_id}.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0 if closure["overall_pass"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--matrix-manifest", type=Path, required=True)
    parser.add_argument("--pilot-decision", type=Path, required=True)
    parser.add_argument("--completion-index", type=Path, required=True)
    parser.add_argument("--anchor-catalog", type=Path, required=True)
    parser.add_argument("--weather-map", type=Path, required=True)
    parser.add_argument("--physical-map", type=Path, required=True)
    parser.add_argument("--cmip-root", type=Path, required=True)
    parser.add_argument("--base-launcher", type=Path, required=True)
    parser.add_argument("--authorization-runner", type=Path, required=True)
    parser.add_argument("--production-runner", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-metrics", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--idf", type=Path, required=True)
    parser.add_argument("--environment-record", type=Path, required=True)
    parser.add_argument("--eplus-root", type=Path, required=True)
    parser.add_argument("--mechanism-report", type=Path, required=True)
    parser.add_argument("--smoke-closure", type=Path, required=True)
    parser.add_argument("--affected24-closure", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=MAX_CONCURRENT_CELLS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="Rehash and validate the exact plan without filesystem mutation/subprocesses.",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Create the one fresh root and launch 96 isolated production subprocesses.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = preflight(args)
    summary = preflight_summary(args, result)
    if args.preflight_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    return execute(args, result)


if __name__ == "__main__":
    raise SystemExit(main())
