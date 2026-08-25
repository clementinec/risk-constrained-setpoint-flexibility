#!/usr/bin/env python3
"""Validate the annual Rev01 submitted-settings sentinel against SCALED-96-028."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


SCHEMA_VERSION = "paperb_enb_rev01_annual_parity_closure_v1"
REBUILD_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ACCEPTED_CELL = (
    REBUILD_ROOT
    / "06_runs/scaled/20260803_routeS96_v2_envscoped/cells/SCALED-96-028"
)
EXPECTED_RUNNER_SHA256 = "4c061fc3b25f7ee6fe66df00cfacfab42f1d3ab171185306fe1ce161a84a0baf"
EXPECTED_RUNNER_HELPER_SHA256 = "b941c0bfc4beb3dc44557ed341bb4e018992620b0d3ad4f3bcb6973558d78bf7"
SQL_PARITY_TABLES = (
    "ComponentSizes",
    "SystemSizes",
    "ZoneSizes",
    "TabularDataWithStrings",
    "Errors",
)


class ParityError(RuntimeError):
    """The annual parity contract failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def one_file(root: Path, pattern: str, *, name: str) -> Path:
    paths = sorted(root.glob(pattern))
    if len(paths) != 1:
        raise ParityError(f"Expected exactly one {name} under {root}: {paths}")
    return paths[0]


def compare_dataframes(
    accepted: pd.DataFrame,
    revision: pd.DataFrame,
    *,
    name: str,
) -> dict[str, Any]:
    missing = [column for column in accepted.columns if column not in revision.columns]
    if missing:
        raise ParityError(f"Revision {name} dropped accepted columns: {missing}")
    selected = revision.loc[:, list(accepted.columns)]
    try:
        pd.testing.assert_frame_equal(
            accepted,
            selected,
            check_dtype=True,
            check_exact=True,
            check_column_type=True,
            check_index_type=True,
        )
    except AssertionError as exc:
        raise ParityError(f"Canonical {name} differs: {exc}") from exc
    return {
        "accepted_rows": int(len(accepted)),
        "revision_rows": int(len(revision)),
        "accepted_columns": int(len(accepted.columns)),
        "revision_columns": int(len(revision.columns)),
        "accepted_column_subset_exact": True,
        "revision_only_columns": sorted(set(revision.columns) - set(accepted.columns)),
    }


def compare_json_contracts(
    accepted_path: Path,
    revision_path: Path,
    *,
    ignored_keys: set[str] | None = None,
) -> dict[str, Any]:
    ignored = ignored_keys or set()
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    revision = json.loads(revision_path.read_text(encoding="utf-8"))
    for key in ignored:
        accepted.pop(key, None)
        revision.pop(key, None)
    if accepted != revision:
        raise ParityError(f"Control JSON contract differs: {accepted_path.name}")
    return {
        "accepted_path": str(accepted_path.resolve()),
        "revision_path": str(revision_path.resolve()),
        "ignored_path_only_keys": sorted(ignored),
        "canonical_contract_exact": True,
    }


def sqlite_table_sha256(path: Path, table: str) -> str:
    digest = hashlib.sha256()
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
        schema = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
        if not schema:
            raise ParityError(f"Missing SQL parity table {table}: {path}")
        digest.update(json.dumps(schema, separators=(",", ":")).encode("utf-8"))
        query = f'SELECT * FROM "{table}"'
        if table == "TabularDataWithStrings":
            query += (
                " WHERE NOT (ReportName='InputVerificationandResultsSummary'"
                " AND TableName='General' AND RowName='Program Version and Build')"
            )
        cursor = connection.execute(query)
        encoded_rows = sorted(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            for row in cursor
        )
        for encoded in encoded_rows:
            digest.update(encoded.encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def validate(
    accepted_cell: Path,
    new_batch_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    if accepted_cell.resolve() != DEFAULT_ACCEPTED_CELL.resolve():
        raise ParityError("Accepted parity comparator is not the pinned SCALED-96-028 cell")
    batch_closure_path = new_batch_root / "BATCH_CLOSURE.json"
    batch = json.loads(batch_closure_path.read_text(encoding="utf-8"))
    if batch.get("all_gates_passed") is not True or batch.get("batch_id") != "rev01_parity1":
        raise ParityError("New parity batch does not have a passing exact closure")
    if batch.get("planned_cells") != 1 or batch.get("passed_cells") != 1:
        raise ParityError("New parity batch is not exactly one passing cell")
    if not (
        batch.get("runner_sha256") == EXPECTED_RUNNER_SHA256
        and batch.get("runner_helper_sha256") == EXPECTED_RUNNER_HELPER_SHA256
    ):
        raise ParityError("Parity batch used an unauthorized runner/helper identity")
    new_cell = new_batch_root / "cells/REV01-PARITY-001"
    accepted_trace = one_file(accepted_cell / "traces", "*.parquet", name="accepted trace")
    revision_trace = one_file(new_cell / "traces", "*.parquet", name="revision trace")
    accepted_summary = accepted_cell / "summary/medium_office_trace_summary.csv"
    revision_summary = new_cell / "summary/medium_office_trace_summary.csv"
    accepted_idf = one_file(accepted_cell / "model", "*.idf", name="accepted generated IDF")
    revision_idf = one_file(new_cell / "model", "*.idf", name="revision generated IDF")
    accepted_eplus = one_file(
        accepted_cell / "energyplus", "**/eplusout.sql", name="accepted EnergyPlus SQL"
    ).parent
    revision_eplus = one_file(
        new_cell / "energyplus", "**/eplusout.sql", name="revision EnergyPlus SQL"
    ).parent

    accepted_trace_frame = pq.read_table(
        accepted_trace, page_checksum_verification=True
    ).to_pandas()
    revision_trace_frame = pq.read_table(
        revision_trace, page_checksum_verification=True
    ).to_pandas()
    trace = compare_dataframes(
        accepted_trace_frame,
        revision_trace_frame,
        name="annual trace",
    )
    summary = compare_dataframes(
        pd.read_csv(accepted_summary),
        pd.read_csv(revision_summary),
        name="summary",
    )
    accepted_idf_hash = sha256_file(accepted_idf)
    revision_idf_hash = sha256_file(revision_idf)
    if accepted_idf_hash != revision_idf_hash:
        raise ParityError("Generated IDF hash differs")
    warmup = compare_dataframes(
        pd.read_csv(accepted_eplus / "paperb_weather_warmup_setpoint_audit.csv"),
        pd.read_csv(revision_eplus / "paperb_weather_warmup_setpoint_audit.csv"),
        name="warmup setpoint audit",
    )
    environment = compare_json_contracts(
        accepted_eplus / "PAPERB_CONTROL_ENVIRONMENT_AUDIT.json",
        revision_eplus / "PAPERB_CONTROL_ENVIRONMENT_AUDIT.json",
        ignored_keys={"warmup_audit_csv"},
    )
    sql_tables: dict[str, Any] = {}
    for table in SQL_PARITY_TABLES:
        accepted_hash = sqlite_table_sha256(accepted_eplus / "eplusout.sql", table)
        revision_hash = sqlite_table_sha256(revision_eplus / "eplusout.sql", table)
        if accepted_hash != revision_hash:
            raise ParityError(f"Canonical EnergyPlus SQL table differs: {table}")
        sql_tables[table] = {
            "canonical_sha256": accepted_hash,
            "exact": True,
        }
    output_dir.mkdir(parents=True, exist_ok=False)
    closure = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "all_gates_passed": True,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "accepted_cell": str(accepted_cell.resolve()),
        "accepted_cell_id": "SCALED-96-028",
        "accepted_cell_pinned": True,
        "revision_batch": str(new_batch_root.resolve()),
        "revision_cell_id": "REV01-PARITY-001",
        "batch_closure_path": str(batch_closure_path.resolve()),
        "batch_closure_sha256": sha256_file(batch_closure_path),
        "trace": trace,
        "summary": summary,
        "generated_idf_sha256": accepted_idf_hash,
        "generated_idf_exact": True,
        "accepted_trace_sha256": sha256_file(accepted_trace),
        "revision_trace_sha256": sha256_file(revision_trace),
        "parquet_byte_hash_expected_to_differ": True,
        "canonical_intersection_exact": True,
        "full_revision_trace_validated": True,
        "warmup_audit_exact": warmup["accepted_column_subset_exact"],
        "warmup_audit": warmup,
        "environment_control_contract_exact": environment[
            "canonical_contract_exact"
        ],
        "environment_control_contract": environment,
        "sql_sizing_tables_exact": True,
        "sql_parity_tables": sql_tables,
    }
    (output_dir / "PARITY_CLOSURE.json").write_text(
        json.dumps(closure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        {
            "accepted_column": list(accepted_trace_frame.columns),
            "present_in_revision": [
                column in revision_trace_frame.columns for column in accepted_trace_frame.columns
            ],
        }
    ).to_csv(output_dir / "trace_column_parity.csv", index=False)
    return closure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-cell", type=Path, default=DEFAULT_ACCEPTED_CELL)
    parser.add_argument("--new-batch-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate(args.accepted_cell, args.new_batch_root, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
