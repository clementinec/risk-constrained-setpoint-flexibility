#!/usr/bin/env python3
"""Validated Parquet storage primitives for Paper B ENB Rev01.

This module is intentionally narrower than a general CSV converter.  It stages
only the five known EnergyPlus CSV exports, preserves every source until the
entire stage has passed, and requires a separate explicit finalization before
source CSVs may be removed.  It also provides the native trace writer that the
versioned Rev01 runner will use.

The CLI is restricted to *open staging roots* below ``09_rev01/05_reruns``.
Accepted Fresh96 outputs, the completed hydration smoke, and any already closed
Rev01 batch are therefore outside its mutation boundary.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any
import uuid

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA_VERSION = "paperb_enb_rev01_storage_v1"
STAGE_CLOSURE_SCHEMA = "paperb_enb_rev01_storage_stage_closure_v1"
FINAL_CLOSURE_SCHEMA = "paperb_enb_rev01_storage_final_closure_v1"
CONVERTER_VERSION = "1.0.0"

TARGET_NAMES = frozenset(
    {
        "eplusout.csv",
        "eplusmtr.csv",
        "eplustbl.csv",
        "epluszsz.csv",
        "eplusssz.csv",
    }
)
RAGGED_NAMES = frozenset({"eplustbl.csv", "epluszsz.csv", "eplusssz.csv"})

# These remain in their native formats by scientific contract.  The list is
# documentary as well as testable: arbitrary ``*.csv`` recursion is forbidden.
PATH_BOUND_CSV_NAMES = frozenset(
    {
        "paperb_weather_warmup_setpoint_audit.csv",
        "medium_office_trace_summary.csv",
    }
)

PARQUET_VERSION = "2.6"
PARQUET_COMPRESSION = "zstd"
ENERGYPLUS_COMPRESSION_LEVEL = 3
TRACE_COMPRESSION_LEVEL = 7
ENERGYPLUS_ROW_GROUP_SIZE = 65_536
TRACE_ROW_GROUP_SIZE = 8_192
PAGE_SIZE_BYTES = 1_048_576

CLOSED_MARKER_NAMES = frozenset(
    {
        "SMOKE_CLOSURE.json",
        "FRESH96_CLOSURE.json",
        "BATCH_CLOSURE.json",
        "COMPACT_CLOSURE.json",
        "FINAL_CLOSURE.json",
    }
)


class StorageContractError(RuntimeError):
    """A preflight, conversion, or finalization condition failed closed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    physical_lines = 0
    final_byte = b""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
            physical_lines += block.count(b"\n")
            final_byte = block[-1:]
    stat = path.stat()
    if stat.st_size and final_byte != b"\n":
        physical_lines += 1
    return {
        "sha256": digest.hexdigest(),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "physical_line_count": int(physical_lines),
    }


def fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def place_new_atomic(temporary: Path, destination: Path) -> None:
    """Atomically publish a same-directory file without replacement semantics."""

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to replace destination: {destination}")
    os.link(temporary, destination)
    temporary.unlink()
    fsync_directory(destination.parent)


def write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path: Path, value: Any) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StorageContractError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StorageContractError(f"JSON document must be an object: {path}")
    return value


def converter_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def relative_posix(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise StorageContractError(f"Path escapes conversion root: {path}") from exc


def attach_metadata(table: pa.Table, metadata: dict[str, str]) -> pa.Table:
    existing = dict(table.schema.metadata or {})
    existing.update(
        {f"paperb.{key}".encode("utf-8"): str(value).encode("utf-8") for key, value in metadata.items()}
    )
    return table.replace_schema_metadata(existing)


def parquet_write_options(
    table: pa.Table,
    *,
    compression_level: int,
    row_group_size: int,
) -> dict[str, Any]:
    numeric = [
        field.name
        for field in table.schema
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type)
    ]
    dictionary = [field.name for field in table.schema if field.name not in numeric]
    return {
        "version": PARQUET_VERSION,
        "compression": PARQUET_COMPRESSION,
        "compression_level": compression_level,
        "use_dictionary": dictionary,
        "use_byte_stream_split": numeric,
        "row_group_size": row_group_size,
        "write_statistics": True,
        "data_page_size": PAGE_SIZE_BYTES,
        "dictionary_pagesize_limit": PAGE_SIZE_BYTES,
        "store_schema": True,
        "use_compliant_nested_type": True,
        "write_page_checksum": True,
    }


def assert_float_bits_equal(left: pd.DataFrame, right: pd.DataFrame) -> None:
    for column in left.columns:
        if not pd.api.types.is_float_dtype(left[column].dtype):
            continue
        left_values = left[column].to_numpy(dtype=np.float64, copy=False)
        right_values = right[column].to_numpy(dtype=np.float64, copy=False)
        left_nan = np.isnan(left_values)
        right_nan = np.isnan(right_values)
        if not np.array_equal(left_nan, right_nan):
            raise AssertionError(f"NaN mask differs after Parquet round trip: {column}")
        comparable = ~left_nan
        if not np.array_equal(
            left_values[comparable].view(np.uint64),
            right_values[comparable].view(np.uint64),
        ):
            raise AssertionError(f"Float64 bits differ after Parquet round trip: {column}")


def read_rectangular_csv(source: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise StorageContractError(f"Rectangular CSV is empty: {source}") from exc
        if len(header) != len(set(header)):
            raise StorageContractError(f"Rectangular CSV header is not unique: {source}")
        data_rows = 0
        minimum_fields = len(header)
        maximum_fields = len(header)
        for row in reader:
            data_rows += 1
            minimum_fields = min(minimum_fields, len(row))
            maximum_fields = max(maximum_fields, len(row))
    if minimum_fields != len(header) or maximum_fields != len(header):
        raise StorageContractError(
            f"Rectangular target has ragged rows ({minimum_fields}..{maximum_fields}, "
            f"header {len(header)}): {source}"
        )
    frame = pd.read_csv(source, low_memory=False, float_precision="round_trip")
    if list(frame.columns) != header or len(frame) != data_rows:
        raise StorageContractError(f"Rectangular CSV parse identity failed: {source}")
    return frame, {
        "row_count": int(data_rows),
        "column_count": int(len(header)),
        "minimum_field_count": int(minimum_fields),
        "maximum_field_count": int(maximum_fields),
    }


def build_ragged_table(source: Path) -> tuple[pa.Table, list[list[str]], dict[str, Any]]:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    table = pa.table(
        {
            "row_number": pa.array(range(1, len(rows) + 1), type=pa.int64()),
            "fields": pa.array(rows, type=pa.list_(pa.string())),
        }
    )
    field_counts = [len(row) for row in rows]
    return table, rows, {
        "row_count": int(len(rows)),
        "column_count": None,
        "minimum_field_count": int(min(field_counts, default=0)),
        "maximum_field_count": int(max(field_counts, default=0)),
    }


def validate_parquet_metadata(path: Path, expected_source_sha256: str) -> None:
    metadata = pq.read_metadata(path).metadata or {}
    observed = metadata.get(b"paperb.source_sha256", b"").decode("utf-8")
    if observed != expected_source_sha256:
        raise AssertionError(f"Embedded source hash differs in {path}")


def write_energyplus_conversion(source: Path, root: Path) -> dict[str, Any]:
    source = source.resolve()
    root = root.resolve()
    if source.name.casefold() not in TARGET_NAMES:
        raise StorageContractError(f"CSV is not in the EnergyPlus allowlist: {source}")
    destination = source.with_suffix(".parquet")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to replace Parquet destination: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Temporary destination already exists: {temporary}")

    identity = source_identity(source)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "conversion_mode": "",
        "source_relative_path": relative_posix(source, root),
        "source_sha256": identity["sha256"],
        "source_bytes": identity["bytes"],
        "source_mtime_ns": identity["mtime_ns"],
        "source_physical_line_count": identity["physical_line_count"],
        "source_encoding": "utf-8-sig",
        "source_csv_parser": "python.csv.reader",
        "compression": f"zstd_level_{ENERGYPLUS_COMPRESSION_LEVEL}",
        "parquet_version": PARQUET_VERSION,
        "row_group_size": ENERGYPLUS_ROW_GROUP_SIZE,
        "page_checksum": "true",
        "converter_version": CONVERTER_VERSION,
        "converter_sha256": converter_sha256(),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "pyarrow_version": pa.__version__,
    }

    try:
        if source.name.casefold() in RAGGED_NAMES:
            table, original_rows, details = build_ragged_table(source)
            mode = "ragged_csv_cells"
            metadata["conversion_mode"] = mode
            metadata.update({key: value for key, value in details.items() if value is not None})
            table = attach_metadata(table, metadata)
            pq.write_table(
                table,
                temporary,
                **parquet_write_options(
                    table,
                    compression_level=ENERGYPLUS_COMPRESSION_LEVEL,
                    row_group_size=ENERGYPLUS_ROW_GROUP_SIZE,
                ),
            )
            restored = pq.read_table(temporary, page_checksum_verification=True)
            if restored.column("row_number").to_pylist() != list(range(1, len(original_rows) + 1)):
                raise AssertionError(f"Ragged row numbers differ after round trip: {source}")
            if restored.column("fields").to_pylist() != original_rows:
                raise AssertionError(f"Ragged parsed cells differ after round trip: {source}")
            validation = "exact_parsed_cell_roundtrip"
        else:
            frame, details = read_rectangular_csv(source)
            mode = "rectangular_typed"
            metadata["conversion_mode"] = mode
            metadata.update(details)
            table = attach_metadata(pa.Table.from_pandas(frame, preserve_index=False), metadata)
            pq.write_table(
                table,
                temporary,
                **parquet_write_options(
                    table,
                    compression_level=ENERGYPLUS_COMPRESSION_LEVEL,
                    row_group_size=ENERGYPLUS_ROW_GROUP_SIZE,
                ),
            )
            restored = pq.read_table(temporary, page_checksum_verification=True).to_pandas()
            pd.testing.assert_frame_equal(frame, restored, check_dtype=True, check_exact=True)
            assert_float_bits_equal(frame, restored)
            validation = "exact_dataframe_and_float_bit_roundtrip"

        fsync_file(temporary)
        validate_parquet_metadata(temporary, identity["sha256"])
        parquet_hash = sha256_file(temporary)
        parquet_bytes = temporary.stat().st_size
        place_new_atomic(temporary, destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    if sha256_file(source) != identity["sha256"]:
        raise StorageContractError(f"Source changed during conversion: {source}")
    if sha256_file(destination) != parquet_hash:
        raise StorageContractError(f"Parquet changed after atomic placement: {destination}")

    return {
        "schema_version": SCHEMA_VERSION,
        "conversion_mode": mode,
        "source_relative_path": relative_posix(source, root),
        "parquet_relative_path": relative_posix(destination, root),
        "source_sha256": identity["sha256"],
        "source_bytes": identity["bytes"],
        "source_mtime_ns": identity["mtime_ns"],
        "source_physical_line_count": identity["physical_line_count"],
        "parquet_sha256": parquet_hash,
        "parquet_bytes": int(parquet_bytes),
        "bytes_saved_after_finalize": int(identity["bytes"] - parquet_bytes),
        "source_retained_during_stage": True,
        "validation": validation,
        "converted_at_utc": utc_now(),
        **details,
    }


def write_native_trace_parquet(frame: pd.DataFrame, destination: Path) -> dict[str, Any]:
    """Write a new Rev01 trace directly to the optimized final Parquet contract."""

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to replace trace: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    table = pa.Table.from_pandas(frame, preserve_index=False)
    try:
        pq.write_table(
            table,
            temporary,
            **parquet_write_options(
                table,
                compression_level=TRACE_COMPRESSION_LEVEL,
                row_group_size=TRACE_ROW_GROUP_SIZE,
            ),
        )
        restored = pq.read_table(temporary, page_checksum_verification=True).to_pandas()
        pd.testing.assert_frame_equal(frame, restored, check_dtype=True, check_exact=True)
        assert_float_bits_equal(frame, restored)
        fsync_file(temporary)
        final_hash = sha256_file(temporary)
        final_bytes = temporary.stat().st_size
        place_new_atomic(temporary, destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return {
        "schema_version": SCHEMA_VERSION,
        "path": str(destination.resolve()),
        "sha256": final_hash,
        "bytes": int(final_bytes),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "compression": f"zstd_level_{TRACE_COMPRESSION_LEVEL}",
        "row_group_size": TRACE_ROW_GROUP_SIZE,
        "page_checksum": True,
        "validation": "exact_dataframe_and_float_bit_roundtrip",
    }


def scan_candidates(root: Path) -> list[Path]:
    root = root.resolve()
    return sorted(
        path.resolve()
        for path in root.rglob("*.csv")
        if path.name.casefold() in TARGET_NAMES
    )


def scan_temp_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob(".*.tmp-*") if path.is_file())


def preflight_summary(root: Path) -> dict[str, Any]:
    candidates = scan_candidates(root)
    collisions = [path.with_suffix(".parquet") for path in candidates if path.with_suffix(".parquet").exists()]
    return {
        "schema_version": SCHEMA_VERSION,
        "root": str(root.resolve()),
        "candidate_count": len(candidates),
        "candidate_bytes": int(sum(path.stat().st_size for path in candidates)),
        "destination_collision_count": len(collisions),
        "temporary_file_count": len(scan_temp_files(root)),
        "source_paths": [relative_posix(path, root) for path in candidates],
        "preflight_passed": not collisions and not scan_temp_files(root),
        "filesystem_mutated": False,
    }


def write_manifest_parquet(records: list[dict[str, Any]], path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to replace storage manifest: {path}")
    frame = pd.DataFrame.from_records(records)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    pq.write_table(
        table,
        temporary,
        **parquet_write_options(
            table,
            compression_level=ENERGYPLUS_COMPRESSION_LEVEL,
            row_group_size=ENERGYPLUS_ROW_GROUP_SIZE,
        ),
    )
    restored = pq.read_table(temporary, page_checksum_verification=True).to_pandas()
    pd.testing.assert_frame_equal(frame, restored, check_dtype=True, check_exact=True)
    fsync_file(temporary)
    place_new_atomic(temporary, path)


def stage_conversions(root: Path, manifest_dir: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_dir = manifest_dir.resolve()
    if manifest_dir.exists() or manifest_dir.is_symlink():
        raise FileExistsError(f"Stage manifest directory must be absent: {manifest_dir}")
    if root not in manifest_dir.parents:
        raise StorageContractError("Stage manifest directory must be inside the staging root")
    summary = preflight_summary(root)
    if not summary["preflight_passed"]:
        raise StorageContractError(f"Storage preflight failed: {summary}")
    if summary["candidate_count"] == 0:
        raise StorageContractError("Storage stage requires at least one allowlisted CSV")
    manifest_dir.mkdir(parents=True, exist_ok=False)
    journal = manifest_dir / "STAGE_JOURNAL.jsonl"
    records: list[dict[str, Any]] = []
    try:
        for source in scan_candidates(root):
            record = write_energyplus_conversion(source, root)
            append_jsonl(journal, record)
            records.append(record)
        manifest_path = manifest_dir / "STORAGE_MANIFEST.parquet"
        write_manifest_parquet(records, manifest_path)
        independent_failures: list[str] = []
        for record in records:
            source = root / record["source_relative_path"]
            parquet = root / record["parquet_relative_path"]
            if sha256_file(source) != record["source_sha256"]:
                independent_failures.append(f"source_hash:{record['source_relative_path']}")
            if sha256_file(parquet) != record["parquet_sha256"]:
                independent_failures.append(f"parquet_hash:{record['parquet_relative_path']}")
            validate_parquet_metadata(parquet, record["source_sha256"])
        temporary_file_count = len(scan_temp_files(root))
        if temporary_file_count:
            independent_failures.append(f"temporary_files:{temporary_file_count}")
        closure = {
            "schema_version": STAGE_CLOSURE_SCHEMA,
            "status": "PASS" if not independent_failures else "FAIL",
            "all_gates_passed": not independent_failures,
            "created_utc": utc_now(),
            "root": str(root),
            "candidate_count": len(records),
            "source_csvs_retained": len(records),
            "parquets_staged": len(records),
            "source_bytes": int(sum(record["source_bytes"] for record in records)),
            "parquet_bytes": int(sum(record["parquet_bytes"] for record in records)),
            "projected_bytes_saved_after_finalize": int(
                sum(record["bytes_saved_after_finalize"] for record in records)
            ),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "journal_path": str(journal),
            "journal_sha256": sha256_file(journal) if journal.exists() else None,
            "independent_failures": independent_failures,
            "temporary_file_count": temporary_file_count,
        }
        write_json_new(manifest_dir / "STAGE_CLOSURE.json", closure)
    except Exception as exc:
        write_json_new(
            manifest_dir / "STAGE_FAILURE.json",
            {
                "schema_version": STAGE_CLOSURE_SCHEMA,
                "status": "FAIL",
                "all_gates_passed": False,
                "created_utc": utc_now(),
                "root": str(root),
                "completed_records": len(records),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "sources_removed": 0,
            },
        )
        raise
    if not closure["all_gates_passed"]:
        raise StorageContractError(f"Independent stage closure failed: {closure}")
    return closure


def manifest_records(path: Path) -> list[dict[str, Any]]:
    frame = pq.read_table(path, page_checksum_verification=True).to_pandas()
    return frame.where(pd.notna(frame), None).to_dict(orient="records")


def resolve_record_path(root: Path, value: Any, *, kind: str) -> Path:
    relative = Path(str(value))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise StorageContractError(f"Unsafe {kind} relative path: {value!r}")
    candidate = (root / relative).resolve()
    if root not in candidate.parents:
        raise StorageContractError(f"{kind} path escapes finalization root: {value!r}")
    return candidate


def finalize_conversions(
    root: Path,
    manifest_dir: Path,
    *,
    remove_source_csvs: bool,
) -> dict[str, Any]:
    if not remove_source_csvs:
        raise StorageContractError("Finalization requires explicit remove_source_csvs=True")
    root = root.resolve()
    manifest_dir = manifest_dir.resolve()
    stage_closure_path = manifest_dir / "STAGE_CLOSURE.json"
    stage = read_json_object(stage_closure_path)
    if stage.get("schema_version") != STAGE_CLOSURE_SCHEMA or stage.get("all_gates_passed") is not True:
        raise StorageContractError("A passing storage stage closure is required")
    if Path(str(stage.get("root", ""))).resolve() != root:
        raise StorageContractError("Stage closure root does not match finalization root")
    if manifest_dir.parent != root and root not in manifest_dir.parents:
        raise StorageContractError("Storage manifest directory escapes finalization root")
    final_path = manifest_dir / "FINAL_CLOSURE.json"
    if final_path.exists() or final_path.is_symlink():
        raise FileExistsError(f"Final closure already exists: {final_path}")
    manifest_path = Path(str(stage["manifest_path"])).resolve()
    expected_manifest_path = (manifest_dir / "STORAGE_MANIFEST.parquet").resolve()
    if manifest_path != expected_manifest_path:
        raise StorageContractError("Stage closure points to an unexpected storage manifest")
    if sha256_file(manifest_path) != stage["manifest_sha256"]:
        raise StorageContractError("Storage manifest changed after staging")
    records = manifest_records(manifest_path)
    if len(records) != int(stage.get("candidate_count", -1)):
        raise StorageContractError("Storage manifest record count differs from stage closure")
    seen_sources: set[Path] = set()
    seen_parquets: set[Path] = set()

    # Phase-B preflight validates the entire batch before the first source is removed.
    for record in records:
        source = resolve_record_path(
            root, record["source_relative_path"], kind="source"
        )
        parquet = resolve_record_path(
            root, record["parquet_relative_path"], kind="parquet"
        )
        if source in seen_sources or parquet in seen_parquets:
            raise StorageContractError("Storage manifest contains duplicate paths")
        seen_sources.add(source)
        seen_parquets.add(parquet)
        if source.name.casefold() not in TARGET_NAMES or source.suffix.casefold() != ".csv":
            raise StorageContractError(f"Manifest source is outside the CSV allowlist: {source}")
        if parquet != source.with_suffix(".parquet"):
            raise StorageContractError("Manifest source/Parquet path pairing differs")
        if not source.is_file() or sha256_file(source) != record["source_sha256"]:
            raise StorageContractError(f"Staged source missing or changed: {source}")
        if not parquet.is_file() or sha256_file(parquet) != record["parquet_sha256"]:
            raise StorageContractError(f"Staged Parquet missing or changed: {parquet}")
        validate_parquet_metadata(parquet, str(record["source_sha256"]))

    finalize_journal = manifest_dir / "FINALIZE_JOURNAL.jsonl"
    removed = 0
    for record in records:
        source = resolve_record_path(
            root, record["source_relative_path"], kind="source"
        )
        source.unlink()
        fsync_directory(source.parent)
        append_jsonl(
            finalize_journal,
            {
                "source_relative_path": record["source_relative_path"],
                "source_sha256": record["source_sha256"],
                "parquet_relative_path": record["parquet_relative_path"],
                "parquet_sha256": record["parquet_sha256"],
                "removed_at_utc": utc_now(),
            },
        )
        removed += 1

    remaining = [
        record["source_relative_path"]
        for record in records
        if resolve_record_path(
            root, record["source_relative_path"], kind="source"
        ).exists()
    ]
    parquet_failures = [
        record["parquet_relative_path"]
        for record in records
        if not resolve_record_path(
            root, record["parquet_relative_path"], kind="parquet"
        ).is_file()
        or sha256_file(
            resolve_record_path(
                root, record["parquet_relative_path"], kind="parquet"
            )
        )
        != record["parquet_sha256"]
    ]
    temporary_file_count = len(scan_temp_files(root))
    closure = {
        "schema_version": FINAL_CLOSURE_SCHEMA,
        "status": "PASS"
        if not remaining and not parquet_failures and temporary_file_count == 0
        else "FAIL",
        "all_gates_passed": not remaining
        and not parquet_failures
        and temporary_file_count == 0,
        "created_utc": utc_now(),
        "root": str(root),
        "staged_count": len(records),
        "source_csvs_removed": removed,
        "remaining_source_csvs": remaining,
        "parquet_failures": parquet_failures,
        "bytes_saved": int(sum(int(record["bytes_saved_after_finalize"]) for record in records)),
        "stage_closure_path": str(stage_closure_path),
        "stage_closure_sha256": sha256_file(stage_closure_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "finalize_journal_path": str(finalize_journal),
        "finalize_journal_sha256": sha256_file(finalize_journal),
        "temporary_file_count": temporary_file_count,
    }
    write_json_new(final_path, closure)
    if not closure["all_gates_passed"]:
        raise StorageContractError(f"Storage finalization failed: {closure}")
    return closure


def rev01_rerun_root() -> Path:
    return Path(__file__).resolve().parents[1]


def validate_cli_staging_root(root: Path) -> Path:
    base = rev01_rerun_root().resolve()
    lexical = root.absolute()
    try:
        lexical_parts = lexical.relative_to(base).parts
    except ValueError as exc:
        raise StorageContractError("Lexical root is outside the Rev01 rerun directory") from exc
    current = base
    for part in lexical_parts:
        current = current / part
        if current.is_symlink():
            raise StorageContractError(f"Symlink path component is forbidden: {current}")
    try:
        resolved = root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise StorageContractError(f"Staging root does not exist: {root}") from exc
    if not resolved.is_dir():
        raise StorageContractError(f"Staging root is not a directory: {root}")
    if resolved == base or base not in resolved.parents:
        raise StorageContractError(f"Root must be a descendant of {base}: {root}")
    for marker in CLOSED_MARKER_NAMES:
        if any(resolved.rglob(marker)):
            raise StorageContractError(
                f"Refusing to mutate a root containing final closure marker {marker}: {resolved}"
            )
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--stage", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    parser.add_argument(
        "--remove-source-csvs",
        action="store_true",
        help="Required only with --finalize after a passing two-phase stage.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = validate_cli_staging_root(args.root)
    manifest_dir = (
        args.manifest_dir.resolve()
        if args.manifest_dir is not None
        else root / "storage_migration"
    )
    if args.preflight_only:
        if args.remove_source_csvs:
            raise StorageContractError("Preflight cannot request source removal")
        print(json.dumps(preflight_summary(root), indent=2, sort_keys=True))
        return 0
    if args.stage:
        if args.remove_source_csvs:
            raise StorageContractError("Stage always retains source CSVs")
        result = stage_conversions(root, manifest_dir)
    else:
        result = finalize_conversions(
            root,
            manifest_dir,
            remove_source_csvs=args.remove_source_csvs,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
