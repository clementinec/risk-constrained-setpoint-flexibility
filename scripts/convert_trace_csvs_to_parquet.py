#!/usr/bin/env python3
"""Convert Paper B trace CSV files to compressed Parquet tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS = ROOT / "Probabilities_ENB" / "paperB_control" / "runs"


def human_size(n_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}TB"


def convert_one(csv_path: Path, delete_csv: bool, overwrite: bool) -> tuple[int, int]:
    parquet_path = csv_path.with_suffix(".parquet")
    csv_size = csv_path.stat().st_size
    if parquet_path.exists() and not overwrite:
        parquet_size = parquet_path.stat().st_size
        print(
            f"[exists] {csv_path.name}: csv={human_size(csv_size)} "
            f"parquet={human_size(parquet_size)}"
        )
        return csv_size, parquet_size

    df = pd.read_csv(csv_path)
    df.to_parquet(parquet_path, index=False, compression="zstd")
    parquet_size = parquet_path.stat().st_size
    check_rows = pq.ParquetFile(parquet_path).metadata.num_rows
    if check_rows != len(df):
        raise RuntimeError(f"Row-count mismatch for {csv_path}: {len(df)} != {check_rows}")
    print(
        f"[converted] {csv_path.name}: csv={human_size(csv_size)} "
        f"parquet={human_size(parquet_size)} ratio={parquet_size / csv_size:.3f}"
    )
    if delete_csv:
        csv_path.unlink()
        print(f"[deleted-csv] {csv_path}")
    return csv_size, parquet_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Specific run-directory name to convert. May be repeated. Defaults to all runs.",
    )
    parser.add_argument(
        "--include-combined",
        action="store_true",
        help="Also convert medium_office_control_traces.csv combined traces.",
    )
    parser.add_argument("--delete-csv", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dirs = (
        [args.runs_dir / name for name in args.run]
        if args.run
        else [path for path in args.runs_dir.iterdir() if path.is_dir()]
    )
    csv_paths: list[Path] = []
    for run_dir in sorted(run_dirs):
        traces_dir = run_dir / "traces"
        if not traces_dir.exists():
            continue
        for csv_path in sorted(traces_dir.glob("*.csv")):
            if not args.include_combined and csv_path.name == "medium_office_control_traces.csv":
                continue
            csv_paths.append(csv_path)

    total_csv = sum(path.stat().st_size for path in csv_paths)
    print(f"[selected] {len(csv_paths)} CSV traces, total={human_size(total_csv)}")
    if args.dry_run:
        for path in csv_paths[:20]:
            print(f"  - {human_size(path.stat().st_size):>8} {path}")
        if len(csv_paths) > 20:
            print(f"  ... {len(csv_paths) - 20} more")
        return 0

    total_parquet = 0
    for csv_path in csv_paths:
        _csv_size, parquet_size = convert_one(
            csv_path,
            delete_csv=args.delete_csv,
            overwrite=args.overwrite,
        )
        total_parquet += parquet_size
    print(
        f"[done] csv_total={human_size(total_csv)} parquet_total={human_size(total_parquet)} "
        f"saved_if_csv_deleted={human_size(max(total_csv - total_parquet, 0))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
