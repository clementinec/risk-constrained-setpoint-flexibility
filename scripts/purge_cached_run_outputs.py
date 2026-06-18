#!/usr/bin/env python3
"""Purge bulky Paper B run artifacts after compact summaries exist.

This script is intentionally conservative. By default it only reports candidate
deletions. Use --apply to remove files/directories.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS = ROOT / "Probabilities_ENB" / "paperB_control" / "runs"


def size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def human_size(n_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}TB"


def has_summary(run_dir: Path) -> bool:
    summary_dir = run_dir / "summary"
    return summary_dir.exists() and any(summary_dir.glob("*.csv"))


def candidate_paths(run_dir: Path, keep_case_traces: bool) -> list[Path]:
    paths: list[Path] = []
    energyplus_dir = run_dir / "energyplus"
    if energyplus_dir.exists():
        paths.append(energyplus_dir)

    traces_dir = run_dir / "traces"
    if traces_dir.exists():
        combined = traces_dir / "medium_office_control_traces.csv"
        if combined.exists():
            paths.append(combined)
        if not keep_case_traces:
            for suffix in ("*.csv", "*.parquet"):
                for trace in traces_dir.glob(suffix):
                    if trace != combined:
                        paths.append(trace)
    return paths


def purge_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Specific run-directory name to inspect. May be repeated. Defaults to all runs.",
    )
    parser.add_argument(
        "--keep-case-traces",
        action="store_true",
        help="Keep per-case trace CSVs and purge only EnergyPlus outputs plus combined traces.",
    )
    parser.add_argument(
        "--allow-without-summary",
        action="store_true",
        help="Allow purging a run even if no CSV exists under its summary directory.",
    )
    parser.add_argument("--apply", action="store_true", help="Actually delete candidate artifacts.")
    args = parser.parse_args()

    run_dirs = (
        [args.runs_dir / name for name in args.run]
        if args.run
        else [path for path in args.runs_dir.iterdir() if path.is_dir()]
    )

    total = 0
    for run_dir in sorted(run_dirs):
        if not run_dir.exists():
            print(f"[missing] {run_dir}")
            continue
        if not args.allow_without_summary and not has_summary(run_dir):
            print(f"[skip-no-summary] {run_dir}")
            continue
        paths = candidate_paths(run_dir, keep_case_traces=args.keep_case_traces)
        reclaim = sum(size_bytes(path) for path in paths)
        total += reclaim
        print(f"[candidate] {run_dir.name}: {human_size(reclaim)}")
        for path in paths:
            print(f"  - {human_size(size_bytes(path)):>8} {path.relative_to(args.runs_dir)}")
        if args.apply:
            for path in paths:
                purge_path(path)
            print(f"[purged] {run_dir.name}")

    print(f"[total-candidate] {human_size(total)}")
    if not args.apply:
        print("[dry-run] no files deleted; rerun with --apply to purge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
