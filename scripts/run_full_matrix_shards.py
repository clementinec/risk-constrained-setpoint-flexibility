#!/usr/bin/env python3
"""Run Paper B annual controller matrix in city-scenario shards."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "Probabilities_ENB" / "paperB_control" / "scripts" / "run_medium_office_paperB_control.py"
DEFAULT_MANIFEST = (
    ROOT
    / "Probabilities_ENB"
    / "paperB_control"
    / "diagnostics"
    / "paperb_full_144_weather_manifest.csv"
)
DEFAULT_RUN_ROOT = ROOT / "Probabilities_ENB" / "paperB_control" / "runs" / "full_matrix_realmet"
DEFAULT_SHARD_MANIFEST_DIR = (
    ROOT / "Probabilities_ENB" / "paperB_control" / "diagnostics" / "full_matrix_shards"
)

STRATEGIES = [
    "reference",
    "diagnostic_reference",
    "paperb_adaptive_band_relax",
    "paperb_pmv_exceedance_guard_relax",
    "paperb_ppd_guard_relax",
    "paperb_mu_relax",
    "paperb_gate_tail_asym_relax",
]


def shard_complete(output_dir: Path, n_weather: int, strategies: list[str]) -> bool:
    summary = output_dir / "summary" / "medium_office_trace_summary.csv"
    if not summary.exists():
        return False
    try:
        df = pd.read_csv(summary)
    except Exception:
        return False
    expected = n_weather * len(strategies)
    return len(df) == expected and set(df["strategy"]) == set(strategies)


def run_shard(
    shard: pd.DataFrame,
    output_dir: Path,
    shard_manifest: Path,
    paperb_met: float,
    people_activity_w: float,
    n_estimators: int,
    strategies: list[str],
    paperb_tail_threshold: float,
    log_dir: Path | None = None,
) -> None:
    shard_manifest.parent.mkdir(parents=True, exist_ok=True)
    shard.to_csv(shard_manifest, index=False)
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--output-dir",
        str(output_dir),
        "--manifest",
        str(shard_manifest),
        "--stage",
        "full",
        "--begin-month",
        "1",
        "--begin-day",
        "1",
        "--end-month",
        "12",
        "--end-day",
        "31",
        "--strategies",
        *strategies,
        "--paperb-met",
        f"{paperb_met:.6f}",
        "--paperb-people-activity-w",
        f"{people_activity_w:.3f}",
        "--n-estimators",
        str(n_estimators),
        "--paperb-tail-threshold",
        f"{paperb_tail_threshold:.6f}",
        "--trace-format",
        "parquet",
        "--skip-combined-trace",
        "--purge-energyplus-after-trace",
        "--purge-case-traces-after-summary",
        "--skip-plot",
    ]
    print("[run] " + " ".join(cmd), flush=True)
    if log_dir is None:
        subprocess.run(cmd, check=True)
    else:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{output_dir.name}.log"
        with log_path.open("w") as log:
            log.write("[run] " + " ".join(cmd) + "\n")
            log.flush()
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--shard-manifest-dir", type=Path, default=DEFAULT_SHARD_MANIFEST_DIR)
    parser.add_argument("--paperb-met", type=float, default=0.854333)
    parser.add_argument("--people-activity-w", type=float, default=93.2)
    parser.add_argument("--paperb-tail-threshold", type=float, default=0.20)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument(
        "--strategy",
        action="append",
        default=[],
        help=(
            "Strategy to run. Repeat for multiple strategies. "
            "Defaults to the full Paper B production strategy list."
        ),
    )
    parser.add_argument("--city", action="append", default=[])
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of city-scenario shards to run concurrently.",
    )
    parser.add_argument(
        "--parallel-log-dir",
        type=Path,
        default=None,
        help="Directory for per-shard logs when --workers is greater than 1.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    strategies = args.strategy or STRATEGIES

    manifest = pd.read_csv(args.manifest)
    if args.city:
        manifest = manifest[manifest["city"].isin(args.city)].copy()
    if args.scenario:
        manifest = manifest[manifest["scenario"].isin(args.scenario)].copy()
    if manifest.empty:
        raise RuntimeError("No manifest rows selected.")

    jobs = []
    for (country, city, scenario), shard in manifest.groupby(["country", "city", "scenario"]):
        slug = f"{city.lower()}_{scenario}"
        output_dir = args.run_root / slug
        shard_manifest = args.shard_manifest_dir / f"{slug}_manifest.csv"
        if not args.force and shard_complete(output_dir, n_weather=len(shard), strategies=strategies):
            print(f"[skip-complete] {slug}", flush=True)
            continue
        print(
            f"[shard] {country} / {city} / {scenario}: "
            f"{len(shard)} weather files x {len(strategies)} strategies",
            flush=True,
        )
        jobs.append(
            {
                "slug": slug,
                "shard": shard.copy(),
                "output_dir": output_dir,
                "shard_manifest": shard_manifest,
            }
        )
    if not jobs:
        print("[done] no incomplete shards selected", flush=True)
        return 0
    if args.workers <= 1:
        for job in jobs:
            run_shard(
                shard=job["shard"],
                output_dir=job["output_dir"],
                shard_manifest=job["shard_manifest"],
                paperb_met=args.paperb_met,
                people_activity_w=args.people_activity_w,
                n_estimators=args.n_estimators,
                strategies=strategies,
                paperb_tail_threshold=args.paperb_tail_threshold,
                log_dir=args.parallel_log_dir,
            )
        return 0

    print(f"[parallel] running {len(jobs)} shards with {args.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_slug = {
            executor.submit(
                run_shard,
                shard=job["shard"],
                output_dir=job["output_dir"],
                shard_manifest=job["shard_manifest"],
                paperb_met=args.paperb_met,
                people_activity_w=args.people_activity_w,
                n_estimators=args.n_estimators,
                strategies=strategies,
                paperb_tail_threshold=args.paperb_tail_threshold,
                log_dir=args.parallel_log_dir,
            ): job["slug"]
            for job in jobs
        }
        for future in as_completed(future_to_slug):
            slug = future_to_slug[future]
            future.result()
            print(f"[complete] {slug}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
