#!/usr/bin/env python3
"""Offline replay of controller signal logic over a fixed Paper A trace.

This script does not change EnergyPlus states. It reads fixed-reference probability
outputs and records what several controller families would request under simple
dwell and offset constraints.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean


CONTROLLERS = (
    "reference",
    "pmv_rule",
    "expected_tsv_rule",
    "total_tail_mu",
    "tail_asym",
    "gate_tail_asym",
)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def zone_prefixes(header: list[str]) -> list[str]:
    prefixes = []
    for name in header:
        if name.startswith("zone_") and name.endswith("_p_disc"):
            prefixes.append(name[: -len("_p_disc")])
    return prefixes


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def aggregate_signal(row: dict[str, str], prefixes: list[str], mode: str) -> dict[str, float]:
    if mode == "mean":
        p_tail = f(row, "discomfort_probability")
        warm = f(row, "warm_discomfort_probability")
        cold = f(row, "cold_discomfort_probability")
        mu = f(row, "expected_tsv")
        return {"p_tail": p_tail, "warm": warm, "cold": cold, "d_tail": warm - cold, "mu": mu}

    zone_rows = []
    for prefix in prefixes:
        p_tail = f(row, f"{prefix}_p_disc")
        warm = f(row, f"{prefix}_warm_tail")
        cold = f(row, f"{prefix}_cold_tail")
        mu = f(row, f"{prefix}_expected_tsv")
        zone_rows.append(
            {"p_tail": p_tail, "warm": warm, "cold": cold, "d_tail": warm - cold, "mu": mu}
        )
    if mode == "max":
        return max(zone_rows, key=lambda x: x["p_tail"])
    if mode == "p90":
        target = percentile([z["p_tail"] for z in zone_rows], 0.90)
        return min(zone_rows, key=lambda x: abs(x["p_tail"] - target))
    raise ValueError(f"Unknown aggregation mode: {mode}")


def requested_direction(
    controller: str,
    signal: dict[str, float],
    row: dict[str, str],
    tail_threshold: float,
    asym_threshold: float,
    pmv_threshold: float,
    tsv_threshold: float,
) -> int:
    """Return +1 for warmer setpoint, -1 for cooler setpoint, 0 for no request."""
    if controller == "reference":
        return 0
    if controller == "pmv_rule":
        pmv = f(row, "mean_pmv")
        if pmv > pmv_threshold:
            return -1
        if pmv < -pmv_threshold:
            return 1
        return 0
    if controller == "expected_tsv_rule":
        mu = signal["mu"]
        if mu > tsv_threshold:
            return -1
        if mu < -tsv_threshold:
            return 1
        return 0
    if controller == "total_tail_mu":
        if signal["p_tail"] < tail_threshold:
            return 0
        if signal["mu"] > 0:
            return -1
        if signal["mu"] < 0:
            return 1
        return 0
    if controller == "tail_asym":
        d_tail = signal["d_tail"]
        if d_tail > asym_threshold:
            return -1
        if d_tail < -asym_threshold:
            return 1
        return 0
    if controller == "gate_tail_asym":
        if signal["p_tail"] < tail_threshold:
            return 0
        return requested_direction(
            "tail_asym", signal, row, tail_threshold, asym_threshold, pmv_threshold, tsv_threshold
        )
    raise ValueError(f"Unknown controller: {controller}")


def replay(
    rows: list[dict[str, str]],
    prefixes: list[str],
    controller: str,
    aggregation: str,
    tail_threshold: float,
    asym_threshold: float,
    dwell_steps: int,
    step_c: float,
    min_offset: float,
    max_offset: float,
    pmv_threshold: float,
    tsv_threshold: float,
) -> dict[str, float | int | str]:
    offset = 0.0
    last_action_idx = -10**9
    requests = 0
    applied = 0
    warm_requests = 0
    cold_requests = 0
    blocked_dwell = 0
    blocked_bounds = 0
    reversals_2h = 0
    action_history: list[tuple[int, int]] = []
    offsets = [offset]
    p_tail_values = []
    high_tail = 0

    for idx, row in enumerate(rows):
        signal = aggregate_signal(row, prefixes, aggregation)
        p_tail_values.append(signal["p_tail"])
        high_tail += int(signal["p_tail"] >= tail_threshold)
        direction = requested_direction(
            controller, signal, row, tail_threshold, asym_threshold, pmv_threshold, tsv_threshold
        )
        if direction:
            requests += 1
            warm_requests += int(direction == -1)
            cold_requests += int(direction == 1)
        if not direction:
            offsets.append(offset)
            continue
        if idx - last_action_idx < dwell_steps:
            blocked_dwell += 1
            offsets.append(offset)
            continue
        candidate = offset + direction * step_c
        if candidate < min_offset or candidate > max_offset:
            blocked_bounds += 1
            offsets.append(offset)
            continue
        if action_history and idx - action_history[-1][0] <= 8 and direction != action_history[-1][1]:
            reversals_2h += 1
        offset = candidate
        last_action_idx = idx
        action_history.append((idx, direction))
        applied += 1
        offsets.append(offset)

    occupied = len(rows)
    return {
        "controller": controller,
        "aggregation": aggregation,
        "tail_threshold": tail_threshold,
        "asym_threshold": asym_threshold,
        "dwell_minutes": dwell_steps * 15,
        "occupied_records": occupied,
        "request_records": requests,
        "request_share_pct": 100 * requests / occupied if occupied else 0,
        "warm_request_share_pct": 100 * warm_requests / requests if requests else 0,
        "applied_actions": applied,
        "actions_per_occupied_day": applied / (occupied / 96) if occupied else 0,
        "blocked_by_dwell": blocked_dwell,
        "blocked_by_bounds": blocked_bounds,
        "reversals_within_2h": reversals_2h,
        "min_offset_c": min(offsets),
        "max_offset_c": max(offsets),
        "mean_signal_p_tail": mean(p_tail_values) if p_tail_values else 0,
        "p95_signal_p_tail": percentile(p_tail_values, 0.95),
        "high_tail_share_pct": 100 * high_tail / occupied if occupied else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tail-thresholds", default="0.10,0.20,0.30")
    parser.add_argument("--asym-thresholds", default="0.05,0.10")
    parser.add_argument("--dwell-minutes", default="0,60,120")
    parser.add_argument("--aggregations", default="mean,p90,max")
    parser.add_argument("--step-c", type=float, default=0.5)
    parser.add_argument("--min-offset-c", type=float, default=-2.0)
    parser.add_argument("--max-offset-c", type=float, default=2.0)
    parser.add_argument("--pmv-threshold", type=float, default=0.5)
    parser.add_argument("--tsv-threshold", type=float, default=0.5)
    args = parser.parse_args()

    with args.trace.open(newline="") as f_in:
        reader = csv.DictReader(f_in)
        header = reader.fieldnames or []
        prefixes = zone_prefixes(header)
        rows = [row for row in reader if row.get("occupied") == "True"]

    tail_thresholds = [float(x) for x in args.tail_thresholds.split(",")]
    asym_thresholds = [float(x) for x in args.asym_thresholds.split(",")]
    dwell_steps_values = [int(int(x) / 15) for x in args.dwell_minutes.split(",")]
    aggregations = args.aggregations.split(",")

    summaries = []
    for aggregation in aggregations:
        for controller in CONTROLLERS:
            controller_tail_thresholds = tail_thresholds if "tail" in controller else [0.20]
            controller_asym_thresholds = asym_thresholds if "asym" in controller else [0.10]
            for tail_threshold in controller_tail_thresholds:
                for asym_threshold in controller_asym_thresholds:
                    for dwell_steps in dwell_steps_values:
                        summaries.append(
                            replay(
                                rows,
                                prefixes,
                                controller,
                                aggregation,
                                tail_threshold,
                                asym_threshold,
                                dwell_steps,
                                args.step_c,
                                args.min_offset_c,
                                args.max_offset_c,
                                args.pmv_threshold,
                                args.tsv_threshold,
                            )
                        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote {len(summaries)} replay rows to {args.output}")


if __name__ == "__main__":
    main()
