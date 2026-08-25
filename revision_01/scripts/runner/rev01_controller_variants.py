#!/usr/bin/env python3
"""Pure, fail-closed helpers for the Paper B Rev01 controller variants.

This module deliberately contains no EnergyPlus imports and performs no file
writes.  The derived runner imports it for variant selection and manifest
validation; the unit tests exercise the same functions directly.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


VARIANT_MANIFEST_SCHEMA = "paperb_enb_rev01_variant_manifest_v1"
PARENT_RUNNER_SHA256 = "a9693042883ed5c20edad6fcbc757c62c7216d5abc120ad18de1a003932848a4"
ACCEPTED_SOURCE_IDF_SHA256 = (
    "1144b58b848992d1730e49ef9c252569e3a515d82a2f99b2c0233352f625a7e4"
)
CITY_DDY_EXCLUDED_OBJECT_TYPES = frozenset(
    {"site:location", "sizingperiod:designday"}
)


class VariantContractError(ValueError):
    """Raised when a Rev01 variant or manifest fails its frozen contract."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise VariantContractError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise VariantContractError(f"{name} must be a finite number")
    return result


def validate_probability(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if not 0.0 <= result <= 1.0:
        raise VariantContractError(f"{name} must be within [0, 1], got {result}")
    return result


def nearest_quantile_index(values: np.ndarray, quantile: float) -> tuple[int, float]:
    """Return the stable original-order index nearest NumPy's linear quantile.

    This reproduces the accepted p90 selector at q=0.90. ``np.argmin`` gives a
    deterministic first-index tie break in the frozen zone order.
    """

    q = validate_probability(quantile, "spatial_quantile")
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise VariantContractError("quantile input must be a nonempty 1-D array")
    if not np.isfinite(array).all():
        raise VariantContractError("quantile input contains nonfinite values")
    target = float(np.quantile(array, q))
    index = int(np.argmin(np.abs(array - target)))
    return index, target


def select_pmv_quantile_guard(
    zone_pmv: np.ndarray, quantile: float = 0.90
) -> dict[str, float | int]:
    """Select an actual zone by a quantile of absolute PMV, retaining its sign."""

    pmv = np.asarray(zone_pmv, dtype=float)
    index, target = nearest_quantile_index(np.abs(pmv), quantile)
    return {
        "index": index,
        "target_abs_pmv": target,
        "selected_abs_pmv": float(abs(pmv[index])),
        "selected_signed_pmv": float(pmv[index]),
    }


def select_learned_tail_guard(
    zone_cold_tail: np.ndarray,
    zone_warm_tail: np.ndarray,
    quantile: float = 0.90,
) -> dict[str, float | int]:
    """Select the actual learned-risk zone used by the configurable guard."""

    cold = np.asarray(zone_cold_tail, dtype=float)
    warm = np.asarray(zone_warm_tail, dtype=float)
    if cold.shape != warm.shape or cold.ndim != 1 or len(cold) == 0:
        raise VariantContractError("cold/warm tail arrays must be equal nonempty 1-D arrays")
    if not (np.isfinite(cold).all() and np.isfinite(warm).all()):
        raise VariantContractError("tail arrays contain nonfinite values")
    if (cold < 0).any() or (warm < 0).any() or (cold + warm > 1.0 + 1e-12).any():
        raise VariantContractError("tail arrays violate probability bounds")
    p_tail = cold + warm
    index, target = nearest_quantile_index(p_tail, quantile)
    return {
        "index": index,
        "target_p_tail": target,
        "selected_p_tail": float(p_tail[index]),
        "selected_d_tail": float(warm[index] - cold[index]),
    }


def attenuate_probabilities(
    probabilities: np.ndarray,
    alpha: float,
    *,
    neutral_index: int = 3,
) -> np.ndarray:
    """Shrink class probabilities deterministically toward neutral TSV.

    Alpha=1 returns the exact input object to protect accepted default parity;
    alpha=0 returns a neutral one-hot distribution.  No random perturbation or
    data-dependent recalibration is performed.
    """

    scale = validate_probability(alpha, "signal_alpha")
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != 7:
        raise VariantContractError("probabilities must have shape (n, 7)")
    if not np.isfinite(probs).all():
        raise VariantContractError("probabilities contain nonfinite values")
    if (probs < -1e-12).any() or (probs > 1.0 + 1e-12).any():
        raise VariantContractError("probabilities fall outside [0, 1]")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-9, rtol=0.0):
        raise VariantContractError("probability rows do not sum to one")
    if not 0 <= neutral_index < probs.shape[1]:
        raise VariantContractError("neutral_index is outside the class range")
    if scale == 1.0:
        return probabilities
    neutral = np.zeros_like(probs)
    neutral[:, neutral_index] = 1.0
    if scale == 0.0:
        return neutral
    return scale * probs + (1.0 - scale) * neutral


def adaptive_comfort_bounds(
    running_mean_outdoor_c: float,
    *,
    clamp_low_c: float | None = None,
    clamp_high_c: float | None = None,
    half_width_90_c: float = 2.5,
    half_width_80_c: float = 3.5,
) -> dict[str, float | bool]:
    """Calculate raw or explicitly clamped adaptive-band bounds."""

    raw = _finite_float(running_mean_outdoor_c, "running_mean_outdoor_c")
    used = raw
    if (clamp_low_c is None) != (clamp_high_c is None):
        raise VariantContractError("adaptive clamp requires both low and high bounds")
    if clamp_low_c is not None and clamp_high_c is not None:
        low = _finite_float(clamp_low_c, "adaptive_clamp_low_c")
        high = _finite_float(clamp_high_c, "adaptive_clamp_high_c")
        if low >= high:
            raise VariantContractError("adaptive clamp low must be below high")
        used = float(np.clip(raw, low, high))
    width90 = _finite_float(half_width_90_c, "adaptive_half_width_90_c")
    width80 = _finite_float(half_width_80_c, "adaptive_half_width_80_c")
    if not (0.0 < width90 < width80):
        raise VariantContractError("adaptive half-widths must satisfy 0 < 90% < 80%")
    center = 0.31 * used + 17.8
    return {
        "raw_running_mean_c": raw,
        "used_running_mean_c": used,
        "was_clamped": used != raw,
        "center": center,
        "low_90": center - width90,
        "high_90": center + width90,
        "low_80": center - width80,
        "high_80": center + width80,
    }


def validate_controller_bounds(
    *,
    reference_heat_c: float,
    reference_cool_c: float,
    save_heat_c: float,
    save_cool_c: float,
    warm_protect_cool_c: float,
    cold_protect_heat_c: float,
) -> dict[str, float]:
    values = {
        "reference_heat_c": _finite_float(reference_heat_c, "reference_heat_c"),
        "reference_cool_c": _finite_float(reference_cool_c, "reference_cool_c"),
        "save_heat_c": _finite_float(save_heat_c, "save_heat_c"),
        "save_cool_c": _finite_float(save_cool_c, "save_cool_c"),
        "warm_protect_cool_c": _finite_float(
            warm_protect_cool_c, "warm_protect_cool_c"
        ),
        "cold_protect_heat_c": _finite_float(
            cold_protect_heat_c, "cold_protect_heat_c"
        ),
    }
    if values["reference_cool_c"] - values["reference_heat_c"] < 2.0:
        raise VariantContractError("reference setpoints require at least a 2 C deadband")
    if values["save_heat_c"] > values["reference_heat_c"]:
        raise VariantContractError("save heating target must not exceed reference heating")
    if values["save_cool_c"] < values["reference_cool_c"]:
        raise VariantContractError("save cooling target must not be below reference cooling")
    if values["save_cool_c"] - values["save_heat_c"] < 2.0:
        raise VariantContractError("saving targets require at least a 2 C deadband")
    return values


def parse_idf_objects(path: Path) -> list[list[str]]:
    objects: list[list[str]] = []
    text = path.read_text(encoding="utf-8", errors="strict")
    for raw in text.split(";"):
        uncommented = " ".join(line.split("!", 1)[0] for line in raw.splitlines())
        fields = [field.strip() for field in uncommented.split(",") if field.strip()]
        if fields:
            objects.append(fields)
    return objects


def idf_non_sizing_semantic_sha256(path: Path) -> str:
    retained = [
        fields
        for fields in parse_idf_objects(path)
        if fields[0].casefold() not in CITY_DDY_EXCLUDED_OBJECT_TYPES
    ]
    return canonical_json_sha256(retained)


def inspect_city_ddy_idf(path: Path) -> dict[str, Any]:
    objects = parse_idf_objects(path)
    locations = [x for x in objects if x[0].casefold() == "site:location"]
    design_days = [x for x in objects if x[0].casefold() == "sizingperiod:designday"]
    if len(locations) != 1:
        raise VariantContractError(f"city-DDY IDF must contain one Site:Location, got {len(locations)}")
    if len(design_days) < 2:
        raise VariantContractError("city-DDY IDF must contain at least two design days")
    names = [fields[1] for fields in design_days if len(fields) >= 2]
    if len(names) != len(design_days) or len(set(names)) != len(names):
        raise VariantContractError("city-DDY design-day names are missing or duplicated")
    return {
        "source_idf_sha256": file_sha256(path),
        "site_location_name": locations[0][1] if len(locations[0]) >= 2 else "",
        "design_day_names": names,
        "design_day_count": len(names),
        "non_sizing_semantic_sha256": idf_non_sizing_semantic_sha256(path),
    }


def _canonical_contract(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise VariantContractError("variant contract is not canonical JSON") from exc


def validate_variant_manifest(
    manifest_path: Path,
    *,
    variant_id: str,
    effective_config: Mapping[str, Any],
    source_idf_path: Path,
) -> dict[str, Any]:
    """Validate one exact variant entry and any city-DDY source contract."""

    if not manifest_path.is_file():
        raise VariantContractError(f"missing variant manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VariantContractError(f"invalid variant manifest: {manifest_path}") from exc
    if manifest.get("schema_version") != VARIANT_MANIFEST_SCHEMA:
        raise VariantContractError("unexpected Rev01 variant-manifest schema")
    if manifest.get("parent_runner_sha256") != PARENT_RUNNER_SHA256:
        raise VariantContractError("variant manifest does not pin the accepted parent runner")
    variants = manifest.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise VariantContractError("variant manifest must contain a nonempty variants object")
    if any(not isinstance(key, str) or not key for key in variants):
        raise VariantContractError("variant-manifest keys must be nonempty strings")
    if variant_id not in variants or not isinstance(variants[variant_id], dict):
        raise VariantContractError(f"variant ID is not declared: {variant_id}")
    entry = variants[variant_id]
    declared = entry.get("effective_config")
    if not isinstance(declared, dict):
        raise VariantContractError("variant entry lacks effective_config")
    if _canonical_contract(declared) != _canonical_contract(effective_config):
        raise VariantContractError("effective runner configuration differs from manifest")
    if entry.get("strategy") != effective_config.get("strategy"):
        raise VariantContractError("manifest strategy differs from effective strategy")

    sizing = entry.get("sizing")
    if not isinstance(sizing, dict):
        raise VariantContractError("variant entry lacks sizing contract")
    mode = effective_config.get("sizing_contract")
    if sizing.get("mode") != mode:
        raise VariantContractError("sizing mode differs from effective configuration")
    idf_audit: dict[str, Any]
    if mode == "accepted_denver":
        observed = file_sha256(source_idf_path)
        expected = sizing.get("source_idf_sha256", ACCEPTED_SOURCE_IDF_SHA256)
        if observed != expected:
            raise VariantContractError("accepted-Denver source IDF hash mismatch")
        idf_audit = {"source_idf_sha256": observed, "mode": mode}
    elif mode == "city_ddy":
        idf_audit = inspect_city_ddy_idf(source_idf_path)
        required = {
            "source_idf_sha256": idf_audit["source_idf_sha256"],
            "site_location_name": idf_audit["site_location_name"],
            "design_day_names": idf_audit["design_day_names"],
            "non_sizing_semantic_sha256": idf_audit["non_sizing_semantic_sha256"],
        }
        for key, observed in required.items():
            if sizing.get(key) != observed:
                raise VariantContractError(f"city-DDY manifest mismatch for {key}")
        if idf_audit["non_sizing_semantic_sha256"] != sizing.get(
            "accepted_base_non_sizing_semantic_sha256"
        ):
            raise VariantContractError(
                "city-DDY IDF changes objects outside Site:Location/design days"
            )
        idf_audit["mode"] = mode
        idf_audit["city"] = sizing.get("city")
    else:
        raise VariantContractError(f"unsupported sizing contract: {mode}")

    return {
        "schema_version": "paperb_enb_rev01_variant_audit_v1",
        "variant_id": variant_id,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "parent_runner_sha256": PARENT_RUNNER_SHA256,
        "effective_config_sha256": canonical_json_sha256(effective_config),
        "effective_config": dict(effective_config),
        "sizing_audit": idf_audit,
        "all_gates_passed": True,
    }
