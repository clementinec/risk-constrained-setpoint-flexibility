#!/usr/bin/env python3
"""Compile the frozen REV01-C2 city-design-condition sensitivity.

The compiler verifies the author amendment and official raw-DDY identities,
extracts exactly one companion Site:Location plus annual heating DB 99.6% and
cooling DB=>MWB 0.4% objects, replaces only those three accepted objects, and
freezes the 12-cell policy matrix before simulation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA = "paperb_enb_rev01_c2_compiled_plan_v1"
MATRIX_SCHEMA = "paperb_enb_rev01_cell_matrix_v1"
VARIANT_SCHEMA = "paperb_enb_rev01_variant_manifest_v1"
BATCH_ID = "REV01-C2-012-20260825-v2"
EXPECTED_AMENDMENT_SHA256 = "067cdfd7127ccae6051472c77ac14eb9374ed48bc9c8629b119173340a0ff932"
EXPECTED_ACCEPTED_MATRIX_SHA256 = "fa03767f0c3d581e65e00ff6149546c44fe3e936608da73ad2f5485bf19895e0"
EXPECTED_PARENT_RUNNER_SHA256 = "a9693042883ed5c20edad6fcbc757c62c7216d5abc120ad18de1a003932848a4"
EXPECTED_BASE_IDF_SHA256 = "1144b58b848992d1730e49ef9c252569e3a515d82a2f99b2c0233352f625a7e4"
EXPECTED_MODEL_SHA256 = "6fbeb06644a36b226b17824a1fca7526bd518e0a9b76a41f878fb1c27efcd619"
EXPECTED_ROWS = 35_040
EXPECTED_OCCUPIED_ROWS = 16_704

# Public-package sanitization: source DDYs, the accepted IDF/matrix, and the
# author amendment are intentionally not redistributed.  Callers must supply
# those hash-pinned inputs explicitly; no workstation/archive layout is assumed.

ANCHORS = {
    "Beijing": "beijing_ssp585_present_typical",
    "Guangzhou": "guangzhou_ssp585_future_extreme",
    "Phoenix": "phoenix_ssp585_future_extreme",
}
EXPECTED_COMPANION_LOCATIONS = {
    "Beijing": (39.80, 116.47, 8.00, 32.0),
    "Guangzhou": (23.13, 113.32, 8.00, 8.0),
    "Phoenix": (33.45, -111.98, -7.00, 337.0),
}
EXPECTED_RUNTIME_EPW_LOCATIONS = {
    "Beijing": (39.9042, 116.4074, 8.0, 44.0),
    "Guangzhou": (23.1291, 113.2644, 8.0, 21.0),
    "Phoenix": (33.4484, -112.074, -7.0, 331.0),
}
POLICIES = (
    ("fixed", "diagnostic_reference"),
    ("pmv", "paperb_pmv_relax"),
    ("adaptive_band", "paperb_adaptive_band_relax"),
    ("learned_p90", "paperb_p90_tail_asym_relax"),
)
DEFAULTS: dict[str, Any] = {
    "paperb_spatial_quantile": 0.90,
    "paperb_signal_alpha": 1.0,
    "paperb_tail_threshold": 0.20,
    "paperb_asym_threshold": 0.10,
    "paperb_save_heat_c": 20.0,
    "paperb_save_cool_c": 26.0,
    "paperb_warm_protect_cool_c": 23.25,
    "paperb_cold_protect_heat_c": 23.25,
    "paperb_tighten_dwell_steps": 4,
    "paperb_relax_dwell_steps": 1,
    "paperb_pmv_threshold": 0.50,
    "paperb_adaptive_rm_clamp_low_c": 10.0,
    "paperb_adaptive_rm_clamp_high_c": 33.5,
    "paperb_met": 0.854333,
    "paperb_people_activity_w": 93.2,
    "controller_semantics": "corrected",
    "inference_preprocessing": "training_contract",
    "sizing_contract": "city_ddy",
}


class C2ContractError(RuntimeError):
    """A frozen C2 compilation gate failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def object_spans(text: str, object_type: str) -> list[tuple[int, int]]:
    """Locate IDF object spans while ignoring semicolons inside comments."""

    target = object_type.casefold()
    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    position = 0
    for line in lines:
        offsets.append(position)
        position += len(line)
    spans: list[tuple[int, int]] = []
    active: int | None = None
    for index, line in enumerate(lines):
        code = line.split("!", 1)[0]
        if active is None:
            first = code.split(",", 1)[0].strip().casefold()
            if first == target:
                active = offsets[index] + (len(code) - len(code.lstrip()))
        if active is not None and ";" in code:
            semicolon = line.index(";")
            spans.append((active, offsets[index] + semicolon + 1))
            active = None
    if active is not None:
        raise C2ContractError(f"Unterminated {object_type} object")
    return spans


def object_fields(raw: str) -> list[str]:
    code = " ".join(line.split("!", 1)[0] for line in raw.splitlines())
    return [field.strip() for field in code.rstrip().rstrip(";").split(",")]


def parsed_objects(text: str) -> list[list[str]]:
    # Remove comments line-by-line first so comment text such as
    # ``Degrees; N=0`` cannot be mistaken for an object terminator.
    uncommented = "\n".join(line.split("!", 1)[0] for line in text.splitlines())
    result: list[list[str]] = []
    for raw in uncommented.split(";"):
        fields = object_fields(raw + ";")
        if fields and fields[0]:
            result.append(fields)
    return result


def authorized_objects_removed(text: str) -> str:
    spans = sorted(
        object_spans(text, "Site:Location")
        + object_spans(text, "SizingPeriod:DesignDay")
    )
    pieces: list[str] = []
    cursor = 0
    for start, end in spans:
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def canonical_object(fields: list[str]) -> str:
    """Render exact object fields without comments for unambiguous parsing."""

    # The replacement span starts at the object token; indentation immediately
    # before it belongs to the untouched accepted-IDF text.
    lines = [f"{fields[0]},"]
    for index, field in enumerate(fields[1:], start=1):
        terminator = ";" if index == len(fields) - 1 else ","
        lines.append(f"    {field}{terminator}")
    return "\n".join(lines)


def runner_non_sizing_semantic_sha(text: str) -> str:
    """Reproduce the frozen runner helper's semantic IDF checksum."""

    retained: list[list[str]] = []
    for raw in text.split(";"):
        uncommented = " ".join(line.split("!", 1)[0] for line in raw.splitlines())
        fields = [field.strip() for field in uncommented.split(",") if field.strip()]
        if fields and fields[0].casefold() not in {"site:location", "sizingperiod:designday"}:
            retained.append(fields)
    payload = json.dumps(
        retained, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_design_days(source_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for start, end in object_spans(source_text, "SizingPeriod:DesignDay"):
        raw = source_text[start:end]
        fields = object_fields(raw)
        if len(fields) < 22:
            raise C2ContractError("Design-day object has too few fields")
        objects.append({"raw": raw, "fields": fields, "name": fields[1]})
    heating = [x for x in objects if x["name"].casefold().endswith("ann htg 99.6% condns db")]
    cooling = [x for x in objects if x["name"].casefold().endswith("ann clg .4% condns db=>mwb")]
    if len(heating) != 1 or len(cooling) != 1:
        raise C2ContractError(
            f"Ambiguous frozen DDY selection: heating={len(heating)}, cooling={len(cooling)}"
        )
    return heating[0], cooling[0]


def select_site_location(source_text: str) -> dict[str, Any]:
    spans = object_spans(source_text, "Site:Location")
    if len(spans) != 1:
        raise C2ContractError(f"Companion DDY must contain one Site:Location, got {len(spans)}")
    start, end = spans[0]
    raw = source_text[start:end]
    fields = object_fields(raw)
    if len(fields) != 6 or fields[0].casefold() != "site:location":
        raise C2ContractError("Companion Site:Location fields differ or Keep flag is explicit")
    return {"raw": raw, "fields": fields, "name": fields[1]}


def design_day_metadata(item: dict[str, Any]) -> dict[str, Any]:
    fields = item["fields"]
    return {
        "name": fields[1],
        "month": int(fields[2]),
        "day_of_month": int(fields[3]),
        "day_type": fields[4],
        "maximum_dry_bulb_c": float(fields[5]),
        "daily_dry_bulb_range_c": float(fields[6]),
        "humidity_condition_type": fields[9],
        "wetbulb_or_dewpoint_at_max_db_c": float(fields[10]),
        "barometric_pressure_pa": float(fields[15]),
        "wind_speed_m_s": float(fields[16]),
        "wind_direction_deg": float(fields[17]),
        "solar_model": fields[21],
        "canonical_fields_sha256": canonical_sha(fields),
    }


def site_location_metadata(item: dict[str, Any]) -> dict[str, Any]:
    fields = item["fields"]
    return {
        "name": fields[1],
        "latitude_deg": float(fields[2]),
        "longitude_deg": float(fields[3]),
        "time_zone_hours": float(fields[4]),
        "elevation_m": float(fields[5]),
        "keep_site_location_information": None,
        "canonical_fields_sha256": canonical_sha(fields),
    }


def epw_location(path: Path) -> dict[str, Any]:
    header = path.open("r", encoding="utf-8", errors="strict").readline().rstrip("\r\n")
    fields = [field.strip() for field in header.split(",")]
    if len(fields) < 10 or fields[0].casefold() != "location":
        raise C2ContractError(f"Invalid EPW LOCATION header: {path}")
    return {
        "name": " ".join(part for part in fields[1:6] if part),
        "latitude_deg": float(fields[6]),
        "longitude_deg": float(fields[7]),
        "time_zone_hours": float(fields[8]),
        "elevation_m": float(fields[9]),
        "header_sha256": hashlib.sha256(header.encode("utf-8")).hexdigest(),
    }


def derive_city_idf(
    base_path: Path,
    source_path: Path,
    destination: Path,
) -> dict[str, Any]:
    base_text = base_path.read_text(encoding="utf-8", errors="strict")
    source_text = source_path.read_text(encoding="latin-1", errors="strict")
    base_location_spans = object_spans(base_text, "Site:Location")
    base_spans = object_spans(base_text, "SizingPeriod:DesignDay")
    if len(base_location_spans) != 1 or len(base_spans) != 2:
        raise C2ContractError(
            f"Accepted base must contain one location/two design days, got "
            f"{len(base_location_spans)}/{len(base_spans)}"
        )
    location = select_site_location(source_text)
    heating, cooling = select_design_days(source_text)
    # Inline DDY comments contain semicolons (for example ``Degrees; N=0``),
    # which are legal human annotations but ambiguous to the frozen helper's
    # intentionally small parser.  Render the exact selected fields without
    # comments; no scientific value is changed.
    replacement_pairs = [
        (base_location_spans[0], canonical_object(location["fields"])),
        (base_spans[0], canonical_object(heating["fields"])),
        (base_spans[1], canonical_object(cooling["fields"])),
    ]
    pieces: list[str] = []
    cursor = 0
    for (start, end), replacement in sorted(replacement_pairs):
        pieces.extend((base_text[cursor:start], replacement))
        cursor = end
    pieces.append(base_text[cursor:])
    derived_text = "".join(pieces)
    base_objects = parsed_objects(base_text)
    derived_objects = parsed_objects(derived_text)
    authorized_types = {"site:location", "sizingperiod:designday"}
    base_retained = [x for x in base_objects if x[0].casefold() not in authorized_types]
    derived_retained = [x for x in derived_objects if x[0].casefold() not in authorized_types]
    base_locations = [x for x in base_objects if x[0].casefold() == "site:location"]
    derived_locations = [x for x in derived_objects if x[0].casefold() == "site:location"]
    derived_days = [x for x in derived_objects if x[0].casefold() == "sizingperiod:designday"]
    expected_days = [heating["fields"], cooling["fields"]]
    gates = {
        "exactly_one_base_site_location": len(base_location_spans) == 1,
        "exactly_one_derived_site_location": len(derived_locations) == 1,
        "selected_site_location_exact": derived_locations == [location["fields"]],
        "exactly_two_base_design_days": len(base_spans) == 2,
        "exactly_two_derived_design_days": len(derived_days) == 2,
        "selected_objects_exact": derived_days == expected_days,
        "all_non_design_day_objects_semantically_exact": base_retained == derived_retained,
        "all_text_outside_three_authorized_spans_byte_exact": (
            authorized_objects_removed(base_text).encode("utf-8")
            == authorized_objects_removed(derived_text).encode("utf-8")
        ),
    }
    if not all(gates.values()):
        raise C2ContractError(f"Alternative-IDF semantic diff failed: {gates}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.write_text(derived_text, encoding="utf-8", newline="")
    return {
        "source_idf_sha256": sha256_file(destination),
        "site_location_name": location["name"],
        "site_location": site_location_metadata(location),
        "design_day_names": [heating["name"], cooling["name"]],
        "heating_99p6_db": design_day_metadata(heating),
        "cooling_0p4_db_mwb": design_day_metadata(cooling),
        "accepted_base_non_sizing_semantic_sha256": runner_non_sizing_semantic_sha(base_text),
        "non_sizing_semantic_sha256": runner_non_sizing_semantic_sha(derived_text),
        "strict_non_design_day_objects_sha256": canonical_sha(derived_retained),
        "site_location_canonical_sha256": canonical_sha(location["fields"]),
        "text_outside_authorized_objects_sha256": hashlib.sha256(
            authorized_objects_removed(base_text).encode("utf-8")
        ).hexdigest(),
        "semantic_diff_gates": gates,
    }


def validate_accepted_matrix(path: Path) -> pd.DataFrame:
    if sha256_file(path) != EXPECTED_ACCEPTED_MATRIX_SHA256:
        raise C2ContractError("Accepted Fresh96 matrix hash differs")
    frame = pd.read_csv(path)
    subset = frame.loc[frame["anchor_id"].astype(str).isin(ANCHORS.values())]
    if len(subset) != 12 or subset.groupby("anchor_id")["strategy"].nunique().to_dict() != {
        anchor: 4 for anchor in ANCHORS.values()
    }:
        raise C2ContractError("The three frozen anchors do not map to exactly 12 accepted cells")
    return frame


def effective_config(strategy: str) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "spatial_quantile": DEFAULTS["paperb_spatial_quantile"],
        "signal_alpha": DEFAULTS["paperb_signal_alpha"],
        "tail_threshold": DEFAULTS["paperb_tail_threshold"],
        "asym_threshold": DEFAULTS["paperb_asym_threshold"],
        "save_heat_c": DEFAULTS["paperb_save_heat_c"],
        "save_cool_c": DEFAULTS["paperb_save_cool_c"],
        "warm_protect_cool_c": DEFAULTS["paperb_warm_protect_cool_c"],
        "cold_protect_heat_c": DEFAULTS["paperb_cold_protect_heat_c"],
        "tighten_dwell_steps": DEFAULTS["paperb_tighten_dwell_steps"],
        "relax_dwell_steps": DEFAULTS["paperb_relax_dwell_steps"],
        "pmv_threshold": DEFAULTS["paperb_pmv_threshold"],
        "adaptive_rm_clamp_low_c": DEFAULTS["paperb_adaptive_rm_clamp_low_c"],
        "adaptive_rm_clamp_high_c": DEFAULTS["paperb_adaptive_rm_clamp_high_c"],
        "paperb_met": DEFAULTS["paperb_met"],
        "people_activity_w_per_person": DEFAULTS["paperb_people_activity_w"],
        "controller_semantics": DEFAULTS["controller_semantics"],
        "inference_preprocessing": DEFAULTS["inference_preprocessing"],
        "sizing_contract": "city_ddy",
    }


def write_parquet_exact(frame: pd.DataFrame, path: Path) -> None:
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False), path, version="2.6",
        compression="zstd", compression_level=3, row_group_size=65_536,
        write_page_checksum=True,
    )
    restored = pq.read_table(path, page_checksum_verification=True).to_pandas()
    pd.testing.assert_frame_equal(frame, restored, check_dtype=True, check_exact=True)


def compile_plan(args: argparse.Namespace) -> dict[str, Any]:
    if sha256_file(args.amendment) != EXPECTED_AMENDMENT_SHA256:
        raise C2ContractError("Author amendment hash differs")
    if sha256_file(args.base_idf) != EXPECTED_BASE_IDF_SHA256:
        raise C2ContractError("Accepted Denver source IDF hash differs")
    retrieval = json.loads(args.retrieval.read_text(encoding="utf-8"))
    if retrieval.get("schema_version") != "paperb_enb_rev01_c2_ddy_retrieval_v1":
        raise C2ContractError("DDY retrieval record schema differs")
    accepted = validate_accepted_matrix(args.accepted_matrix)
    city_audits: dict[str, Any] = {}
    for city in ANCHORS:
        declared = retrieval["sources"][city]
        source = args.source_root / declared["filename"]
        if not source.is_file() or source.stat().st_size != declared["bytes"]:
            raise C2ContractError(f"Missing/changed raw DDY for {city}")
        if sha256_file(source) != declared["sha256"]:
            raise C2ContractError(f"Raw DDY checksum differs for {city}")
        derived = args.derived_root / f"{city.casefold()}_city_package_location_99p6H_0p4C.idf"
        audit = derive_city_idf(args.base_idf, source, derived)
        audit.update({
            "city": city,
            "derived_idf_path": str(derived.resolve()),
            "raw_ddy_sha256": declared["sha256"],
            "raw_ddy_bytes": declared["bytes"],
            "retrieved_utc": declared["retrieved_utc"],
            "station": declared["station"],
            "url": declared["url"],
        })
        companion_tuple = tuple(
            audit["site_location"][key]
            for key in ("latitude_deg", "longitude_deg", "time_zone_hours", "elevation_m")
        )
        if companion_tuple != EXPECTED_COMPANION_LOCATIONS[city]:
            raise C2ContractError(f"Companion Site:Location differs from amendment: {city}")
        city_audits[city] = audit

    rows: list[dict[str, Any]] = []
    variants: dict[str, Any] = {}
    sequence = 0
    for city, anchor in ANCHORS.items():
        accepted_anchor = accepted.loc[accepted["anchor_id"].astype(str).eq(anchor)]
        audit = city_audits[city]
        for label, strategy in POLICIES:
            sequence += 1
            source_rows = accepted_anchor.loc[accepted_anchor["strategy"].astype(str).eq(strategy)]
            if len(source_rows) != 1:
                raise C2ContractError(f"Accepted comparator is not unique: {anchor}, {strategy}")
            source = source_rows.iloc[0]
            runtime_location = epw_location(Path(str(source["execution_epw_path"])))
            runtime_tuple = tuple(
                runtime_location[key]
                for key in ("latitude_deg", "longitude_deg", "time_zone_hours", "elevation_m")
            )
            if runtime_tuple != EXPECTED_RUNTIME_EPW_LOCATIONS[city]:
                raise C2ContractError(f"Runtime EPW location differs from amendment gate: {city}")
            variant_id = f"rev01_c2_{city.casefold()}_{label}"
            variants[variant_id] = {
                "strategy": strategy,
                "effective_config": effective_config(strategy),
                "sizing": {
                    "mode": "city_ddy",
                    "city": city,
                    "source_idf_sha256": audit["source_idf_sha256"],
                    "site_location_name": audit["site_location_name"],
                    "design_day_names": audit["design_day_names"],
                    "non_sizing_semantic_sha256": audit["non_sizing_semantic_sha256"],
                    "accepted_base_non_sizing_semantic_sha256": audit["accepted_base_non_sizing_semantic_sha256"],
                },
            }
            row = {
                "matrix_schema_version": MATRIX_SCHEMA,
                "batch_id": BATCH_ID,
                "cell_id": f"REV01-C2-012-{sequence:03d}",
                "experiment": "C2",
                "variant_id": variant_id,
                "policy_label": label,
                "strategy": strategy,
                "anchor_id": anchor,
                "anchor_order": int(source["anchor_order"]),
                "anchor_role": str(source["anchor_role"]),
                "city": city,
                "climate_role": str(source["climate_role"]),
                "scenario": str(source["scenario"]),
                "time_slice": str(source["time_slice"]),
                "selector_role": str(source["selector_role"]),
                "weather_year": int(source["recomputed_weather_year"]),
                "physical_condition_id": str(source["physical_condition_id"]),
                "selector_label_id": str(source["selector_label_id"]),
                "weather_path": str(Path(str(source["execution_epw_path"])).resolve()),
                "weather_sha256": str(source["epw_sha256"]),
                "weather_data_sha256": str(source["epw_data_sha256"]),
                "weather_stem": Path(str(source["execution_epw_path"])).stem,
                "source_idf_path": audit["derived_idf_path"],
                "source_idf_sha256": audit["source_idf_sha256"],
                "source_ddy_sha256": audit["raw_ddy_sha256"],
                "companion_site_location_name": audit["site_location"]["name"],
                "companion_latitude_deg": audit["site_location"]["latitude_deg"],
                "companion_longitude_deg": audit["site_location"]["longitude_deg"],
                "companion_time_zone_hours": audit["site_location"]["time_zone_hours"],
                "companion_elevation_m": audit["site_location"]["elevation_m"],
                "runtime_epw_location_name": runtime_location["name"],
                "runtime_epw_latitude_deg": runtime_location["latitude_deg"],
                "runtime_epw_longitude_deg": runtime_location["longitude_deg"],
                "runtime_epw_time_zone_hours": runtime_location["time_zone_hours"],
                "runtime_epw_elevation_m": runtime_location["elevation_m"],
                "runtime_epw_location_header_sha256": runtime_location["header_sha256"],
                "heating_design_day_name": audit["design_day_names"][0],
                "cooling_design_day_name": audit["design_day_names"][1],
                "accepted_comparator_strategy": strategy,
                "accepted_comparator_cell_id": str(source["scaled_run_id"]),
                "expected_trace_rows": EXPECTED_ROWS,
                "expected_occupied_rows": EXPECTED_OCCUPIED_ROWS,
                "accepted_matrix_sha256": EXPECTED_ACCEPTED_MATRIX_SHA256,
                "accepted_idf_sha256": EXPECTED_BASE_IDF_SHA256,
                "accepted_model_sha256": EXPECTED_MODEL_SHA256,
                "resume_permitted": False,
                "reuse_or_splice_permitted": False,
                "purge_permitted": False,
                **DEFAULTS,
            }
            rows.append(row)
    matrix = pd.DataFrame.from_records(rows)
    if len(matrix) != 12 or matrix["cell_id"].nunique() != 12:
        raise C2ContractError("C2 matrix did not close at exactly 12 unique cells")
    if matrix.groupby("city")["strategy"].nunique().to_dict() != {city: 4 for city in ANCHORS}:
        raise C2ContractError("C2 city-policy coverage differs from the amendment")
    manifest = {
        "schema_version": VARIANT_SCHEMA,
        "parent_runner_sha256": EXPECTED_PARENT_RUNNER_SHA256,
        "variants": variants,
    }
    public_provenance = {
        "schema_version": "paperb_enb_rev01_c2_ddy_provenance_v1",
        "created_utc": utc_now(),
        "batch_id": BATCH_ID,
        "raw_source_contents_published": False,
        "accepted_base_idf_sha256": EXPECTED_BASE_IDF_SHA256,
        "cities": {
            city: {key: value for key, value in audit.items() if key != "derived_idf_path"}
            for city, audit in city_audits.items()
        },
    }
    return {"matrix": matrix, "manifest": manifest, "provenance": public_provenance}


def freeze(args: argparse.Namespace, compiled: dict[str, Any]) -> dict[str, Any]:
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    matrix: pd.DataFrame = compiled["matrix"]
    matrix_csv = args.output_dir / "c2_city_ddy12.csv"
    matrix_parquet = args.output_dir / "c2_city_ddy12.parquet"
    manifest_path = args.output_dir / "VARIANT_MANIFEST.json"
    provenance_path = args.output_dir / "DDY_PROVENANCE.json"
    matrix.to_csv(matrix_csv, index=False)
    write_parquet_exact(matrix, matrix_parquet)
    write_json(manifest_path, compiled["manifest"])
    write_json(provenance_path, compiled["provenance"])
    closure = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "all_gates_passed": True,
        "created_utc": utc_now(),
        "batch_id": BATCH_ID,
        "author_amendment_path": str(args.amendment.resolve()),
        "author_amendment_sha256": sha256_file(args.amendment),
        "accepted_matrix_path": str(args.accepted_matrix.resolve()),
        "accepted_matrix_sha256": sha256_file(args.accepted_matrix),
        "accepted_base_idf_sha256": sha256_file(args.base_idf),
        "raw_source_ddys_published": False,
        "planned_cells": 12,
        "planned_cities": 3,
        "planned_policies": 4,
        "matrix_path": str(matrix_csv.resolve()),
        "matrix_sha256": sha256_file(matrix_csv),
        "matrix_parquet_path": str(matrix_parquet.resolve()),
        "matrix_parquet_sha256": sha256_file(matrix_parquet),
        "variant_manifest_path": str(manifest_path.resolve()),
        "variant_manifest_sha256": sha256_file(manifest_path),
        "ddy_provenance_path": str(provenance_path.resolve()),
        "ddy_provenance_sha256": sha256_file(provenance_path),
        "conditional_batches_inferred_at_runtime": False,
        "matrix_frozen_before_outcome_inspection": True,
    }
    write_json(args.output_dir / "PLAN_CLOSURE.json", closure)
    return closure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--base-idf", type=Path, required=True)
    parser.add_argument("--accepted-matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--freeze", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not all(path.is_file() for path in (args.amendment, args.retrieval, args.base_idf, args.accepted_matrix)):
        raise FileNotFoundError("A required C2 authority/input file is missing")
    compiled = compile_plan(args)
    if args.preflight_only:
        print(json.dumps({
            "schema_version": SCHEMA,
            "status": "PASS",
            "filesystem_mutated": True,
            "derived_idfs_created": 3,
            "planned_cells": len(compiled["matrix"]),
            "cities": sorted(compiled["matrix"]["city"].unique()),
        }, indent=2, sort_keys=True))
        return 0
    if args.output_dir is None:
        raise C2ContractError("--freeze requires --output-dir")
    print(json.dumps(freeze(args, compiled), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
