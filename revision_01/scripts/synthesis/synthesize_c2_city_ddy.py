#!/usr/bin/env python3
"""Fail-closed synthesis for the targeted C2 city-design-condition sensitivity.

This program is read-only with respect to both the accepted Fresh96 cells and
the 12-cell C2 batch.  It validates the closed batch and its accepted evidence
snapshot before computing matched sizing, unmet-load, energy, and occupied
operative-temperature diagnostics.  The two weather strata remain a stress-
test grid, and temperature degree-hours are descriptive physical diagnostics.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA = "paperb_enb_rev01_c2_synthesis_v2"
EXPECTED_BATCH_ID = "REV01-C2-012-20260825-v2"
EXPECTED_BATCH_CLOSURE_SHA256 = "9ce111074c22c0da5c4924369a595a0702ab3b3201ed53b7ecf94e48d0a7e7b8"
EXPECTED_FREEZE_SHA256 = "150ba8ae1300715a763536e7a0f062be3b59f43da669d5234dc8ddeea2da5b3f"
EXPECTED_CELL_INDEX_SHA256 = "edff8f9e640f7b5806baa593254e007a818d1b915bf290ffce0fdea84d315fa1"
EXPECTED_PLAN_CLOSURE_SHA256 = "73f402563cdfe8784e3c02012bf6f95f8bd2f642b958aa5b7887a95c91727149"
EXPECTED_MATRIX_SHA256 = "2f578dc9335b31d2ad7bef9f0f20e6536c1aaedfff2d667abdce0b620250fa97"
EXPECTED_MANIFEST_SHA256 = "b01b8857958a3a29c82557225de393ed5ebb3066e62056c021e845d78d658ae6"
EXPECTED_ACCEPTED_MATRIX_SHA256 = "fa03767f0c3d581e65e00ff6149546c44fe3e936608da73ad2f5485bf19895e0"
EXPECTED_LAUNCHER_SHA256 = "114d2f9583d3c6e84828c30ca8e1f25e2a8725acb830a2f74dca7c297be56edb"
EXPECTED_RUNNER_SHA256 = "4c061fc3b25f7ee6fe66df00cfacfab42f1d3ab171185306fe1ce161a84a0baf"
EXPECTED_HELPER_SHA256 = "b941c0bfc4beb3dc44557ed341bb4e018992620b0d3ad4f3bcb6973558d78bf7"
EXPECTED_STORAGE_MAPPING_SHA256 = "40d8b59cd2f2085bef8aafe16634fe1364c2aa5f317e0e62986c83aca5b65238"
EXPECTED_ROWS = 35_040
EXPECTED_OCCUPIED_ROWS = 16_704
EXPECTED_WINDOW_ROWS = 1_344
EXPECTED_WINDOW_OCCUPIED_ROWS = 640
STEP_HOURS = 0.25
JOULES_PER_KWH = 3_600_000.0
THRESHOLDS = (26.0, 28.0, 30.0)
POLICY_ORDER = ("fixed", "pmv", "adaptive_band", "learned_p90")
STRATEGY_TO_POLICY = {
    "diagnostic_reference": "fixed",
    "paperb_pmv_relax": "pmv",
    "paperb_adaptive_band_relax": "adaptive_band",
    "paperb_p90_tail_asym_relax": "learned_p90",
}

# These are the predeclared numeric endpoints.  Keep this list explicit so a
# missing SQL value cannot silently disappear from a dtype-driven analysis.
ANALYSIS_METRICS = (
    "annual_electricity_kwh_delivered",
    "annual_natural_gas_kwh_delivered_including_service_water",
    "annual_total_delivered_site_energy_kwh",
    "extreme_336h_electricity_kwh_delivered",
    "extreme_336h_natural_gas_kwh_delivered_including_service_water",
    "extreme_336h_total_delivered_site_energy_kwh",
    "annual_occupied_worst_zone_degree_hours_above_26c",
    "annual_occupied_worst_zone_degree_hours_above_28c",
    "annual_occupied_worst_zone_degree_hours_above_30c",
    "extreme_336h_occupied_worst_zone_degree_hours_above_26c",
    "extreme_336h_occupied_worst_zone_degree_hours_above_28c",
    "extreme_336h_occupied_worst_zone_degree_hours_above_30c",
    "dx_high_speed_cooling_capacity_w",
    "fuel_heating_capacity_w",
    "electric_reheat_capacity_w",
    "combined_fuel_plus_reheat_capacity_w",
    "dx_high_speed_rated_airflow_m3_s",
    "zone_cooling_user_design_load_w_sum",
    "zone_cooling_user_design_flow_m3_s_sum",
    "zone_heating_user_design_load_w_sum",
    "zone_heating_user_design_flow_m3_s_sum",
    "system_cooling_user_design_capacity_w_sum",
    "system_cooling_user_design_flow_m3_s_sum",
    "system_heating_user_design_capacity_w_sum",
    "system_heating_user_design_flow_m3_s_sum",
    "facility_occupied_heating_setpoint_not_met_h",
    "facility_occupied_cooling_setpoint_not_met_h",
    "cooling_unmet_degree_hours_sum",
    "heating_unmet_degree_hours_sum",
    "cooling_occupied_unmet_degree_hours_sum",
    "heating_occupied_unmet_degree_hours_sum",
    "cooling_occupied_unmet_degree_hours_max_zone",
    "heating_occupied_unmet_degree_hours_max_zone",
)
CAPACITY_METRICS = tuple(
    metric for metric in ANALYSIS_METRICS
    if any(token in metric for token in ("capacity", "rated_airflow", "design_load", "design_flow"))
)

REBUILD_ROOT = Path(__file__).resolve().parents[2]
RERUN_ROOT = REBUILD_ROOT / "09_rev01/05_reruns"
DEFAULT_BATCH = RERUN_ROOT / "scientific_runs/rev01_c2_city_ddy12_20260825_v2"
DEFAULT_PLAN = RERUN_ROOT / "plans/rev01_c2_city_ddy_plan_20260825_v4"
DEFAULT_ACCEPTED_MATRIX = (
    REBUILD_ROOT
    / "06_runs/scaled/20260803_routeS96_v2_envscoped/freeze/authorization/execution"
    / "scaled_matrix_fresh96_v2_envscoped.csv"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "c2_city_ddy_sensitivity_20260825_v2"

CAPACITY_WHERE = """
    (ReportName='ComponentSizingSummary' AND TableName IN
      ('Coil:Cooling:DX:TwoSpeed','Coil:Heating:Fuel','Coil:Heating:Electric'))
 OR (ReportName='EquipmentSummary' AND TableName IN ('Cooling Coils','Heating Coils'))
 OR (ReportName='HVACSizingSummary' AND TableName='Coil Sizing Summary')
"""
UNMET_WHERE = """
    (ReportName='AnnualBuildingUtilityPerformanceSummary' AND
      TableName IN ('Comfort and Setpoint Not Met Summary','Setpoint Not Met Criteria'))
 OR (ReportName='SystemSummary' AND TableName='Time Setpoint Not Met')
 OR (ReportName='AnnualThermalResilienceSummary' AND TableName='Unmet Degree-Hours')
"""


class SynthesisError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SynthesisError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    workspace = REBUILD_ROOT.parent.resolve()
    try:
        return resolved.relative_to(workspace).as_posix()
    except ValueError as exc:
        raise SynthesisError(f"Artifact is outside the workspace: {resolved}") from exc


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON is not an object: {path}")
    return value


def one_path(root: Path, pattern: str, label: str) -> Path:
    paths = sorted(root.glob(pattern))
    require(len(paths) == 1, f"Expected one {label} below {root}: {paths}")
    return paths[0]


def verify(path: Path, expected: str, label: str) -> str:
    require(path.is_file(), f"Missing {label}: {path}")
    observed = sha256_file(path)
    require(observed == expected, f"Hash differs for {label}: {path}")
    return observed


def numeric(value: Any) -> float | None:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def strict_occupied(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series.dtype):
        result = series.to_numpy(bool)
    elif pd.api.types.is_numeric_dtype(series.dtype):
        numeric_values = pd.to_numeric(series, errors="coerce").to_numpy(float)
        require(np.isfinite(numeric_values).all(), "Occupied field contains nonfinite values")
        require(set(np.unique(numeric_values)) <= {0.0, 1.0},
                "Occupied numeric field is outside {0,1}")
        result = numeric_values.astype(bool)
    else:
        values = series.astype(str).str.strip().str.casefold()
        require(set(values.unique()) <= {"true", "false"},
                "Occupied text field is outside {true,false}")
        result = values.eq("true").to_numpy()
    return result


def validate_trace_cadence(series: pd.Series) -> None:
    elapsed = pd.to_numeric(series, errors="coerce").to_numpy(float)
    require(np.isfinite(elapsed).all(), "sim_time_hours contains nonfinite values")
    require(len(elapsed) == EXPECTED_ROWS, "sim_time_hours row count differs")
    differences = np.diff(elapsed)
    require((differences > 0).all(), "sim_time_hours is not strictly increasing")
    require(np.allclose(differences, STEP_HOURS, rtol=0.0, atol=1e-9),
            "Trace cadence is not uniformly 15 minutes")


def read_sql_facility_meters(sql_path: Path) -> dict[str, dict[str, Any]]:
    uri = f"{sql_path.resolve().as_uri()}?mode=ro"
    query = """
      SELECT d.VariableName,d.ReportingFrequency,d.VariableUnits,
             COUNT(*) AS n,SUM(r.VariableValue) AS joules
      FROM ReportMeterData AS r
      JOIN ReportMeterDataDictionary AS d
        USING (ReportMeterDataDictionaryIndex)
      WHERE d.VariableName IN ('Electricity:Facility','NaturalGas:Facility')
        AND d.ReportingFrequency='Hourly'
      GROUP BY d.VariableName,d.ReportingFrequency,d.VariableUnits
    """
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            rows = connection.execute(query).fetchall()
    except sqlite3.Error as exc:
        raise SynthesisError(f"Cannot read facility meters from {sql_path}: {exc}") from exc
    result = {
        str(name): {
            "frequency": str(frequency), "units": str(units),
            "rows": int(count), "joules": float(joules),
        }
        for name, frequency, units, count, joules in rows
    }
    require(set(result) == {"Electricity:Facility", "NaturalGas:Facility"},
            f"SQL facility-meter coverage differs: {sql_path}")
    for name, item in result.items():
        require(item["frequency"] == "Hourly", f"SQL meter frequency differs: {name}")
        require(item["units"] == "J", f"SQL meter units differ: {name}")
        require(item["rows"] == 8760, f"SQL meter does not have 8760 rows: {name}")
        require(math.isfinite(item["joules"]) and item["joules"] > 0,
                f"SQL meter total is invalid: {name}")
    return result


def sql_tabular_rows(connection: sqlite3.Connection, where: str) -> list[dict[str, Any]]:
    query = f"""
      SELECT ReportName,TableName,RowName,ColumnName,Units,Value
      FROM TabularDataWithStrings WHERE {where}
      ORDER BY ReportName,TableName,RowName,ColumnName
    """
    columns = ("report_name", "table_name", "row_name", "column_name", "units", "value")
    rows: list[dict[str, Any]] = []
    for values in connection.execute(query):
        row = dict(zip(columns, values))
        row["numeric_value"] = numeric(row["value"])
        rows.append(row)
    return rows


def select_value(
    rows: list[dict[str, Any]], table: str, row: str, column: str, expected_units: str
) -> float:
    matches = [
        item for item in rows
        if item["table_name"] == table and item["row_name"] == row
        and item["column_name"] == column
    ]
    require(len(matches) == 1, f"Expected one SQL value: {table}/{row}/{column}")
    match = matches[0]
    require(match["units"] == expected_units,
            f"SQL units differ: {table}/{row}/{column}: {match['units']} != {expected_units}")
    require(match["numeric_value"] is not None,
            f"SQL value is nonnumeric: {table}/{row}/{column}")
    return float(match["numeric_value"])


def sum_capacity(
    rows: list[dict[str, Any]], table: str, column: str, expected_units: str
) -> float:
    matches = [
        item for item in rows
        if item["report_name"] == "ComponentSizingSummary"
        and item["table_name"] == table and item["column_name"] == column
    ]
    require(matches, f"No component sizing values: {table}/{column}")
    require(all(item["units"] == expected_units for item in matches),
            f"Component sizing units differ: {table}/{column}")
    require(all(item["numeric_value"] is not None for item in matches),
            f"Component sizing value is nonnumeric: {table}/{column}")
    return float(sum(float(item["numeric_value"]) for item in matches))


def discover_zones(columns: list[str]) -> list[str]:
    available = set(columns)
    zones = sorted(
        column[len("zone_"):-len("_ta_c")]
        for column in columns
        if column.startswith("zone_") and column.endswith("_ta_c")
        and f"zone_{column[len('zone_'):-len('_ta_c')]}_tr_c" in available
    )
    require(len(zones) == 15, f"Expected 15 operative-temperature zones, got {len(zones)}")
    return zones


def time_masks(
    frame: pd.DataFrame,
    weather_year: int,
    event_start: str,
    event_end: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    occupied = strict_occupied(frame["occupied"])
    require(int(occupied.sum()) == EXPECTED_OCCUPIED_ROWS, "Occupied-row contract differs")
    base = pd.to_datetime({
        "year": pd.Series(weather_year, index=frame.index),
        "month": pd.to_numeric(frame["month"], errors="raise").astype(int),
        "day": pd.to_numeric(frame["day"], errors="raise").astype(int),
    })
    interval_end = base + pd.to_timedelta(
        pd.to_numeric(frame["current_time"], errors="raise"), unit="h"
    )
    interval_start = interval_end - pd.to_timedelta(STEP_HOURS, unit="h")
    require(not interval_end.isna().any(), "Calendar reconstruction contains null timestamps")
    require(interval_end.is_monotonic_increasing and interval_end.is_unique,
            "Reconstructed trace calendar is not strictly increasing and unique")
    calendar_differences = interval_end.diff().dropna().dt.total_seconds().to_numpy(float) / 3600.0
    require(np.allclose(calendar_differences, STEP_HOURS, rtol=0.0, atol=1e-9),
            "Reconstructed trace calendar is not continuous at 15-minute cadence")
    start_of_year = pd.Timestamp(year=weather_year, month=1, day=1)
    calendar_elapsed = (
        (interval_end - start_of_year).dt.total_seconds().to_numpy(float) / 3600.0
    )
    sim_elapsed = pd.to_numeric(frame["sim_time_hours"], errors="raise").to_numpy(float)
    require(np.allclose(calendar_elapsed, sim_elapsed, rtol=0.0, atol=1e-9),
            "Reconstructed calendar and sim_time_hours differ")
    require(math.isclose(calendar_elapsed[0], STEP_HOURS, abs_tol=1e-9),
            "Trace does not begin at the first 15-minute interval")
    require(math.isclose(calendar_elapsed[-1], 8760.0, abs_tol=1e-9),
            "Trace does not close at 8760 hours")
    event = (
        (interval_start >= pd.Timestamp(event_start))
        & (interval_end <= pd.Timestamp(event_end))
    ).to_numpy()
    require(int(event.sum()) == EXPECTED_WINDOW_ROWS, f"Event row count differs: {event.sum()}")
    event_occupied = occupied & event
    require(int(event_occupied.sum()) == EXPECTED_WINDOW_OCCUPIED_ROWS,
            f"Event occupied-row count differs: {event_occupied.sum()}")
    return occupied, event, event_occupied


def worst_zone_degree_hours(
    frame: pd.DataFrame, zones: list[str], mask: np.ndarray, threshold: float
) -> tuple[float, str]:
    values: dict[str, float] = {}
    for zone in zones:
        air = pd.to_numeric(frame[f"zone_{zone}_ta_c"], errors="raise").to_numpy(float)
        radiant = pd.to_numeric(frame[f"zone_{zone}_tr_c"], errors="raise").to_numpy(float)
        operative = (air + radiant) / 2.0
        require(np.isfinite(operative[mask]).all(), f"Nonfinite operative temperature: {zone}")
        values[zone] = float(np.maximum(operative[mask] - threshold, 0.0).sum() * STEP_HOURS)
    return sorted(values.items(), key=lambda item: (-item[1], item[0]))[0][1], sorted(
        values.items(), key=lambda item: (-item[1], item[0])
    )[0][0]


def artifact_paths_from_snapshot(snapshot: dict[str, Any], cell_id: str) -> dict[str, Path]:
    evidence = snapshot.get(cell_id)
    require(isinstance(evidence, dict), f"Accepted snapshot lacks {cell_id}")
    result: dict[str, Path] = {}
    for role in ("trace", "summary", "sql", "eio", "generated_idf", "cell_result"):
        identity = evidence.get(role)
        require(isinstance(identity, dict), f"Accepted snapshot lacks {cell_id}/{role}")
        path = Path(str(identity["path"])).resolve()
        require(path.is_file(), f"Accepted artifact is missing: {path}")
        require(path.stat().st_size == int(identity["bytes"]), f"Accepted size differs: {path}")
        verify(path, str(identity["sha256"]), f"accepted {cell_id}/{role}")
        result[role] = path
    return result


def revision_paths(batch: Path, row: pd.Series) -> dict[str, Path]:
    cell = batch / "cells" / str(row["cell_id"])
    result = read_json(cell / "CELL_RESULT.json")
    require(result.get("all_preliminary_gates_passed") is True, f"Revision cell not passing: {cell}")
    require(result.get("batch_id") == EXPECTED_BATCH_ID, f"Revision batch differs: {cell}")
    require(result.get("cell_id") == str(row["cell_id"]), f"Revision cell identity differs: {cell}")
    require(result.get("strategy") == str(row["strategy"]), f"Revision strategy differs: {cell}")
    require(
        result.get("accepted_comparator_cell_id") == str(row["accepted_comparator_cell_id"]),
        f"Revision accepted comparator differs: {cell}",
    )
    require(
        result.get("c2_runtime_contract", {}).get("all_gates_passed") is True,
        f"Revision C2 runtime gate not passing: {cell}",
    )
    trace = one_path(cell / "traces", "*.parquet", "revision trace")
    summary = cell / "summary/medium_office_trace_summary.csv"
    sql = one_path(cell / "energyplus", "**/eplusout.sql", "revision SQL")
    eio = one_path(cell / "energyplus", "**/eplusout.eio", "revision EIO")
    generated_idf = one_path(cell / "model", "*.idf", "revision generated IDF")
    verify(trace, result["outputs"]["trace"]["sha256"], "revision trace")
    verify(summary, result["outputs"]["summary"]["sha256"], "revision summary")
    retained = result["outputs"]["retained_energyplus_sha256"]
    verify(sql, retained["eplusout.sql"], "revision SQL")
    verify(eio, retained["eplusout.eio"], "revision EIO")
    verify(generated_idf, result["outputs"]["generated_idf_sha256"], "revision generated IDF")
    return {
        "trace": trace, "summary": summary, "sql": sql, "eio": eio,
        "generated_idf": generated_idf, "cell_result": cell / "CELL_RESULT.json",
    }


def read_sql_metrics(
    sql_path: Path,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    uri = f"{sql_path.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        capacities = sql_tabular_rows(connection, CAPACITY_WHERE)
        unmet = sql_tabular_rows(connection, UNMET_WHERE)
        component_columns = [x[1] for x in connection.execute("PRAGMA table_info(ComponentSizes)")]
        zone_columns = [x[1] for x in connection.execute("PRAGMA table_info(ZoneSizes)")]
        system_columns = [x[1] for x in connection.execute("PRAGMA table_info(SystemSizes)")]
        component = [dict(zip(component_columns, row)) for row in connection.execute("SELECT * FROM ComponentSizes")]
        zones = [dict(zip(zone_columns, row)) for row in connection.execute("SELECT * FROM ZoneSizes")]
        systems = [dict(zip(system_columns, row)) for row in connection.execute("SELECT * FROM SystemSizes")]
    for row in zones:
        row["sizing_table"] = "ZoneSizes"
    for row in systems:
        row["sizing_table"] = "SystemSizes"
    for collection in (capacities, unmet, component, zones, systems):
        for row in collection:
            row.update(metadata)
    metrics = {
        "dx_high_speed_cooling_capacity_w": sum_capacity(
            capacities, "Coil:Cooling:DX:TwoSpeed",
            "Design Size High Speed Gross Rated Total Cooling Capacity",
            "W",
        ),
        "fuel_heating_capacity_w": sum_capacity(
            capacities, "Coil:Heating:Fuel", "Design Size Nominal Capacity", "W"
        ),
        "electric_reheat_capacity_w": sum_capacity(
            capacities, "Coil:Heating:Electric", "Design Size Nominal Capacity", "W"
        ),
        "dx_high_speed_rated_airflow_m3_s": sum_capacity(
            capacities, "Coil:Cooling:DX:TwoSpeed",
            "Design Size High Speed Rated Air Flow Rate",
            "m3/s",
        ),
        "facility_occupied_heating_setpoint_not_met_h": select_value(
            unmet, "Comfort and Setpoint Not Met Summary",
            "Time Setpoint Not Met During Occupied Heating", "Facility",
            "Hours",
        ),
        "facility_occupied_cooling_setpoint_not_met_h": select_value(
            unmet, "Comfort and Setpoint Not Met Summary",
            "Time Setpoint Not Met During Occupied Cooling", "Facility",
            "Hours",
        ),
        "cooling_unmet_degree_hours_sum": select_value(
            unmet, "Unmet Degree-Hours", "Sum", "Cooling Setpoint Unmet Degree-Hours",
            "°C·hr",
        ),
        "heating_unmet_degree_hours_sum": select_value(
            unmet, "Unmet Degree-Hours", "Sum", "Heating Setpoint Unmet Degree-Hours",
            "°C·hr",
        ),
        "cooling_occupied_unmet_degree_hours_sum": select_value(
            unmet, "Unmet Degree-Hours", "Sum", "Cooling Setpoint Unmet Occupied Degree-Hours",
            "°C·hr",
        ),
        "heating_occupied_unmet_degree_hours_sum": select_value(
            unmet, "Unmet Degree-Hours", "Sum", "Heating Setpoint Unmet Occupied Degree-Hours",
            "°C·hr",
        ),
        "cooling_occupied_unmet_degree_hours_max_zone": select_value(
            unmet, "Unmet Degree-Hours", "Max", "Cooling Setpoint Unmet Occupied Degree-Hours",
            "°C·hr",
        ),
        "heating_occupied_unmet_degree_hours_max_zone": select_value(
            unmet, "Unmet Degree-Hours", "Max", "Heating Setpoint Unmet Occupied Degree-Hours",
            "°C·hr",
        ),
    }
    metrics["combined_fuel_plus_reheat_capacity_w"] = (
        metrics["fuel_heating_capacity_w"] + metrics["electric_reheat_capacity_w"]
    )
    zone_frame = pd.DataFrame(zones)
    system_frame = pd.DataFrame(systems)
    for load in ("Cooling", "Heating"):
        tag = load.casefold()
        zone_load = zone_frame.loc[zone_frame["LoadType"].eq(load), "UserDesLoad"].astype(float)
        zone_flow = zone_frame.loc[zone_frame["LoadType"].eq(load), "UserDesFlow"].astype(float)
        system_cap = system_frame.loc[system_frame["LoadType"].eq(load), "UserDesCap"].astype(float)
        system_flow = system_frame.loc[system_frame["LoadType"].eq(load), "UserDesVolFlow"].astype(float)
        require(len(zone_load) == 15 and len(system_cap) == 3, f"Sizing coverage differs: {load}")
        metrics[f"zone_{tag}_user_design_load_w_sum"] = float(zone_load.sum())
        metrics[f"zone_{tag}_user_design_flow_m3_s_sum"] = float(zone_flow.sum())
        metrics[f"system_{tag}_user_design_capacity_w_sum"] = float(system_cap.sum())
        metrics[f"system_{tag}_user_design_flow_m3_s_sum"] = float(system_flow.sum())
    return metrics, capacities, unmet, component, zones + systems


def compute_artifact_metrics(
    paths: dict[str, Path],
    *,
    weather_year: int,
    event_start: str,
    event_end: str,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    schema = pq.read_schema(paths["trace"])
    zones = discover_zones(schema.names)
    required = {
        "occupied", "month", "day", "current_time", "sim_time_hours", "electricity_facility_j",
        "natural_gas_facility_j",
    }
    for zone in zones:
        required.update({f"zone_{zone}_ta_c", f"zone_{zone}_tr_c"})
    require(required <= set(schema.names), f"Trace lacks metrics: {sorted(required - set(schema.names))}")
    frame = pd.read_parquet(paths["trace"], columns=sorted(required))
    require(len(frame) == EXPECTED_ROWS, f"Trace row count differs: {paths['trace']}")
    validate_trace_cadence(frame["sim_time_hours"])
    occupied, event, event_occupied = time_masks(frame, weather_year, event_start, event_end)
    electricity = pd.to_numeric(frame["electricity_facility_j"], errors="raise").to_numpy(float)
    gas = pd.to_numeric(frame["natural_gas_facility_j"], errors="raise").to_numpy(float)
    require(np.isfinite(electricity).all() and (electricity >= 0).all(), "Invalid electricity trace")
    require(np.isfinite(gas).all() and (gas >= 0).all(), "Invalid gas trace")
    trace_electricity_j = float(electricity.sum())
    trace_gas_j = float(gas.sum())
    require(trace_electricity_j > 0 and trace_gas_j > 0,
            "Facility energy totals must be positive")
    sql_meters = read_sql_facility_meters(paths["sql"])
    require(math.isclose(
        trace_electricity_j, float(sql_meters["Electricity:Facility"]["joules"]),
        rel_tol=1e-12, abs_tol=0.1,
    ), "Trace/SQL facility electricity differs")
    require(math.isclose(
        trace_gas_j, float(sql_meters["NaturalGas:Facility"]["joules"]),
        rel_tol=1e-12, abs_tol=0.1,
    ), "Trace/SQL facility natural gas differs")
    metrics: dict[str, Any] = {
        **metadata,
        "annual_electricity_kwh_delivered": trace_electricity_j / JOULES_PER_KWH,
        "annual_natural_gas_kwh_delivered_including_service_water": trace_gas_j / JOULES_PER_KWH,
        "extreme_336h_electricity_kwh_delivered": float(electricity[event].sum() / JOULES_PER_KWH),
        "extreme_336h_natural_gas_kwh_delivered_including_service_water": float(gas[event].sum() / JOULES_PER_KWH),
        "annual_occupied_hours": float(occupied.sum() * STEP_HOURS),
        "extreme_336h_total_hours": float(event.sum() * STEP_HOURS),
        "extreme_336h_occupied_hours": float(event_occupied.sum() * STEP_HOURS),
        "sql_hourly_electricity_meter_rows": int(sql_meters["Electricity:Facility"]["rows"]),
        "sql_hourly_natural_gas_meter_rows": int(sql_meters["NaturalGas:Facility"]["rows"]),
        "trace_minus_sql_electricity_j": (
            trace_electricity_j - float(sql_meters["Electricity:Facility"]["joules"])
        ),
        "trace_minus_sql_natural_gas_j": (
            trace_gas_j - float(sql_meters["NaturalGas:Facility"]["joules"])
        ),
    }
    metrics["annual_total_delivered_site_energy_kwh"] = (
        metrics["annual_electricity_kwh_delivered"]
        + metrics["annual_natural_gas_kwh_delivered_including_service_water"]
    )
    metrics["extreme_336h_total_delivered_site_energy_kwh"] = (
        metrics["extreme_336h_electricity_kwh_delivered"]
        + metrics["extreme_336h_natural_gas_kwh_delivered_including_service_water"]
    )
    for threshold in THRESHOLDS:
        tag = f"{threshold:g}c"
        annual_value, annual_zone = worst_zone_degree_hours(frame, zones, occupied, threshold)
        event_value, event_zone = worst_zone_degree_hours(frame, zones, event_occupied, threshold)
        metrics[f"annual_occupied_worst_zone_degree_hours_above_{tag}"] = annual_value
        metrics[f"annual_occupied_worst_zone_degree_hours_above_{tag}_zone"] = annual_zone
        metrics[f"extreme_336h_occupied_worst_zone_degree_hours_above_{tag}"] = event_value
        metrics[f"extreme_336h_occupied_worst_zone_degree_hours_above_{tag}_zone"] = event_zone
    sql_metrics, capacities, unmet, components, sizing = read_sql_metrics(paths["sql"], metadata)
    metrics.update(sql_metrics)
    summary = pd.read_csv(paths["summary"])
    require(len(summary) == 1, f"Summary is not one row: {paths['summary']}")
    require(math.isclose(
        metrics["annual_electricity_kwh_delivered"], float(summary.iloc[0]["electricity_kwh"]),
        rel_tol=1e-10, abs_tol=1e-5,
    ), "Trace/summary electricity differs")
    require(math.isclose(
        metrics["annual_natural_gas_kwh_delivered_including_service_water"],
        float(summary.iloc[0]["natural_gas_kwh"]), rel_tol=1e-10, abs_tol=1e-5,
    ), "Trace/summary gas differs")
    return metrics, {
        "capacity_tabular": capacities,
        "unmet_tabular": unmet,
        "component_sizes": components,
        "zone_system_sizes": sizing,
    }


def validate_endpoint_contract(frame: pd.DataFrame) -> list[str]:
    missing = sorted(set(ANALYSIS_METRICS) - set(frame.columns))
    require(not missing, f"Predeclared endpoints are absent: {missing}")
    for metric in ANALYSIS_METRICS:
        values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(float)
        require(np.isfinite(values).all(), f"Predeclared endpoint is null/nonfinite: {metric}")
    require(
        (frame[list(CAPACITY_METRICS)] >= 0).all(axis=None),
        "A capacity/design endpoint is negative",
    )
    return list(ANALYSIS_METRICS)


def paired_delta_table(city: pd.DataFrame, accepted: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    keys = ["city", "anchor_id", "policy", "strategy", "accepted_comparator_cell_id"]
    merged = city.merge(accepted, on=keys, suffixes=("_city_ddy", "_accepted_denver"), validate="one_to_one")
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        result = {key: row[key] for key in keys}
        for metric in metrics:
            city_value = float(row[f"{metric}_city_ddy"])
            accepted_value = float(row[f"{metric}_accepted_denver"])
            result[f"city_ddy_{metric}"] = city_value
            result[f"accepted_denver_{metric}"] = accepted_value
            result[f"delta_city_ddy_minus_denver_{metric}"] = city_value - accepted_value
            result[f"pct_city_ddy_minus_denver_{metric}"] = (
                100.0 * (city_value - accepted_value) / accepted_value
                if accepted_value != 0 else np.nan
            )
        rows.append(result)
    return pd.DataFrame(rows)


def within_contract_contrasts(cells: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (city, contract), group in cells.groupby(["city", "sizing_contract"], sort=True):
        fixed = group.loc[group["policy"].eq("fixed")]
        require(len(fixed) == 1 and len(group) == 4, f"Policy coverage differs: {city}/{contract}")
        fixed_row = fixed.iloc[0]
        for _, row in group.iterrows():
            result = {"city": city, "anchor_id": row["anchor_id"], "sizing_contract": contract,
                      "policy": row["policy"], "strategy": row["strategy"]}
            for metric in metrics:
                value = float(row[metric])
                baseline = float(fixed_row[metric])
                result[f"policy_{metric}"] = value
                result[f"fixed_{metric}"] = baseline
                result[f"delta_policy_minus_fixed_{metric}"] = value - baseline
                result[f"pct_policy_minus_fixed_{metric}"] = (
                    100.0 * (value - baseline) / baseline if baseline != 0 else np.nan
                )
            rows.append(result)
    return pd.DataFrame(rows)


def direction(value: float, tolerance: float = 1e-9) -> str:
    if value < -tolerance:
        return "lower"
    if value > tolerance:
        return "higher"
    return "tied"


def differential_contrasts(within: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    city = within.loc[within["sizing_contract"].eq("city_ddy")]
    denver = within.loc[within["sizing_contract"].eq("accepted_denver")]
    merged = city.merge(denver, on=["city", "anchor_id", "policy", "strategy"],
                        suffixes=("_city_ddy", "_accepted_denver"), validate="one_to_one")
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        result = {key: row[key] for key in ("city", "anchor_id", "policy", "strategy")}
        for metric in metrics:
            field = f"delta_policy_minus_fixed_{metric}"
            city_delta = float(row[f"{field}_city_ddy"])
            denver_delta = float(row[f"{field}_accepted_denver"])
            result[f"city_ddy_policy_minus_fixed_{metric}"] = city_delta
            result[f"accepted_denver_policy_minus_fixed_{metric}"] = denver_delta
            result[f"differential_of_contrasts_{metric}"] = city_delta - denver_delta
            result[f"direction_city_ddy_{metric}"] = direction(city_delta)
            result[f"direction_accepted_denver_{metric}"] = direction(denver_delta)
            result[f"direction_changed_{metric}"] = direction(city_delta) != direction(denver_delta)
        rows.append(result)
    return pd.DataFrame(rows)


RANK_METRICS = (
    "annual_total_delivered_site_energy_kwh",
    "extreme_336h_total_delivered_site_energy_kwh",
    "annual_occupied_worst_zone_degree_hours_above_26c",
    "annual_occupied_worst_zone_degree_hours_above_28c",
    "annual_occupied_worst_zone_degree_hours_above_30c",
    "extreme_336h_occupied_worst_zone_degree_hours_above_26c",
    "extreme_336h_occupied_worst_zone_degree_hours_above_28c",
    "extreme_336h_occupied_worst_zone_degree_hours_above_30c",
    "facility_occupied_cooling_setpoint_not_met_h",
    "facility_occupied_heating_setpoint_not_met_h",
    "cooling_occupied_unmet_degree_hours_sum",
    "heating_occupied_unmet_degree_hours_sum",
)


def build_rankings(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (city, contract), group in cells.groupby(["city", "sizing_contract"], sort=True):
        for metric in RANK_METRICS:
            ranks = group[metric].rank(method="min", ascending=True)
            for index, rank in ranks.items():
                rows.append({
                    "city": city, "sizing_contract": contract, "metric": metric,
                    "policy": group.loc[index, "policy"], "value": float(group.loc[index, metric]),
                    "lower_is_preferred": True, "rank": int(rank),
                })
    ranking = pd.DataFrame(rows)
    city = ranking.loc[ranking["sizing_contract"].eq("city_ddy")]
    denver = ranking.loc[ranking["sizing_contract"].eq("accepted_denver")]
    stability = city.merge(denver, on=["city", "metric", "policy"],
                           suffixes=("_city_ddy", "_accepted_denver"), validate="one_to_one")
    stability["rank_change_city_ddy_minus_denver"] = (
        stability["rank_city_ddy"] - stability["rank_accepted_denver"]
    )
    stability["rank_changed"] = stability["rank_change_city_ddy_minus_denver"].ne(0)
    return ranking, stability


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, path, version="2.6", compression="zstd", compression_level=7,
                   row_group_size=65_536, write_page_checksum=True)
    restored = pq.read_table(path, page_checksum_verification=True).to_pandas()
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), restored, check_dtype=True, check_exact=True)


def write_small_table(frame: pd.DataFrame, output: Path, stem: str) -> list[Path]:
    ordered = frame.reset_index(drop=True)
    csv_path = output / f"{stem}.csv"
    parquet_path = output / f"{stem}.parquet"
    ordered.to_csv(csv_path, index=False, lineterminator="\n", float_format="%.12g")
    write_parquet(ordered, parquet_path)
    require(len(pd.read_csv(csv_path)) == len(ordered), f"CSV row count differs: {stem}")
    return [csv_path, parquet_path]


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def build_report(
    cell_metrics: pd.DataFrame,
    matched: pd.DataFrame,
    within: pd.DataFrame,
    differential: pd.DataFrame,
    rankings: pd.DataFrame,
    stability: pd.DataFrame,
) -> str:
    lines = [
        "# C2 city-design-condition sensitivity", "", f"Generated: {utc_now()}", "",
        "## Closure", "",
        "The frozen 12-cell batch completed 12/12 with all execution, runtime-location, storage, and accepted-evidence gates passing. The alternative IDF substitutes the geographically matched companion Site:Location and two selected annual design days while retaining the identical annual EPW, building, HVAC, schedules, controller, and policy settings. Because Keep Site Location Information remains at its default No, the annual weather run uses the unchanged EPW location; the companion location makes the design-day package internally city-consistent and removes the textual Denver location/elevation from the alternative IDF.", "",
        "The results are a representative current city-design-condition sensitivity. They are not future-climate resizing, proof of equipment saturation, or a multi-building/general climate-model robustness test. NaturalGas:Facility includes service-water heating. Operative-temperature degree-hours are descriptive warm-exposure diagnostics, not observed satisfaction, health, or standards compliance.", "",
        "## Fixed-policy sizing changes", "",
        "The capacity values are policy-invariant within every city and sizing contract; the fixed row therefore provides one nonduplicated capacity comparison per city.", "",
        "| City | City-DDY DX cooling | Denver DX cooling | Delta | City-DDY fuel heat | Denver fuel heat | Delta | City-DDY electric reheat | Denver electric reheat | Delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    fixed = matched.loc[matched["policy"].eq("fixed")]
    for _, row in fixed.sort_values("city").iterrows():
        lines.append(
            "| {city} | {city_dx:.2f} kW | {denver_dx:.2f} kW | {dx:+.2f}% | {city_fuel:.2f} kW | {denver_fuel:.2f} kW | {fuel:+.2f}% | {city_reheat:.2f} kW | {denver_reheat:.2f} kW | {reheat:+.2f}% |".format(
                city=row["city"],
                city_dx=row["city_ddy_dx_high_speed_cooling_capacity_w"] / 1000.0,
                denver_dx=row["accepted_denver_dx_high_speed_cooling_capacity_w"] / 1000.0,
                dx=row["pct_city_ddy_minus_denver_dx_high_speed_cooling_capacity_w"],
                city_fuel=row["city_ddy_fuel_heating_capacity_w"] / 1000.0,
                denver_fuel=row["accepted_denver_fuel_heating_capacity_w"] / 1000.0,
                fuel=row["pct_city_ddy_minus_denver_fuel_heating_capacity_w"],
                city_reheat=row["city_ddy_electric_reheat_capacity_w"] / 1000.0,
                denver_reheat=row["accepted_denver_electric_reheat_capacity_w"] / 1000.0,
                reheat=row["pct_city_ddy_minus_denver_electric_reheat_capacity_w"],
            )
        )
    lines.extend([
        "", "## Annual absolute outcomes", "",
        "Delivered energy is reported at the site boundary. Natural gas includes service-water heating.", "",
        "| City | Sizing | Policy | Electricity | Natural gas | Total site energy | DH>26 | DH>28 | DH>30 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    policy_sort = {policy: index for index, policy in enumerate(POLICY_ORDER)}
    ordered_cells = cell_metrics.assign(
        _policy_order=cell_metrics["policy"].map(policy_sort)
    ).sort_values(["city", "sizing_contract", "_policy_order"])
    for _, row in ordered_cells.iterrows():
        lines.append(
            "| {city} | {contract} | {policy} | {electricity:.2f} kWh | {gas:.2f} kWh | {total:.2f} kWh | {dh26:.2f} C h | {dh28:.2f} C h | {dh30:.2f} C h |".format(
                city=row["city"], contract=row["sizing_contract"], policy=row["policy"],
                electricity=row["annual_electricity_kwh_delivered"],
                gas=row["annual_natural_gas_kwh_delivered_including_service_water"],
                total=row["annual_total_delivered_site_energy_kwh"],
                dh26=row["annual_occupied_worst_zone_degree_hours_above_26c"],
                dh28=row["annual_occupied_worst_zone_degree_hours_above_28c"],
                dh30=row["annual_occupied_worst_zone_degree_hours_above_30c"],
            )
        )
    lines.extend([
        "", "## Extreme-window absolute outcomes", "",
        "Each event window contains exactly 336 hours (1,344 quarter-hour records), including exactly 160 occupied hours (640 records); degree-hours use occupied records only.", "",
        "| City | Sizing | Policy | Electricity | Natural gas | Total site energy | DH>26 | DH>28 | DH>30 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in ordered_cells.iterrows():
        lines.append(
            "| {city} | {contract} | {policy} | {electricity:.2f} kWh | {gas:.2f} kWh | {total:.2f} kWh | {dh26:.2f} C h | {dh28:.2f} C h | {dh30:.2f} C h |".format(
                city=row["city"], contract=row["sizing_contract"], policy=row["policy"],
                electricity=row["extreme_336h_electricity_kwh_delivered"],
                gas=row["extreme_336h_natural_gas_kwh_delivered_including_service_water"],
                total=row["extreme_336h_total_delivered_site_energy_kwh"],
                dh26=row["extreme_336h_occupied_worst_zone_degree_hours_above_26c"],
                dh28=row["extreme_336h_occupied_worst_zone_degree_hours_above_28c"],
                dh30=row["extreme_336h_occupied_worst_zone_degree_hours_above_30c"],
            )
        )
    lines.extend([
        "", "## Occupied setpoint-not-met evidence", "",
        "These annual SQL diagnostics establish setpoint mismatch, not simultaneous component saturation.", "",
        "| City | Sizing | Policy | Cooling not met | Heating not met | Cooling occupied DH sum | Heating occupied DH sum | Cooling occupied DH max zone | Heating occupied DH max zone |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in ordered_cells.iterrows():
        lines.append(
            "| {city} | {contract} | {policy} | {cool_time:.2f} h | {heat_time:.2f} h | {cool_sum:.2f} C h | {heat_sum:.2f} C h | {cool_max:.2f} C h | {heat_max:.2f} C h |".format(
                city=row["city"], contract=row["sizing_contract"], policy=row["policy"],
                cool_time=row["facility_occupied_cooling_setpoint_not_met_h"],
                heat_time=row["facility_occupied_heating_setpoint_not_met_h"],
                cool_sum=row["cooling_occupied_unmet_degree_hours_sum"],
                heat_sum=row["heating_occupied_unmet_degree_hours_sum"],
                cool_max=row["cooling_occupied_unmet_degree_hours_max_zone"],
                heat_max=row["heating_occupied_unmet_degree_hours_max_zone"],
            )
        )
    lines.extend([
        "", "## Policy-minus-fixed contrasts", "",
        "Negative values mean the policy is lower than fixed under the same sizing contract.", "",
        "| City | Sizing | Policy | Annual site energy | Extreme site energy | Annual DH>28 | Extreme DH>28 | Cooling not-met time | Heating not-met time |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    nonfixed_within = within.loc[~within["policy"].eq("fixed")].assign(
        _policy_order=lambda frame: frame["policy"].map(policy_sort)
    ).sort_values(["city", "sizing_contract", "_policy_order"])
    for _, row in nonfixed_within.iterrows():
        lines.append(
            "| {city} | {contract} | {policy} | {annual_energy:+.2f} kWh | {extreme_energy:+.2f} kWh | {annual_dh:+.2f} C h | {extreme_dh:+.2f} C h | {cool_time:+.2f} h | {heat_time:+.2f} h |".format(
                city=row["city"], contract=row["sizing_contract"], policy=row["policy"],
                annual_energy=row["delta_policy_minus_fixed_annual_total_delivered_site_energy_kwh"],
                extreme_energy=row["delta_policy_minus_fixed_extreme_336h_total_delivered_site_energy_kwh"],
                annual_dh=row["delta_policy_minus_fixed_annual_occupied_worst_zone_degree_hours_above_28c"],
                extreme_dh=row["delta_policy_minus_fixed_extreme_336h_occupied_worst_zone_degree_hours_above_28c"],
                cool_time=row["delta_policy_minus_fixed_facility_occupied_cooling_setpoint_not_met_h"],
                heat_time=row["delta_policy_minus_fixed_facility_occupied_heating_setpoint_not_met_h"],
            )
        )
    lines.extend([
        "", "## Lower-is-better policy rankings", "",
        "Rank 1 policy or tied policies for each city and sizing contract:", "",
        "| Metric | Beijing city-DDY | Beijing Denver | Guangzhou city-DDY | Guangzhou Denver | Phoenix city-DDY | Phoenix Denver |",
        "|---|---|---|---|---|---|---|",
    ])
    for metric in RANK_METRICS:
        winners: list[str] = []
        for city in ("Beijing", "Guangzhou", "Phoenix"):
            for contract in ("city_ddy", "accepted_denver"):
                selection = rankings.loc[
                    rankings["city"].eq(city)
                    & rankings["sizing_contract"].eq(contract)
                    & rankings["metric"].eq(metric)
                    & rankings["rank"].eq(1), "policy"
                ]
                require(len(selection) >= 1, f"No rank-1 policy: {city}/{contract}/{metric}")
                winners.append(" + ".join(sorted(selection, key=policy_sort.get)))
        lines.append("| {metric} | {values} |".format(metric=metric, values=" | ".join(winners)))
    key_metrics = (
        "annual_total_delivered_site_energy_kwh",
        "annual_occupied_worst_zone_degree_hours_above_28c",
        "extreme_336h_occupied_worst_zone_degree_hours_above_28c",
    )
    nonfixed = differential.loc[~differential["policy"].eq("fixed")]
    changes = {
        metric: int(nonfixed[f"direction_changed_{metric}"].sum()) for metric in key_metrics
    }
    rank_changes = int(stability["rank_changed"].sum())
    lines.extend([
        "", "## Policy-attribution robustness", "",
        "The differential-of-contrasts table asks whether changing the sizing contract changes each policy-minus-fixed effect. Direction changes among the nine non-fixed city-policy comparisons were: annual site energy {energy}/9, annual DH>28 {annual}/9, and extreme-period DH>28 {extreme}/9. Across the 144 matched city/metric/policy ranking entries, {rank_changes} ranks changed (ties use minimum rank).".format(
            energy=changes[key_metrics[0]], annual=changes[key_metrics[1]],
            extreme=changes[key_metrics[2]], rank_changes=rank_changes,
        ), "",
        "Interpret direction changes together with their magnitudes in `differential_of_contrasts`; a sign change near zero does not automatically constitute a material reversal. Lower energy and lower degree-hours refer to different objectives and should not be collapsed into a single comfort-performance claim.", "",
        "The complete city-DDY-minus-Denver substitutions for every declared endpoint are retained in `matched_city_ddy_minus_denver`; the tables above show the absolute energy/fuel, all three warm thresholds, annual occupied unmet-load evidence, and the principal policy contrasts without suppressing zero-valued outcomes.", "",
        "## Files", "",
        "- `cell_metrics`: absolute accepted-Denver and city-design-condition endpoints.",
        "- `matched_city_ddy_minus_denver`: matched sizing-contract substitutions.",
        "- `within_contract_policy_minus_fixed`: policy effects separately under each sizing contract.",
        "- `differential_of_contrasts`: change in policy-minus-fixed effects caused by sizing substitution.",
        "- `policy_rankings` and `policy_ranking_stability`: within-city rankings and changes.",
        "- Long SQL sizing/unmet tables are stored as compressed Parquet only.", "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--accepted-matrix", type=Path, default=DEFAULT_ACCEPTED_MATRIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    require(not args.output.exists() and not args.output.is_symlink(), f"Fresh output required: {args.output}")

    batch_closure_path = args.batch / "BATCH_CLOSURE.json"
    verify(batch_closure_path, EXPECTED_BATCH_CLOSURE_SHA256, "C2 batch closure")
    batch_closure = read_json(batch_closure_path)
    require(batch_closure.get("all_gates_passed") is True, "C2 batch is not passing")
    require(batch_closure.get("status") == "PASS", "C2 batch status is not PASS")
    require(batch_closure.get("batch_id") == EXPECTED_BATCH_ID, "C2 batch ID differs")
    require(batch_closure.get("planned_cells") == batch_closure.get("passed_cells") == 12, "C2 is not 12/12")
    required_gates = {
        "accepted_comparator_evidence_unchanged", "all_cells_final_validated",
        "all_cells_fresh_preliminary_validated", "authorized_sources_unchanged_at_close",
        "c2_runtime_contracts_validated", "exact_cell_id_coverage",
        "exact_frozen_runner_per_subprocess", "frozen_inputs_unchanged_at_close",
        "maximum_four_workers", "one_strategy_per_subprocess",
        "parity_authorization_present_when_required", "shared_model_once_per_batch",
        "storage_finalize_passed", "storage_stage_passed", "zero_resume_reuse_splice_purge",
    }
    require(set(batch_closure["closure_gates"]) == required_gates, "Batch closure-gate set differs")
    require(all(batch_closure["closure_gates"].values()), "At least one batch gate failed")
    require(batch_closure["launcher_sha256"] == EXPECTED_LAUNCHER_SHA256, "Launcher hash differs")
    require(batch_closure["runner_sha256"] == EXPECTED_RUNNER_SHA256, "Runner hash differs")
    require(batch_closure["runner_helper_sha256"] == EXPECTED_HELPER_SHA256, "Helper hash differs")
    require(batch_closure["matrix_sha256"] == EXPECTED_MATRIX_SHA256, "Matrix binding differs")
    require(batch_closure["variant_manifest_sha256"] == EXPECTED_MANIFEST_SHA256, "Manifest binding differs")
    require(batch_closure["storage_mapping_sha256"] == EXPECTED_STORAGE_MAPPING_SHA256, "Storage map differs")
    require(batch_closure["storage_final"]["all_gates_passed"] is True, "Storage finalization failed")
    freeze_path = args.batch / "FREEZE.json"
    cell_index_path = args.batch / "CELL_INDEX.parquet"
    verify(freeze_path, EXPECTED_FREEZE_SHA256, "C2 batch freeze")
    verify(cell_index_path, EXPECTED_CELL_INDEX_SHA256, "C2 final cell index")
    require(batch_closure["freeze_sha256"] == EXPECTED_FREEZE_SHA256, "Freeze binding differs")
    require(batch_closure["cell_index"] == {
        "path": str(cell_index_path.resolve()), "rows": 12,
        "sha256": EXPECTED_CELL_INDEX_SHA256,
    }, "Final cell-index binding differs")
    cell_index = pd.read_parquet(cell_index_path)
    require(len(cell_index) == cell_index["cell_id"].nunique() == 12, "Cell index coverage differs")
    require(cell_index["status"].eq("complete_final_validated").all(),
            "A final cell-index row is not validated")

    plan_closure_path = args.plan / "PLAN_CLOSURE.json"
    matrix_path = args.plan / "c2_city_ddy12.csv"
    manifest_path = args.plan / "VARIANT_MANIFEST.json"
    verify(plan_closure_path, EXPECTED_PLAN_CLOSURE_SHA256, "C2 plan closure")
    verify(matrix_path, EXPECTED_MATRIX_SHA256, "C2 matrix")
    verify(manifest_path, EXPECTED_MANIFEST_SHA256, "C2 variant manifest")
    verify(args.accepted_matrix, EXPECTED_ACCEPTED_MATRIX_SHA256, "accepted matrix")
    matrix = pd.read_csv(matrix_path)
    accepted_matrix = pd.read_csv(args.accepted_matrix)
    require(len(matrix) == matrix["cell_id"].nunique() == 12, "C2 matrix coverage differs")
    require(matrix.groupby("city")["strategy"].nunique().to_dict() == {
        "Beijing": 4, "Guangzhou": 4, "Phoenix": 4,
    }, "C2 city-policy coverage differs")

    freeze = read_json(freeze_path)
    accepted_snapshot = freeze.get("accepted_comparator_snapshot")
    require(isinstance(accepted_snapshot, dict) and len(accepted_snapshot) == 12,
            "Accepted comparator snapshot is not 12 cells")
    require(
        set(accepted_snapshot) == set(matrix["accepted_comparator_cell_id"].astype(str)),
        "Accepted comparator snapshot identities differ from the C2 matrix",
    )
    accepted_weather = accepted_matrix.loc[
        accepted_matrix["anchor_id"].astype(str).isin(matrix["anchor_id"].astype(str).unique())
    ].copy()
    require(len(accepted_weather) == 12, "Accepted event-metadata coverage differs")

    cell_rows: list[dict[str, Any]] = []
    details: dict[str, list[dict[str, Any]]] = {
        "capacity_tabular": [], "unmet_tabular": [], "component_sizes": [], "zone_system_sizes": [],
    }
    artifact_manifest: list[dict[str, Any]] = []
    for _, row in matrix.sort_values("cell_id").iterrows():
        policy = STRATEGY_TO_POLICY[str(row["strategy"])]
        event_rows = accepted_weather.loc[
            accepted_weather["anchor_id"].astype(str).eq(str(row["anchor_id"]))
            & accepted_weather["strategy"].astype(str).eq(str(row["strategy"]))
        ]
        require(len(event_rows) == 1, f"Accepted event metadata is not unique: {row['cell_id']}")
        event = event_rows.iloc[0]
        require(int(event["recomputed_weather_year"]) == int(row["weather_year"]),
                f"Accepted weather year differs: {row['cell_id']}")
        common = {
            "city": str(row["city"]), "anchor_id": str(row["anchor_id"]),
            "policy": policy, "strategy": str(row["strategy"]),
            "weather_year": int(row["weather_year"]),
            "climate_role": str(row["climate_role"]),
            "time_slice": str(row["time_slice"]),
            "selector_role": str(row["selector_role"]),
            "accepted_comparator_cell_id": str(row["accepted_comparator_cell_id"]),
            "event_window_start": str(event["event_window_start"]),
            "event_window_end_exclusive": str(event["event_window_end_exclusive"]),
        }
        path_sets = {
            "city_ddy": revision_paths(args.batch, row),
            "accepted_denver": artifact_paths_from_snapshot(
                accepted_snapshot, str(row["accepted_comparator_cell_id"])
            ),
        }
        indexed = cell_index.loc[cell_index["cell_id"].astype(str).eq(str(row["cell_id"]))]
        require(len(indexed) == 1, f"Final cell index lacks {row['cell_id']}")
        indexed_row = indexed.iloc[0]
        require(
            sha256_file(path_sets["city_ddy"]["trace"]) == str(indexed_row["trace_sha256"]),
            f"Final cell-index trace binding differs: {row['cell_id']}",
        )
        require(
            sha256_file(path_sets["city_ddy"]["summary"]) == str(indexed_row["summary_sha256"]),
            f"Final cell-index summary binding differs: {row['cell_id']}",
        )
        for contract, paths in path_sets.items():
            artifact_cell = str(row["cell_id"] if contract == "city_ddy" else row["accepted_comparator_cell_id"])
            metadata = {**common, "sizing_contract": contract, "artifact_cell_id": artifact_cell}
            metrics, long_tables = compute_artifact_metrics(
                paths, weather_year=int(row["weather_year"]),
                event_start=common["event_window_start"], event_end=common["event_window_end_exclusive"],
                metadata=metadata,
            )
            cell_rows.append(metrics)
            for name, values in long_tables.items():
                details[name].extend(values)
            for role, path in paths.items():
                artifact_manifest.append({
                    **metadata, "artifact_role": role, "path": portable_path(path),
                    "bytes": int(path.stat().st_size), "sha256": sha256_file(path),
                })

    cell_metrics = pd.DataFrame(cell_rows).sort_values(
        ["city", "sizing_contract", "policy"]
    ).reset_index(drop=True)
    require(len(cell_metrics) == 24, "Expected 24 role-specific cell rows")
    require(
        cell_metrics.groupby(["city", "sizing_contract"])["policy"].nunique().eq(4).all(),
        "Absolute cell table lacks a complete policy set",
    )
    metrics = validate_endpoint_contract(cell_metrics)
    for (city, contract), group in cell_metrics.groupby(["city", "sizing_contract"]):
        for metric in CAPACITY_METRICS:
            values = group[metric].to_numpy(float)
            require(
                np.allclose(values, values[0], rtol=0.0, atol=1e-9),
                f"Sizing endpoint varies by policy: {city}/{contract}/{metric}",
            )
    city_cells = cell_metrics.loc[cell_metrics["sizing_contract"].eq("city_ddy")]
    accepted_cells = cell_metrics.loc[cell_metrics["sizing_contract"].eq("accepted_denver")]
    matched = paired_delta_table(city_cells, accepted_cells, metrics)
    within = within_contract_contrasts(cell_metrics, metrics)
    differential = differential_contrasts(within, metrics)
    rankings, stability = build_rankings(cell_metrics)

    args.output.mkdir(parents=True, exist_ok=False)
    outputs: list[Path] = []
    for frame, stem in (
        (cell_metrics, "cell_metrics"),
        (matched, "matched_city_ddy_minus_denver"),
        (within, "within_contract_policy_minus_fixed"),
        (differential, "differential_of_contrasts"),
        (rankings, "policy_rankings"),
        (stability, "policy_ranking_stability"),
        (pd.DataFrame(artifact_manifest).sort_values(["artifact_cell_id", "artifact_role"]), "input_artifact_manifest"),
    ):
        outputs.extend(write_small_table(frame, args.output, stem))
    for name, values in details.items():
        frame = pd.DataFrame(values)
        path = args.output / f"{name}.parquet"
        write_parquet(frame.reset_index(drop=True), path)
        outputs.append(path)
    report_path = args.output / "SYNTHESIS_REPORT.md"
    report_path.write_text(
        build_report(cell_metrics, matched, within, differential, rankings, stability),
        encoding="utf-8",
    )
    outputs.append(report_path)

    # Re-verify every input after all output writes.
    require(sha256_file(batch_closure_path) == EXPECTED_BATCH_CLOSURE_SHA256,
            "Batch closure changed during synthesis")
    manifest_frame = pd.DataFrame(artifact_manifest)
    require(all(
        (REBUILD_ROOT.parent / path).is_file()
        and sha256_file(REBUILD_ROOT.parent / path) == digest
        for path, digest in zip(manifest_frame["path"], manifest_frame["sha256"])
    ), "At least one input artifact changed during synthesis")
    output_manifest = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(outputs)
    }
    closure = {
        "schema_version": SCHEMA,
        "created_utc": utc_now(), "status": "PASS", "all_gates_passed": True,
        "script_path": portable_path(Path(__file__)),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "batch_id": EXPECTED_BATCH_ID,
        "batch_closure_path": portable_path(batch_closure_path),
        "batch_closure_sha256": EXPECTED_BATCH_CLOSURE_SHA256,
        "freeze_sha256": EXPECTED_FREEZE_SHA256,
        "cell_index_sha256": EXPECTED_CELL_INDEX_SHA256,
        "plan_closure_sha256": EXPECTED_PLAN_CLOSURE_SHA256,
        "matrix_sha256": EXPECTED_MATRIX_SHA256,
        "variant_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "accepted_matrix_sha256": EXPECTED_ACCEPTED_MATRIX_SHA256,
        "absolute_cell_metric_rows": len(cell_metrics),
        "matched_sizing_contract_rows": len(matched),
        "within_contract_policy_contrast_rows": len(within),
        "differential_of_contrast_rows": len(differential),
        "ranking_rows": len(rankings),
        "ranking_stability_rows": len(stability),
        "predeclared_numeric_endpoints": list(ANALYSIS_METRICS),
        "predeclared_ranking_endpoints": list(RANK_METRICS),
        "input_artifacts": len(artifact_manifest),
        "event_window_rows_per_cell": EXPECTED_WINDOW_ROWS,
        "event_window_occupied_rows_per_cell": EXPECTED_WINDOW_OCCUPIED_ROWS,
        "annual_trace_rows_per_cell": EXPECTED_ROWS,
        "annual_occupied_rows_per_cell": EXPECTED_OCCUPIED_ROWS,
        "capacity_policy_invariance_passed": True,
        "trace_calendar_continuity_passed": True,
        "trace_sql_facility_meter_reconciliation_passed": True,
        "sql_tabular_unit_contract_passed": True,
        "public_manifest_paths_are_workspace_relative": True,
        "energyplus_invocations": 0,
        "accepted_outputs_modified": False,
        "c2_outputs_modified": False,
        "raw_source_ddys_published": False,
        "output_manifest": output_manifest,
        "interpretation_limits": [
            "Representative current city-design-condition sensitivity only.",
            "Not future-climate resizing or proof of component saturation.",
            "NaturalGas:Facility includes service-water heating.",
            "Operative-temperature degree-hours are descriptive physical exposure diagnostics.",
        ],
    }
    closure_path = args.output / "SYNTHESIS_CLOSURE.json"
    closure_path.write_text(json.dumps(closure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(closure, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
