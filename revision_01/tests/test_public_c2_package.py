from __future__ import annotations

from argparse import Namespace
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative: str, name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


compiler = load_module(
    "scripts/automation/compile_c2_city_ddy_plan.py", "public_c2_compiler"
)
runner = load_module(
    "scripts/automation/run_c2_city_ddy_batch.py", "public_c2_batch_runner"
)
synth = load_module(
    "scripts/synthesis/synthesize_c2_city_ddy.py", "public_c2_synthesizer"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_c2_public_matrix_is_path_free_and_complete() -> None:
    matrix = pd.read_csv(ROOT / "configs/c2_city_ddy_public_matrix.csv")
    assert len(matrix) == matrix["cell_id"].nunique() == 12
    assert "weather_path" not in matrix
    assert "source_idf_path" not in matrix
    assert set(matrix["batch_id"]) == {"REV01-C2-012-20260825-v2"}
    assert matrix.groupby("city")["strategy"].nunique().to_dict() == {
        "Beijing": 4,
        "Guangzhou": 4,
        "Phoenix": 4,
    }
    assert set(matrix["sizing_contract"]) == {"city_ddy"}
    assert set(matrix["expected_trace_rows"]) == {35040}
    assert set(matrix["expected_occupied_rows"]) == {16704}
    assert matrix["source_idf_sha256"].nunique() == 3
    assert matrix["source_ddy_sha256"].nunique() == 3
    for field in ("resume_permitted", "reuse_or_splice_permitted", "purge_permitted"):
        assert not matrix[field].map(bool).any()


def test_c2_public_provenance_binds_code_and_outputs() -> None:
    record = json.loads(
        (ROOT / "diagnostics/c2_city_ddy/C2_PUBLIC_PROVENANCE.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["status"] == "PASS"
    assert record["scientific_cells"] == {
        "planned": 12,
        "passed": 12,
        "failed": 0,
        "accepted_denver_comparator_cells_reused": 12,
    }
    for relative, expected in record["public_code_hashes"].items():
        assert sha256(ROOT / relative) == expected
    for relative, expected in record["public_output_hashes"].items():
        assert sha256(ROOT / relative) == expected
    assert sha256(ROOT / "configs/c2_city_ddy_public_matrix.csv") == record[
        "frozen_bindings"
    ]["public_path_free_matrix_sha256"]
    assert sha256(ROOT / "configs/variant_manifest_c2_city_ddy.json") == record[
        "frozen_bindings"
    ]["variant_manifest_sha256"]


def test_c2_processed_tables_have_declared_coverage_and_findings() -> None:
    cells = pd.read_csv(ROOT / "summary_outputs/c2_cell_metrics.csv")
    matched = pd.read_csv(
        ROOT / "summary_outputs/c2_matched_city_ddy_minus_denver.csv"
    )
    within = pd.read_csv(
        ROOT / "summary_outputs/c2_within_contract_policy_minus_fixed.csv"
    )
    differential = pd.read_csv(
        ROOT / "summary_outputs/c2_differential_of_contrasts.csv"
    )
    rankings = pd.read_csv(ROOT / "summary_outputs/c2_policy_rankings.csv")
    stability = pd.read_csv(
        ROOT / "summary_outputs/c2_policy_ranking_stability.csv"
    )
    assert len(cells) == 24
    assert len(matched) == 12
    assert len(within) == 24
    assert len(differential) == 12
    assert len(rankings) == 288
    assert len(stability) == 144
    assert cells.groupby(["city", "sizing_contract"])["policy"].nunique().eq(4).all()
    active = differential.loc[differential["policy"].ne("fixed")]
    assert len(active) == 9
    assert not active[
        "direction_changed_annual_total_delivered_site_energy_kwh"
    ].any()
    assert not active[
        "direction_changed_annual_occupied_worst_zone_degree_hours_above_28c"
    ].any()
    assert int(
        active[
            "direction_changed_extreme_336h_occupied_worst_zone_degree_hours_above_28c"
        ].sum()
    ) == 5
    assert int(stability["rank_changed"].sum()) == 49


def test_c2_provenance_has_three_sources_and_semantic_diff_gates() -> None:
    provenance = json.loads(
        (ROOT / "diagnostics/c2_city_ddy/C2_DDY_PROVENANCE.json").read_text(
            encoding="utf-8"
        )
    )
    assert provenance["raw_source_contents_published"] is False
    assert set(provenance["cities"]) == {"Beijing", "Guangzhou", "Phoenix"}
    for city, item in provenance["cities"].items():
        assert item["city"] == city
        assert item["url"].startswith("https://energyplus-weather.s3.amazonaws.com/")
        assert len(item["raw_ddy_sha256"]) == 64
        assert len(item["source_idf_sha256"]) == 64
        assert len(item["design_day_names"]) == 2
        assert all(item["semantic_diff_gates"].values())
        assert (
            item["accepted_base_non_sizing_semantic_sha256"]
            == item["non_sizing_semantic_sha256"]
        )


def test_c2_compiler_comment_parser_and_frozen_constants() -> None:
    text = (
        "Site:Location, X, 1, 2, 3, 4; ! Degrees; ignored\n"
        "RunPeriod, R, 1, 1, , 12, 31;\n"
    )
    spans = compiler.object_spans(text, "Site:Location")
    assert len(spans) == 1
    assert text[slice(*spans[0])].endswith("4;")
    assert compiler.BATCH_ID == "REV01-C2-012-20260825-v2"
    assert compiler.EXPECTED_ACCEPTED_MATRIX_SHA256 == (
        "fa03767f0c3d581e65e00ff6149546c44fe3e936608da73ad2f5485bf19895e0"
    )


def test_c2_runner_uses_row_resolved_idf() -> None:
    row = pd.read_csv(ROOT / "configs/c2_city_ddy_public_matrix.csv").iloc[0].copy()
    source_idf = Path("/tmp/c2-source.idf").resolve()
    weather_source = Path("/tmp/c2-weather.epw").resolve()
    row["source_idf_path"] = str(source_idf)
    row["weather_path"] = str(weather_source)
    frozen_idf = Path("/tmp/c2-frozen.idf")
    frozen = SimpleNamespace(
        idf_by_source={str(source_idf): frozen_idf},
        weather_by_source={str(weather_source): Path("/tmp/c2-frozen.epw")},
        model=Path("/tmp/model.joblib"),
        model_metrics=Path("/tmp/model-metrics.json"),
        variant_manifest=Path("/tmp/manifest.json"),
        runner=Path("/tmp/runner.py"),
    )
    args = Namespace(
        python=Path("/tmp/python"),
        training_data=Path("/tmp/training.csv"),
        eplus_root=Path("/tmp/eplus"),
        idf=Path("/tmp/accepted-denver.idf"),
    )
    command = runner.command_for_cell(
        row, cell_dir=Path("/tmp/c2-cell"), frozen=frozen, args=args
    )
    assert command[command.index("--idf") + 1] == str(frozen_idf)
    assert command[command.index("--sizing-contract") + 1] == "city_ddy"
    assert str(args.idf) not in command


def test_c2_runtime_contract_on_synthetic_outputs(tmp_path: Path) -> None:
    cell = tmp_path / "cell"
    model = cell / "model"
    energyplus = cell / "energyplus" / "case"
    model.mkdir(parents=True)
    energyplus.mkdir(parents=True)
    (model / "generated.idf").write_text(
        "Site:Location, BEIJING_CHN Design_Conditions, 39.8, 116.47, 8.0, 32.0;\n"
        "SizingPeriod:DesignDay, BEIJING Ann Htg 99.6% Condns DB;\n"
        "SizingPeriod:DesignDay, BEIJING Ann Clg .4% Condns DB=>MWB;\n",
        encoding="utf-8",
    )
    (energyplus / "eplusout.eio").write_text(
        "Site:Location,Beijing Synthetic,39.90,116.41,8.00,44.00,100798,1.1980\n",
        encoding="utf-8",
    )
    (energyplus / "eplusout.err").write_text(
        "EnergyPlus Completed Successfully-- 1 Warning; 0 Severe Errors\n",
        encoding="utf-8",
    )
    with sqlite3.connect(energyplus / "eplusout.sql") as connection:
        connection.execute("CREATE TABLE ZoneSizes (DesDayName TEXT)")
        connection.execute("CREATE TABLE SystemSizes (DesDayName TEXT)")
        for table in ("ZoneSizes", "SystemSizes"):
            connection.executemany(
                f"INSERT INTO {table} VALUES (?)",
                [
                    ("BEIJING ANN HTG 99.6% CONDNS DB",),
                    ("BEIJING ANN CLG .4% CONDNS DB=>MWB",),
                ],
            )
    row = pd.Series(
        {
            "batch_id": "REV01-C2-012-20260825-v2",
            "companion_site_location_name": "BEIJING_CHN Design_Conditions",
            "companion_latitude_deg": 39.8,
            "companion_longitude_deg": 116.47,
            "companion_time_zone_hours": 8.0,
            "companion_elevation_m": 32.0,
            "heating_design_day_name": "BEIJING Ann Htg 99.6% Condns DB",
            "cooling_design_day_name": "BEIJING Ann Clg .4% Condns DB=>MWB",
            "runtime_epw_latitude_deg": 39.9042,
            "runtime_epw_longitude_deg": 116.4074,
            "runtime_epw_time_zone_hours": 8.0,
            "runtime_epw_elevation_m": 44.0,
            "runtime_epw_location_header_sha256": "header-pin",
        }
    )
    result = runner.validate_c2_runtime_contract(cell, row)
    assert result["all_gates_passed"] is True
    assert result["epw_override_warning_present_diagnostic"] is False


def test_c2_synthesizer_contrast_and_ranking_helpers() -> None:
    metric = "annual_total_delivered_site_energy_kwh"
    rows = []
    for contract, pmv_delta, learned_delta in (
        ("city_ddy", 2.0, -1.0),
        ("accepted_denver", 1.0, 1.0),
    ):
        for policy, delta in (("pmv", pmv_delta), ("learned_p90", learned_delta)):
            rows.append(
                {
                    "city": "TestCity",
                    "anchor_id": "anchor",
                    "policy": policy,
                    "strategy": f"strategy_{policy}",
                    "sizing_contract": contract,
                    f"delta_policy_minus_fixed_{metric}": delta,
                }
            )
    result = synth.differential_contrasts(pd.DataFrame(rows), [metric])
    learned = result.loc[result["policy"].eq("learned_p90")].iloc[0]
    assert learned[f"differential_of_contrasts_{metric}"] == -2.0
    assert bool(learned[f"direction_changed_{metric}"])

    rank_rows = []
    for contract in ("city_ddy", "accepted_denver"):
        for index, policy in enumerate(synth.POLICY_ORDER):
            row = {"city": "TestCity", "sizing_contract": contract, "policy": policy}
            for ranking_metric in synth.RANK_METRICS:
                row[ranking_metric] = 0.0 if index < 2 else float(index)
            rank_rows.append(row)
    ranking, stability = synth.build_rankings(pd.DataFrame(rank_rows))
    assert len(ranking.loc[ranking["rank"].eq(1)]) == 2 * len(synth.RANK_METRICS) * 2
    assert not stability["rank_changed"].any()


def test_c2_public_payload_is_trace_free_and_has_no_workstation_paths() -> None:
    public_files = [
        ROOT / "configs/c2_city_ddy_public_matrix.csv",
        ROOT / "configs/variant_manifest_c2_city_ddy.json",
        *sorted((ROOT / "diagnostics/c2_city_ddy").glob("*")),
        *sorted((ROOT / "summary_outputs").glob("c2_*.csv")),
        ROOT / "scripts/automation/compile_c2_city_ddy_plan.py",
        ROOT / "scripts/automation/run_c2_city_ddy_batch.py",
        ROOT / "scripts/synthesis/synthesize_c2_city_ddy.py",
    ]
    forbidden = (
        "/Users/",
        "\\\\Users\\",
        "paperB_ENB_Rebuild/09_rev01/05_reruns/private_inputs/",
    )
    for path in public_files:
        assert path.is_file(), path
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path
    assert not list((ROOT / "diagnostics/c2_city_ddy").rglob("*.ddy"))
    assert not list((ROOT / "diagnostics/c2_city_ddy").rglob("*.idf"))
    assert not list((ROOT / "diagnostics/c2_city_ddy").rglob("*.sql"))
    assert not list((ROOT / "diagnostics/c2_city_ddy").rglob("*.eio"))
    assert not list((ROOT / "summary_outputs").glob("c2_*.parquet"))
