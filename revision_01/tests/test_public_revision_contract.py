from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts/runner/rev01_controller_variants.py"
SPEC = importlib.util.spec_from_file_location("rev01_controller_variants_public_test", HELPER_PATH)
assert SPEC is not None and SPEC.loader is not None
helper = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = helper
SPEC.loader.exec_module(helper)

D13_PATH = ROOT / "scripts/diagnostics/build_d13_spatial_trace.py"
D13_SPEC = importlib.util.spec_from_file_location(
    "build_d13_spatial_trace_public_test", D13_PATH
)
assert D13_SPEC is not None and D13_SPEC.loader is not None
d13 = importlib.util.module_from_spec(D13_SPEC)
sys.modules[D13_SPEC.name] = d13
D13_SPEC.loader.exec_module(d13)


def test_probability_attenuation_endpoints_and_mass() -> None:
    probabilities = np.array(
        [[0.05, 0.05, 0.10, 0.50, 0.10, 0.10, 0.10]], dtype=float
    )
    assert helper.attenuate_probabilities(probabilities, 1.0) is probabilities
    neutral = helper.attenuate_probabilities(probabilities, 0.0)
    np.testing.assert_array_equal(
        neutral, [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    )
    half = helper.attenuate_probabilities(probabilities, 0.5)
    np.testing.assert_allclose(half.sum(axis=1), 1.0, rtol=0.0, atol=0.0)


def test_spatial_selector_and_adaptive_clamp_contracts() -> None:
    values = np.array([0.1, 0.3, 0.3, 0.9])
    index, target = helper.nearest_quantile_index(values, 0.5)
    assert target == float(np.quantile(values, 0.5))
    assert index == 1
    clamped = helper.adaptive_comfort_bounds(
        40.0, clamp_low_c=10.0, clamp_high_c=33.5
    )
    assert clamped["used_running_mean_c"] == 33.5
    assert clamped["was_clamped"] is True


def test_public_manifests_cover_all_revision_experiments() -> None:
    core = json.loads(
        (ROOT / "configs/variant_manifest_core_adaptive.json").read_text(
            encoding="utf-8"
        )
    )["variants"]
    learned = json.loads(
        (ROOT / "configs/variant_manifest_learned_full_grid.json").read_text(
            encoding="utf-8"
        )
    )["variants"]
    assert "rev01_p90_abs_pmv_relax" in core
    assert "rev01_adaptive_rm_clamp_10_33p5" in core
    assert "rev01_p90_signal_a050" in core
    assert "rev01_p90_signal_a000" in core
    assert "rev01_learned_mean_tail_full24" in learned
    assert "rev01_p90_signal_a000_full24" in learned


def test_compact_evidence_has_exact_scientific_coverage() -> None:
    inventory = pd.read_csv(ROOT / "RUN_INVENTORY.csv")
    assert int(inventory["planned_cells"].sum()) == 148
    assert int(inventory["passed_cells"].sum()) == 148
    assert int(inventory["failed_cells"].sum()) == 0
    d13_inventory = inventory.loc[inventory["experiment"].eq("D13")]
    assert len(d13_inventory) == 1
    assert int(d13_inventory.iloc[0]["planned_cells"]) == 0
    c2_inventory = inventory.loc[inventory["experiment"].eq("C2")]
    assert len(c2_inventory) == 1
    assert int(c2_inventory.iloc[0]["planned_cells"]) == 12
    assert int(c2_inventory.iloc[0]["passed_cells"]) == 12
    provenance = json.loads((ROOT / "RUN_PROVENANCE.json").read_text(encoding="utf-8"))
    assert provenance["scientific_cells"]["planned"] == 148
    assert provenance["scientific_cells"]["passed"] == 148
    assert provenance["scientific_cells"]["attribution_and_robustness_cells"]["passed"] == 136
    assert provenance["scientific_cells"]["c2_city_design_condition_cells"]["passed"] == 12
    cells = pd.read_csv(ROOT / "summary_outputs/matched_cell_metrics.csv")
    assert len(cells) == 136
    assert cells.groupby("experiment").size().to_dict() == {
        "C1": 24,
        "M1": 24,
        "M2": 32,
        "M3": 8,
        "M4": 24,
        "M5": 24,
    }


def test_weather_archive_matches_frozen_manifest() -> None:
    manifest = pd.read_csv(
        ROOT / "weather/weather_manifest.csv", keep_default_na=False
    )
    assert len(manifest) == 24
    assert manifest["archive_relative_path"].nunique() == 24
    assert manifest["anchor_id"].nunique() == 24
    assert manifest.groupby("scenario").size().to_dict() == {
        "ssp245": 12,
        "ssp585": 12,
    }
    assert manifest.groupby("anchor_role").size().to_dict() == {
        "future_extreme": 12,
        "present_typical": 12,
    }
    assert set(manifest["gcm"]) == {"MPI-ESM1-2-LR"}
    assert set(manifest["rcm"]) == {"N/A"}

    for row in manifest.itertuples(index=False):
        path = ROOT / row.archive_relative_path
        assert path.is_file(), path
        assert path.stat().st_size == int(row.size_bytes), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row.sha256, path
        with path.open(encoding="utf-8") as stream:
            hour_records = sum(1 for _ in stream) - 8
        assert hour_records == int(row.epw_hour_records), path

    lineage = pd.read_csv(
        ROOT / "weather/provenance/source_lineage_manifest.csv",
        keep_default_na=False,
    )
    deltas = pd.read_csv(
        ROOT / "weather/provenance/climate_delta_metadata_48.csv"
    )
    assert len(lineage) == 12
    assert len(deltas) == 48
    assert lineage.groupby("scenario").size().to_dict() == {
        "ssp245": 6,
        "ssp585": 6,
    }


def test_primary_cold_diagnostic_has_exact_accepted_panel_coverage() -> None:
    cells = pd.read_csv(
        ROOT / "summary_outputs/d03_primary_cold_cell_metrics.csv"
    )
    pairs = pd.read_csv(
        ROOT / "summary_outputs/d03_primary_cold_paired_cells.csv"
    )
    policy_summary = pd.read_csv(
        ROOT / "summary_outputs/d03_primary_cold_policy_summary.csv"
    )
    paired_summary = pd.read_csv(
        ROOT / "summary_outputs/d03_primary_cold_paired_summary.csv"
    )

    expected_strategies = {
        "diagnostic_reference",
        "paperb_pmv_relax",
        "paperb_adaptive_band_relax",
        "paperb_p90_tail_asym_relax",
    }
    expected_comparators = expected_strategies - {"paperb_p90_tail_asym_relax"}
    assert len(cells) == 96
    assert cells["anchor_id"].nunique() == 24
    assert cells.groupby("strategy").size().to_dict() == {
        strategy: 24 for strategy in expected_strategies
    }
    assert len(pairs) == 72
    assert pairs.groupby("comparator_strategy").size().to_dict() == {
        strategy: 24 for strategy in expected_comparators
    }
    assert set(policy_summary["strategy"]) == expected_strategies
    assert set(paired_summary["comparator_strategy"]) == expected_comparators
    assert (policy_summary["n_scenarios"] == 24).all()
    assert (policy_summary["median_dh_below_18c"] == 0.0).all()
    assert (policy_summary["median_dh_below_16c"] == 0.0).all()
    assert np.isclose(
        policy_summary.loc[
            policy_summary["strategy"].eq("paperb_p90_tail_asym_relax"),
            "mean_dh_below_18c",
        ].iloc[0],
        8.31317814073,
        rtol=0.0,
        atol=1e-10,
    )
    for prefix in ("dh_below_18c", "dh_below_16c"):
        direction_total = paired_summary[
            [f"lower_count_{prefix}", f"near_zero_count_{prefix}", f"higher_count_{prefix}"]
        ].sum(axis=1)
        assert (direction_total == 24).all()
    evidence_map = pd.read_csv(ROOT / "summary_outputs/reviewer_evidence_map.csv")
    cold_rows = evidence_map.loc[
        evidence_map["reviewer_comment_id"].isin(["R2-M4", "R3-7"])
    ]
    assert len(cold_rows) == 2
    assert cold_rows["evidence_or_artifact"].str.contains("D03").all()


def test_common_scale_thermal_figure_package_is_reproducible() -> None:
    source = pd.read_csv(
        ROOT / "summary_outputs/d07_common_scale_annual_dh28_source.csv"
    )
    summary = pd.read_csv(
        ROOT / "summary_outputs/d07_common_scale_annual_dh28_summary.csv"
    )
    assert len(source) == 72
    assert source["anchor_id"].nunique() == 24
    assert source.groupby("comparator").size().to_dict() == {
        "diagnostic_reference": 24,
        "paperb_adaptive_band_relax": 24,
        "paperb_pmv_relax": 24,
    }
    expected_means = {
        "diagnostic_reference": 411.668838105,
        "paperb_pmv_relax": 11.9677636744,
        "paperb_adaptive_band_relax": 24.7072911627,
    }
    for comparator, expected in expected_means.items():
        recorded = summary.loc[
            summary["comparator"].eq(comparator), "mean_delta_dh_above_28c"
        ].iloc[0]
        assert np.isclose(recorded, expected, rtol=0.0, atol=1e-9)

    script = ROOT / "scripts/figures/plot_common_scale_annual_dh28.py"
    retained_figure = ROOT / "figures/fig_s_common_scale_annual_dh28.pdf"
    assert script.is_file()
    assert retained_figure.read_bytes().startswith(b"%PDF-")
    evidence_map = pd.read_csv(ROOT / "summary_outputs/reviewer_evidence_map.csv")
    figure_evidence = evidence_map.loc[
        evidence_map["reviewer_comment_id"].eq("R3-M8"), "evidence_or_artifact"
    ].iloc[0]
    assert "D07" in figure_evidence
    with tempfile.TemporaryDirectory(prefix="paperb_rev01_public_d07_") as temp_dir:
        relocated_root = Path(temp_dir) / "revision_01"
        relocated_script = (
            relocated_root / "scripts/figures/plot_common_scale_annual_dh28.py"
        )
        relocated_source = (
            relocated_root / "summary_outputs/d07_common_scale_annual_dh28_source.csv"
        )
        relocated_summary = (
            relocated_root / "summary_outputs/d07_common_scale_annual_dh28_summary.csv"
        )
        relocated_script.parent.mkdir(parents=True)
        relocated_source.parent.mkdir(parents=True)
        shutil.copyfile(script, relocated_script)
        shutil.copyfile(
            ROOT / "summary_outputs/d07_common_scale_annual_dh28_source.csv",
            relocated_source,
        )
        shutil.copyfile(
            ROOT / "summary_outputs/d07_common_scale_annual_dh28_summary.csv",
            relocated_summary,
        )
        subprocess.run(
            [sys.executable, str(relocated_script)],
            check=True,
            capture_output=True,
            text=True,
        )
        regenerated = relocated_root / "figures/fig_s_common_scale_annual_dh28.pdf"
        assert regenerated.read_bytes().startswith(b"%PDF-")
        assert regenerated.stat().st_size > 10_000
        assert hashlib.sha256(regenerated.read_bytes()).hexdigest() == hashlib.sha256(
            retained_figure.read_bytes()
        ).hexdigest()


def test_d13_compact_spatial_trace_package_is_reproducible() -> None:
    verified = d13.verify_package(ROOT)
    assert verified == {
        "status": "PASS",
        "scenarios": 24,
        "occupied_rows": 400896,
        "published_outputs": 6,
    }

    manifest = pd.read_csv(ROOT / "summary_outputs/d13_input_manifest.csv")
    scenarios = pd.read_csv(ROOT / "summary_outputs/d13_scenario_summary.csv")
    groups = pd.read_csv(ROOT / "summary_outputs/d13_group_summary.csv")
    zones = pd.read_csv(ROOT / "summary_outputs/d13_zone_frequency.csv")
    parity = pd.read_csv(
        ROOT / "summary_outputs/d13_reconstruction_validation.csv"
    )
    assert len(manifest) == 24
    assert manifest["trace_sha256_recorded"].eq(
        manifest["trace_sha256_computed"]
    ).all()
    assert len(scenarios) == 24
    assert (scenarios["unique_selected_zones"] == 15).all()
    assert len(groups) == 5
    assert len(zones.loc[zones["scope"].eq("scenario")]) == 360
    pooled_zones = zones.loc[zones["scope"].eq("all_24_scenarios")]
    assert len(pooled_zones) == 15
    assert int(pooled_zones["selected_count"].sum()) == 400896
    assert int(pooled_zones["warm_protection_selected_count"].sum()) == 43295
    assert int(parity.iloc[0]["index_mismatch_count"]) == 0
    assert int(parity.iloc[0]["label_mismatch_count"]) == 0

    values = np.vstack([np.arange(15, dtype=float), np.arange(14, -1, -1)])
    indices, targets = d13.selector_indices(values)
    np.testing.assert_array_equal(indices, [13, 1])
    np.testing.assert_allclose(targets, [12.6, 12.6], rtol=0.0, atol=1e-12)
    episodes, switches, opportunities = d13.persistence_lengths(
        np.array([1, 1, 2, 3]), np.array([1, 2, 4, 5])
    )
    np.testing.assert_array_equal(episodes, [2, 1, 1])
    assert switches == 1
    assert opportunities == 2

    assert not list((ROOT / "diagnostics/d13_spatial_trace").rglob("*.parquet"))
    assert not list((ROOT / "summary_outputs").glob("d13_*.parquet"))

    provenance = json.loads(
        (ROOT / "diagnostics/d13_spatial_trace/D13_PUBLIC_PROVENANCE.json").read_text(
            encoding="utf-8"
        )
    )
    assert provenance["source_diagnostic"]["public_portable_script_sha256"] == hashlib.sha256(
        D13_PATH.read_bytes()
    ).hexdigest()

    with tempfile.TemporaryDirectory(prefix="paperb_rev01_public_d13_") as temp_dir:
        relocated_root = Path(temp_dir) / "revision_01"
        relocated_script = (
            relocated_root / "scripts/diagnostics/build_d13_spatial_trace.py"
        )
        relocated_script.parent.mkdir(parents=True)
        shutil.copyfile(D13_PATH, relocated_script)
        provenance_relative = Path(
            "diagnostics/d13_spatial_trace/D13_PUBLIC_PROVENANCE.json"
        )
        relocated_provenance = relocated_root / provenance_relative
        relocated_provenance.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / provenance_relative, relocated_provenance)
        for relative in provenance["published_outputs"]:
            source = ROOT / relative
            target = relocated_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        completed = subprocess.run(
            [sys.executable, str(relocated_script), "--verify-package"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert json.loads(completed.stdout)["status"] == "PASS"


def test_checksum_manifest_covers_public_addendum() -> None:
    manifest_path = ROOT / "CHECKSUMS_SHA256.txt"
    records: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(None, 1)
        records[relative] = digest
        path = ROOT.parent / relative
        assert path.is_file(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, relative

    retained = set()
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == manifest_path:
            continue
        relative_parts = path.relative_to(ROOT).parts
        if "__pycache__" in relative_parts or ".pytest_cache" in relative_parts:
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        retained.add(f"revision_01/{path.relative_to(ROOT).as_posix()}")
    assert set(records) == retained


if __name__ == "__main__":
    test_probability_attenuation_endpoints_and_mass()
    test_spatial_selector_and_adaptive_clamp_contracts()
    test_public_manifests_cover_all_revision_experiments()
    test_compact_evidence_has_exact_scientific_coverage()
    test_weather_archive_matches_frozen_manifest()
    test_primary_cold_diagnostic_has_exact_accepted_panel_coverage()
    test_common_scale_thermal_figure_package_is_reproducible()
    test_d13_compact_spatial_trace_package_is_reproducible()
    test_checksum_manifest_covers_public_addendum()
