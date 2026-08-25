from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts/runner/rev01_controller_variants.py"
SPEC = importlib.util.spec_from_file_location("rev01_controller_variants_public_test", HELPER_PATH)
assert SPEC is not None and SPEC.loader is not None
helper = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = helper
SPEC.loader.exec_module(helper)


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
    assert int(inventory["planned_cells"].sum()) == 136
    assert int(inventory["passed_cells"].sum()) == 136
    assert int(inventory["failed_cells"].sum()) == 0
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


if __name__ == "__main__":
    test_probability_attenuation_endpoints_and_mass()
    test_spatial_selector_and_adaptive_clamp_contracts()
    test_public_manifests_cover_all_revision_experiments()
    test_compact_evidence_has_exact_scientific_coverage()
    test_weather_archive_matches_frozen_manifest()
