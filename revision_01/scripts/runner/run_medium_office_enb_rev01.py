#!/usr/bin/env python
"""Run the Rev01-derived, environment-scoped Medium Office controller.

The HPH project is used only as a source of TSV training data and EPW files.
The simulated building is the DOE Medium Office prototype shipped with the
local EnergyPlus install.  This v2 identity makes two initialization corrections:
control callbacks are inert outside the weather-file RunPeriod, and formal
warmup uses the same occupied/unoccupied baseline setpoints as reported hours.

This file was derived byte-for-byte from accepted runner SHA-256
``a9693042883ed5c20edad6fcbc757c62c7216d5abc120ad18de1a003932848a4``.
Accepted defaults and strategies retain their original controller behavior.
Reviewer-only variants are additive and require an exact Rev01 manifest match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/control_probabilities_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/control_probabilities_cache")
os.environ.setdefault("FC_CACHEDIR", "/private/tmp/control_probabilities_fontconfig")
for cache_dir in (
    os.environ["MPLCONFIGDIR"],
    os.environ["XDG_CACHE_HOME"],
    os.environ["FC_CACHEDIR"],
):
    os.makedirs(cache_dir, exist_ok=True)
warnings.filterwarnings("ignore", message="The py23 module has been deprecated")
warnings.filterwarnings("ignore", message="is_sparse is deprecated.*", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message="Passing a BlockManager to DataFrame is deprecated.*",
    category=DeprecationWarning,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


if not hasattr(np, "unicode_"):
    # LightGBM 4.6 still imports np.unicode_ on NumPy 2.x.
    np.unicode_ = np.str_

import lightgbm as lgb
from pythermalcomfort.models import pmv_ppd

from rev01_controller_variants import (
    PARENT_RUNNER_SHA256,
    VariantContractError,
    adaptive_comfort_bounds as rev01_adaptive_comfort_bounds,
    attenuate_probabilities,
    file_sha256 as rev01_file_sha256,
    select_learned_tail_guard,
    select_pmv_quantile_guard,
    validate_controller_bounds,
    validate_probability,
    validate_variant_manifest,
)


CONTROL_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CONTROL_DIR.parents[1]
DEFAULT_DATA = WORKSPACE_ROOT / "TCN/newin_with_bmr.csv"
DEFAULT_OUT = CONTROL_DIR / "runs" / "guangzhou_closed_loop_smoke"
DEFAULT_IDF = Path(
    "/Applications/EnergyPlus-25-1-0/ExampleFiles/ASHRAE901_OfficeMedium_STD2019_Denver.idf"
)
DEFAULT_EPLUS = Path("/Applications/EnergyPlus-25-1-0")
DEFAULT_WEATHER = (
    WORKSPACE_ROOT
    / "HPH_Carbon_Entitlement/weather/cmip_selected_years/ssp585/Phoenix/"
    / "phoenix_ssp585_late_2080s_heatwave_extreme_2085.epw"
)

FEATURE_COLUMNS_FULL = [
    "top",
    "v",
    "rh",
    "met_c",
    "clo_c",
    "met_x_clo",
    "BSA_m2",
    "width_diff",
    "s_mono",
    "e_neg",
    "pmv",
    "s",
]
FEATURE_COLUMNS_NO_PMV = [col for col in FEATURE_COLUMNS_FULL if col != "pmv"]
FEATURE_COLUMNS = FEATURE_COLUMNS_FULL
TSV_VALUES = np.arange(-3, 4, dtype=float)
GRID_STRATEGIES = {"grid_naive", "grid_gated"}
PAPERB_STRATEGIES = {
    "paperb_pmv_relax",
    "paperb_adaptive_band_relax",
    "paperb_ppd_guard_relax",
    "paperb_pmv_exceedance_guard_relax",
    "paperb_pmv_extreme_guard_relax",
    "paperb_mu_relax",
    "paperb_gate_tail_asym_relax",
    "paperb_p90_tail_asym_relax",
    "paperb_p90_abs_pmv_relax",
    "paperb_adaptive_band_clamped_relax",
}
GRID_FULL_SHED_DELTA_C = 1.5
GRID_MILD_SHED_DELTA_C = 0.75
GRID_WARM_RISK_SOFT = 0.20
GRID_WARM_RISK_BLOCK = 0.35
MAX_WARMUP_DAYS = 50
WEATHER_FILE_RUN_PERIOD_KIND = 3
WARMUP_SETPOINT_CONTRACT = "production_occupancy_baseline_22_24_else_12_30"
PAPERB_REF_HEAT_C = 22.0
PAPERB_REF_COOL_C = 24.0
PAPERB_SAVE_HEAT_C = 20.0
PAPERB_SAVE_COOL_C = 26.0
PAPERB_WARM_PROTECT_COOL_C = 23.25
PAPERB_COLD_PROTECT_HEAT_C = 23.25
PAPERB_TIGHTEN_STEP_C = 0.5
PAPERB_RELAX_STEP_C = 0.5
PAPERB_TIGHTEN_DWELL_STEPS = 4  # 60 minutes at 15-min timesteps
PAPERB_RELAX_DWELL_STEPS = 1  # allow faster movement toward energy-saving setpoints
PAPERB_TAIL_THRESHOLD = 0.20
PAPERB_ASYM_THRESHOLD = 0.10
PAPERB_PMV_THRESHOLD = 0.50
PAPERB_PMV_EXTREME_THRESHOLD = 1.00
PAPERB_PPD_HOLD_THRESHOLD = 10.0
PAPERB_PPD_PROTECT_THRESHOLD = 25.0
PAPERB_ADAPTIVE_90_HALF_WIDTH_C = 2.5
PAPERB_ADAPTIVE_80_HALF_WIDTH_C = 3.5
PAPERB_TSV_THRESHOLD = 0.50
PAPERB_MET = 1.10
PAPERB_PEOPLE_ACTIVITY_W_PER_PERSON = 120.0
CONTROLLER_SEMANTICS = "legacy"
INFERENCE_PREPROCESSING = "legacy"
PAPERB_SPATIAL_QUANTILE = 0.90
PAPERB_SIGNAL_ALPHA = 1.0
PAPERB_ADAPTIVE_RM_CLAMP_LOW_C = 10.0
PAPERB_ADAPTIVE_RM_CLAMP_HIGH_C = 33.5
REV01_VARIANT_ID = "accepted_default_parity"
REV01_VARIANT_MANIFEST_SHA256 = ""
REV01_EFFECTIVE_CONFIG_SHA256 = ""
SIZING_CONTRACT = "accepted_denver"
ZONE_NAMES = [
    "Core_bottom",
    "Core_mid",
    "Core_top",
    "Perimeter_top_ZN_3",
    "Perimeter_top_ZN_2",
    "Perimeter_top_ZN_1",
    "Perimeter_top_ZN_4",
    "Perimeter_bot_ZN_3",
    "Perimeter_bot_ZN_2",
    "Perimeter_bot_ZN_1",
    "Perimeter_bot_ZN_4",
    "Perimeter_mid_ZN_3",
    "Perimeter_mid_ZN_2",
    "Perimeter_mid_ZN_1",
    "Perimeter_mid_ZN_4",
]
ZONE_FIELD_NAMES = [
    re.sub(r"[^a-z0-9]+", "_", zone.lower()).strip("_") for zone in ZONE_NAMES
]


def paperb_met_tag() -> str:
    return f"met{PAPERB_MET:.3f}".replace(".", "p")


def people_activity_tag() -> str:
    return f"act{PAPERB_PEOPLE_ACTIVITY_W_PER_PERSON:.1f}W".replace(".", "p")


@dataclass
class FeatureSpec:
    medians: dict[str, float]
    winsor: dict[str, tuple[float, float]]
    met_mean: float
    clo_mean: float


@dataclass
class PredictorBundle:
    spec: FeatureSpec
    scaler: StandardScaler
    nominal: CalibratedClassifierCV
    ordinal: list[CalibratedClassifierCV]
    feature_columns: list[str]

    def predict_nominal(self, features: pd.DataFrame) -> np.ndarray:
        x = self.scaler.transform(features[self.feature_columns].to_numpy(float))
        probs = self.nominal.predict_proba(x)
        return normalize_probs(align_nominal_classes(probs, self.nominal.classes_))

    def predict_ordinal(self, features: pd.DataFrame) -> np.ndarray:
        x = self.scaler.transform(features[self.feature_columns].to_numpy(float))
        cumulative = []
        for model in self.ordinal:
            p_gt = model.predict_proba(x)[:, 1]
            cumulative.append(p_gt)
        p_gt = np.column_stack(cumulative)
        p_gt = np.minimum.accumulate(np.clip(p_gt, 0.0, 1.0), axis=1)
        probs = np.empty((x.shape[0], 7), dtype=float)
        probs[:, 0] = 1.0 - p_gt[:, 0]
        probs[:, 1:6] = p_gt[:, :-1] - p_gt[:, 1:]
        probs[:, 6] = p_gt[:, 5]
        return normalize_probs(probs)


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    totals = probs.sum(axis=1, keepdims=True)
    totals[totals <= 0] = 1.0
    return probs / totals


def align_nominal_classes(probs: np.ndarray, classes: np.ndarray) -> np.ndarray:
    aligned = np.zeros((probs.shape[0], 7), dtype=float)
    for col, cls in enumerate(classes):
        aligned[:, int(cls)] = probs[:, col]
    return aligned


def round_tsv(series: pd.Series) -> np.ndarray:
    return np.clip(np.rint(series.to_numpy(float)), -3, 3).astype(int) + 3


def read_training_data(path: Path, sample_limit: int | None = None) -> pd.DataFrame:
    cols = [
        "thermal_sensation",
        "ta",
        "mean_radiant_temperature",
        "vel",
        "rh",
        "metabolic_rate",
        "clothing_insulation",
        "height_cm",
        "weight_kg",
        "bsa_m2",
        "outdoor_air_temp",
        "prevailing_outdoor_mean",
    ]
    df = pd.read_csv(path, usecols=cols)
    if sample_limit and sample_limit < len(df):
        y = round_tsv(df["thermal_sensation"])
        df, _ = train_test_split(
            df,
            train_size=sample_limit,
            random_state=42,
            stratify=y,
        )
    return df.reset_index(drop=True)


def fit_feature_spec(train_df: pd.DataFrame) -> FeatureSpec:
    med_cols = [
        "ta",
        "mean_radiant_temperature",
        "vel",
        "rh",
        "metabolic_rate",
        "clothing_insulation",
        "height_cm",
        "weight_kg",
        "bsa_m2",
        "outdoor_air_temp",
        "prevailing_outdoor_mean",
    ]
    medians = {}
    for col in med_cols:
        med = pd.to_numeric(train_df[col], errors="coerce").median()
        if not np.isfinite(med):
            med = 0.0
        medians[col] = float(med)
    medians["vel"] = max(medians.get("vel", 0.01), 0.01)
    medians["rh"] = float(np.clip(medians.get("rh", 50.0), 1.0, 100.0))
    if not (1.0 <= medians.get("bsa_m2", 0.0) <= 3.0):
        medians["bsa_m2"] = 1.8

    raw = coerce_raw_inputs(train_df, medians)
    winsor = {}
    for col in [
        "ta",
        "tr",
        "top",
        "v",
        "rh",
        "met",
        "clo",
        "BSA_m2",
        "rm_out",
    ]:
        lo, hi = np.nanquantile(raw[col], [0.01, 0.99])
        winsor[col] = (float(lo), float(hi))

    met_mean = float(np.nanmean(np.clip(raw["met"], *winsor["met"])))
    clo_mean = float(np.nanmean(np.clip(raw["clo"], *winsor["clo"])))
    return FeatureSpec(medians=medians, winsor=winsor, met_mean=met_mean, clo_mean=clo_mean)


def coerce_raw_inputs(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    ta = numeric_or_median(df, "ta", medians)
    tr = numeric_or_median(df, "mean_radiant_temperature", medians)
    tr = tr.where(np.isfinite(tr), ta)
    tr = tr.fillna(ta)
    out["ta"] = ta
    out["tr"] = tr
    out["top"] = (ta + tr) / 2.0
    out["v"] = numeric_or_median(df, "vel", medians).clip(lower=0.01, upper=2.0)
    out["rh"] = numeric_or_median(df, "rh", medians).clip(lower=1.0, upper=100.0)
    out["met"] = numeric_or_median(df, "metabolic_rate", medians).clip(lower=0.5, upper=4.0)
    out["clo"] = numeric_or_median(df, "clothing_insulation", medians).clip(lower=0.0, upper=3.0)

    bsa = numeric_or_median(df, "bsa_m2", medians)
    height = numeric_or_median(df, "height_cm", medians)
    weight = numeric_or_median(df, "weight_kg", medians)
    dubois = 0.007184 * np.power(height.clip(lower=100, upper=230), 0.725) * np.power(
        weight.clip(lower=30, upper=200), 0.425
    )
    out["BSA_m2"] = bsa.where(bsa.between(1.0, 3.0), dubois).fillna(medians.get("bsa_m2", 1.8))

    rm = numeric_or_median(df, "prevailing_outdoor_mean", medians)
    oat = numeric_or_median(df, "outdoor_air_temp", medians)
    out["rm_out"] = rm.where(np.isfinite(rm), oat).fillna(oat)
    return out


def numeric_or_median(df: pd.DataFrame, col: str, medians: dict[str, float]) -> pd.Series:
    if col in df:
        s = pd.to_numeric(df[col], errors="coerce")
    else:
        s = pd.Series(np.nan, index=df.index)
    return s.fillna(medians.get(col, 0.0))


def build_features_from_raw(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    raw = coerce_raw_inputs(df, spec.medians)
    for col, (lo, hi) in spec.winsor.items():
        if col in raw:
            raw[col] = raw[col].clip(lo, hi)
    return build_features_from_arrays(
        ta=raw["ta"].to_numpy(float),
        tr=raw["tr"].to_numpy(float),
        v=raw["v"].to_numpy(float),
        rh=raw["rh"].to_numpy(float),
        met=raw["met"].to_numpy(float),
        clo=raw["clo"].to_numpy(float),
        bsa=raw["BSA_m2"].to_numpy(float),
        rm_out=raw["rm_out"].to_numpy(float),
        spec=spec,
    )


def build_features_from_arrays(
    *,
    ta: np.ndarray,
    tr: np.ndarray,
    v: np.ndarray,
    rh: np.ndarray,
    met: np.ndarray,
    clo: np.ndarray,
    bsa: np.ndarray,
    rm_out: np.ndarray,
    spec: FeatureSpec,
) -> pd.DataFrame:
    ta = np.asarray(ta, dtype=float)
    tr = np.asarray(tr, dtype=float)
    v = np.clip(np.asarray(v, dtype=float), 0.01, 2.0)
    rh = np.clip(np.asarray(rh, dtype=float), 1.0, 100.0)
    met = np.clip(np.asarray(met, dtype=float), 0.5, 4.0)
    clo = np.clip(np.asarray(clo, dtype=float), 0.0, 3.0)
    bsa = np.clip(np.asarray(bsa, dtype=float), 1.0, 3.0)
    rm_out = np.asarray(rm_out, dtype=float)
    top = (ta + tr) / 2.0

    pmv = compute_pmv_array(ta, tr, v, rh, met, clo)
    t_comf = 0.31 * rm_out + 17.8
    warm_lift = np.zeros_like(top)
    warm_mask = (v >= 0.3) & (top > 25.0)
    warm_lift[warm_mask] = 1.2
    warm_lift[(v >= 0.9) & (top > 25.0)] = 1.8
    warm_lift[(v >= 1.2) & (top > 25.0)] = 2.2
    width_cold = np.full_like(top, 3.5)
    width_hot = 3.5 + warm_lift
    width = np.where(top >= t_comf, width_hot, width_cold)
    s = (top - t_comf) / np.maximum(width, 0.1)
    met_c = met - spec.met_mean
    clo_c = clo - spec.clo_mean

    features = pd.DataFrame(
        {
            "top": top,
            "v": v,
            "rh": rh,
            "met_c": met_c,
            "clo_c": clo_c,
            "met_x_clo": met_c * clo_c,
            "BSA_m2": bsa,
            "width_diff": width_hot - width_cold,
            "s_mono": np.clip(s, -1.0, 1.0),
            "e_neg": np.maximum(-s - 1.0, 0.0),
            "pmv": pmv,
            "s": s,
        }
    )
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def preprocess_runtime_inputs(
    *,
    ta: np.ndarray,
    tr: np.ndarray,
    v: np.ndarray,
    rh: np.ndarray,
    met: np.ndarray,
    clo: np.ndarray,
    bsa: np.ndarray,
    rm_out: np.ndarray,
    spec: FeatureSpec,
    mode: str | None = None,
) -> dict[str, np.ndarray]:
    """Apply either the legacy or training-time raw-input contract at runtime."""
    selected_mode = mode or INFERENCE_PREPROCESSING
    if selected_mode not in {"legacy", "training_contract"}:
        raise ValueError(f"Unknown inference preprocessing mode: {selected_mode}")
    arrays = {
        "ta": np.asarray(ta, dtype=float),
        "tr": np.asarray(tr, dtype=float),
        "v": np.asarray(v, dtype=float),
        "rh": np.asarray(rh, dtype=float),
        "met": np.asarray(met, dtype=float),
        "clo": np.asarray(clo, dtype=float),
        "BSA_m2": np.asarray(bsa, dtype=float),
        "rm_out": np.asarray(rm_out, dtype=float),
    }
    if selected_mode == "training_contract":
        for name, values in arrays.items():
            if name in spec.winsor:
                lower, upper = spec.winsor[name]
                arrays[name] = np.clip(values, lower, upper)
    return arrays


def compute_pmv_array(
    ta: np.ndarray,
    tr: np.ndarray,
    v: np.ndarray,
    rh: np.ndarray,
    met: np.ndarray,
    clo: np.ndarray,
) -> np.ndarray:
    try:
        result = pmv_ppd(
            tdb=ta,
            tr=tr,
            vr=v,
            rh=rh,
            met=met,
            clo=clo,
            standard="ISO",
            units="SI",
            limit_inputs=False,
        )
        return np.asarray(result["pmv"], dtype=float)
    except Exception:
        return np.zeros_like(np.asarray(ta, dtype=float))


def train_predictors(
    data_path: Path,
    model_path: Path,
    metrics_path: Path,
    n_estimators: int,
    sample_limit: int | None,
    feature_columns: list[str] | None = None,
) -> PredictorBundle:
    feature_columns = feature_columns or FEATURE_COLUMNS_FULL
    print(f"[train] reading TSV source: {data_path}")
    df = read_training_data(data_path, sample_limit=sample_limit)
    y = round_tsv(df["thermal_sensation"])
    train_df, hold_df, y_train, y_hold = train_test_split(
        df,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )
    cal_df, test_df, y_cal, y_test = train_test_split(
        hold_df,
        y_hold,
        test_size=0.50,
        random_state=42,
        stratify=y_hold,
    )

    spec = fit_feature_spec(train_df)
    x_train_df = build_features_from_raw(train_df, spec)
    x_cal_df = build_features_from_raw(cal_df, spec)
    x_test_df = build_features_from_raw(test_df, spec)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_df[feature_columns].to_numpy(float))
    x_cal = scaler.transform(x_cal_df[feature_columns].to_numpy(float))
    x_test = scaler.transform(x_test_df[feature_columns].to_numpy(float))

    common = dict(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=0.0,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    print(f"[train] fitting nominal LightGBM with {n_estimators} trees")
    nominal_base = lgb.LGBMClassifier(objective="multiclass", num_class=7, **common)
    nominal_base.fit(x_train, y_train)
    nominal = CalibratedClassifierCV(
        estimator=nominal_base,
        method="isotonic",
        cv="prefit",
        ensemble=False,
    )
    nominal.fit(x_cal, y_cal)

    print("[train] fitting six cumulative ordinal LightGBM heads")
    ordinal_models: list[CalibratedClassifierCV] = []
    for threshold in range(6):
        y_train_bin = (y_train > threshold).astype(int)
        y_cal_bin = (y_cal > threshold).astype(int)
        base = lgb.LGBMClassifier(objective="binary", **common)
        base.fit(x_train, y_train_bin)
        cal = CalibratedClassifierCV(
            estimator=base,
            method="isotonic",
            cv="prefit",
            ensemble=False,
        )
        cal.fit(x_cal, y_cal_bin)
        ordinal_models.append(cal)

    bundle = PredictorBundle(
        spec=spec,
        scaler=scaler,
        nominal=nominal,
        ordinal=ordinal_models,
        feature_columns=feature_columns,
    )
    metrics = evaluate_bundle(bundle, x_test_df, y_test)
    metrics.update(
        {
            "n_total": int(len(df)),
            "n_train": int(len(train_df)),
            "n_calibration": int(len(cal_df)),
            "n_test": int(len(test_df)),
            "n_estimators": int(n_estimators),
            "sample_limit": sample_limit,
            "feature_columns": list(feature_columns),
            "feature_set": "no_pmv" if "pmv" not in feature_columns else "full",
        }
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[train] saved model: {model_path}")
    print(f"[train] saved metrics: {metrics_path}")
    return bundle


def evaluate_bundle(bundle: PredictorBundle, x_test_df: pd.DataFrame, y_test: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, probs in [
        ("nominal", bundle.predict_nominal(x_test_df)),
        ("ordinal", bundle.predict_ordinal(x_test_df)),
    ]:
        pred = probs.argmax(axis=1)
        tail_true = np.isin(y_test, [0, 1, 5, 6]).astype(float)
        tail_prob = probs[:, [0, 1, 5, 6]].sum(axis=1)
        mu = probs @ TSV_VALUES
        y_signed = y_test - 3
        out[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "log_loss": float(log_loss(y_test, probs, labels=np.arange(7))),
            "tail_probability_mae": float(np.mean(np.abs(tail_prob - tail_true))),
            "expected_tsv_mae": float(np.mean(np.abs(mu - y_signed))),
        }
    return out


def patch_idf_for_control(
    source_idf: Path,
    target_idf: Path,
    begin_month: int,
    begin_day: int,
    end_month: int,
    end_day: int,
    people_activity_w_per_person: float = 120.0,
) -> None:
    text = source_idf.read_text(errors="ignore")
    patched_objects: list[str] = []
    inserted_runperiod = False
    for obj in text.split(";"):
        if not obj.strip():
            continue
        fields = parse_idf_fields(obj)
        class_name = fields[0].lower() if fields else ""
        if class_name == "building" and len(fields) >= 9:
            patched_objects.append(
                f"""
  Building,
    {fields[1]},             !- Name
    {fields[2]},             !- North Axis {{deg}}
    {fields[3]},             !- Terrain
    {fields[4]},             !- Loads Convergence Tolerance Value {{W}}
    {fields[5]},             !- Temperature Convergence Tolerance Value {{deltaC}}
    {fields[6]},             !- Solar Distribution
    {MAX_WARMUP_DAYS},       !- Maximum Number of Warmup Days
    {fields[8]}"""
            )
            continue
        if class_name == "runperiod":
            if not inserted_runperiod:
                patched_objects.append(
                    f"""
  RunPeriod,
    OTC_CONTROL_TRACE,       !- Name
    {begin_month},           !- Begin Month
    {begin_day},             !- Begin Day of Month
    ,                        !- Begin Year
    {end_month},             !- End Month
    {end_day},               !- End Day of Month
    ,                        !- End Year
    Monday,                  !- Day of Week for Start Day
    No,                      !- Use Weather File Holidays and Special Days
    No,                      !- Use Weather File Daylight Saving Period
    No,                      !- Apply Weekend Holiday Rule
    Yes,                     !- Use Weather File Rain Indicators
    Yes"""
                )
                inserted_runperiod = True
            continue
        if class_name == "thermostatsetpoint:dualsetpoint":
            if len(fields) >= 2:
                name = fields[1]
                patched_objects.append(
                    f"""
  ThermostatSetpoint:DualSetpoint,
    {name},                  !- Name
    OTC_HEATING_SETPOINT,    !- Heating Setpoint Temperature Schedule Name
    OTC_COOLING_SETPOINT"""
                )
                continue
        if class_name == "schedule:compact" and len(fields) >= 2 and fields[1] == "ACTIVITY_SCH":
            patched_objects.append(
                f"""
  Schedule:Compact,
    ACTIVITY_SCH,            !- Name
    Any Number,              !- Schedule Type Limits Name
    Through: 12/31,          !- Field 1
    For: AllDays,            !- Field 2
    Until: 24:00,{people_activity_w_per_person:.6g}"""
            )
            continue
        patched_objects.append(obj)

    patched_objects.append(
        """

!-   ===========  OTC CONTROL API SCHEDULES ===========

  Schedule:Constant,
    OTC_HEATING_SETPOINT,    !- Name
    Temperature,             !- Schedule Type Limits Name
    22.0;                    !- Hourly Value

  Schedule:Constant,
    OTC_COOLING_SETPOINT,    !- Name
    Temperature,             !- Schedule Type Limits Name
    24.0;                    !- Hourly Value

  Output:SQLite,
    SimpleAndTabular;

  Output:Meter,
    Electricity:Facility,
    Hourly;

  Output:Meter,
    NaturalGas:Facility,
    Hourly;
"""
    )
    target_idf.parent.mkdir(parents=True, exist_ok=True)
    target_idf.write_text(";\n".join(patched_objects) + "\n")


def parse_idf_fields(obj: str) -> list[str]:
    no_comments = []
    for line in obj.splitlines():
        no_comments.append(line.split("!", 1)[0])
    raw = "\n".join(no_comments).replace("\n", " ")
    return [f.strip() for f in raw.split(",") if f.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_generated_idf_contract(path: Path) -> dict[str, Any]:
    """Fail closed on every numerical/model field implicated by the rerun."""

    objects_by_type: dict[str, list[list[str]]] = {}
    for raw in path.read_text(errors="strict").split(";"):
        fields = parse_idf_fields(raw)
        if not fields:
            continue
        objects_by_type.setdefault(fields[0].casefold(), []).append(fields)

    def exactly_one(object_type: str) -> list[str]:
        found = objects_by_type.get(object_type.casefold(), [])
        if len(found) != 1:
            raise RuntimeError(f"Expected one {object_type} in {path}, found {len(found)}")
        return found[0]

    timestep = exactly_one("Timestep")
    convergence = exactly_one("ConvergenceLimits")
    building = exactly_one("Building")
    nightcycle = objects_by_type.get("availabilitymanager:nightcycle", [])

    if timestep != ["Timestep", "4"]:
        raise RuntimeError(f"Unexpected Timestep contract: {timestep}")
    if convergence != ["ConvergenceLimits", "15", "5"]:
        raise RuntimeError(f"Unexpected ConvergenceLimits contract: {convergence}")
    if len(building) < 9 or (
        float(building[4]),
        float(building[5]),
        building[6].casefold(),
        int(building[7]),
        int(building[8]),
    ) != (0.04, 0.20, "minimalshadowing", 50, 6):
        raise RuntimeError(f"Unexpected Building contract: {building[:9]}")
    if len(nightcycle) != 3:
        raise RuntimeError(f"Expected three NightCycle objects, found {len(nightcycle)}")
    expected_names = {
        "PACU_VAV_bot Availability Manager",
        "PACU_VAV_mid Availability Manager",
        "PACU_VAV_top Availability Manager",
    }
    observed_names: set[str] = set()
    for fields in nightcycle:
        if len(fields) != 8:
            raise RuntimeError(f"Malformed NightCycle object: {fields}")
        observed_names.add(fields[1])
        observed = (
            fields[2].casefold(),
            fields[3].casefold(),
            fields[4].casefold(),
            float(fields[5]),
            fields[6].casefold(),
            int(fields[7]),
        )
        expected = (
            "always_on",
            "hvacoperationschd",
            "cycleonany",
            1.0,
            "fixedruntime",
            1800,
        )
        if observed != expected:
            raise RuntimeError(f"Unexpected NightCycle contract: {fields}")
    if observed_names != expected_names:
        raise RuntimeError(f"Unexpected NightCycle names: {sorted(observed_names)}")

    return {
        "idf_sha256": file_sha256(path),
        "timestep_per_hour": 4,
        "minimum_system_timestep_minutes": 15,
        "maximum_hvac_iterations": 5,
        "loads_convergence_tolerance": 0.04,
        "temperature_convergence_tolerance_c": 0.20,
        "solar_distribution": "MinimalShadowing",
        "maximum_warmup_days": 50,
        "minimum_warmup_days": 6,
        "nightcycle_object_count": 3,
        "nightcycle_names": sorted(observed_names),
        "nightcycle_contract": "Always_On/HVACOperationSchd/CycleOnAny/1.0/FixedRunTime/1800",
    }


@dataclass
class ControlState:
    strategy: str
    bundle: PredictorBundle | None
    heat_sp: float = 22.0
    cool_sp: float = 24.0
    rm_out: float | None = None
    last_control_key: tuple[int, int, int, int, float] | None = None
    last_record_key: tuple[int, int, int, int, float] | None = None
    initialized: bool = False
    handles: dict[str, Any] | None = None
    records: list[dict[str, Any]] | None = None
    grid_signal: dict[tuple[int, int, int], dict[str, float | int]] | None = None
    current_grid_signal: dict[str, float | int] | None = None
    last_paperb_action_step: int = -10**9
    pending_signal: dict[str, float | int] | None = None
    active_environment_num: int | None = None
    environment_events: list[dict[str, Any]] | None = None
    weather_environment_nums: list[int] | None = None
    warmup_setpoint_records: list[dict[str, Any]] | None = None
    audit_counts: dict[str, int] | None = None
    formal_weather_step: int = 0
    last_raw_probabilities: np.ndarray | None = None
    last_effective_probabilities: np.ndarray | None = None


def run_energyplus_strategy(
    *,
    strategy: str,
    bundle: PredictorBundle | None,
    idf_path: Path,
    weather_path: Path,
    eplus_root: Path,
    out_dir: Path,
    begin_month: int,
    begin_day: int,
    end_month: int,
    end_day: int,
    resume: bool = False,
    purge_energyplus_after_trace: bool = False,
    trace_format: str = "csv",
) -> Path:
    if resume:
        raise RuntimeError(
            "The v2 environment-scoped runner refuses resume; use a fresh versioned root."
        )
    if purge_energyplus_after_trace:
        raise RuntimeError(
            "The v2 environment-scoped runner preserves EnergyPlus and audit outputs."
        )
    if str(eplus_root) not in sys.path:
        sys.path.insert(0, str(eplus_root))
    from pyenergyplus.api import EnergyPlusAPI

    api = EnergyPlusAPI()
    state = api.state_manager.new_state()
    met_tag = paperb_met_tag()
    heat_tag = people_activity_tag()
    run_dir = out_dir / "energyplus" / weather_path.stem / met_tag / heat_tag / strategy
    trace_suffix = ".parquet" if trace_format == "parquet" else ".csv"
    trace_path = out_dir / "traces" / f"{weather_path.stem}_{met_tag}_{heat_tag}_{strategy}{trace_suffix}"
    if run_dir.exists() or trace_path.exists():
        raise FileExistsError(
            f"The v2 runner requires fresh per-strategy outputs: {run_dir} / {trace_path}"
        )
    idf_contract = validate_generated_idf_contract(idf_path)
    run_dir.mkdir(parents=True, exist_ok=False)
    for zone in ZONE_NAMES:
        for variable in [
            "Zone Mean Air Temperature",
            "Zone Mean Radiant Temperature",
            "Zone Air Relative Humidity",
            "Zone Air System Sensible Heating Rate",
            "Zone Air System Sensible Cooling Rate",
        ]:
            api.exchange.request_variable(state, variable, zone)
    api.exchange.request_variable(state, "Site Outdoor Air Drybulb Temperature", "Environment")

    grid_signal = (
        build_microgrid_signal_schedule(weather_path, begin_month, begin_day, end_month, end_day)
        if strategy in GRID_STRATEGIES
        else None
    )
    ctl = ControlState(
        strategy=strategy,
        bundle=bundle,
        records=[],
        grid_signal=grid_signal,
        environment_events=[],
        weather_environment_nums=[],
        warmup_setpoint_records=[],
        audit_counts={
            "begin_environment_callbacks": 0,
            "non_weather_apply_callbacks_skipped": 0,
            "weather_warmup_control_callbacks": 0,
            "weather_regular_control_callbacks": 0,
            "weather_record_callbacks": 0,
            "warmup_setpoint_mismatches": 0,
            "warmup_controller_state_mutations": 0,
            "after_weather_warmup_callbacks": 0,
            "actuator_resets_after_weather_warmup": 0,
        },
    )

    def reset_dynamic_control_state() -> None:
        """Start each weather environment from one explicit controller state."""

        ctl.heat_sp = PAPERB_REF_HEAT_C
        ctl.cool_sp = PAPERB_REF_COOL_C
        ctl.rm_out = None
        ctl.last_control_key = None
        ctl.last_record_key = None
        ctl.current_grid_signal = None
        ctl.last_paperb_action_step = -10**9
        ctl.pending_signal = None
        ctl.formal_weather_step = 0
        ctl.last_raw_probabilities = None
        ctl.last_effective_probabilities = None

    def enter_environment(st: Any, *, source: str) -> None:
        environment_num = int(api.exchange.current_environment_num(st))
        kind = int(api.exchange.kind_of_sim(st))
        if ctl.active_environment_num == environment_num:
            return
        ctl.active_environment_num = environment_num
        ctl.audit_counts["begin_environment_callbacks"] += 1
        ctl.environment_events.append(
            {
                "source": source,
                "environment_num": environment_num,
                "kind_of_sim": kind,
                "weather_file_run_period": kind == WEATHER_FILE_RUN_PERIOD_KIND,
            }
        )
        if kind == WEATHER_FILE_RUN_PERIOD_KIND:
            ctl.weather_environment_nums.append(environment_num)
            reset_dynamic_control_state()

    def begin_new_environment(st: Any) -> None:
        enter_environment(st, source="callback_begin_new_environment")

    def after_new_environment_warmup_complete(st: Any) -> None:
        if not is_weather_file_run_period(api, st):
            return
        enter_environment(st, source="after_warmup_fallback")
        initialize_handles(st)
        if not ctl.initialized:
            raise RuntimeError("Actuator handles unavailable after weather warmup.")
        api.exchange.reset_actuator(st, ctl.handles["heat_act"])
        api.exchange.reset_actuator(st, ctl.handles["cool_act"])
        ctl.audit_counts["actuator_resets_after_weather_warmup"] += 2
        reset_dynamic_control_state()
        ctl.audit_counts["after_weather_warmup_callbacks"] += 1

    def initialize_handles(st: Any) -> None:
        if ctl.initialized or not api.exchange.api_data_fully_ready(st):
            return
        handles: dict[str, Any] = {
            "heat_act": api.exchange.get_actuator_handle(
                st, "Schedule:Constant", "Schedule Value", "OTC_HEATING_SETPOINT"
            ),
            "cool_act": api.exchange.get_actuator_handle(
                st, "Schedule:Constant", "Schedule Value", "OTC_COOLING_SETPOINT"
            ),
            "oat": api.exchange.get_variable_handle(
                st, "Site Outdoor Air Drybulb Temperature", "Environment"
            ),
            "electricity": api.exchange.get_meter_handle(st, "Electricity:Facility"),
            "electricity_hvac": api.exchange.get_meter_handle(st, "Electricity:HVAC"),
            "electricity_cooling": api.exchange.get_meter_handle(st, "Cooling:Electricity"),
            "electricity_heating": api.exchange.get_meter_handle(st, "Heating:Electricity"),
            "electricity_fans": api.exchange.get_meter_handle(st, "Fans:Electricity"),
            "electricity_pumps": api.exchange.get_meter_handle(st, "Pumps:Electricity"),
            "gas": api.exchange.get_meter_handle(st, "NaturalGas:Facility"),
            "zones": {},
        }
        for zone in ZONE_NAMES:
            handles["zones"][zone] = {
                "ta": api.exchange.get_variable_handle(st, "Zone Mean Air Temperature", zone),
                "tr": api.exchange.get_variable_handle(st, "Zone Mean Radiant Temperature", zone),
                "rh": api.exchange.get_variable_handle(st, "Zone Air Relative Humidity", zone),
                "heat_rate": api.exchange.get_variable_handle(
                    st, "Zone Air System Sensible Heating Rate", zone
                ),
                "cool_rate": api.exchange.get_variable_handle(
                    st, "Zone Air System Sensible Cooling Rate", zone
                ),
            }
        if handles["heat_act"] < 0 or handles["cool_act"] < 0:
            raise RuntimeError("Could not acquire OTC thermostat schedule actuators.")
        ctl.handles = handles
        ctl.initialized = True

    def apply_control(st: Any) -> None:
        kind = int(api.exchange.kind_of_sim(st))
        if not is_weather_file_run_period(api, st):
            ctl.audit_counts["non_weather_apply_callbacks_skipped"] += 1
            return
        enter_environment(st, source="apply_control_fallback")
        initialize_handles(st)
        if not ctl.initialized:
            return
        key = current_key(api, st)
        if ctl.last_control_key == key:
            return
        ctl.last_control_key = key

        if api.exchange.warmup_flag(st):
            occupied = is_occupied(api, st)
            heat, cool = baseline_setpoints(occupied)
            state_before = {
                "rm_out": ctl.rm_out,
                "last_paperb_action_step": ctl.last_paperb_action_step,
                "current_grid_signal": ctl.current_grid_signal,
                "pending_signal": ctl.pending_signal,
            }
            set_api_setpoints(api, st, ctl, heat, cool)
            expected_heat, expected_cool = baseline_setpoints(occupied)
            applied_heat = float(
                api.exchange.get_actuator_value(st, ctl.handles["heat_act"])
            )
            applied_cool = float(
                api.exchange.get_actuator_value(st, ctl.handles["cool_act"])
            )
            mismatch = (
                heat != expected_heat
                or cool != expected_cool
                or applied_heat != expected_heat
                or applied_cool != expected_cool
            )
            state_after = {
                "rm_out": ctl.rm_out,
                "last_paperb_action_step": ctl.last_paperb_action_step,
                "current_grid_signal": ctl.current_grid_signal,
                "pending_signal": ctl.pending_signal,
            }
            state_mutated = state_before != state_after
            ctl.audit_counts["weather_warmup_control_callbacks"] += 1
            ctl.audit_counts["warmup_setpoint_mismatches"] += int(mismatch)
            ctl.audit_counts["warmup_controller_state_mutations"] += int(state_mutated)
            ctl.warmup_setpoint_records.append(
                {
                    "environment_num": int(api.exchange.current_environment_num(st)),
                    "kind_of_sim": kind,
                    "month": int(api.exchange.month(st)),
                    "day": int(api.exchange.day_of_month(st)),
                    "day_of_week": int(api.exchange.day_of_week(st)),
                    "hour": int(api.exchange.hour(st)),
                    "current_time": float(api.exchange.current_time(st)),
                    "occupied": bool(occupied),
                    "requested_heating_setpoint_c": float(heat),
                    "requested_cooling_setpoint_c": float(cool),
                    "applied_heating_actuator_value_c": applied_heat,
                    "applied_cooling_actuator_value_c": applied_cool,
                    "setpoint_contract": WARMUP_SETPOINT_CONTRACT,
                    "setpoint_mismatch": bool(mismatch),
                    "controller_state_mutated": bool(state_mutated),
                }
            )
            return

        ctl.audit_counts["weather_regular_control_callbacks"] += 1
        ctl.formal_weather_step += 1
        values = read_zone_values(api, st, ctl)
        values["sim_step"] = ctl.formal_weather_step
        oat = read_handle(api, st, ctl.handles["oat"], default=np.nan)
        if not np.isfinite(oat):
            oat = values["ta_mean"]
        ctl.rm_out = update_running_mean(ctl.rm_out, oat)
        ctl.current_grid_signal = lookup_microgrid_signal(api, st, ctl)
        occupied = is_occupied(api, st)

        if not occupied:
            heat, cool = baseline_setpoints(False)
            signal = default_control_signal(values["pmv_mean"], ctl.current_grid_signal)
        elif strategy == "reference":
            heat, cool = baseline_setpoints(True)
            signal = default_control_signal(values["pmv_mean"], ctl.current_grid_signal)
        elif strategy == "diagnostic_reference":
            heat, cool = baseline_setpoints(True)
            signal = probability_diagnostic_signal(ctl, values, oat, predictor="ordinal")
        else:
            if ctl.heat_sp <= 12.01 or ctl.cool_sp >= 29.99:
                ctl.heat_sp, ctl.cool_sp = 22.0, 24.0
            heat, cool, signal = controller_step(strategy, ctl, values, oat)
        set_api_setpoints(api, st, ctl, heat, cool)
        ctl.pending_signal = signal

    def record(st: Any) -> None:
        if not is_weather_file_run_period(api, st):
            return
        enter_environment(st, source="record_fallback")
        initialize_handles(st)
        if not ctl.initialized or api.exchange.warmup_flag(st):
            return
        if not in_requested_period(api, st, begin_month, begin_day, end_month, end_day):
            return
        key = current_key(api, st)
        if ctl.last_record_key == key:
            return
        ctl.last_record_key = key
        ctl.audit_counts["weather_record_callbacks"] += 1
        values = read_zone_values(api, st, ctl)
        oat = read_handle(api, st, ctl.handles["oat"], default=np.nan)
        signal = ctl.pending_signal or {}
        adaptive_bounds_raw = adaptive_comfort_bounds(
            ctl.rm_out if ctl.rm_out is not None else oat
        )
        adaptive_bounds = {
            key: float(signal.get(f"adaptive_effective_{key}", value))
            for key, value in adaptive_bounds_raw.items()
        }
        rec = {
            "strategy": strategy,
            "weather": weather_path.stem,
            "rev01_parent_runner_sha256": PARENT_RUNNER_SHA256,
            "rev01_variant_id": REV01_VARIANT_ID,
            "rev01_variant_manifest_sha256": REV01_VARIANT_MANIFEST_SHA256,
            "rev01_effective_config_sha256": REV01_EFFECTIVE_CONFIG_SHA256,
            "sizing_contract": SIZING_CONTRACT,
            "environment_num": int(api.exchange.current_environment_num(st)),
            "kind_of_sim": int(api.exchange.kind_of_sim(st)),
            "formal_weather_step": int(ctl.formal_weather_step),
            "calendar_year": int(api.exchange.calendar_year(st)),
            "month": int(api.exchange.month(st)),
            "day": int(api.exchange.day_of_month(st)),
            "day_of_week": int(api.exchange.day_of_week(st)),
            "hour": int(api.exchange.hour(st)),
            "current_time": float(api.exchange.current_time(st)),
            "sim_time_days": float(api.exchange.current_sim_time(st)),
            "sim_time_hours": float(api.exchange.current_sim_time(st)),
            "occupied": bool(is_occupied(api, st)),
            "outdoor_temp_c": float(oat),
            "running_mean_outdoor_c": float(ctl.rm_out if ctl.rm_out is not None else oat),
            "adaptive_running_mean_used_c": float(
                signal.get(
                    "adaptive_running_mean_used_c",
                    ctl.rm_out if ctl.rm_out is not None else oat,
                )
            ),
            "adaptive_running_mean_was_clamped": bool(
                signal.get("adaptive_running_mean_was_clamped", False)
            ),
            "paperb_met": float(PAPERB_MET),
            "paperb_people_activity_w_per_person": float(PAPERB_PEOPLE_ACTIVITY_W_PER_PERSON),
            "paperb_spatial_quantile": float(PAPERB_SPATIAL_QUANTILE),
            "paperb_signal_alpha": float(PAPERB_SIGNAL_ALPHA),
            "paperb_tail_threshold": float(PAPERB_TAIL_THRESHOLD),
            "paperb_asym_threshold": float(PAPERB_ASYM_THRESHOLD),
            "paperb_save_heat_c": float(PAPERB_SAVE_HEAT_C),
            "paperb_save_cool_c": float(PAPERB_SAVE_COOL_C),
            "paperb_warm_protect_cool_c": float(PAPERB_WARM_PROTECT_COOL_C),
            "paperb_cold_protect_heat_c": float(PAPERB_COLD_PROTECT_HEAT_C),
            "paperb_tighten_dwell_steps": int(PAPERB_TIGHTEN_DWELL_STEPS),
            "paperb_relax_dwell_steps": int(PAPERB_RELAX_DWELL_STEPS),
            "paperb_pmv_threshold": float(PAPERB_PMV_THRESHOLD),
            "paperb_adaptive_rm_clamp_low_c": float(PAPERB_ADAPTIVE_RM_CLAMP_LOW_C),
            "paperb_adaptive_rm_clamp_high_c": float(PAPERB_ADAPTIVE_RM_CLAMP_HIGH_C),
            "controller_semantics": CONTROLLER_SEMANTICS,
            "inference_preprocessing": INFERENCE_PREPROCESSING,
            "adaptive_comfort_center_c": float(adaptive_bounds["center"]),
            "adaptive_90_low_c": float(adaptive_bounds["low_90"]),
            "adaptive_90_high_c": float(adaptive_bounds["high_90"]),
            "adaptive_80_low_c": float(adaptive_bounds["low_80"]),
            "adaptive_80_high_c": float(adaptive_bounds["high_80"]),
            "comfort_low_c": float(adaptive_bounds["low_90"]),
            "comfort_high_c": float(adaptive_bounds["high_90"]),
            "mean_air_temp_c": float(values["ta_mean"]),
            "mean_mrt_c": float(values["tr_mean"]),
            "mean_operative_temp_c": float(values["top_mean"]),
            "mean_rh_pct": float(values["rh_mean"]),
            "mean_pmv": float(signal.get("mean_pmv", values["pmv_mean"])),
            "mean_ppd_pct": float(signal.get("ppd_pct", ppd_from_pmv(float(values["pmv_mean"])))),
            "expected_tsv": float(signal.get("expected_tsv", np.nan)),
            "discomfort_probability": float(signal.get("discomfort_probability", np.nan)),
            "warm_discomfort_probability": float(
                signal.get("warm_discomfort_probability", np.nan)
            ),
            "cold_discomfort_probability": float(
                signal.get("cold_discomfort_probability", np.nan)
            ),
            "raw_expected_tsv": float(signal.get("raw_expected_tsv", np.nan)),
            "raw_discomfort_probability": float(
                signal.get("raw_discomfort_probability", np.nan)
            ),
            "raw_warm_discomfort_probability": float(
                signal.get("raw_warm_discomfort_probability", np.nan)
            ),
            "raw_cold_discomfort_probability": float(
                signal.get("raw_cold_discomfort_probability", np.nan)
            ),
            "guard_zone": str(signal.get("guard_zone", "")),
            "guard_zone_index": int(signal.get("guard_zone_index", -1)),
            "guard_target_quantile_value": float(
                signal.get("guard_target_quantile_value", np.nan)
            ),
            "guard_selected_pmv": float(signal.get("guard_selected_pmv", np.nan)),
            "guard_selected_p_tail": float(signal.get("guard_selected_p_tail", np.nan)),
            "guard_selected_d_tail": float(signal.get("guard_selected_d_tail", np.nan)),
            "action_delta_c": float(signal.get("action_delta", 0.0)),
            "action_direction": int(signal.get("action_direction", 0)),
            "setpoint_shift_c": float(signal.get("setpoint_shift", 0.0)),
            "grid_event": int(signal.get("grid_event", 0)),
            "grid_stress_score": float(signal.get("grid_stress_score", np.nan)),
            "grid_oat_c": float(signal.get("grid_oat_c", np.nan)),
            "grid_ghi_w_m2": float(signal.get("grid_ghi_w_m2", np.nan)),
                "grid_requested_delta_c": float(signal.get("grid_requested_delta", 0.0)),
                "grid_served_delta_c": float(signal.get("grid_served_delta", 0.0)),
                "grid_rejected": int(signal.get("grid_rejected", 0)),
            "paperb_requested_direction": int(signal.get("paperb_requested_direction", 0)),
            "paperb_request_source": str(signal.get("paperb_request_source", "")),
            "paperb_request_branch": str(signal.get("paperb_request_branch", "not_recorded")),
            "paperb_request_outcome": str(signal.get("paperb_request_outcome", "not_recorded")),
            "paperb_setpoint_moved": int(signal.get("paperb_setpoint_moved", 0)),
            "paperb_guard_classification": str(
                signal.get("paperb_guard_classification", "not_a_tail_guard")
            ),
            "paperb_hold_guard": int(signal.get("paperb_hold_guard", 0)),
            "paperb_heat_target_c": float(signal.get("paperb_heat_target_c", np.nan)),
            "paperb_cool_target_c": float(signal.get("paperb_cool_target_c", np.nan)),
                "heating_setpoint_c": float(ctl.heat_sp),
                "cooling_setpoint_c": float(ctl.cool_sp),
            "zone_heating_rate_w": float(values["heat_rate_sum"]),
            "zone_cooling_rate_w": float(values["cool_rate_sum"]),
            "hvac_on": bool(values["heat_rate_sum"] > 10.0 or values["cool_rate_sum"] > 10.0),
            "electricity_facility_j": float(
                read_handle(api, st, ctl.handles["electricity"], default=0.0, meter=True)
            ),
            "electricity_hvac_j": float(
                read_handle(api, st, ctl.handles["electricity_hvac"], default=0.0, meter=True)
            ),
            "electricity_cooling_j": float(
                read_handle(api, st, ctl.handles["electricity_cooling"], default=0.0, meter=True)
            ),
            "electricity_heating_j": float(
                read_handle(api, st, ctl.handles["electricity_heating"], default=0.0, meter=True)
            ),
            "electricity_fans_j": float(
                read_handle(api, st, ctl.handles["electricity_fans"], default=0.0, meter=True)
            ),
            "electricity_pumps_j": float(
                read_handle(api, st, ctl.handles["electricity_pumps"], default=0.0, meter=True)
            ),
            "natural_gas_facility_j": float(
                read_handle(api, st, ctl.handles["gas"], default=0.0, meter=True)
            ),
        }
        add_zone_environment_record_fields(rec, values)
        add_zone_probability_record_fields(rec, signal)
        ctl.records.append(rec)

    def progress(_pct: int) -> None:
        return

    api.runtime.callback_begin_new_environment(state, begin_new_environment)
    api.runtime.callback_after_new_environment_warmup_complete(
        state, after_new_environment_warmup_complete
    )
    api.runtime.callback_begin_zone_timestep_after_init_heat_balance(state, apply_control)
    api.runtime.callback_end_zone_timestep_after_zone_reporting(state, record)
    api.runtime.callback_progress(state, progress)

    print(f"[simulate] {strategy}: {weather_path.name}")
    try:
        exit_code = api.runtime.run_energyplus(
            state,
            ["-w", str(weather_path), "-d", str(run_dir), str(idf_path)],
        )
    finally:
        api.state_manager.delete_state(state)
        api.runtime.clear_callbacks()

    warmup_audit_path = run_dir / "paperb_weather_warmup_setpoint_audit.csv"
    pd.DataFrame(ctl.warmup_setpoint_records).to_csv(warmup_audit_path, index=False)
    unique_weather_environments = sorted(set(ctl.weather_environment_nums))
    warmup_days = parse_eio_warmup_days(run_dir / "eplusout.eio")
    expected_warmup_callbacks = warmup_days * 24 * 4
    error_gate = parse_energyplus_error_gate(run_dir / "eplusout.err")
    audit_gates = {
        "exit_code_zero": exit_code == 0,
        "energyplus_completed_successfully": error_gate["completed_successfully"],
        "zero_severe": error_gate["severe_error_count"] == 0,
        "zero_fatal": error_gate["fatal_error_count"] == 0,
        "no_check_warmup_convergence": not error_gate[
            "check_warmup_convergence_present"
        ],
        "warmup_days_within_original_ceiling_50": warmup_days <= MAX_WARMUP_DAYS,
        "exactly_one_weather_file_run_period": len(unique_weather_environments) == 1,
        "weather_warmup_callbacks_present": ctl.audit_counts[
            "weather_warmup_control_callbacks"
        ]
        > 0,
        "every_weather_warmup_timestep_audited": ctl.audit_counts[
            "weather_warmup_control_callbacks"
        ]
        == expected_warmup_callbacks,
        "warmup_setpoints_exact": ctl.audit_counts["warmup_setpoint_mismatches"] == 0,
        "warmup_controller_state_inert": ctl.audit_counts[
            "warmup_controller_state_mutations"
        ]
        == 0,
        "weather_regular_callbacks_present": ctl.audit_counts[
            "weather_regular_control_callbacks"
        ]
        > 0,
        "one_post_warmup_reset": ctl.audit_counts[
            "after_weather_warmup_callbacks"
        ]
        == 1,
        "both_setpoint_actuators_released_after_warmup": ctl.audit_counts[
            "actuator_resets_after_weather_warmup"
        ]
        == 2,
        "control_and_record_callback_counts_match": ctl.audit_counts[
            "weather_record_callbacks"
        ]
        == ctl.audit_counts["weather_regular_control_callbacks"]
        == len(ctl.records),
    }
    environment_audit = {
        "schema_version": "paperb_enb_environment_scoped_control_audit_v1",
        "strategy": strategy,
        "weather": weather_path.stem,
        "weather_file_run_period_kind": WEATHER_FILE_RUN_PERIOD_KIND,
        "warmup_setpoint_contract": WARMUP_SETPOINT_CONTRACT,
        "generated_idf_contract": idf_contract,
        "control_scope": "kind_of_sim_3_only",
        "environment_events": ctl.environment_events,
        "weather_environment_nums": unique_weather_environments,
        "callback_counts": ctl.audit_counts,
        "weather_runperiod_warmup_days": warmup_days,
        "expected_weather_warmup_callbacks": expected_warmup_callbacks,
        "energyplus_error_gate": error_gate,
        "warmup_audit_csv": str(warmup_audit_path.resolve()),
        "audit_gates": audit_gates,
        "all_audit_gates_passed": all(audit_gates.values()),
    }
    (run_dir / "PAPERB_CONTROL_ENVIRONMENT_AUDIT.json").write_text(
        json.dumps(environment_audit, indent=2, sort_keys=True) + "\n"
    )
    if not all(audit_gates.values()):
        failed = sorted(name for name, passed in audit_gates.items() if not passed)
        raise RuntimeError(
            f"Environment-scoped control audit failed for {strategy}: {failed}"
        )

    trace_df = pd.DataFrame(ctl.records)
    trace_df = attach_hourly_meters(trace_df, run_dir / "eplusout.mtr")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    write_trace_table(trace_df, trace_path)
    print(f"[simulate] wrote trace: {trace_path}")
    return trace_path


def read_trace_table(path: Path, **kwargs: Any) -> pd.DataFrame:
    if path.suffix == ".parquet":
        if "nrows" in kwargs:
            nrows = kwargs.pop("nrows")
            if nrows == 0:
                return pd.read_parquet(path).head(0)
        return pd.read_parquet(path, **kwargs)
    return pd.read_csv(path, **kwargs)


def write_trace_table(df: pd.DataFrame, path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to replace trace table: {path}")
    if path.suffix == ".parquet":
        temporary = path.with_name(
            f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        )
        table = pa.Table.from_pandas(df, preserve_index=False)
        numeric_columns = [
            field.name
            for field in table.schema
            if pa.types.is_integer(field.type) or pa.types.is_floating(field.type)
        ]
        dictionary_columns = [
            field.name for field in table.schema if field.name not in numeric_columns
        ]
        try:
            pq.write_table(
                table,
                temporary,
                version="2.6",
                compression="zstd",
                compression_level=7,
                use_dictionary=dictionary_columns,
                use_byte_stream_split=numeric_columns,
                row_group_size=8192,
                write_statistics=True,
                data_page_size=1_048_576,
                dictionary_pagesize_limit=1_048_576,
                store_schema=True,
                use_compliant_nested_type=True,
                write_page_checksum=True,
            )
            restored = pq.read_table(
                temporary, page_checksum_verification=True
            ).to_pandas()
            pd.testing.assert_frame_equal(
                df, restored, check_dtype=True, check_exact=True
            )
            for column in df.columns:
                if not pd.api.types.is_float_dtype(df[column].dtype):
                    continue
                left = df[column].to_numpy(dtype=np.float64, copy=False)
                right = restored[column].to_numpy(dtype=np.float64, copy=False)
                left_nan = np.isnan(left)
                right_nan = np.isnan(right)
                if not np.array_equal(left_nan, right_nan):
                    raise AssertionError(f"Trace NaN mask differs: {column}")
                comparable = ~left_nan
                if not np.array_equal(
                    left[comparable].view(np.uint64),
                    right[comparable].view(np.uint64),
                ):
                    raise AssertionError(f"Trace float bits differ: {column}")
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.link(temporary, path)
            temporary.unlink()
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
    else:
        with path.open("x", encoding="utf-8", newline="") as handle:
            df.to_csv(handle, index=False)


def default_control_signal(
    mean_pmv: float, grid_signal: dict[str, float | int] | None = None
) -> dict[str, float | int]:
    signal: dict[str, float | int] = {
        "action_delta": 0.0,
        "action_direction": 0,
        "setpoint_shift": 0.0,
        "mean_pmv": float(mean_pmv),
        "expected_tsv": np.nan,
        "discomfort_probability": np.nan,
        "warm_discomfort_probability": np.nan,
        "cold_discomfort_probability": np.nan,
    }
    signal.update(format_grid_signal(grid_signal, requested=0.0, served=0.0, rejected=0))
    return signal


def format_grid_signal(
    grid_signal: dict[str, float | int] | None,
    *,
    requested: float,
    served: float,
    rejected: int,
) -> dict[str, float | int]:
    grid_signal = grid_signal or {}
    return {
        "grid_event": int(grid_signal.get("grid_event", 0)),
        "grid_stress_score": float(grid_signal.get("grid_stress_score", np.nan)),
        "grid_oat_c": float(grid_signal.get("grid_oat_c", np.nan)),
        "grid_ghi_w_m2": float(grid_signal.get("grid_ghi_w_m2", np.nan)),
        "grid_requested_delta": float(requested),
        "grid_served_delta": float(served),
        "grid_rejected": int(rejected),
    }


def build_microgrid_signal_schedule(
    weather_path: Path,
    begin_month: int,
    begin_day: int,
    end_month: int,
    end_day: int,
) -> dict[tuple[int, int, int], dict[str, float | int]]:
    weather = read_epw_microgrid_weather(weather_path)
    if weather.empty:
        return {}
    period = weather[
        weather.apply(
            lambda row: (begin_month, begin_day)
            <= (int(row["month"]), int(row["day"]))
            <= (end_month, end_day),
            axis=1,
        )
    ].copy()
    if period.empty:
        period = weather.copy()
    occupied = period[(period["hour"] >= 6) & (period["hour"] < 22)].copy()
    if occupied.empty:
        occupied = period.copy()

    oat_q10 = float(occupied["drybulb_c"].quantile(0.10))
    oat_q90 = float(occupied["drybulb_c"].quantile(0.90))
    ghi_max = max(float(occupied["ghi_w_m2"].quantile(0.95)), 1.0)
    oat_norm = ((period["drybulb_c"] - oat_q10) / max(oat_q90 - oat_q10, 1e-6)).clip(0, 1)
    pv_deficit = (1.0 - (period["ghi_w_m2"] / ghi_max).clip(0, 1)).clip(0, 1)
    evening = period["hour"].map(evening_ramp_weight).astype(float)
    period["grid_stress_score"] = 0.60 * oat_norm + 0.30 * pv_deficit + 0.10 * evening

    event_pool = period[(period["hour"] >= 6) & (period["hour"] < 22)].copy()
    if event_pool.empty:
        event_pool = period.copy()
    threshold = float(event_pool["grid_stress_score"].quantile(0.85))
    hot_floor = float(event_pool["drybulb_c"].quantile(0.60))
    period["grid_event"] = (
        (period["grid_stress_score"] >= threshold)
        & (period["drybulb_c"] >= hot_floor)
        & (period["hour"] >= 12)
        & (period["hour"] < 22)
    ).astype(int)

    schedule: dict[tuple[int, int, int], dict[str, float | int]] = {}
    for row in period.itertuples(index=False):
        schedule[(int(row.month), int(row.day), int(row.hour))] = {
            "grid_event": int(row.grid_event),
            "grid_stress_score": float(row.grid_stress_score),
            "grid_oat_c": float(row.drybulb_c),
            "grid_ghi_w_m2": float(row.ghi_w_m2),
        }
    event_hours = sum(v["grid_event"] for v in schedule.values())
    print(
        f"[grid] {weather_path.stem}: selected {event_hours} event hours "
        f"from {len(schedule)} simulated weather hours"
    )
    return schedule


def read_epw_microgrid_weather(weather_path: Path) -> pd.DataFrame:
    rows = []
    with weather_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for i, raw in enumerate(handle):
            if i < 8:
                continue
            parts = raw.strip().split(",")
            if len(parts) < 16:
                continue
            try:
                rows.append(
                    {
                        "month": int(float(parts[1])),
                        "day": int(float(parts[2])),
                        "hour": int(float(parts[3])),
                        "drybulb_c": float(parts[6]),
                        "ghi_w_m2": max(float(parts[13]), 0.0),
                    }
                )
            except ValueError:
                continue
    return pd.DataFrame(rows)


def evening_ramp_weight(hour: int) -> float:
    if 15 <= hour < 20:
        return 1.0
    if 13 <= hour < 15 or 20 <= hour < 22:
        return 0.5
    return 0.0


def lookup_microgrid_signal(
    api: Any, st: Any, ctl: ControlState
) -> dict[str, float | int] | None:
    if not ctl.grid_signal:
        return None
    month = int(api.exchange.month(st))
    day = int(api.exchange.day_of_month(st))
    meter_hour = int(math.ceil(float(api.exchange.current_time(st))))
    meter_hour = max(1, min(24, meter_hour))
    return ctl.grid_signal.get((month, day, meter_hour))


def current_key(api: Any, st: Any) -> tuple[int, int, int, int, float]:
    return (
        int(api.exchange.current_environment_num(st)),
        int(api.exchange.month(st)),
        int(api.exchange.day_of_month(st)),
        int(api.exchange.hour(st)),
        round(float(api.exchange.current_time(st)), 6),
    )


def parse_eio_warmup_days(path: Path) -> int:
    """Read the single weather-RunPeriod warmup count from an EnergyPlus EIO."""

    matches = re.findall(
        r"(?im)^\s*Environment:WarmupDays\s*,\s*(\d+)\s*$",
        path.read_text(errors="strict"),
    )
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Environment:WarmupDays row in {path}, got {matches}")
    return int(matches[0])


def parse_energyplus_error_gate(path: Path) -> dict[str, Any]:
    """Return the strict production error gate from an EnergyPlus ERR file."""

    text = path.read_text(errors="strict")
    severe_lines = [line.strip() for line in text.splitlines() if "** Severe  **" in line]
    fatal_lines = [line.strip() for line in text.splitlines() if "**  Fatal  **" in line]
    return {
        "completed_successfully": "EnergyPlus Completed Successfully" in text,
        "severe_error_count": len(severe_lines),
        "severe_lines": severe_lines,
        "fatal_error_count": len(fatal_lines),
        "fatal_lines": fatal_lines,
        "check_warmup_convergence_present": "CheckWarmupConvergence" in text,
    }


def is_weather_file_run_period(api: Any, st: Any) -> bool:
    """True only in the weather-file RunPeriod (EnergyPlus kind-of-sim 3)."""

    return int(api.exchange.kind_of_sim(st)) == WEATHER_FILE_RUN_PERIOD_KIND


def attach_hourly_meters(trace_df: pd.DataFrame, mtr_path: Path) -> pd.DataFrame:
    if trace_df.empty or not mtr_path.exists():
        return trace_df
    hourly = parse_mtr_hourly(mtr_path)
    if hourly.empty:
        return trace_df
    out = trace_df.copy()
    out["meter_hour"] = np.ceil(out["current_time"].astype(float)).astype(int).clip(1, 24)
    merged = out.merge(hourly, on=["month", "day", "meter_hour"], how="left")
    counts = merged.groupby(["month", "day", "meter_hour"])["strategy"].transform("size").clip(lower=1)
    for col in ["electricity_facility_j", "natural_gas_facility_j"]:
        hourly_col = f"{col}_hourly"
        if hourly_col in merged:
            step_values = merged[hourly_col] / counts
            merged[col] = np.where(step_values.notna(), step_values, merged[col])
    return merged.drop(columns=[c for c in ["meter_hour"] if c in merged])


def parse_mtr_hourly(mtr_path: Path) -> pd.DataFrame:
    meter_ids: dict[str, str] = {}
    rows: list[dict[str, float | int]] = []
    current: dict[str, int] | None = None
    in_dictionary = True
    for raw in mtr_path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        if in_dictionary:
            if line == "End of Data Dictionary":
                in_dictionary = False
                continue
            match = re.match(r"^(\d+),\d+,(.+?) \[J\] !Hourly", line)
            if match:
                meter_id, name = match.groups()
                if name == "Electricity:Facility":
                    meter_ids[meter_id] = "electricity_facility_j_hourly"
                elif name == "NaturalGas:Facility":
                    meter_ids[meter_id] = "natural_gas_facility_j_hourly"
            continue
        parts = [p.strip() for p in line.split(",")]
        if not parts:
            continue
        if parts[0] == "2" and len(parts) >= 8:
            if current is not None:
                rows.append(current)
            current = {
                "month": int(float(parts[2])),
                "day": int(float(parts[3])),
                "meter_hour": int(float(parts[5])),
            }
        elif current is not None and parts[0] in meter_ids and len(parts) >= 2:
            current[meter_ids[parts[0]]] = float(parts[1])
    if current is not None:
        rows.append(current)
    return pd.DataFrame(rows)


def in_requested_period(
    api: Any,
    st: Any,
    begin_month: int,
    begin_day: int,
    end_month: int,
    end_day: int,
) -> bool:
    cur = (int(api.exchange.month(st)), int(api.exchange.day_of_month(st)))
    return (begin_month, begin_day) <= cur <= (end_month, end_day)


def read_handle(api: Any, st: Any, handle: int, default: float, meter: bool = False) -> float:
    if handle is None or handle < 0:
        return default
    try:
        if meter:
            val = api.exchange.get_meter_value(st, handle)
        else:
            val = api.exchange.get_variable_value(st, handle)
        return float(val)
    except Exception:
        return default


def read_zone_values(api: Any, st: Any, ctl: ControlState) -> dict[str, float | np.ndarray]:
    ta_vals = []
    tr_vals = []
    rh_vals = []
    heat_rates = []
    cool_rates = []
    for zone in ZONE_NAMES:
        handles = ctl.handles["zones"][zone]
        ta = read_handle(api, st, handles["ta"], default=np.nan)
        tr = read_handle(api, st, handles["tr"], default=ta)
        rh = read_handle(api, st, handles["rh"], default=50.0)
        ta_vals.append(ta)
        tr_vals.append(tr if np.isfinite(tr) else ta)
        rh_vals.append(rh if np.isfinite(rh) else 50.0)
        heat_rates.append(read_handle(api, st, handles["heat_rate"], default=0.0))
        cool_rates.append(read_handle(api, st, handles["cool_rate"], default=0.0))
    ta_arr = np.asarray(ta_vals, dtype=float)
    tr_arr = np.asarray(tr_vals, dtype=float)
    rh_arr = np.asarray(rh_vals, dtype=float)
    bad = ~np.isfinite(ta_arr)
    if bad.any():
        ta_arr[bad] = np.nanmean(ta_arr[~bad]) if (~bad).any() else 24.0
    bad = ~np.isfinite(tr_arr)
    if bad.any():
        tr_arr[bad] = ta_arr[bad]
    rh_arr = np.clip(np.nan_to_num(rh_arr, nan=50.0), 1.0, 100.0)
    top = (ta_arr + tr_arr) / 2.0
    pmv = compute_pmv_array(
        ta=ta_arr,
        tr=tr_arr,
        v=np.full_like(ta_arr, 0.10),
        rh=rh_arr,
        met=np.full_like(ta_arr, PAPERB_MET),
        clo=np.full_like(ta_arr, 0.65),
    )
    return {
        "ta": ta_arr,
        "tr": tr_arr,
        "rh": rh_arr,
        "top": top,
        "pmv": pmv,
        "ta_mean": float(np.mean(ta_arr)),
        "tr_mean": float(np.mean(tr_arr)),
        "top_mean": float(np.mean(top)),
        "rh_mean": float(np.mean(rh_arr)),
        "pmv_mean": float(np.mean(pmv)),
        "heat_rate_sum": float(np.nansum(heat_rates)),
        "cool_rate_sum": float(np.nansum(cool_rates)),
    }


def update_running_mean(previous: float | None, oat: float) -> float:
    if previous is None or not np.isfinite(previous):
        return float(oat)
    alpha_step = 0.8 ** (1.0 / 96.0)
    return float(alpha_step * previous + (1.0 - alpha_step) * oat)


def adaptive_comfort_bounds(running_mean_outdoor_c: float) -> dict[str, float]:
    """ASHRAE-style adaptive comfort center with 90% and 80% half-width bands.

    This is used as a scalar benchmark, not as a compliance claim for the
    mechanically conditioned Medium Office prototype.
    """
    center = 0.31 * float(running_mean_outdoor_c) + 17.8
    return {
        "center": center,
        "low_90": center - PAPERB_ADAPTIVE_90_HALF_WIDTH_C,
        "high_90": center + PAPERB_ADAPTIVE_90_HALF_WIDTH_C,
        "low_80": center - PAPERB_ADAPTIVE_80_HALF_WIDTH_C,
        "high_80": center + PAPERB_ADAPTIVE_80_HALF_WIDTH_C,
    }


def is_occupied(api: Any, st: Any) -> bool:
    day = int(api.exchange.day_of_week(st))
    current_time = float(api.exchange.current_time(st))
    hour = int(math.floor(current_time))
    weekday = 2 <= day <= 6
    return bool(weekday and 6 <= hour < 22)


def baseline_setpoints(occupied: bool) -> tuple[float, float]:
    """Return the one baseline contract used in warmup and reported operation."""

    if occupied:
        return PAPERB_REF_HEAT_C, PAPERB_REF_COOL_C
    return 12.0, 30.0


def controller_step(
    strategy: str,
    ctl: ControlState,
    values: dict[str, float | np.ndarray],
    oat: float,
) -> tuple[float, float, dict[str, float | int]]:
    direction = 0
    delta = 0.0
    setpoint_shift = 0.0
    signal: dict[str, float | int] = default_control_signal(
        float(values["pmv_mean"]), ctl.current_grid_signal
    )

    if strategy in GRID_STRATEGIES:
        return grid_controller_step(strategy, ctl, values, oat, signal)
    if strategy in PAPERB_STRATEGIES:
        return paperb_asymmetric_relax_controller_step(strategy, ctl, values, oat, signal)

    if strategy == "pmv":
        pmv_signal = float(values["pmv_mean"])
        if abs(pmv_signal) > 1.5:
            delta = 1.25
        elif abs(pmv_signal) > 0.5:
            delta = 0.5
        direction = int(np.sign(pmv_signal))
        signal["expected_tsv"] = np.nan
        signal["discomfort_probability"] = np.nan
    else:
        if ctl.bundle is None:
            raise RuntimeError(f"{strategy} requires trained predictors.")
        probs = predict_zone_probabilities(ctl, values, oat, predictor=strategy)
        zone_mu, zone_cold_tail, zone_warm_tail = zone_probability_signals(probs)
        add_raw_probability_diagnostics(ctl, signal)
        mu = float(np.mean(zone_mu))
        cold_tail = float(np.mean(zone_cold_tail))
        warm_tail = float(np.mean(zone_warm_tail))
        p_disc = cold_tail + warm_tail
        if p_disc >= 0.35:
            delta = 1.25
        elif p_disc > 0.065:
            delta = 0.5
        direction = int(np.sign(mu)) if delta > 0 else 0
        signal["expected_tsv"] = mu
        signal["discomfort_probability"] = p_disc
        signal["warm_discomfort_probability"] = warm_tail
        signal["cold_discomfort_probability"] = cold_tail
        signal["zone_expected_tsv"] = zone_mu
        signal["zone_cold_tail"] = zone_cold_tail
        signal["zone_warm_tail"] = zone_warm_tail

    if direction == 0 or delta == 0.0:
        heat, cool = ctl.heat_sp, ctl.cool_sp
    else:
        # direction is the thermal sensation side; setpoints move oppositely.
        setpoint_shift = -delta if direction > 0 else delta
        heat, cool = apply_setpoint_shift(ctl.heat_sp, ctl.cool_sp, setpoint_shift)

    signal["action_delta"] = delta
    signal["action_direction"] = direction
    signal["setpoint_shift"] = setpoint_shift
    return heat, cool, signal


def ppd_from_pmv(pmv: float) -> float:
    pmv_abs = abs(float(pmv))
    return float(100.0 - 95.0 * math.exp(-0.03353 * pmv_abs**4 - 0.2179 * pmv_abs**2))


def classify_tail_guard(
    guard_p_disc: float,
    guard_d_tail: float,
    *,
    semantics: str | None = None,
) -> tuple[int, str]:
    """Classify tail risk before applying dwell or setpoint bounds.

    Directions follow the production contract: +1 warm protection, -1 cold
    protection, 99 hold, and 0 energy-saving relaxation. The legacy rule
    relaxes in high-tail states with ambiguous warm/cold asymmetry; the
    corrected rule holds in precisely those states.
    """
    selected_semantics = semantics or CONTROLLER_SEMANTICS
    if selected_semantics not in {"legacy", "corrected"}:
        raise ValueError(f"Unknown controller semantics: {selected_semantics}")
    if guard_p_disc < PAPERB_TAIL_THRESHOLD:
        return 0, "below_tail_threshold"
    if guard_d_tail > PAPERB_ASYM_THRESHOLD:
        return 1, "high_tail_warm"
    if guard_d_tail < -PAPERB_ASYM_THRESHOLD:
        return -1, "high_tail_cold"
    if selected_semantics == "corrected":
        return 99, "high_tail_ambiguous_hold"
    return 0, "high_tail_ambiguous_relax"


def paperb_asymmetric_relax_controller_step(
    strategy: str,
    ctl: ControlState,
    values: dict[str, float | np.ndarray],
    oat: float,
    signal: dict[str, float | int],
) -> tuple[float, float, dict[str, float | int]]:
    """Building-level constrained controller with faster energy-saving relaxation.

    The policy is intentionally simple for the first Paper B smoke test. It is
    designed to test operational behavior, not to claim optimal control.
    """
    step_idx = int(round(float(values.get("sim_step", 0.0))))
    direction = 0
    request_delta = 0.0
    request_source = "none"

    if strategy in {
        "paperb_pmv_relax",
        "paperb_p90_abs_pmv_relax",
        "paperb_adaptive_band_relax",
        "paperb_adaptive_band_clamped_relax",
        "paperb_ppd_guard_relax",
        "paperb_pmv_exceedance_guard_relax",
        "paperb_pmv_extreme_guard_relax",
    }:
        pmv_signal = float(values["pmv_mean"])
        ppd_signal = ppd_from_pmv(pmv_signal)
        signal["ppd_pct"] = ppd_signal
        if ctl.bundle is not None:
            probs = predict_zone_probabilities(ctl, values, oat, predictor="ordinal")
            zone_mu, zone_cold_tail, zone_warm_tail = zone_probability_signals(probs)
            add_raw_probability_diagnostics(ctl, signal)
            cold_tail = float(np.mean(zone_cold_tail))
            warm_tail = float(np.mean(zone_warm_tail))
            signal["expected_tsv"] = float(np.mean(zone_mu))
            signal["discomfort_probability"] = cold_tail + warm_tail
            signal["warm_discomfort_probability"] = warm_tail
            signal["cold_discomfort_probability"] = cold_tail
            signal["zone_expected_tsv"] = zone_mu
            signal["zone_cold_tail"] = zone_cold_tail
            signal["zone_warm_tail"] = zone_warm_tail
        else:
            signal["expected_tsv"] = np.nan
            signal["discomfort_probability"] = np.nan
            signal["warm_discomfort_probability"] = np.nan
            signal["cold_discomfort_probability"] = np.nan
        if strategy in {
            "paperb_adaptive_band_relax",
            "paperb_adaptive_band_clamped_relax",
        }:
            running_mean = ctl.rm_out if ctl.rm_out is not None else oat
            if strategy == "paperb_adaptive_band_clamped_relax":
                adaptive_bounds = rev01_adaptive_comfort_bounds(
                    running_mean,
                    clamp_low_c=PAPERB_ADAPTIVE_RM_CLAMP_LOW_C,
                    clamp_high_c=PAPERB_ADAPTIVE_RM_CLAMP_HIGH_C,
                    half_width_90_c=PAPERB_ADAPTIVE_90_HALF_WIDTH_C,
                    half_width_80_c=PAPERB_ADAPTIVE_80_HALF_WIDTH_C,
                )
                signal["adaptive_running_mean_used_c"] = float(
                    adaptive_bounds["used_running_mean_c"]
                )
                signal["adaptive_running_mean_was_clamped"] = bool(
                    adaptive_bounds["was_clamped"]
                )
                for key in ("center", "low_90", "high_90", "low_80", "high_80"):
                    signal[f"adaptive_effective_{key}"] = float(adaptive_bounds[key])
            else:
                adaptive_bounds = adaptive_comfort_bounds(running_mean)
                signal["adaptive_running_mean_used_c"] = float(running_mean)
                signal["adaptive_running_mean_was_clamped"] = False
            top_signal = float(values["top_mean"])
            signal["adaptive_comfort_center_c"] = float(adaptive_bounds["center"])
            signal["adaptive_90_low_c"] = float(adaptive_bounds["low_90"])
            signal["adaptive_90_high_c"] = float(adaptive_bounds["high_90"])
            signal["adaptive_80_low_c"] = float(adaptive_bounds["low_80"])
            signal["adaptive_80_high_c"] = float(adaptive_bounds["high_80"])
            signal["adaptive_warm_slack_90_c"] = float(adaptive_bounds["high_90"] - top_signal)
            signal["adaptive_warm_slack_80_c"] = float(adaptive_bounds["high_80"] - top_signal)
            signal["adaptive_cold_slack_90_c"] = float(top_signal - adaptive_bounds["low_90"])
            signal["adaptive_cold_slack_80_c"] = float(top_signal - adaptive_bounds["low_80"])
            if top_signal > adaptive_bounds["high_80"]:
                direction = 1
                request_source = "adaptive_80_warm"
            elif top_signal < adaptive_bounds["low_80"]:
                direction = -1
                request_source = "adaptive_80_cold"
            elif top_signal > adaptive_bounds["high_90"] or top_signal < adaptive_bounds["low_90"]:
                direction = 99
                request_source = "adaptive_90_hold"
        elif strategy in {"paperb_pmv_relax", "paperb_p90_abs_pmv_relax"}:
            if strategy == "paperb_p90_abs_pmv_relax":
                selected = select_pmv_quantile_guard(
                    np.asarray(values["pmv"], dtype=float),
                    PAPERB_SPATIAL_QUANTILE,
                )
                guard_index = int(selected["index"])
                pmv_signal = float(selected["selected_signed_pmv"])
                signal["guard_zone"] = ZONE_NAMES[guard_index]
                signal["guard_zone_index"] = guard_index
                signal["guard_target_quantile_value"] = float(
                    selected["target_abs_pmv"]
                )
                signal["guard_selected_pmv"] = pmv_signal
            if pmv_signal > PAPERB_PMV_THRESHOLD:
                direction = 1
                request_source = (
                    "pmv_p90_warm"
                    if strategy == "paperb_p90_abs_pmv_relax"
                    else "pmv_warm"
                )
            elif pmv_signal < -PAPERB_PMV_THRESHOLD:
                direction = -1
                request_source = (
                    "pmv_p90_cold"
                    if strategy == "paperb_p90_abs_pmv_relax"
                    else "pmv_cold"
                )
        elif strategy == "paperb_ppd_guard_relax":
            if ppd_signal > PAPERB_PPD_PROTECT_THRESHOLD:
                direction = int(np.sign(pmv_signal))
                request_source = "ppd_protect_warm" if direction > 0 else "ppd_protect_cold"
            elif ppd_signal > PAPERB_PPD_HOLD_THRESHOLD:
                direction = 99
                request_source = "ppd_hold"
        else:
            if pmv_signal > PAPERB_PMV_EXTREME_THRESHOLD:
                direction = 1
                request_source = "pmv_exceedance_warm"
            elif pmv_signal < -PAPERB_PMV_EXTREME_THRESHOLD:
                direction = -1
                request_source = "pmv_exceedance_cold"
            elif abs(pmv_signal) > PAPERB_PMV_THRESHOLD:
                direction = 99
                request_source = "pmv_mild_hold"
    else:
        if ctl.bundle is None:
            raise RuntimeError(f"{strategy} requires trained predictors.")
        probs = predict_zone_probabilities(ctl, values, oat, predictor="ordinal")
        zone_mu, zone_cold_tail, zone_warm_tail = zone_probability_signals(probs)
        add_raw_probability_diagnostics(ctl, signal)
        mu = float(np.mean(zone_mu))
        cold_tail = float(np.mean(zone_cold_tail))
        warm_tail = float(np.mean(zone_warm_tail))
        p_disc = cold_tail + warm_tail
        d_tail = warm_tail - cold_tail
        signal["expected_tsv"] = mu
        signal["discomfort_probability"] = p_disc
        signal["warm_discomfort_probability"] = warm_tail
        signal["cold_discomfort_probability"] = cold_tail
        signal["zone_expected_tsv"] = zone_mu
        signal["zone_cold_tail"] = zone_cold_tail
        signal["zone_warm_tail"] = zone_warm_tail
        if strategy == "paperb_mu_relax":
            if mu > PAPERB_TSV_THRESHOLD:
                direction = 1
                request_source = "mu_warm"
            elif mu < -PAPERB_TSV_THRESHOLD:
                direction = -1
                request_source = "mu_cold"
        elif strategy in {"paperb_gate_tail_asym_relax", "paperb_p90_tail_asym_relax"}:
            guard_p_disc = p_disc
            guard_d_tail = d_tail
            if strategy == "paperb_p90_tail_asym_relax":
                selected = select_learned_tail_guard(
                    zone_cold_tail,
                    zone_warm_tail,
                    PAPERB_SPATIAL_QUANTILE,
                )
                p90_idx = int(selected["index"])
                p90_target = float(selected["target_p_tail"])
                guard_p_disc = float(selected["selected_p_tail"])
                guard_d_tail = float(selected["selected_d_tail"])
                signal["p90_guard_zone"] = ZONE_NAMES[p90_idx]
                signal["p90_guard_p_tail"] = guard_p_disc
                signal["p90_guard_d_tail"] = guard_d_tail
                signal["guard_zone"] = ZONE_NAMES[p90_idx]
                signal["guard_zone_index"] = p90_idx
                signal["guard_target_quantile_value"] = p90_target
                signal["guard_selected_p_tail"] = guard_p_disc
                signal["guard_selected_d_tail"] = guard_d_tail
            direction, guard_classification = classify_tail_guard(guard_p_disc, guard_d_tail)
            signal["paperb_guard_classification"] = guard_classification
            if direction == 99:
                request_source = (
                    "p90_tail_ambiguous_hold"
                    if strategy == "paperb_p90_tail_asym_relax"
                    else "tail_ambiguous_hold"
                )
            elif direction > 0:
                request_source = "p90_tail_warm" if strategy == "paperb_p90_tail_asym_relax" else "tail_warm"
            elif direction < 0:
                request_source = "p90_tail_cold" if strategy == "paperb_p90_tail_asym_relax" else "tail_cold"
        else:
            raise ValueError(f"Unknown Paper B strategy: {strategy}")

    heat = float(ctl.heat_sp)
    cool = float(ctl.cool_sp)
    action_direction = 0
    request_branch = (
        "risk_hold"
        if direction == 99
        else "warm_protection"
        if direction > 0
        else "cold_protection"
        if direction < 0
        else "relax"
    )
    request_outcome = "no_move"

    if direction == 99:
        # Mild scalar discomfort: stop relaxation without adding a protection move.
        request_delta = 0.0
        action_direction = 0
        request_outcome = "hold"
    elif direction > 0:
        # Warm-risk protection: lower cooling setpoint slowly.
        request_delta = PAPERB_TIGHTEN_STEP_C
        if step_idx - ctl.last_paperb_action_step >= PAPERB_TIGHTEN_DWELL_STEPS:
            new_cool = max(PAPERB_WARM_PROTECT_COOL_C, cool - PAPERB_TIGHTEN_STEP_C)
            if new_cool != cool:
                cool = new_cool
                ctl.last_paperb_action_step = step_idx
                action_direction = 1
                request_outcome = "applied"
            else:
                request_outcome = "bound_saturated"
        else:
            request_outcome = "dwell_blocked"
    elif direction < 0:
        # Cold-risk protection: raise heating setpoint slowly.
        request_delta = PAPERB_TIGHTEN_STEP_C
        if step_idx - ctl.last_paperb_action_step >= PAPERB_TIGHTEN_DWELL_STEPS:
            new_heat = min(PAPERB_COLD_PROTECT_HEAT_C, heat + PAPERB_TIGHTEN_STEP_C)
            if new_heat != heat:
                heat = new_heat
                ctl.last_paperb_action_step = step_idx
                action_direction = -1
                request_outcome = "applied"
            else:
                request_outcome = "bound_saturated"
        else:
            request_outcome = "dwell_blocked"
    else:
        # No active discomfort request: relax faster toward energy-saving deadband.
        if step_idx - ctl.last_paperb_action_step >= PAPERB_RELAX_DWELL_STEPS:
            new_heat = max(PAPERB_SAVE_HEAT_C, heat - PAPERB_RELAX_STEP_C)
            new_cool = min(PAPERB_SAVE_COOL_C, cool + PAPERB_RELAX_STEP_C)
            if new_heat != heat or new_cool != cool:
                heat, cool = new_heat, new_cool
                ctl.last_paperb_action_step = step_idx
                action_direction = 2
                request_delta = PAPERB_RELAX_STEP_C
                request_source = "energy_relax"
                request_outcome = "applied"
            else:
                request_outcome = "bound_saturated"
        else:
            request_outcome = "dwell_blocked"

    if cool - heat < 2.0:
        midpoint = (heat + cool) / 2.0
        heat = midpoint - 1.0
        cool = midpoint + 1.0

    signal["action_delta"] = request_delta
    signal["action_direction"] = action_direction
    signal["setpoint_shift"] = cool - PAPERB_REF_COOL_C
    signal["paperb_requested_direction"] = 0 if direction == 99 else direction
    signal["paperb_hold_guard"] = 1 if direction == 99 else 0
    signal["paperb_request_source"] = request_source
    signal["paperb_request_branch"] = request_branch
    signal["paperb_request_outcome"] = request_outcome
    signal["paperb_setpoint_moved"] = 1 if action_direction in {-1, 1, 2} else 0
    signal["paperb_heat_target_c"] = heat
    signal["paperb_cool_target_c"] = cool
    return heat, cool, signal


def probability_diagnostic_signal(
    ctl: ControlState,
    values: dict[str, float | np.ndarray],
    oat: float,
    *,
    predictor: str = "ordinal",
) -> dict[str, float | int]:
    """Compute probability diagnostics without changing thermostat setpoints."""
    if ctl.bundle is None:
        raise RuntimeError("diagnostic_reference requires trained predictors.")
    signal: dict[str, float | int] = default_control_signal(
        float(values["pmv_mean"]), ctl.current_grid_signal
    )
    probs = predict_zone_probabilities(ctl, values, oat, predictor=predictor)
    zone_mu, zone_cold_tail, zone_warm_tail = zone_probability_signals(probs)
    add_raw_probability_diagnostics(ctl, signal)
    cold_tail = float(np.mean(zone_cold_tail))
    warm_tail = float(np.mean(zone_warm_tail))
    signal["expected_tsv"] = float(np.mean(zone_mu))
    signal["discomfort_probability"] = cold_tail + warm_tail
    signal["warm_discomfort_probability"] = warm_tail
    signal["cold_discomfort_probability"] = cold_tail
    signal["zone_expected_tsv"] = zone_mu
    signal["zone_cold_tail"] = zone_cold_tail
    signal["zone_warm_tail"] = zone_warm_tail
    return signal


def grid_controller_step(
    strategy: str,
    ctl: ControlState,
    values: dict[str, float | np.ndarray],
    oat: float,
    signal: dict[str, float | int],
) -> tuple[float, float, dict[str, float | int]]:
    grid_signal = ctl.current_grid_signal or {}
    grid_event = int(grid_signal.get("grid_event", 0))
    requested = GRID_FULL_SHED_DELTA_C if grid_event else 0.0
    served = requested
    rejected = 0

    signal["expected_tsv"] = np.nan
    signal["discomfort_probability"] = np.nan
    signal["warm_discomfort_probability"] = np.nan
    signal["cold_discomfort_probability"] = np.nan

    if strategy == "grid_gated" and grid_event:
        if ctl.bundle is None:
            raise RuntimeError(f"{strategy} requires trained predictors.")
        probs = predict_zone_probabilities(ctl, values, oat, predictor="ordinal")
        zone_mu, zone_cold_tail, zone_warm_tail = zone_probability_signals(probs)
        add_raw_probability_diagnostics(ctl, signal)
        mu = float(np.mean(zone_mu))
        cold_tail = float(np.mean(zone_cold_tail))
        warm_tail = float(np.mean(zone_warm_tail))
        p_disc = cold_tail + warm_tail
        if warm_tail >= GRID_WARM_RISK_BLOCK:
            served = 0.0
            rejected = 1
        elif warm_tail >= GRID_WARM_RISK_SOFT:
            served = GRID_MILD_SHED_DELTA_C
        signal["expected_tsv"] = mu
        signal["discomfort_probability"] = p_disc
        signal["warm_discomfort_probability"] = warm_tail
        signal["cold_discomfort_probability"] = cold_tail
        signal["zone_expected_tsv"] = zone_mu
        signal["zone_cold_tail"] = zone_cold_tail
        signal["zone_warm_tail"] = zone_warm_tail

    heat, cool = apply_setpoint_shift(22.0, 24.0, served) if served > 0 else (22.0, 24.0)
    signal["action_delta"] = abs(served)
    signal["action_direction"] = -1 if served > 0 else 0
    signal["setpoint_shift"] = served
    signal.update(
        format_grid_signal(grid_signal, requested=requested, served=served, rejected=rejected)
    )
    return heat, cool, signal


def predict_zone_probabilities(
    ctl: ControlState,
    values: dict[str, float | np.ndarray],
    oat: float,
    *,
    predictor: str,
) -> np.ndarray:
    if ctl.bundle is None:
        raise RuntimeError("Predictor bundle is required.")
    n = len(values["ta"])
    runtime_inputs = preprocess_runtime_inputs(
        ta=np.asarray(values["ta"], dtype=float),
        tr=np.asarray(values["tr"], dtype=float),
        v=np.full(n, 0.10),
        rh=np.asarray(values["rh"], dtype=float),
        met=np.full(n, PAPERB_MET),
        clo=np.full(n, 0.65),
        bsa=np.full(n, 1.80),
        rm_out=np.full(n, ctl.rm_out if ctl.rm_out is not None else oat),
        spec=ctl.bundle.spec,
    )
    features = build_features_from_arrays(
        ta=runtime_inputs["ta"],
        tr=runtime_inputs["tr"],
        v=runtime_inputs["v"],
        rh=runtime_inputs["rh"],
        met=runtime_inputs["met"],
        clo=runtime_inputs["clo"],
        bsa=runtime_inputs["BSA_m2"],
        rm_out=runtime_inputs["rm_out"],
        spec=ctl.bundle.spec,
    )
    if predictor == "nominal":
        raw = ctl.bundle.predict_nominal(features)
    elif predictor == "ordinal":
        raw = ctl.bundle.predict_ordinal(features)
    else:
        raise ValueError(f"Unknown predictor: {predictor}")
    effective = attenuate_probabilities(raw, PAPERB_SIGNAL_ALPHA)
    ctl.last_raw_probabilities = raw
    ctl.last_effective_probabilities = effective
    return effective


def zone_probability_signals(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zone_mu = probs @ TSV_VALUES
    zone_cold_tail = probs[:, [0, 1]].sum(axis=1)
    zone_warm_tail = probs[:, [5, 6]].sum(axis=1)
    return zone_mu, zone_cold_tail, zone_warm_tail


def add_raw_probability_diagnostics(
    ctl: ControlState, signal: dict[str, Any]
) -> None:
    """Attach frozen-model probabilities before deterministic Rev01 attenuation."""

    if ctl.last_raw_probabilities is None:
        return
    raw_mu, raw_cold, raw_warm = zone_probability_signals(ctl.last_raw_probabilities)
    signal["raw_expected_tsv"] = float(np.mean(raw_mu))
    signal["raw_discomfort_probability"] = float(np.mean(raw_cold + raw_warm))
    signal["raw_warm_discomfort_probability"] = float(np.mean(raw_warm))
    signal["raw_cold_discomfort_probability"] = float(np.mean(raw_cold))
    signal["raw_zone_expected_tsv"] = raw_mu
    signal["raw_zone_cold_tail"] = raw_cold
    signal["raw_zone_warm_tail"] = raw_warm


def add_zone_probability_record_fields(
    rec: dict[str, Any],
    signal: dict[str, Any],
) -> None:
    if not all(
        key in signal
        for key in ("zone_expected_tsv", "zone_cold_tail", "zone_warm_tail")
    ):
        return
    zone_mu = np.asarray(signal["zone_expected_tsv"], dtype=float)
    zone_cold_tail = np.asarray(signal["zone_cold_tail"], dtype=float)
    zone_warm_tail = np.asarray(signal["zone_warm_tail"], dtype=float)
    if not (
        len(zone_mu) == len(zone_cold_tail) == len(zone_warm_tail) == len(ZONE_FIELD_NAMES)
    ):
        return
    zone_p_disc = zone_cold_tail + zone_warm_tail
    zone_d_tail = zone_warm_tail - zone_cold_tail
    for idx, slug in enumerate(ZONE_FIELD_NAMES):
        rec[f"zone_{slug}_expected_tsv"] = float(zone_mu[idx])
        rec[f"zone_{slug}_p_disc"] = float(zone_p_disc[idx])
        rec[f"zone_{slug}_warm_tail"] = float(zone_warm_tail[idx])
        rec[f"zone_{slug}_cold_tail"] = float(zone_cold_tail[idx])
        rec[f"zone_{slug}_d_tail"] = float(zone_d_tail[idx])
    if all(
        key in signal
        for key in ("raw_zone_expected_tsv", "raw_zone_cold_tail", "raw_zone_warm_tail")
    ):
        raw_mu = np.asarray(signal["raw_zone_expected_tsv"], dtype=float)
        raw_cold = np.asarray(signal["raw_zone_cold_tail"], dtype=float)
        raw_warm = np.asarray(signal["raw_zone_warm_tail"], dtype=float)
        if len(raw_mu) == len(raw_cold) == len(raw_warm) == len(ZONE_FIELD_NAMES):
            raw_p_tail = raw_cold + raw_warm
            raw_d_tail = raw_warm - raw_cold
            for idx, slug in enumerate(ZONE_FIELD_NAMES):
                rec[f"raw_zone_{slug}_expected_tsv"] = float(raw_mu[idx])
                rec[f"raw_zone_{slug}_p_disc"] = float(raw_p_tail[idx])
                rec[f"raw_zone_{slug}_warm_tail"] = float(raw_warm[idx])
                rec[f"raw_zone_{slug}_cold_tail"] = float(raw_cold[idx])
                rec[f"raw_zone_{slug}_d_tail"] = float(raw_d_tail[idx])


def add_zone_environment_record_fields(
    rec: dict[str, Any],
    values: dict[str, Any],
) -> None:
    if not all(key in values for key in ("ta", "tr", "rh")):
        return
    ta = np.asarray(values["ta"], dtype=float)
    tr = np.asarray(values["tr"], dtype=float)
    rh = np.asarray(values["rh"], dtype=float)
    if not (len(ta) == len(tr) == len(rh) == len(ZONE_FIELD_NAMES)):
        return
    for idx, slug in enumerate(ZONE_FIELD_NAMES):
        rec[f"zone_{slug}_ta_c"] = float(ta[idx])
        rec[f"zone_{slug}_tr_c"] = float(tr[idx])
        rec[f"zone_{slug}_rh_pct"] = float(rh[idx])


def apply_setpoint_shift(heat: float, cool: float, shift: float) -> tuple[float, float]:
    heat += shift
    cool += shift
    heat = float(np.clip(heat, 12.0, 23.25))
    cool = float(np.clip(cool, 23.25, 30.0))
    if cool - heat < 2.0:
        if shift >= 0:
            cool = min(30.0, heat + 2.0)
            heat = min(heat, cool - 2.0)
        else:
            heat = max(12.0, cool - 2.0)
            cool = max(cool, heat + 2.0)
    return heat, cool


def set_api_setpoints(api: Any, st: Any, ctl: ControlState, heat: float, cool: float) -> None:
    ctl.heat_sp = float(heat)
    ctl.cool_sp = float(cool)
    api.exchange.set_actuator_value(st, ctl.handles["heat_act"], ctl.heat_sp)
    api.exchange.set_actuator_value(st, ctl.handles["cool_act"], ctl.cool_sp)


def run_simulations(
    *,
    bundle: PredictorBundle | None,
    output_dir: Path,
    source_idf: Path,
    weather_paths: list[Path],
    eplus_root: Path,
    begin_month: int,
    begin_day: int,
    end_month: int,
    end_day: int,
    strategies: list[str],
    resume: bool = False,
    purge_energyplus_after_trace: bool = False,
    skip_combined_trace: bool = False,
    trace_format: str = "csv",
) -> list[Path]:
    idf_path = output_dir / "model" / f"medium_office_otc_control_{people_activity_tag()}.idf"
    contract_path = output_dir / "model" / "PAPERB_GENERATED_IDF_CONTRACT.json"
    if idf_path.exists() or contract_path.exists():
        raise FileExistsError(
            f"The v2 runner refuses to replace a generated model: {idf_path}"
        )
    patch_idf_for_control(
        source_idf,
        idf_path,
        begin_month,
        begin_day,
        end_month,
        end_day,
        people_activity_w_per_person=PAPERB_PEOPLE_ACTIVITY_W_PER_PERSON,
    )
    generated_contract = validate_generated_idf_contract(idf_path)
    contract_path.write_text(json.dumps(generated_contract, indent=2, sort_keys=True) + "\n")
    trace_paths = []
    total_runs = len(weather_paths) * len(strategies)
    run_idx = 0
    for weather in weather_paths:
        for strategy in strategies:
            run_idx += 1
            print(f"[simulate] case {run_idx}/{total_runs}: {weather.stem} / {strategy}")
            trace_paths.append(
                run_energyplus_strategy(
                    strategy=strategy,
                    bundle=bundle,
                    idf_path=idf_path,
                    weather_path=weather,
                    eplus_root=eplus_root,
                    out_dir=output_dir,
                    begin_month=begin_month,
                    begin_day=begin_day,
                    end_month=end_month,
                    end_day=end_day,
                    resume=resume,
                    purge_energyplus_after_trace=purge_energyplus_after_trace,
                    trace_format=trace_format,
                )
            )
    if not skip_combined_trace:
        combined_path = output_dir / "traces" / "medium_office_control_traces.csv"
        write_combined_traces(trace_paths, combined_path)
        print(f"[simulate] wrote combined traces: {combined_path}")
    return trace_paths


def write_combined_traces(trace_paths: list[Path], combined_path: Path) -> None:
    """Concatenate trace CSVs using the union of columns.

    Paper B smoke tests intentionally mix scalar and probabilistic controllers.
    Scalar traces do not have zone-probability fields, so exact-header matching is
    too strict here.
    """
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    headers: list[str] = []
    for trace_path in trace_paths:
        cols = list(read_trace_table(trace_path, nrows=0).columns)
        for col in cols:
            if col not in headers:
                headers.append(col)
    with combined_path.open("w", encoding="utf-8", newline="") as out:
        out.write(",".join(headers) + "\n")
        for trace_path in trace_paths:
            if trace_path.suffix == ".parquet":
                chunk = read_trace_table(trace_path).reindex(columns=headers)
                chunk.to_csv(out, index=False, header=False)
            else:
                for chunk in pd.read_csv(trace_path, chunksize=10000):
                    chunk = chunk.reindex(columns=headers)
                    chunk.to_csv(out, index=False, header=False)


def add_trace_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "calendar_year" in df and df["calendar_year"].notna().any():
        years = df["calendar_year"].replace(0, 2001).fillna(2001).astype(int)
    else:
        years = pd.Series(2001, index=df.index)
    hour_float = df["current_time"].fillna(df["hour"]).astype(float)
    hour = np.floor(hour_float).astype(int)
    minute = np.rint((hour_float - hour) * 60).astype(int)
    hour = np.clip(hour, 0, 23)
    minute = np.clip(minute, 0, 59)
    df["timestamp"] = pd.to_datetime(
        {
            "year": years,
            "month": df["month"].astype(int),
            "day": df["day"].astype(int),
            "hour": hour,
            "minute": minute,
        },
        errors="coerce",
    )
    fallback = pd.Timestamp(2001, 1, 1) + pd.to_timedelta(
        np.arange(len(df)) * 15, unit="min"
    )
    df["timestamp"] = df["timestamp"].fillna(pd.Series(fallback, index=df.index))
    return df


def select_plot_window(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    if df.empty:
        return df
    reference = df[df["strategy"] == "reference"].copy()
    if reference.empty:
        reference = df.copy()
    reference["date"] = reference["timestamp"].dt.floor("D")
    daily = reference.groupby("date")["outdoor_temp_c"].mean().sort_index()
    if len(daily) <= days:
        start = daily.index.min()
    else:
        rolling = daily.rolling(days, min_periods=days).mean().dropna()
        end = rolling.idxmax()
        start = end - pd.Timedelta(days=days - 1)
    stop = start + pd.Timedelta(days=days)
    return df[(df["timestamp"] >= start) & (df["timestamp"] < stop)].copy()


def make_temporal_plot(output_dir: Path, weather_stem: str) -> Path:
    combined_path = output_dir / "traces" / "medium_office_control_traces.csv"
    df = pd.read_csv(combined_path)
    df = df[df["weather"] == weather_stem].copy()
    df = add_trace_datetime(df)
    df = select_plot_window(df, days=7)

    strategies = ["reference", "diagnostic_reference", "pmv", "nominal", "ordinal"]
    titles = {
        "reference": "Reference",
        "diagnostic_reference": "Diagnostic Reference",
        "pmv": "PMV",
        "nominal": "Nominal Prob.",
        "ordinal": "Ordinal Prob.",
    }
    fig, axes = plt.subplots(
        4,
        len(strategies),
        figsize=(17.5, 10.5),
        sharex=True,
    )
    fig.subplots_adjust(top=0.86, hspace=0.22, wspace=0.16)

    for col, strategy in enumerate(strategies):
        sdf = df[df["strategy"] == strategy].sort_values("timestamp")
        if sdf.empty:
            continue
        x = sdf["timestamp"]

        ax = axes[0, col]
        ax.fill_between(
            x,
            sdf["comfort_low_c"],
            sdf["comfort_high_c"],
            color="#d8ead2",
            alpha=0.55,
            linewidth=0,
            label="Adaptive 90% band",
        )
        ax.plot(x, sdf["outdoor_temp_c"], color="#8f3d2f", lw=1.0, alpha=0.85, label="Outdoor")
        ax.plot(x, sdf["mean_operative_temp_c"], color="#1f5a85", lw=1.4, label="Mean operative")
        ax.plot(x, sdf["heating_setpoint_c"], color="#bf7b30", lw=1.0, ls="--", label="Heating SP")
        ax.plot(x, sdf["cooling_setpoint_c"], color="#455aa0", lw=1.0, ls="--", label="Cooling SP")
        ax.set_title(titles[strategy], fontsize=11, fontweight="bold")
        ax.set_ylabel("Temp. (C)" if col == 0 else "")
        ax.grid(True, color="#d9d9d9", lw=0.5)

        ax = axes[1, col]
        ax.axhspan(-0.5, 0.5, color="#e5f0df", alpha=0.6, linewidth=0)
        ax.plot(x, sdf["mean_pmv"], color="#3f6d3a", lw=1.2, label="PMV")
        if strategy in {"diagnostic_reference", "nominal", "ordinal"}:
            ax2 = ax.twinx()
            ax2.plot(
                x,
                sdf["discomfort_probability"],
                color="#872f59",
                lw=1.1,
                alpha=0.9,
                label="P(|TSV|>=2)",
            )
            ax2.axhline(0.065, color="#872f59", lw=0.8, ls=":")
            ax2.axhline(0.35, color="#872f59", lw=0.8, ls="--")
            ax2.set_ylim(0, max(0.45, float(sdf["discomfort_probability"].max()) * 1.15))
            if col == len(strategies) - 1:
                ax2.set_ylabel("Tail prob.")
        ax.axhline(-0.5, color="#777777", lw=0.7, ls=":")
        ax.axhline(0.5, color="#777777", lw=0.7, ls=":")
        ax.set_ylabel("PMV" if col == 0 else "")
        ax.grid(True, color="#d9d9d9", lw=0.5)

        ax = axes[2, col]
        if strategy in {"diagnostic_reference", "nominal", "ordinal"}:
            ax.plot(x, sdf["expected_tsv"], color="#514c85", lw=1.2, label="E[TSV]")
            ax.axhline(0, color="#666666", lw=0.8)
            ax.set_ylim(-2.8, 2.8)
        else:
            demand = np.where(
                sdf["mean_operative_temp_c"] > sdf["cooling_setpoint_c"],
                1,
                np.where(sdf["mean_operative_temp_c"] < sdf["heating_setpoint_c"], -1, 0),
            )
            ax.step(x, demand, where="post", color="#514c85", lw=1.2, label="Thermostat demand")
            ax.set_ylim(-1.4, 1.4)
        ax.bar(
            x,
            np.where(sdf["hvac_on"], 0.18, 0.0),
            width=0.008,
            bottom=ax.get_ylim()[0],
            color="#2b6f6b",
            alpha=0.35,
            label="HVAC on",
        )
        ax.set_ylabel("Signal" if col == 0 else "")
        ax.grid(True, color="#d9d9d9", lw=0.5)

        ax = axes[3, col]
        step_hours = 0.25
        elec_kw = sdf["electricity_facility_j"] / (step_hours * 3600.0 * 1000.0)
        ax.plot(x, elec_kw, color="#303030", lw=1.0, label="Facility electric")
        ax.set_ylabel("kW" if col == 0 else "")
        ax.grid(True, color="#d9d9d9", lw=0.5)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(axis="x", rotation=30)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        f"Medium Office 15-min supervisory control trace\n{weather_stem}",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    fig_path = output_dir / "figs" / f"medium_office_temporal_grid_{weather_stem}.png"
    pdf_path = fig_path.with_suffix(".pdf")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[plot] wrote figure: {fig_path}")
    print(f"[plot] wrote figure: {pdf_path}")
    return fig_path


def summarize_trace_paths(
    trace_paths: list[Path],
    output_dir: Path,
    purge_case_traces_after_summary: bool = False,
) -> Path:
    rows = []
    for trace_path in trace_paths:
        df = read_trace_table(trace_path)
        for (weather, strategy), sdf in df.groupby(["weather", "strategy"]):
            occ = sdf[sdf["occupied"]]
            p_tail = pd.to_numeric(occ.get("discomfort_probability"), errors="coerce")
            warm_tail = pd.to_numeric(occ.get("warm_discomfort_probability"), errors="coerce")
            cold_tail = pd.to_numeric(occ.get("cold_discomfort_probability"), errors="coerce")
            zone_p_cols = [
                col for col in occ.columns if col.startswith("zone_") and col.endswith("_p_disc")
            ]
            zone_p_tail = (
                occ[zone_p_cols].apply(pd.to_numeric, errors="coerce")
                if zone_p_cols
                else pd.DataFrame(index=occ.index)
            )
            max_zone_p_tail = (
                zone_p_tail.max(axis=1)
                if zone_p_cols
                else pd.Series(np.nan, index=occ.index)
            )
            p90_zone_p_tail = (
                zone_p_tail.quantile(0.90, axis=1)
                if zone_p_cols
                else pd.Series(np.nan, index=occ.index)
            )
            action_delta = pd.to_numeric(sdf.get("action_delta_c"), errors="coerce").fillna(0.0)
            hold_guard = pd.to_numeric(sdf.get("paperb_hold_guard"), errors="coerce").fillna(0)
            action_direction = pd.to_numeric(sdf.get("action_direction"), errors="coerce").fillna(0)
            requested_direction = pd.to_numeric(
                sdf.get("paperb_requested_direction", pd.Series(0, index=sdf.index)),
                errors="coerce",
            ).fillna(0)
            has_explicit_process_events = {
                "paperb_request_branch",
                "paperb_request_outcome",
            }.issubset(sdf.columns)
            request_branch = sdf.get(
                "paperb_request_branch",
                pd.Series("legacy_unrecorded", index=sdf.index),
            ).astype(str)
            request_outcome = sdf.get(
                "paperb_request_outcome",
                pd.Series("legacy_unrecorded", index=sdf.index),
            ).astype(str)
            guard_classification = sdf.get(
                "paperb_guard_classification",
                pd.Series("not_recorded", index=sdf.index),
            ).astype(str)
            heating_rate_w = pd.to_numeric(sdf.get("zone_heating_rate_w"), errors="coerce").fillna(0.0)
            cooling_rate_w = pd.to_numeric(sdf.get("zone_cooling_rate_w"), errors="coerce").fillna(0.0)
            zone_hvac_w = (
                heating_rate_w
                + cooling_rate_w
            )
            step_hours = 0.25
            elec_hvac_j = pd.to_numeric(
                sdf.get("electricity_hvac_j", pd.Series(0.0, index=sdf.index)),
                errors="coerce",
            ).fillna(0.0)
            elec_cooling_j = pd.to_numeric(
                sdf.get("electricity_cooling_j", pd.Series(0.0, index=sdf.index)),
                errors="coerce",
            ).fillna(0.0)
            elec_heating_j = pd.to_numeric(
                sdf.get("electricity_heating_j", pd.Series(0.0, index=sdf.index)),
                errors="coerce",
            ).fillna(0.0)
            elec_fans_j = pd.to_numeric(
                sdf.get("electricity_fans_j", pd.Series(0.0, index=sdf.index)),
                errors="coerce",
            ).fillna(0.0)
            elec_pumps_j = pd.to_numeric(
                sdf.get("electricity_pumps_j", pd.Series(0.0, index=sdf.index)),
                errors="coerce",
            ).fillna(0.0)
            rows.append(
                {
                    "weather": weather,
                    "strategy": strategy,
                    "rev01_variant_id": str(sdf["rev01_variant_id"].iloc[0])
                    if "rev01_variant_id" in sdf and len(sdf)
                    else "legacy_unrecorded",
                    "rev01_variant_manifest_sha256": str(
                        sdf["rev01_variant_manifest_sha256"].iloc[0]
                    )
                    if "rev01_variant_manifest_sha256" in sdf and len(sdf)
                    else "",
                    "rev01_effective_config_sha256": str(
                        sdf["rev01_effective_config_sha256"].iloc[0]
                    )
                    if "rev01_effective_config_sha256" in sdf and len(sdf)
                    else "",
                    "sizing_contract": str(sdf["sizing_contract"].iloc[0])
                    if "sizing_contract" in sdf and len(sdf)
                    else "legacy_unrecorded",
                    "controller_semantics": str(sdf["controller_semantics"].iloc[0])
                    if "controller_semantics" in sdf and len(sdf)
                    else "legacy_unrecorded",
                    "inference_preprocessing": str(sdf["inference_preprocessing"].iloc[0])
                    if "inference_preprocessing" in sdf and len(sdf)
                    else "legacy_unrecorded",
                    "n_steps": int(len(sdf)),
                    "occupied_steps": int(len(occ)),
                    "paperb_met": float(pd.to_numeric(sdf["paperb_met"], errors="coerce").mean())
                    if "paperb_met" in sdf
                    else float("nan"),
                    "paperb_people_activity_w_per_person": float(
                        pd.to_numeric(
                            sdf["paperb_people_activity_w_per_person"], errors="coerce"
                        ).mean()
                    )
                    if "paperb_people_activity_w_per_person" in sdf
                    else float("nan"),
                    "paperb_spatial_quantile": float(
                        pd.to_numeric(sdf["paperb_spatial_quantile"], errors="coerce").mean()
                    )
                    if "paperb_spatial_quantile" in sdf
                    else float("nan"),
                    "paperb_signal_alpha": float(
                        pd.to_numeric(sdf["paperb_signal_alpha"], errors="coerce").mean()
                    )
                    if "paperb_signal_alpha" in sdf
                    else float("nan"),
                    "paperb_tail_threshold": float(
                        pd.to_numeric(sdf["paperb_tail_threshold"], errors="coerce").mean()
                    )
                    if "paperb_tail_threshold" in sdf
                    else float("nan"),
                    "paperb_asym_threshold": float(
                        pd.to_numeric(sdf["paperb_asym_threshold"], errors="coerce").mean()
                    )
                    if "paperb_asym_threshold" in sdf
                    else float("nan"),
                    "paperb_save_heat_c": float(
                        pd.to_numeric(sdf["paperb_save_heat_c"], errors="coerce").mean()
                    )
                    if "paperb_save_heat_c" in sdf
                    else float("nan"),
                    "paperb_save_cool_c": float(
                        pd.to_numeric(sdf["paperb_save_cool_c"], errors="coerce").mean()
                    )
                    if "paperb_save_cool_c" in sdf
                    else float("nan"),
                    "mean_operative_temp_c_occ": float(occ["mean_operative_temp_c"].mean()),
                    "mean_pmv_occ": float(pd.to_numeric(occ["mean_pmv"], errors="coerce").mean()),
                    "pmv_violation_pct_occ": float(
                        (occ["mean_pmv"].abs() > 0.5).mean() * 100.0
                    ),
                    "pmv_abs_gt_1_0_pct_occ": float(
                        (pd.to_numeric(occ["mean_pmv"], errors="coerce").abs() > 1.0).mean()
                        * 100.0
                    ),
                    "mean_ppd_pct_occ": float(
                        pd.to_numeric(occ["mean_ppd_pct"], errors="coerce").mean()
                    )
                    if "mean_ppd_pct" in occ
                    else float("nan"),
                    "ppd_gt_25_pct_occ": float(
                        (pd.to_numeric(occ["mean_ppd_pct"], errors="coerce") > 25.0).mean()
                        * 100.0
                    )
                    if "mean_ppd_pct" in occ
                    else float("nan"),
                    "adaptive_violation_pct_occ": float(
                        (
                            (occ["mean_operative_temp_c"] < occ["comfort_low_c"])
                            | (occ["mean_operative_temp_c"] > occ["comfort_high_c"])
                        ).mean()
                        * 100.0
                    ),
                    "mean_p_tail_occ": float(p_tail.mean()),
                    "mean_warm_tail_occ": float(warm_tail.mean()),
                    "mean_cold_tail_occ": float(cold_tail.mean()),
                    "p95_p_tail_occ": float(p_tail.quantile(0.95)),
                    "mean_zone_p_tail_ge_0p20_pct_occ": float((p_tail >= 0.20).mean() * 100.0),
                    "max_zone_p_tail_mean_occ": float(max_zone_p_tail.mean()),
                    "max_zone_p_tail_ge_0p20_pct_occ": float(
                        (max_zone_p_tail >= 0.20).mean() * 100.0
                    ),
                    "p90_zone_p_tail_mean_occ": float(p90_zone_p_tail.mean()),
                    "p90_zone_p_tail_ge_0p20_pct_occ": float(
                        (p90_zone_p_tail >= 0.20).mean() * 100.0
                    ),
                    "process_event_schema": (
                        "explicit_branch_outcome_v2"
                        if has_explicit_process_events
                        else "legacy_inferred"
                    ),
                    "relax_request_count": int((request_branch == "relax").sum())
                    if has_explicit_process_events
                    else float("nan"),
                    "relax_applied_count": int(
                        ((request_branch == "relax") & (request_outcome == "applied")).sum()
                    )
                    if has_explicit_process_events
                    else int((action_direction == 2).sum()),
                    "relax_action_count": int((action_direction == 2).sum()),
                    "guard_hold_count": int(hold_guard.astype(bool).sum()),
                    "risk_hold_count": int((request_outcome == "hold").sum())
                    if has_explicit_process_events
                    else int(hold_guard.astype(bool).sum()),
                    "warm_protection_count": int((action_direction == 1).sum()),
                    "cold_protection_count": int((action_direction == -1).sum()),
                    "warm_protection_request_count": int((requested_direction == 1).sum()),
                    "cold_protection_request_count": int((requested_direction == -1).sum()),
                    "warm_protection_applied_count": int(
                        (
                            (request_branch == "warm_protection")
                            & (request_outcome == "applied")
                        ).sum()
                    )
                    if has_explicit_process_events
                    else int((action_direction == 1).sum()),
                    "cold_protection_applied_count": int(
                        (
                            (request_branch == "cold_protection")
                            & (request_outcome == "applied")
                        ).sum()
                    )
                    if has_explicit_process_events
                    else int((action_direction == -1).sum()),
                    "dwell_blocked_protection_request_count": int(
                        (
                            (
                                (request_branch == "warm_protection")
                                | (request_branch == "cold_protection")
                            )
                            & (request_outcome == "dwell_blocked")
                        ).sum()
                    )
                    if has_explicit_process_events
                    else int(
                        (
                            ((requested_direction == 1) | (requested_direction == -1))
                            & (action_direction == 0)
                        ).sum()
                    ),
                    "dwell_blocked_request_count": int(
                        (request_outcome == "dwell_blocked").sum()
                    )
                    if has_explicit_process_events
                    else float("nan"),
                    "bound_saturated_request_count": int(
                        (request_outcome == "bound_saturated").sum()
                    )
                    if has_explicit_process_events
                    else float("nan"),
                    "setpoint_move_applied_count": int((request_outcome == "applied").sum())
                    if has_explicit_process_events
                    else int(action_direction.isin([-1, 1, 2]).sum()),
                    "no_setpoint_move_count": int((request_outcome != "applied").sum())
                    if has_explicit_process_events
                    else int((action_direction == 0).sum()),
                    "unclassified_no_move_count": int((request_outcome == "no_move").sum())
                    if has_explicit_process_events
                    else float("nan"),
                    "relax_dwell_blocked_count": int(
                        (
                            (request_branch == "relax")
                            & (request_outcome == "dwell_blocked")
                        ).sum()
                    )
                    if has_explicit_process_events
                    else float("nan"),
                    "relax_bound_saturated_count": int(
                        (
                            (request_branch == "relax")
                            & (request_outcome == "bound_saturated")
                        ).sum()
                    )
                    if has_explicit_process_events
                    else float("nan"),
                    "protection_bound_saturated_count": int(
                        (
                            request_branch.isin(["warm_protection", "cold_protection"])
                            & (request_outcome == "bound_saturated")
                        ).sum()
                    )
                    if has_explicit_process_events
                    else float("nan"),
                    "ambiguous_high_tail_step_count": int(
                        guard_classification.str.startswith("high_tail_ambiguous").sum()
                    ),
                    "hvac_on_pct_all": float(sdf["hvac_on"].mean() * 100.0),
                    "mean_heating_setpoint_c": float(
                        pd.to_numeric(sdf["heating_setpoint_c"], errors="coerce").mean()
                    ),
                    "mean_cooling_setpoint_c": float(
                        pd.to_numeric(sdf["cooling_setpoint_c"], errors="coerce").mean()
                    ),
                    "electricity_kwh": float(sdf["electricity_facility_j"].sum() / 3.6e6),
                    "hvac_electricity_kwh": float(elec_hvac_j.sum() / 3.6e6),
                    "cooling_electricity_kwh": float(elec_cooling_j.sum() / 3.6e6),
                    "heating_electricity_kwh": float(elec_heating_j.sum() / 3.6e6),
                    "fan_electricity_kwh": float(elec_fans_j.sum() / 3.6e6),
                    "pump_electricity_kwh": float(elec_pumps_j.sum() / 3.6e6),
                    "natural_gas_kwh": float(sdf["natural_gas_facility_j"].sum() / 3.6e6),
                    "zone_heating_kwh_thermal": float(heating_rate_w.sum() * step_hours / 1000.0),
                    "zone_cooling_kwh_thermal": float(cooling_rate_w.sum() * step_hours / 1000.0),
                    "zone_hvac_kwh_thermal": float(zone_hvac_w.sum() * step_hours / 1000.0),
                    "peak_facility_electric_kw_15min": float(
                        pd.to_numeric(sdf["electricity_facility_j"], errors="coerce").max()
                        / 900000.0
                    ),
                    "peak_hvac_electric_kw_15min": float(elec_hvac_j.max() / 900000.0),
                    "peak_cooling_electric_kw_15min": float(elec_cooling_j.max() / 900000.0),
                    "peak_fan_electric_kw_15min": float(elec_fans_j.max() / 900000.0),
                    "peak_zone_hvac_kw_15min": float(zone_hvac_w.max() / 1000.0),
                }
            )
    out = output_dir / "summary" / "medium_office_trace_summary.csv"
    if out.exists():
        raise FileExistsError(f"The v2 runner refuses to replace summary output: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["weather", "strategy"]).to_csv(out, index=False)
    print(f"[summary] wrote: {out}")
    if purge_case_traces_after_summary:
        for trace_path in trace_paths:
            if trace_path.exists():
                trace_path.unlink()
        print(f"[summary] purged {len(trace_paths)} per-case trace tables")
    return out


def select_weather_from_manifest(manifest_path: Path, stage: str) -> list[Path]:
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path)
    stage_col = {
        "smoke": "stage_smoke",
        "typical": "stage_typical",
        "full": "stage_full",
    }[stage]
    selected = manifest[manifest[stage_col].astype(bool)].copy()
    if selected.empty:
        raise ValueError(f"No weather cases selected for stage={stage!r}.")
    paths = [Path(path) for path in selected["epw_path"]]
    missing = [path for path in paths if not path.exists()]
    if missing:
        preview = "\n".join(str(path) for path in missing[:20])
        raise FileNotFoundError(f"Missing EPW files:\n{preview}")
    print(f"[manifest] selected {len(paths)} {stage} weather cases from {manifest_path}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--idf", type=Path, default=DEFAULT_IDF)
    parser.add_argument("--eplus-root", type=Path, default=DEFAULT_EPLUS)
    parser.add_argument("--weather", type=Path, nargs="+", default=[DEFAULT_WEATHER])
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--stage", choices=["smoke", "typical", "full"], default="full")
    parser.add_argument("--begin-month", type=int, default=7)
    parser.add_argument("--begin-day", type=int, default=1)
    parser.add_argument("--end-month", type=int, default=7)
    parser.add_argument("--end-day", type=int, default=14)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Load/store the predictor bundle at this exact path instead of output-dir/models.",
    )
    parser.add_argument(
        "--model-metrics-path",
        type=Path,
        default=None,
        help="Exact metrics/provenance file paired with --model-path.",
    )
    parser.add_argument(
        "--feature-set",
        choices=["full", "no_pmv"],
        default="full",
        help="Predictor feature set. 'no_pmv' trains/applies the ordinal model without PMV as an input feature.",
    )
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-sim", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--purge-energyplus-after-trace",
        action="store_true",
        help=(
            "Delete each per-case EnergyPlus output directory after its trace CSV has "
            "been written successfully. Keeps traces, combined traces, models, and summaries."
        ),
    )
    parser.add_argument(
        "--skip-combined-trace",
        action="store_true",
        help=(
            "Do not concatenate per-case trace CSVs into one large combined trace. "
            "Use this for annual/full-scale panels where the compact summary is the "
            "primary retained artifact."
        ),
    )
    parser.add_argument(
        "--purge-case-traces-after-summary",
        action="store_true",
        help=(
            "Delete per-case trace tables after medium_office_trace_summary.csv is "
            "written. This keeps only summary metrics, model artifacts, and optional "
            "figures."
        ),
    )
    parser.add_argument(
        "--trace-format",
        choices=["csv", "parquet"],
        default="csv",
        help="Per-case trace table format. Parquet uses Zstandard compression.",
    )
    parser.add_argument("--paperb-save-heat-c", type=float, default=PAPERB_SAVE_HEAT_C)
    parser.add_argument("--paperb-save-cool-c", type=float, default=PAPERB_SAVE_COOL_C)
    parser.add_argument("--paperb-warm-protect-cool-c", type=float, default=PAPERB_WARM_PROTECT_COOL_C)
    parser.add_argument("--paperb-cold-protect-heat-c", type=float, default=PAPERB_COLD_PROTECT_HEAT_C)
    parser.add_argument("--paperb-tighten-dwell-steps", type=int, default=PAPERB_TIGHTEN_DWELL_STEPS)
    parser.add_argument("--paperb-relax-dwell-steps", type=int, default=PAPERB_RELAX_DWELL_STEPS)
    parser.add_argument("--paperb-tail-threshold", type=float, default=PAPERB_TAIL_THRESHOLD)
    parser.add_argument("--paperb-asym-threshold", type=float, default=PAPERB_ASYM_THRESHOLD)
    parser.add_argument("--paperb-pmv-threshold", type=float, default=PAPERB_PMV_THRESHOLD)
    parser.add_argument(
        "--paperb-spatial-quantile",
        type=float,
        default=PAPERB_SPATIAL_QUANTILE,
    )
    parser.add_argument(
        "--paperb-signal-alpha",
        type=float,
        default=PAPERB_SIGNAL_ALPHA,
    )
    parser.add_argument(
        "--paperb-adaptive-rm-clamp-low-c",
        type=float,
        default=PAPERB_ADAPTIVE_RM_CLAMP_LOW_C,
    )
    parser.add_argument(
        "--paperb-adaptive-rm-clamp-high-c",
        type=float,
        default=PAPERB_ADAPTIVE_RM_CLAMP_HIGH_C,
    )
    parser.add_argument(
        "--paperb-pmv-extreme-threshold",
        type=float,
        default=PAPERB_PMV_EXTREME_THRESHOLD,
    )
    parser.add_argument("--paperb-ppd-hold-threshold", type=float, default=PAPERB_PPD_HOLD_THRESHOLD)
    parser.add_argument(
        "--paperb-ppd-protect-threshold",
        type=float,
        default=PAPERB_PPD_PROTECT_THRESHOLD,
    )
    parser.add_argument("--paperb-tsv-threshold", type=float, default=PAPERB_TSV_THRESHOLD)
    parser.add_argument(
        "--controller-semantics",
        choices=["legacy", "corrected"],
        default="legacy",
        help=(
            "Legacy relaxes when tail mass is high but warm/cold asymmetry is "
            "ambiguous; corrected holds in that state."
        ),
    )
    parser.add_argument(
        "--inference-preprocessing",
        choices=["legacy", "training_contract"],
        default="legacy",
        help=(
            "Legacy reproduces the submitted runtime without training-time winsorization; "
            "training_contract applies the stored 1st/99th-percentile raw-input bounds."
        ),
    )
    parser.add_argument(
        "--paperb-met",
        type=float,
        default=PAPERB_MET,
        help="Uniform metabolic rate in met units used for PMV and TSV-probability evaluation.",
    )
    parser.add_argument(
        "--paperb-people-activity-w",
        type=float,
        default=PAPERB_PEOPLE_ACTIVITY_W_PER_PERSON,
        help=(
            "EnergyPlus People activity level in W/person. This patches ACTIVITY_SCH, "
            "which supplies occupant sensible/latent heat gains in the Medium Office IDF."
        ),
    )
    parser.add_argument("--rev01-variant-manifest", type=Path, default=None)
    parser.add_argument("--rev01-variant-id", type=str, default=None)
    parser.add_argument(
        "--sizing-contract",
        choices=["accepted_denver", "city_ddy"],
        default="accepted_denver",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["diagnostic_reference"],
        choices=[
            "reference",
            "diagnostic_reference",
            "pmv",
            "nominal",
            "ordinal",
            "grid_naive",
            "grid_gated",
            "paperb_pmv_relax",
            "paperb_adaptive_band_relax",
            "paperb_ppd_guard_relax",
            "paperb_pmv_exceedance_guard_relax",
            "paperb_pmv_extreme_guard_relax",
            "paperb_mu_relax",
            "paperb_gate_tail_asym_relax",
            "paperb_p90_tail_asym_relax",
            "paperb_p90_abs_pmv_relax",
            "paperb_adaptive_band_clamped_relax",
        ],
    )
    return parser.parse_args()


def main() -> int:
    global PAPERB_SAVE_HEAT_C
    global PAPERB_SAVE_COOL_C
    global PAPERB_WARM_PROTECT_COOL_C
    global PAPERB_COLD_PROTECT_HEAT_C
    global PAPERB_TIGHTEN_DWELL_STEPS
    global PAPERB_RELAX_DWELL_STEPS
    global PAPERB_TAIL_THRESHOLD
    global PAPERB_ASYM_THRESHOLD
    global PAPERB_PMV_THRESHOLD
    global PAPERB_PMV_EXTREME_THRESHOLD
    global PAPERB_PPD_HOLD_THRESHOLD
    global PAPERB_PPD_PROTECT_THRESHOLD
    global PAPERB_TSV_THRESHOLD
    global PAPERB_MET
    global PAPERB_PEOPLE_ACTIVITY_W_PER_PERSON
    global CONTROLLER_SEMANTICS
    global INFERENCE_PREPROCESSING
    global PAPERB_SPATIAL_QUANTILE
    global PAPERB_SIGNAL_ALPHA
    global PAPERB_ADAPTIVE_RM_CLAMP_LOW_C
    global PAPERB_ADAPTIVE_RM_CLAMP_HIGH_C
    global REV01_VARIANT_ID
    global REV01_VARIANT_MANIFEST_SHA256
    global REV01_EFFECTIVE_CONFIG_SHA256
    global SIZING_CONTRACT

    args = parse_args()
    PAPERB_SAVE_HEAT_C = float(args.paperb_save_heat_c)
    PAPERB_SAVE_COOL_C = float(args.paperb_save_cool_c)
    PAPERB_WARM_PROTECT_COOL_C = float(args.paperb_warm_protect_cool_c)
    PAPERB_COLD_PROTECT_HEAT_C = float(args.paperb_cold_protect_heat_c)
    PAPERB_TIGHTEN_DWELL_STEPS = int(args.paperb_tighten_dwell_steps)
    PAPERB_RELAX_DWELL_STEPS = int(args.paperb_relax_dwell_steps)
    PAPERB_TAIL_THRESHOLD = float(args.paperb_tail_threshold)
    PAPERB_ASYM_THRESHOLD = float(args.paperb_asym_threshold)
    PAPERB_PMV_THRESHOLD = float(args.paperb_pmv_threshold)
    PAPERB_SPATIAL_QUANTILE = validate_probability(
        args.paperb_spatial_quantile, "paperb_spatial_quantile"
    )
    PAPERB_SIGNAL_ALPHA = validate_probability(
        args.paperb_signal_alpha, "paperb_signal_alpha"
    )
    PAPERB_ADAPTIVE_RM_CLAMP_LOW_C = float(args.paperb_adaptive_rm_clamp_low_c)
    PAPERB_ADAPTIVE_RM_CLAMP_HIGH_C = float(args.paperb_adaptive_rm_clamp_high_c)
    PAPERB_PMV_EXTREME_THRESHOLD = float(args.paperb_pmv_extreme_threshold)
    PAPERB_PPD_HOLD_THRESHOLD = float(args.paperb_ppd_hold_threshold)
    PAPERB_PPD_PROTECT_THRESHOLD = float(args.paperb_ppd_protect_threshold)
    PAPERB_TSV_THRESHOLD = float(args.paperb_tsv_threshold)
    PAPERB_MET = float(args.paperb_met)
    PAPERB_PEOPLE_ACTIVITY_W_PER_PERSON = float(args.paperb_people_activity_w)
    CONTROLLER_SEMANTICS = str(args.controller_semantics)
    INFERENCE_PREPROCESSING = str(args.inference_preprocessing)
    SIZING_CONTRACT = str(args.sizing_contract)

    validate_probability(PAPERB_TAIL_THRESHOLD, "paperb_tail_threshold")
    validate_probability(PAPERB_ASYM_THRESHOLD, "paperb_asym_threshold")
    if PAPERB_TIGHTEN_DWELL_STEPS < 1 or PAPERB_RELAX_DWELL_STEPS < 1:
        raise VariantContractError("controller dwell steps must be positive integers")
    if not (math.isfinite(PAPERB_MET) and PAPERB_MET > 0.0):
        raise VariantContractError("paperb_met must be positive and finite")
    if not (
        math.isfinite(PAPERB_PEOPLE_ACTIVITY_W_PER_PERSON)
        and PAPERB_PEOPLE_ACTIVITY_W_PER_PERSON > 0.0
    ):
        raise VariantContractError("paperb_people_activity_w must be positive and finite")
    if not (
        math.isfinite(PAPERB_ADAPTIVE_RM_CLAMP_LOW_C)
        and math.isfinite(PAPERB_ADAPTIVE_RM_CLAMP_HIGH_C)
        and PAPERB_ADAPTIVE_RM_CLAMP_LOW_C < PAPERB_ADAPTIVE_RM_CLAMP_HIGH_C
    ):
        raise VariantContractError("adaptive clamp bounds must be finite and ordered")
    validate_controller_bounds(
        reference_heat_c=PAPERB_REF_HEAT_C,
        reference_cool_c=PAPERB_REF_COOL_C,
        save_heat_c=PAPERB_SAVE_HEAT_C,
        save_cool_c=PAPERB_SAVE_COOL_C,
        warm_protect_cool_c=PAPERB_WARM_PROTECT_COOL_C,
        cold_protect_heat_c=PAPERB_COLD_PROTECT_HEAT_C,
    )

    if (args.rev01_variant_manifest is None) != (args.rev01_variant_id is None):
        raise VariantContractError(
            "--rev01-variant-manifest and --rev01-variant-id must be supplied together"
        )
    variant_audit: dict[str, Any] | None = None
    if args.rev01_variant_manifest is not None:
        if len(args.strategies) != 1:
            raise VariantContractError("a Rev01 manifest run must contain exactly one strategy")
        effective_config = {
            "strategy": str(args.strategies[0]),
            "spatial_quantile": PAPERB_SPATIAL_QUANTILE,
            "signal_alpha": PAPERB_SIGNAL_ALPHA,
            "tail_threshold": PAPERB_TAIL_THRESHOLD,
            "asym_threshold": PAPERB_ASYM_THRESHOLD,
            "save_heat_c": PAPERB_SAVE_HEAT_C,
            "save_cool_c": PAPERB_SAVE_COOL_C,
            "warm_protect_cool_c": PAPERB_WARM_PROTECT_COOL_C,
            "cold_protect_heat_c": PAPERB_COLD_PROTECT_HEAT_C,
            "tighten_dwell_steps": PAPERB_TIGHTEN_DWELL_STEPS,
            "relax_dwell_steps": PAPERB_RELAX_DWELL_STEPS,
            "pmv_threshold": PAPERB_PMV_THRESHOLD,
            "adaptive_rm_clamp_low_c": PAPERB_ADAPTIVE_RM_CLAMP_LOW_C,
            "adaptive_rm_clamp_high_c": PAPERB_ADAPTIVE_RM_CLAMP_HIGH_C,
            "paperb_met": PAPERB_MET,
            "people_activity_w_per_person": PAPERB_PEOPLE_ACTIVITY_W_PER_PERSON,
            "controller_semantics": CONTROLLER_SEMANTICS,
            "inference_preprocessing": INFERENCE_PREPROCESSING,
            "sizing_contract": SIZING_CONTRACT,
        }
        variant_audit = validate_variant_manifest(
            args.rev01_variant_manifest.resolve(),
            variant_id=str(args.rev01_variant_id),
            effective_config=effective_config,
            source_idf_path=args.idf.resolve(),
        )
        REV01_VARIANT_ID = str(args.rev01_variant_id)
        REV01_VARIANT_MANIFEST_SHA256 = str(variant_audit["manifest_sha256"])
        REV01_EFFECTIVE_CONFIG_SHA256 = str(
            variant_audit["effective_config_sha256"]
        )

    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if variant_audit is not None:
        (args.output_dir / "REV01_VARIANT_AUDIT.json").write_text(
            json.dumps(variant_audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.manifest is not None:
        args.weather = select_weather_from_manifest(args.manifest, args.stage)
    feature_columns = (
        FEATURE_COLUMNS_NO_PMV if args.feature_set == "no_pmv" else FEATURE_COLUMNS_FULL
    )
    model_suffix = "" if args.feature_set == "full" else f"_{args.feature_set}"
    model_path = (
        args.model_path.resolve()
        if args.model_path is not None
        else args.output_dir / "models" / f"control_predictors{model_suffix}.joblib"
    )
    metrics_path = (
        args.model_metrics_path.resolve()
        if args.model_metrics_path is not None
        else args.output_dir / "models" / f"control_predictor_metrics{model_suffix}.json"
    )
    if args.model_path is not None and args.retrain:
        raise VariantContractError("--retrain cannot overwrite an externally supplied model")
    if args.model_metrics_path is not None and not metrics_path.is_file():
        raise FileNotFoundError(f"Missing model metrics artifact: {metrics_path}")

    bundle = None
    needs_predictors = any(
        s
        in {
            "diagnostic_reference",
            "nominal",
            "ordinal",
            "grid_gated",
            "paperb_pmv_relax",
            "paperb_adaptive_band_relax",
            "paperb_ppd_guard_relax",
            "paperb_pmv_exceedance_guard_relax",
            "paperb_pmv_extreme_guard_relax",
            "paperb_mu_relax",
            "paperb_gate_tail_asym_relax",
            "paperb_p90_tail_asym_relax",
            "paperb_p90_abs_pmv_relax",
            "paperb_adaptive_band_clamped_relax",
        }
        for s in args.strategies
    )
    if needs_predictors:
        if args.skip_train and not model_path.exists():
            raise FileNotFoundError(f"Missing model artifact: {model_path}")
        if args.retrain or not model_path.exists():
            bundle = train_predictors(
                data_path=args.data,
                model_path=model_path,
                metrics_path=metrics_path,
                n_estimators=args.n_estimators,
                sample_limit=args.sample_limit,
                feature_columns=feature_columns,
            )
        else:
            print(f"[train] loading existing model: {model_path}")
            bundle = joblib.load(model_path)

    if not args.skip_sim:
        trace_paths = run_simulations(
            bundle=bundle,
            output_dir=args.output_dir,
            source_idf=args.idf,
            weather_paths=args.weather,
            eplus_root=args.eplus_root,
            begin_month=args.begin_month,
            begin_day=args.begin_day,
            end_month=args.end_month,
            end_day=args.end_day,
            strategies=args.strategies,
            resume=args.resume,
            purge_energyplus_after_trace=args.purge_energyplus_after_trace,
            skip_combined_trace=args.skip_combined_trace,
            trace_format=args.trace_format,
        )
        summarize_trace_paths(
            trace_paths,
            args.output_dir,
            purge_case_traces_after_summary=args.purge_case_traces_after_summary,
        )

    if not args.skip_plot:
        for weather in args.weather:
            make_temporal_plot(args.output_dir, weather.stem)

    print(f"[done] elapsed minutes: {(time.time() - start) / 60.0:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
