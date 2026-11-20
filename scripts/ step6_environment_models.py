#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 6 — Environmental response relationships of Zhejiang cephalopods.

This Kaggle-ready script reads the Step-3 analysis master ZIP and performs:

1. Season-specific (April / November) environmental preprocessing.
2. LocationID-by-season climatological means and interannual anomalies for
   SST, SSS and log1p chlorophyll-a.
3. Spline GEE models with exchangeable within-LocationID correlation:
      * occurrence              : binomial-logit
      * total biomass incl. 0   : Tweedie-log
      * taxon richness        : negative-binomial-log
      * positive biomass        : Gamma-log sensitivity model
4. Joint Wald tests for each spline, response curves and grouped CV.
5. Sensitivity models: year trend, Chla >=3 source pixels, no-Chla (n=2100),
   and a raw four-environment-variable association model.
6. Positive-event, all-25-taxa environmental ordination:
      * Bray-Curtis + sqrt-distance euclidification (Lingoes fallback)
      * restricted temporal permutation tests within LocationID
      * station-level spatial depth permutation test

Important interpretation:
    These are observational response relationships, not causal effects.
    Main inference uses environmental anomalies plus depth, separately by
    season. The raw four-variable model is association sensitivity only because
    its predictors are strongly collinear.

Typical Kaggle use:
    import os, runpy
    os.environ["CEPHALOPOD_STEP3_ZIP"] = \
        "/kaggle/input/<dataset>/cephalopod_step3_analysis_master.zip"
    runpy.run_path(
        "/kaggle/input/<code-dataset>/kaggle_step6_environment_response_analysis.py",
        run_name="__main__",
    )

The script intentionally does not install packages during an analysis run.
Kaggle normally includes the required stack. If the dependency check fails,
run the installation command printed by the error in a separate first cell,
restart the session, and then rerun this script.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from packaging.version import Version
from scipy import linalg, stats
from scipy.spatial.distance import pdist, squareform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_gamma_deviance,
    mean_poisson_deviance,
    mean_tweedie_deviance,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold


# ---------------------------------------------------------------------------
# Configuration (every costly setting can be overridden through an env var)
# ---------------------------------------------------------------------------

SCRIPT_VERSION = "6.1.0-EI-revision"
RANDOM_SEED = int(os.environ.get("CEPHALOPOD_STEP6_SEED", "20240901"))
SPLINE_DF = int(os.environ.get("CEPHALOPOD_STEP6_SPLINE_DF", "5"))
YEAR_SPLINE_DF = int(os.environ.get("CEPHALOPOD_STEP6_YEAR_SPLINE_DF", "4"))
GEE_MAXITER = int(os.environ.get("CEPHALOPOD_STEP6_GEE_MAXITER", "200"))
GEE_CTOL = float(os.environ.get("CEPHALOPOD_STEP6_GEE_CTOL", "1e-6"))
CV_FOLDS = int(os.environ.get("CEPHALOPOD_STEP6_CV_FOLDS", "5"))
RUN_GROUPED_CV = os.environ.get("CEPHALOPOD_STEP6_RUN_CV", "1") == "1"
PERMUTATIONS = int(os.environ.get("CEPHALOPOD_STEP6_PERMUTATIONS", "999"))
RUN_ORDINATION = os.environ.get("CEPHALOPOD_STEP6_RUN_ORDINATION", "1") == "1"
PCOA_VARIANCE_THRESHOLD = float(
    os.environ.get("CEPHALOPOD_STEP6_PCOA_VARIANCE", "0.99")
)
MAX_PCOA_AXES = int(os.environ.get("CEPHALOPOD_STEP6_MAX_PCOA_AXES", "400"))
SQRT_BRAY_FOR_ORDINATION = (
    os.environ.get("CEPHALOPOD_STEP6_SQRT_BRAY", "1") == "1"
)
FIGURE_DPI = int(os.environ.get("CEPHALOPOD_STEP6_FIGURE_DPI", "180"))
PREP_ONLY = os.environ.get("CEPHALOPOD_STEP6_PREP_ONLY", "0") == "1"

_power_text = os.environ.get(
    "CEPHALOPOD_STEP6_TWEEDIE_POWERS", "1.3,1.5,1.7"
)
TWEEDIE_POWER_GRID = tuple(float(x.strip()) for x in _power_text.split(",") if x.strip())
if not TWEEDIE_POWER_GRID:
    TWEEDIE_POWER_GRID = (1.5,)
DEFAULT_TWEEDIE_POWER = float(
    os.environ.get("CEPHALOPOD_STEP6_DEFAULT_TWEEDIE_POWER", "1.5")
)

BASE_OUTPUT = Path(
    os.environ.get("CEPHALOPOD_STEP6_OUTPUT_ROOT", "/kaggle/working")
).expanduser()
if not BASE_OUTPUT.exists() and str(BASE_OUTPUT).startswith("/kaggle/"):
    BASE_OUTPUT = Path.cwd()

FINAL_OUTPUT_DIR = BASE_OUTPUT / "cephalopod_step6_environment_response"
FINAL_ZIP = BASE_OUTPUT / "cephalopod_step6_environment_response.zip"

SEASON_LABELS = {4: "April", 11: "November"}
RAW_ENV = ["Depth_m", "SST", "SSS", "log1p_Chla_0125"]
ANOM_BASE = ["SST", "SSS", "log1p_Chla_0125"]
RESPONSE_SPECS = {
    "Occurrence": {
        "column": "Cephalopod_Presence",
        "family": "Binomial",
        "positive_only": False,
    },
    "TotalBiomass": {
        "column": "Total_Biomass_kg_km2",
        "family": "Tweedie",
        "positive_only": False,
    },
    "Richness": {
        "column": "Species_Richness",
        "family": "NegativeBinomial",
        "positive_only": False,
    },
    "PositiveBiomass": {
        "column": "Total_Biomass_kg_km2",
        "family": "Gamma",
        "positive_only": True,
    },
}

REQUIRED_ZIP_MEMBERS = {
    "strict": "03_analysis_master_v1_environment_strict_all25.csv",
    "surveyed": "02_analysis_master_v1_surveyed_all25.csv",
    "taxonomy": "06_taxonomy_analysis_dictionary.csv",
    "chla_min3_x": "18_X_environment_ChlaMin3_sensitivity_raw.csv",
}

START_TIME = time.time()


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def elapsed() -> str:
    return f"{(time.time() - START_TIME) / 60:.1f} min"


def log(message: str) -> None:
    print(f"[{elapsed()}] {message}", flush=True)


def env_bool_series(series: pd.Series) -> pd.Series:
    """Robust conversion of common CSV boolean encodings."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def safe_sd(series: pd.Series) -> float:
    value = float(pd.to_numeric(series, errors="coerce").std(ddof=0))
    if not np.isfinite(value) or value <= 1e-12:
        return 1.0
    return value


def atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if frame.shape[1] == 0:
        frame = pd.DataFrame({"Status": pd.Series(dtype="string")})
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving NaNs."""
    values = np.asarray(p_values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if valid.size == 0:
        return result
    order = valid[np.argsort(values[valid])]
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[order] = np.minimum(adjusted, 1.0)
    return result


def ensure_empty_output_dir(final_dir: Path) -> Path:
    """Create an isolated temporary directory next to the final output."""
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(prefix=f".{final_dir.name}_tmp_", dir=str(final_dir.parent))
    )


def publish_output_dir(temp_dir: Path, final_dir: Path) -> None:
    """Atomically publish output while retaining at most one recoverable backup."""
    backup = final_dir.with_name(final_dir.name + "_previous")
    if backup.exists():
        shutil.rmtree(backup)
    if final_dir.exists():
        os.replace(final_dir, backup)
    os.replace(temp_dir, final_dir)


def build_zip(source_dir: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(source_dir))
    with zipfile.ZipFile(tmp) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Output ZIP CRC check failed at {bad}")
    os.replace(tmp, destination)


# ---------------------------------------------------------------------------
# Input discovery and validation
# ---------------------------------------------------------------------------


def zip_has_required_members(path: Path) -> tuple[bool, str]:
    try:
        if not path.is_file() or not zipfile.is_zipfile(path):
            return False, "not a readable ZIP"
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            missing = sorted(set(REQUIRED_ZIP_MEMBERS.values()) - names)
            if missing:
                return False, "missing: " + ", ".join(missing)
            bad = archive.testzip()
            if bad is not None:
                return False, f"CRC failure: {bad}"
        return True, "OK"
    except Exception as exc:  # pragma: no cover - diagnostic path
        return False, f"{type(exc).__name__}: {exc}"


def discover_step3_zip() -> Path:
    explicit = os.environ.get("CEPHALOPOD_STEP3_ZIP", "").strip()
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        ok, reason = zip_has_required_members(candidate)
        if not ok:
            raise FileNotFoundError(
                f"CEPHALOPOD_STEP3_ZIP is invalid: {candidate}\n{reason}"
            )
        return candidate

    roots = [Path.cwd(), BASE_OUTPUT, Path("/kaggle/input")]
    patterns = [
        "cephalopod_step3_analysis_master.zip",
        "cephalopod_step3_analysis_master*.zip",
    ]
    seen: set[Path] = set()
    valid: list[Path] = []
    rejected: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            iterator = root.rglob(pattern) if root.name == "input" else root.glob(pattern)
            for path in iterator:
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                ok, reason = zip_has_required_members(resolved)
                if ok:
                    valid.append(resolved)
                else:
                    rejected.append(f"{resolved}: {reason}")

    if not valid:
        detail = "\n".join(rejected[:10]) or "No candidate ZIP was found."
        raise FileNotFoundError(
            "Could not find a valid cephalopod Step-3 ZIP. Add it as a Kaggle "
            "input or set:\n"
            "os.environ['CEPHALOPOD_STEP3_ZIP'] = '/kaggle/input/.../"
            "cephalopod_step3_analysis_master.zip'\n\n"
            + detail
        )

    # Prefer the exact canonical name, then the largest file, then stable path.
    valid.sort(
        key=lambda p: (
            p.name == "cephalopod_step3_analysis_master.zip",
            p.stat().st_size,
            str(p),
        ),
        reverse=True,
    )
    if len(valid) > 1:
        log("Multiple valid Step-3 ZIPs found; selected: " + str(valid[0]))
    return valid[0]


def read_step3_tables(path: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(path) as archive:
        for key, member in REQUIRED_ZIP_MEMBERS.items():
            frames[key] = pd.read_csv(archive.open(member), low_memory=False)
    return frames


def validate_and_prepare_inputs(
    frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[str], list[dict[str, Any]]]:
    strict = frames["strict"].copy()
    surveyed = frames["surveyed"].copy()
    taxonomy = frames["taxonomy"].copy()
    chla_min3_ids = set(frames["chla_min3_x"]["EventID"].astype(str))

    required_common = {
        "EventID",
        "Year",
        "Month",
        "LocationID",
        "Depth_m",
        "SST",
        "SSS",
        "log1p_Chla_0125",
        "Total_Biomass_kg_km2",
        "Species_Richness",
        "Cephalopod_Presence",
    }
    missing = sorted(required_common - set(strict.columns))
    if missing:
        raise KeyError("Strict master is missing columns: " + ", ".join(missing))

    biomass_cols = [f"B_Species_{i:02d}_kg_km2" for i in range(1, 26)]
    missing_bio = sorted(set(biomass_cols) - set(strict.columns))
    if missing_bio:
        raise KeyError("Strict master is missing species biomass: " + ", ".join(missing_bio))

    for frame in (strict, surveyed):
        frame["EventID"] = frame["EventID"].astype(str)
        frame["LocationID"] = frame["LocationID"].astype(str)
        for column in [
            "Year",
            "Month",
            "Depth_m",
            "SST",
            "SSS",
            "log1p_Chla_0125",
            "Total_Biomass_kg_km2",
            "Species_Richness",
            "Cephalopod_Presence",
        ]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "Use_NoChla_Sensitivity" in surveyed.columns:
        surveyed = surveyed.loc[
            env_bool_series(surveyed["Use_NoChla_Sensitivity"])
        ].copy()
    else:
        surveyed = surveyed.loc[
            surveyed[["Depth_m", "SST", "SSS"]].notna().all(axis=1)
        ].copy()

    strict["Use_ChlaMin3_Sensitivity"] = strict["EventID"].isin(chla_min3_ids)

    # Independent response reconciliation.
    strict_presence_from_biomass = (strict["Total_Biomass_kg_km2"] > 0).astype(int)
    surveyed_presence_from_biomass = (surveyed["Total_Biomass_kg_km2"] > 0).astype(int)

    qc: list[dict[str, Any]] = []

    def add_qc(check: str, passed: bool, value: Any, expected: Any, severity: str) -> None:
        qc.append(
            {
                "Check": check,
                "Pass": bool(passed),
                "Observed": value,
                "Expected": expected,
                "Severity": severity,
            }
        )

    add_qc("Strict EventID unique", strict["EventID"].is_unique, strict["EventID"].nunique(), len(strict), "FATAL")
    add_qc("NoChla EventID unique", surveyed["EventID"].is_unique, surveyed["EventID"].nunique(), len(surveyed), "FATAL")
    add_qc("Strict rows", len(strict) == 2077, len(strict), 2077, "FATAL")
    add_qc("NoChla rows", len(surveyed) == 2100, len(surveyed), 2100, "FATAL")
    add_qc("ChlaMin3 rows", int(strict["Use_ChlaMin3_Sensitivity"].sum()) == 2017, int(strict["Use_ChlaMin3_Sensitivity"].sum()), 2017, "FATAL")
    add_qc("Strict months are April/November", set(strict["Month"].dropna().astype(int)) == {4, 11}, sorted(strict["Month"].dropna().unique().tolist()), [4, 11], "FATAL")
    add_qc("Strict environment complete", not strict[RAW_ENV].isna().any().any(), int(strict[RAW_ENV].isna().sum().sum()), 0, "FATAL")
    add_qc("Strict responses complete", not strict[["Total_Biomass_kg_km2", "Species_Richness", "Cephalopod_Presence"]].isna().any().any(), int(strict[["Total_Biomass_kg_km2", "Species_Richness", "Cephalopod_Presence"]].isna().sum().sum()), 0, "FATAL")
    add_qc("Strict presence reconciles with biomass", bool((strict_presence_from_biomass == strict["Cephalopod_Presence"].astype(int)).all()), int((strict_presence_from_biomass != strict["Cephalopod_Presence"].astype(int)).sum()), 0, "FATAL")
    add_qc("NoChla presence reconciles with biomass", bool((surveyed_presence_from_biomass == surveyed["Cephalopod_Presence"].astype(int)).all()), int((surveyed_presence_from_biomass != surveyed["Cephalopod_Presence"].astype(int)).sum()), 0, "FATAL")
    add_qc("Species biomass nonnegative", bool((strict[biomass_cols] >= 0).all().all()), int((strict[biomass_cols] < 0).sum().sum()), 0, "FATAL")
    add_qc("All 25 taxonomy rows available", len(taxonomy) == 25, len(taxonomy), 25, "FATAL")

    fatal = [row for row in qc if row["Severity"] == "FATAL" and not row["Pass"]]
    if fatal:
        raise AssertionError("Input QC failed:\n" + pd.DataFrame(fatal).to_string(index=False))

    strict["Cephalopod_Presence"] = strict_presence_from_biomass
    surveyed["Cephalopod_Presence"] = surveyed_presence_from_biomass
    return strict, surveyed, taxonomy, chla_min3_ids, qc


# ---------------------------------------------------------------------------
# Environmental transformation, correlations and VIF
# ---------------------------------------------------------------------------


def add_environment_anomalies(frame: pd.DataFrame, include_chla: bool) -> pd.DataFrame:
    result = frame.copy()
    bases = ["SST", "SSS"] + (["log1p_Chla_0125"] if include_chla else [])
    grouping = ["Month", "LocationID"]
    for column in bases:
        climatology = result.groupby(grouping, observed=True)[column].transform("mean")
        result[f"{column}_Climatology"] = climatology
        result[f"{column}_Anomaly"] = result[column] - climatology
    result["Year_rel"] = result["Year"].astype(float) - float(result["Year"].min())
    return result


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    source: str
    raw_predictors: tuple[str, ...]
    model_predictors: tuple[str, ...]
    include_year: bool = False
    inferential_role: str = "Sensitivity"


SCENARIOS = (
    Scenario(
        name="Main_Anomaly",
        description="Primary: depth + within-location season-specific SST/SSS/Chla anomalies",
        source="strict",
        raw_predictors=(
            "Depth_m",
            "SST_Anomaly",
            "SSS_Anomaly",
            "log1p_Chla_0125_Anomaly",
        ),
        model_predictors=(
            "Depth_z",
            "SST_Anomaly_z",
            "SSS_Anomaly_z",
            "log1p_Chla_0125_Anomaly_z",
        ),
        inferential_role="Primary inference",
    ),
    Scenario(
        name="Main_AnomalyPlusYear",
        description="Sensitivity: primary environmental anomaly model plus nonlinear year trend",
        source="strict",
        raw_predictors=(
            "Depth_m",
            "SST_Anomaly",
            "SSS_Anomaly",
            "log1p_Chla_0125_Anomaly",
            "Year_rel",
        ),
        model_predictors=(
            "Depth_z",
            "SST_Anomaly_z",
            "SSS_Anomaly_z",
            "log1p_Chla_0125_Anomaly_z",
            "Year_z",
        ),
        include_year=True,
    ),
    Scenario(
        name="ChlaMin3_Anomaly",
        description="Sensitivity: primary model restricted to Chla cells with >=3 valid source pixels",
        source="chla_min3",
        raw_predictors=(
            "Depth_m",
            "SST_Anomaly",
            "SSS_Anomaly",
            "log1p_Chla_0125_Anomaly",
        ),
        model_predictors=(
            "Depth_z",
            "SST_Anomaly_z",
            "SSS_Anomaly_z",
            "log1p_Chla_0125_Anomaly_z",
        ),
    ),
    Scenario(
        name="NoChla_2100",
        description="Sensitivity: all 2100 surveyed events, depth + SST/SSS anomalies, no Chla",
        source="no_chla",
        raw_predictors=("Depth_m", "SST_Anomaly", "SSS_Anomaly"),
        model_predictors=("Depth_z", "SST_Anomaly_z", "SSS_Anomaly_z"),
    ),
    Scenario(
        name="RawFull4_Association",
        description="Collinearity association sensitivity only: raw depth, SST, SSS and log1p Chla",
        source="strict",
        raw_predictors=("Depth_m", "SST", "SSS", "log1p_Chla_0125"),
        model_predictors=("Depth_z", "SST_z", "SSS_z", "log1p_Chla_0125_z"),
        inferential_role="Association/collinearity sensitivity only",
    ),
)


def standardize_scenario_frame(
    frame: pd.DataFrame, scenario: Scenario, season_month: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.loc[frame["Month"].astype(int) == season_month].copy()
    scaler_rows: list[dict[str, Any]] = []
    raw_to_model = dict(zip(scenario.raw_predictors, scenario.model_predictors))
    for raw_name, model_name in raw_to_model.items():
        mean = float(result[raw_name].mean())
        sd = safe_sd(result[raw_name])
        result[model_name] = (result[raw_name] - mean) / sd
        scaler_rows.append(
            {
                "Scenario": scenario.name,
                "Month": season_month,
                "Season": SEASON_LABELS[season_month],
                "RawVariable": raw_name,
                "ModelVariable": model_name,
                "Mean": mean,
                "SD": sd,
                "Minimum": float(result[raw_name].min()),
                "Maximum": float(result[raw_name].max()),
                "N": len(result),
            }
        )
    return result, pd.DataFrame(scaler_rows)


def pairwise_correlations(
    frame: pd.DataFrame, variables: Sequence[str], analysis: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month, season in SEASON_LABELS.items():
        subset = frame.loc[frame["Month"].astype(int) == month]
        for i, first in enumerate(variables):
            for second in variables[i + 1 :]:
                pair = subset[[first, second]].dropna()
                pearson = stats.pearsonr(pair[first], pair[second]) if len(pair) >= 3 else (np.nan, np.nan)
                spearman = stats.spearmanr(pair[first], pair[second]) if len(pair) >= 3 else (np.nan, np.nan)
                rows.append(
                    {
                        "Analysis": analysis,
                        "Month": month,
                        "Season": season,
                        "Variable1": first,
                        "Variable2": second,
                        "N": len(pair),
                        "Pearson_r": float(pearson.statistic if hasattr(pearson, "statistic") else pearson[0]),
                        "Pearson_p": float(pearson.pvalue if hasattr(pearson, "pvalue") else pearson[1]),
                        "Spearman_rho": float(spearman.statistic if hasattr(spearman, "statistic") else spearman[0]),
                        "Spearman_p": float(spearman.pvalue if hasattr(spearman, "pvalue") else spearman[1]),
                    }
                )
    return pd.DataFrame(rows)


def compute_vif(frame: pd.DataFrame, variables: Sequence[str], label: str) -> pd.DataFrame:
    clean = frame[list(variables)].dropna().astype(float)
    rows: list[dict[str, Any]] = []
    for target in variables:
        y = clean[target].to_numpy()
        others = [x for x in variables if x != target]
        if not others:
            r2 = 0.0
        else:
            x = np.column_stack([np.ones(len(clean)), clean[others].to_numpy()])
            beta = np.linalg.lstsq(x, y, rcond=None)[0]
            fitted = x @ beta
            denom = float(np.sum((y - y.mean()) ** 2))
            r2 = 0.0 if denom <= 0 else 1.0 - float(np.sum((y - fitted) ** 2)) / denom
        vif = np.inf if r2 >= 1.0 - 1e-12 else 1.0 / (1.0 - r2)
        rows.append(
            {
                "Analysis": label,
                "Variable": target,
                "N": len(clean),
                "R2_with_other_predictors": r2,
                "VIF": vif,
            }
        )
    return pd.DataFrame(rows)


def plot_correlation_heatmaps(
    frame: pd.DataFrame,
    variables: Sequence[str],
    analysis: str,
    output_dir: Path,
) -> None:
    for month, season in SEASON_LABELS.items():
        corr = frame.loc[frame["Month"].astype(int) == month, variables].corr()
        fig, ax = plt.subplots(figsize=(7.2, 6.0))
        sns.heatmap(
            corr,
            vmin=-1,
            vmax=1,
            center=0,
            cmap="vlag",
            annot=True,
            fmt=".2f",
            square=True,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title(f"{season}: {analysis} predictor correlations")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"figure_correlation_{analysis.lower()}_{season.lower()}.png",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# Spline GEE model engine
# ---------------------------------------------------------------------------


def require_model_stack() -> dict[str, str]:
    """Import the formal-model dependencies only when models are requested."""
    global sm, GEE, Exchangeable, Independence, bs, build_design_matrices  # noqa: PLW0603
    try:
        import statsmodels
        import statsmodels.api as sm_module
        import patsy
        from patsy import bs as patsy_bs
        from patsy import build_design_matrices as patsy_build_design_matrices
        from statsmodels.genmod.cov_struct import Exchangeable as SMExchangeable
        from statsmodels.genmod.cov_struct import Independence as SMIndependence
        from statsmodels.genmod.generalized_estimating_equations import GEE as SMGEE
    except Exception as exc:
        raise RuntimeError(
            "Step 6 requires statsmodels and patsy. In a separate first Kaggle "
            "cell run:\n"
            "!pip install -q 'statsmodels>=0.14,<0.15' 'patsy>=0.5'\n"
            "Then choose 'Restart Session', rerun the dependency cell once, and "
            "run Step 6 again. Do not install packages from inside this analysis "
            f"script. Original import error: {type(exc).__name__}: {exc}"
        ) from exc

    if Version(statsmodels.__version__) < Version("0.14"):
        raise RuntimeError(
            f"statsmodels>=0.14 is required; current={statsmodels.__version__}. "
            "Upgrade it in a separate cell and restart the Kaggle session."
        )
    sm = sm_module
    GEE = SMGEE
    Exchangeable = SMExchangeable
    Independence = SMIndependence
    bs = patsy_bs
    build_design_matrices = patsy_build_design_matrices
    return {
        "statsmodels": statsmodels.__version__,
        "patsy": patsy.__version__,
    }


def spline_term(variable, df=None):
    # Natural cubic splines: training-derived knots, centered full-rank basis,
    # linear tails for held-out covariates beyond the training range.
    use_df = YEAR_SPLINE_DF if variable == 'Year_z' else (df or SPLINE_DF)
    return f"cr({variable}, df={use_df}, constraints='center')"


def build_formula(response_column: str, predictors: Sequence[str]) -> str:
    return response_column + " ~ " + " + ".join(spline_term(x) for x in predictors)


def estimate_nb_alpha(response: pd.Series) -> float:
    values = pd.to_numeric(response, errors="coerce").dropna().to_numpy(float)
    mean = float(np.mean(values)) if len(values) else 0.0
    variance = float(np.var(values, ddof=1)) if len(values) > 1 else mean
    if mean <= 1e-12:
        return 1.0
    alpha = (variance - mean) / (mean**2)
    return float(np.clip(alpha, 0.05, 10.0))


def log_link() -> Any:
    # statsmodels 0.14 exposes CamelCase links; older aliases remain supported.
    return sm.families.links.Log()


def make_family(
    family_name: str, response: pd.Series, tweedie_power: float
) -> tuple[Any, dict[str, float]]:
    if family_name == "Binomial":
        return sm.families.Binomial(), {}
    if family_name == "Tweedie":
        return (
            sm.families.Tweedie(
                link=log_link(), var_power=float(tweedie_power), eql=False
            ),
            {"TweediePower": float(tweedie_power)},
        )
    if family_name == "NegativeBinomial":
        alpha = estimate_nb_alpha(response)
        return (
            sm.families.NegativeBinomial(link=log_link(), alpha=alpha),
            {"NB_Alpha": alpha},
        )
    if family_name == "Gamma":
        return sm.families.Gamma(link=log_link()), {}
    raise ValueError(f"Unsupported family: {family_name}")


@dataclass
class FitBundle:
    scenario: Scenario
    month: int
    season: str
    response_key: str
    response_column: str
    family_name: str
    formula: str
    predictors: tuple[str, ...]
    model_data: pd.DataFrame
    result: Any
    family_parameters: dict[str, float]
    scaler: pd.DataFrame


def fit_gee_result(frame, response_key, predictors, tweedie_power):
    spec = RESPONSE_SPECS[response_key]
    response = str(spec['column'])
    d = frame.copy()
    if spec['positive_only']: d = d.loc[d[response] > 0].copy()
    d = d.dropna(subset=[response, 'LocationID', *predictors]).copy()
    formula = build_formula(response, predictors)
    base = dict(Response=response_key, Formula=formula, N=len(d),
                N_LocationID=d.LocationID.nunique(), RandomSeed=RANDOM_SEED,
                EventIDs=d.get('EventID', pd.Series(d.index)).astype(str).tolist())
    if d.LocationID.nunique() < 2: raise ValueError('Too few LocationID clusters')
    if response_key == 'Occurrence' and d[response].nunique() < 2:
        raise ValueError('Occurrence has only one class')
    family, fp = make_family(str(spec['family']), d[response], tweedie_power)
    if response_key == 'Richness':
        # Estimate conditional NB2 alpha from a Poisson mean pilot, not from the
        # unconditional variance of richness. Treat alpha as a working parameter.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pilot = sm.GLM.from_formula(formula, d, family=sm.families.Poisson()).fit(maxiter=GEE_MAXITER)
        mu = np.asarray(pilot.fittedvalues, float)
        rank = np.linalg.matrix_rank(pilot.model.exog)
        raw_alpha = float(np.sum((d[response].to_numpy()-mu)**2-mu) / np.sum(mu**2))
        alpha = max(raw_alpha * len(d) / max(len(d)-rank, 1), 0.0)
        fp = {'NB_Alpha': alpha, 'NB_Theta': (1/alpha if alpha > 0 else np.inf),
              'NB_Alpha_Method': 'conditional_Poisson_pilot_moment_df_adjusted',
              'NB_Alpha_Raw': raw_alpha, 'NB_Alpha_AtBoundary': alpha == 0,
              'WorkingFamily': 'Poisson_limit_of_NB2' if alpha == 0 else 'NegativeBinomial',
              'Pilot_Pearson_Dispersion': float(np.sum((d[response].to_numpy()-mu)**2/mu)/max(len(d)-rank,1))}
        audit_record({**base, 'Attempt':'NB_Poisson_Pilot', 'Converged':bool(pilot.converged),
                      'Warnings':[str(w.message) for w in caught], 'History':pilot.fit_history, **fp})
        if not pilot.converged or not np.all(np.isfinite(mu)):
            raise RuntimeError('NB pilot failed; alpha was not accepted')
        family = (sm.families.Poisson(link=log_link()) if alpha == 0 else sm.families.NegativeBinomial(link=log_link(), alpha=alpha))
    starts = None
    collected = []
    result = None
    for attempt, cov in [('Exchangeable_initial', Exchangeable()),
                         ('Independence_start_only', Independence()),
                         ('Exchangeable_restart', Exchangeable())]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                model = GEE.from_formula(formula, groups='LocationID', data=d, family=family, cov_struct=cov)
                rank = np.linalg.matrix_rank(model.exog)
                if rank != model.exog.shape[1]:
                    raise ValueError(f'Rank-deficient spline design: {rank}/{model.exog.shape[1]}')
                result = model.fit(maxiter=GEE_MAXITER, ctol=GEE_CTOL, cov_type='robust', start_params=starts)
                ok = (bool(getattr(result, 'converged', False)) and
                      np.all(np.isfinite(result.params)) and np.all(np.isfinite(result.cov_params())))
                info = dict(Attempt=attempt, Converged=ok, DesignRank=rank,
                            DesignColumns=model.exog_names, History=result.fit_history,
                            Parameters=result.params, Scale=result.scale, **fp)
                starts = np.asarray(result.params) if np.all(np.isfinite(result.params)) else None
            except Exception as exc:
                ok = False
                info = dict(Attempt=attempt, Converged=False, Error=f'{type(exc).__name__}: {exc}', **fp)
        messages = [str(w.message) for w in caught]
        collected.extend(messages)
        audit_record({**base, **info, 'Warnings':messages})
        if ok and attempt.startswith('Exchangeable'):
            if AUDIT_DIR is not None and FIT_CONTEXT.get('Stage') == 'Final':
                tag = f"{FIT_CONTEXT.get('Scenario')}_{FIT_CONTEXT.get('Month')}_{response_key}"
                design = pd.DataFrame(result.model.exog, columns=result.model.exog_names)
                design.insert(0, 'EventID', d.get('EventID', pd.Series(d.index)).astype(str).to_numpy())
                design.to_csv(AUDIT_DIR / f'design_GEE_{tag}.csv', index=False)
            return result, d, formula, ' | '.join(dict.fromkeys(collected)), fp
    raise RuntimeError('No converged exchangeable GEE; see 25_fit_attempts.jsonl. Inference suppressed.')


def prediction_metrics(response_key, observed, predicted, tweedie_power):
    y, pred = np.asarray(observed, float), np.asarray(predicted, float)
    if not np.all(np.isfinite(pred)): raise ValueError('Nonfinite held-out predictions')
    if response_key != 'Occurrence' and np.any(pred <= 0): raise ValueError('Nonpositive fitted mean')
    metrics = {'MAE':float(np.mean(np.abs(y-pred))),
               'MSE':float(np.mean((y-pred)**2)), 'RMSE':float(np.sqrt(np.mean((y-pred)**2))),
               'MeanBias':float(np.mean(pred-y))}
    if response_key == 'Occurrence':
        if np.any((pred<0)|(pred>1)): raise ValueError('Probability outside [0,1]')
        prob = np.clip(pred, 1e-10, 1-1e-10)
        metrics.update(Brier=float(brier_score_loss(y,prob)),
                       LogLoss=float(log_loss(y,prob,labels=[0,1])),
                       ROC_AUC=float(roc_auc_score(y,prob)) if len(np.unique(y))==2 else np.nan)
    elif response_key == 'Richness': metrics['MeanPoissonDeviance']=float(mean_poisson_deviance(y,pred))
    elif response_key == 'PositiveBiomass': metrics['MeanGammaDeviance']=float(mean_gamma_deviance(y,pred))
    # No power-specific deviance is used to compare Tweedie candidates.
    return metrics


def coefficient_rows(bundle: FitBundle) -> list[dict[str, Any]]:
    result = bundle.result
    names = list(result.model.exog_names)
    params = np.asarray(result.params, dtype=float)
    se = np.asarray(result.bse, dtype=float)
    pvalues = np.asarray(result.pvalues, dtype=float)
    ci = np.asarray(result.conf_int(alpha=0.05), dtype=float)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        rows.append(
            {
                "Scenario": bundle.scenario.name,
                "InferentialRole": bundle.scenario.inferential_role,
                "Month": bundle.month,
                "Season": bundle.season,
                "Response": bundle.response_key,
                "Family": bundle.family_name,
                "Coefficient": name,
                "Estimate": params[index],
                "RobustSE": se[index],
                "CI95_Lower": ci[index, 0],
                "CI95_Upper": ci[index, 1],
                "P_value_component": pvalues[index],
            }
        )
    return rows


def joint_wald_rows(bundle: FitBundle) -> list[dict[str, Any]]:
    result = bundle.result
    names = list(result.model.exog_names)
    params = np.asarray(result.params, dtype=float)
    covariance = np.asarray(result.cov_params(), dtype=float)
    rows: list[dict[str, Any]] = []

    tests: list[tuple[str, list[int]]] = []
    for predictor in bundle.predictors:
        indexes = [
            index
            for index, coefficient in enumerate(names)
            if predictor in coefficient and coefficient != "Intercept"
        ]
        tests.append((predictor, indexes))
    tests.append(
        (
            "AllPredictors",
            [i for i, coefficient in enumerate(names) if coefficient != "Intercept"],
        )
    )

    for term, indexes in tests:
        if not indexes:
            statistic = pvalue = np.nan
            df = 0
            status = "NO_COEFFICIENTS"
        else:
            beta = params[indexes]
            cov = covariance[np.ix_(indexes, indexes)]
            cov = 0.5 * (cov + cov.T)
            inverse = np.linalg.pinv(cov, rcond=1e-12)
            statistic = max(float(beta.T @ inverse @ beta), 0.0)
            df = int(np.linalg.matrix_rank(cov, tol=1e-10))
            pvalue = float(stats.chi2.sf(statistic, df)) if df > 0 else np.nan
            status = "OK" if np.isfinite(statistic) else "NONFINITE"
        rows.append(
            {
                "Scenario": bundle.scenario.name,
                "InferentialRole": bundle.scenario.inferential_role,
                "Month": bundle.month,
                "Season": bundle.season,
                "Response": bundle.response_key,
                "Family": bundle.family_name,
                "Term": term,
                "Wald_chi2": statistic,
                "df": df,
                "P_value": pvalue,
                "Status": status,
            }
        )
    return rows


def dep_parameter(result: Any) -> float:
    cov_struct = getattr(result, "cov_struct", None)
    if cov_struct is None:
        cov_struct = getattr(getattr(result, "model", None), "cov_struct", None)
    value = getattr(cov_struct, "dep_params", np.nan)
    array = np.asarray(value).reshape(-1)
    return float(array[0]) if array.size and np.isfinite(array[0]) else np.nan


def fit_all_gee_models(
    scenario_frames: dict[str, pd.DataFrame],
    scaler_tables: dict[tuple[str, int], pd.DataFrame],
    selected_tweedie: dict[int, float],
) -> tuple[
    list[FitBundle],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
]:
    bundles: list[FitBundle] = []
    summary_rows: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    wald_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for scenario in SCENARIOS:
        source = scenario_frames[scenario.source]
        for month, season in SEASON_LABELS.items():
            prepared, _ = standardize_scenario_frame(source, scenario, month)
            scenario_power = selected_tweedie[month]
            if scenario.name == 'RawFull4_Association':
                scenario_power, _ = tune_power(cv_source(source, scenario, month, 'TotalBiomass'), scenario, month,
                                                {'Stage':'FinalRawTuning', 'OuterFold':0})
            FINAL_POWER_ROWS.append(dict(Scenario=scenario.name, Month=month, SelectedTweediePower=scenario_power,
                                         PowerRole='retuned_raw' if scenario.name=='RawFull4_Association' else 'primary_or_fixed_primary_sensitivity'))
            for response_key, spec in RESPONSE_SPECS.items():
                FIT_CONTEXT.clear(); FIT_CONTEXT.update(Stage='Final', Scenario=scenario.name, Month=month)
                start = time.time()
                log(f"GEE {scenario.name} | {season} | {response_key}")
                try:
                    result, model_data, formula, warning_text, family_parameters = (
                        fit_gee_result(
                            prepared,
                            response_key,
                            scenario.model_predictors,
                            scenario_power,
                        )
                    )
                    predicted = np.asarray(result.predict(model_data), dtype=float)
                    observed = model_data[str(spec["column"])].to_numpy(float)
                    train_metrics = prediction_metrics(
                        response_key,
                        observed,
                        predicted,
                        scenario_power,
                    )
                    converged = bool(getattr(result, "converged", False))
                    status = "OK" if converged else "NOT_CONVERGED"
                    family_parameters = dict(family_parameters)
                    bundle = FitBundle(
                        scenario=scenario,
                        month=month,
                        season=season,
                        response_key=response_key,
                        response_column=str(spec["column"]),
                        family_name=family_parameters.get("WorkingFamily", str(spec["family"])),
                        formula=formula,
                        predictors=scenario.model_predictors,
                        model_data=model_data,
                        result=result,
                        family_parameters=family_parameters,
                        scaler=scaler_tables[(scenario.name, month)],
                    )
                    bundles.append(bundle)
                    coefficients.extend(coefficient_rows(bundle))
                    wald_rows.extend(joint_wald_rows(bundle))
                    summary = {
                        "Scenario": scenario.name,
                        "Description": scenario.description,
                        "InferentialRole": scenario.inferential_role,
                        "Month": month,
                        "Season": season,
                        "Response": response_key,
                        "ResponseColumn": spec["column"],
                        "Family": family_parameters.get("WorkingFamily", spec["family"]),
                        "RequestedFamily": spec["family"],
                        "Status": status,
                        "Converged": converged,
                        "N": len(model_data),
                        "N_LocationID": model_data["LocationID"].nunique(),
                        "N_Zero": int((observed == 0).sum()),
                        "Pearson_WorkingVariance_Ratio": float(np.sum((observed-predicted)**2/np.maximum(result.model.family.variance(predicted),1e-12))/max(len(observed)-len(result.params),1)),
                        "ResponseMean": float(np.mean(observed)),
                        "Formula": formula,
                        "Scale": float(getattr(result, "scale", np.nan)),
                        "ExchangeableCorrelation": dep_parameter(result),
                        "Warnings": warning_text,
                        "ElapsedSeconds": time.time() - start,
                        **family_parameters,
                        **{f"Train_{key}": value for key, value in train_metrics.items()},
                    }
                    summary_rows.append(summary)
                    if not converged:
                        failures.append(
                            f"{scenario.name}/{season}/{response_key}: did not converge"
                        )
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    summary_rows.append(
                        {
                            "Scenario": scenario.name,
                            "Description": scenario.description,
                            "InferentialRole": scenario.inferential_role,
                            "Month": month,
                            "Season": season,
                            "Response": response_key,
                            "ResponseColumn": spec["column"],
                            "Family": spec["family"],
                            "Status": "FAILED",
                            "Converged": False,
                            "N": np.nan,
                            "N_LocationID": np.nan,
                            "Formula": build_formula(
                                str(spec["column"]), scenario.model_predictors
                            ),
                            "Warnings": message,
                            "ElapsedSeconds": time.time() - start,
                        }
                    )
                    failures.append(f"{scenario.name}/{season}/{response_key}: {message}")
                    log("WARNING: " + failures[-1])

    wald = pd.DataFrame(wald_rows)
    if not wald.empty:
        wald["P_FDR_within_model"] = np.nan
        for _, indexes in wald.groupby(
            ["Scenario", "Month", "Response"], observed=True
        ).groups.items():
            idx = list(indexes)
            term_mask = wald.loc[idx, "Term"].ne("AllPredictors")
            term_idx = list(wald.loc[idx].index[term_mask])
            wald.loc[term_idx, "P_FDR_within_model"] = bh_fdr(
                wald.loc[term_idx, "P_value"].to_numpy()
            )
        wald["Significant_FDR_0.05"] = (
            wald["P_FDR_within_model"].lt(0.05).fillna(False)
        )

    return (
        bundles,
        pd.DataFrame(summary_rows),
        pd.DataFrame(coefficients),
        wald,
        failures,
    )


# ---------------------------------------------------------------------------
# Response curves and grouped cross-validation
# ---------------------------------------------------------------------------


CURVE_LABELS = {
    "Depth_m": ("Depth", "m"),
    "SST_Anomaly": ("SST anomaly", "deg C"),
    "SSS_Anomaly": ("SSS anomaly", "salinity units"),
    "log1p_Chla_0125_Anomaly": ("log1p Chla anomaly", "dimensionless"),
}


def inverse_link(result: Any, eta: np.ndarray) -> np.ndarray:
    eta = np.clip(np.asarray(eta, dtype=float), -30, 30)
    return np.asarray(result.model.family.link.inverse(eta), dtype=float)


def response_curve_rows(bundle: FitBundle, points: int = 100) -> pd.DataFrame:
    if bundle.scenario.name != "Main_Anomaly":
        return pd.DataFrame()
    design_info = bundle.result.model.data.design_info
    params = np.asarray(bundle.result.params, dtype=float)
    covariance = np.asarray(bundle.result.cov_params(), dtype=float)
    raw_to_model = dict(
        zip(bundle.scenario.raw_predictors, bundle.scenario.model_predictors)
    )
    scaler = bundle.scaler.set_index("RawVariable")
    rows: list[pd.DataFrame] = []

    for raw_variable, model_variable in raw_to_model.items():
        values = bundle.model_data[raw_variable].dropna().astype(float)
        lower, upper = np.quantile(values, [0.025, 0.975])
        grid = np.linspace(float(lower), float(upper), points)
        new_data = pd.DataFrame(
            {predictor: np.zeros(points, dtype=float) for predictor in bundle.predictors}
        )
        mean = float(scaler.loc[raw_variable, "Mean"])
        sd = float(scaler.loc[raw_variable, "SD"])
        new_data[model_variable] = (grid - mean) / sd
        design = np.asarray(
            build_design_matrices([design_info], new_data, return_type="dataframe")[0],
            dtype=float,
        )
        eta = design @ params
        eta_var = np.einsum("ij,jk,ik->i", design, covariance, design)
        eta_se = np.sqrt(np.clip(eta_var, 0, None))
        predicted = inverse_link(bundle.result, eta)
        lower_ci = inverse_link(bundle.result, eta - 1.96 * eta_se)
        upper_ci = inverse_link(bundle.result, eta + 1.96 * eta_se)
        label, unit = CURVE_LABELS[raw_variable]
        rows.append(
            pd.DataFrame(
                {
                    "Scenario": bundle.scenario.name,
                    "Month": bundle.month,
                    "Season": bundle.season,
                    "Response": bundle.response_key,
                    "Family": bundle.family_name,
                    "Predictor": raw_variable,
                    "PredictorLabel": label,
                    "Unit": unit,
                    "X": grid,
                    "Predicted": predicted,
                    "CI95_Lower": lower_ci,
                    "CI95_Upper": upper_ci,
                    "OtherPredictors": "held at season-specific mean (z=0)",
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def plot_response_curves(curves: pd.DataFrame, output_dir: Path) -> None:
    if curves.empty:
        return
    for (season, response), subset in curves.groupby(
        ["Season", "Response"], observed=True
    ):
        variables = list(dict.fromkeys(subset["Predictor"].tolist()))
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
        axes_flat = axes.ravel()
        for ax, variable in zip(axes_flat, variables):
            data = subset.loc[subset["Predictor"] == variable].sort_values("X")
            x = data["X"].to_numpy(float)
            estimate = data["Predicted"].to_numpy(float)
            lower = data["CI95_Lower"].to_numpy(float)
            upper = data["CI95_Upper"].to_numpy(float)
            ax.plot(x, estimate, color="#176B87", linewidth=2.2)
            ax.fill_between(x, lower, upper, color="#64CCC5", alpha=0.28)
            ax.set_xlabel(f"{data['PredictorLabel'].iloc[0]} ({data['Unit'].iloc[0]})")
            ax.set_ylabel("Model-predicted response")
            ax.grid(alpha=0.2)
        for ax in axes_flat[len(variables) :]:
            ax.set_visible(False)
        fig.suptitle(f"{season} — {response}: primary spline GEE responses", y=1.01)
        fig.tight_layout()
        fig.savefig(
            output_dir
            / f"figure_response_curves_{season.lower()}_{response.lower()}.png",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
        plt.close(fig)


def scale_train_and_test(train, test, scenario):
    tr, te = train.copy(), test.copy()
    for name in scenario.raw_predictors:
        if name.endswith('_Anomaly'):
            source = name[:-len('_Anomaly')]
            reference = tr.groupby('LocationID', observed=True)[source].mean()
            tr[name] = tr[source]-tr.LocationID.map(reference)
            te[name] = te[source]-te.LocationID.map(reference)
        if name == 'Year_rel':
            start = float(tr.Year.min())
            tr[name], te[name] = tr.Year-start, te.Year-start
    for raw, target in zip(scenario.raw_predictors,scenario.model_predictors):
        mean, sd = float(tr[raw].mean()), safe_sd(tr[raw])
        tr[target], te[target] = (tr[raw]-mean)/sd, (te[raw]-mean)/sd
    return tr, te


def run_grouped_cv(source_frame,scenario,month,response_key,tweedie_power):
    # Fixed-power CV helper retained for API compatibility, not used as nested validation.
    d=cv_source(source_frame,scenario,month,response_key);group=cv_group_column(scenario)
    rows=[]
    for fold,(a,b) in enumerate(GroupKFold(min(CV_FOLDS,d[group].nunique())).split(d,groups=d[group]),1):
        rec,met,_=cv_evaluate(d.iloc[a],d.iloc[b],scenario,month,response_key,tweedie_power,
                              {'Stage':'FixedPowerCV','Fold':fold})
        rows.extend([{**rec,'Metric':k,'Value':v} for k,v in met.items()] or [{**rec,'Metric':'Unavailable','Value':np.nan}])
    return pd.DataFrame(rows)


def select_tweedie_power(strict_anomaly,main_scenario):
    if not RUN_GROUPED_CV: raise RuntimeError('Revised submission analysis requires CV enabled; use PREP_ONLY for input audit')
    selected={}; tables=[]
    for month in SEASON_LABELS:
        power,table=tune_power(cv_source(strict_anomaly,main_scenario,month,'TotalBiomass'),main_scenario,month,
                               {'Stage':'FinalTuning','OuterFold':0})
        selected[month]=power;tables.append(table)
        FINAL_POWER_ROWS.append(dict(Scenario=main_scenario.name,Month=month,SelectedTweediePower=power,
                                     Score='pooled held-out MSE on biomass scale',CVGrouping=cv_group_column(main_scenario)))
    return selected,pd.concat(tables,ignore_index=True)


def run_primary_grouped_cv(strict_anomaly,main_scenario,selected_tweedie):
    if not RUN_GROUPED_CV: return pd.DataFrame()
    rows=[]
    raw=next(s for s in SCENARIOS if s.name=='RawFull4_Association')
    for scenario in (main_scenario,raw):
      for month in SEASON_LABELS:
        for response_key in RESPONSE_SPECS:
            d=cv_source(strict_anomaly,scenario,month,response_key);group=cv_group_column(scenario)
            splitter=GroupKFold(min(CV_FOLDS,d[group].nunique()))
            for fold,(a,b) in enumerate(splitter.split(d,groups=d[group]),1):
                train,test=d.iloc[a].copy(),d.iloc[b].copy()
                assert not set(train[group])&set(test[group])
                for role,part in [('train',train),('test',test)]:
                    for event,location,year in part[['EventID','LocationID','Year']].itertuples(index=False,name=None):
                        CV_SPLITS.append(dict(Scenario=scenario.name,Month=month,Response=response_key,Fold=fold,
                                              Role=role,EventID=event,LocationID=location,Year=year,CVGrouping=group))
                ctx={'Stage':'OuterValidation','Fold':fold,'OuterFold':fold}
                try:
                    power=tune_power(train,scenario,month,ctx)[0] if response_key=='TotalBiomass' else DEFAULT_TWEEDIE_POWER
                    rec,met,pred=cv_evaluate(train,test,scenario,month,response_key,power,ctx)
                except Exception as exc:
                    rec=dict(Scenario=scenario.name,Month=month,Response=response_key,Fold=fold,Status='FAILED',Message=str(exc));met={};pred=pd.DataFrame()
                rows.extend([{**rec,'Metric':k,'Value':v,'CVPurpose':'NestedRetrospectiveValidation'} for k,v in met.items()] or
                            [{**rec,'Metric':'Unavailable','Value':np.nan,'CVPurpose':'NestedRetrospectiveValidation'}])
                if not pred.empty: NESTED_PREDICTIONS.append(pred)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# All-25-taxa environmental ordination and restricted permutation tests
# ---------------------------------------------------------------------------


def gower_center(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    squared = np.square(distance_matrix, dtype=float)
    row_mean = squared.mean(axis=1, keepdims=True)
    col_mean = squared.mean(axis=0, keepdims=True)
    grand_mean = float(squared.mean())
    return -0.5 * (squared - row_mean - col_mean + grand_mean)


@dataclass
class PCoAResult:
    coordinates: np.ndarray
    eigenvalues: np.ndarray
    all_positive_eigenvalues: np.ndarray
    corrected_distance: np.ndarray
    lingoes_constant: float
    variance_retained: float
    axes_retained: int


def lingoes_pcoa(distance_matrix: np.ndarray) -> PCoAResult:
    """Principal coordinates with a Lingoes correction if needed."""
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("Distance matrix must be square")
    if not np.allclose(distance_matrix, distance_matrix.T, atol=1e-10):
        raise ValueError("Distance matrix is not symmetric")

    original_b = gower_center(distance_matrix)
    original_eigenvalues = linalg.eigvalsh(original_b, check_finite=False)
    minimum = float(original_eigenvalues.min())
    tolerance = max(float(np.max(np.abs(original_eigenvalues))) * 1e-10, 1e-12)
    constant = abs(minimum) if minimum < -tolerance else 0.0

    corrected = distance_matrix.copy()
    if constant > 0:
        squared = np.square(distance_matrix)
        corrected_squared = squared + 2.0 * constant
        np.fill_diagonal(corrected_squared, 0.0)
        corrected = np.sqrt(np.clip(corrected_squared, 0, None))

    b_matrix = gower_center(corrected)
    eigenvalues, eigenvectors = linalg.eigh(b_matrix, check_finite=False)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive_tolerance = max(float(np.max(np.abs(eigenvalues))) * 1e-10, 1e-12)
    positive = eigenvalues > positive_tolerance
    positive_values = eigenvalues[positive]
    positive_vectors = eigenvectors[:, positive]
    if positive_values.size < 2:
        raise RuntimeError("PCoA produced fewer than two positive axes")

    retain = len(positive_values)  # all numerically positive axes
    selected_values = positive_values[:retain]
    coordinates = positive_vectors[:, :retain] * np.sqrt(selected_values)
    retained_variance = float(selected_values.sum() / positive_values.sum())
    return PCoAResult(
        coordinates=coordinates,
        eigenvalues=selected_values,
        all_positive_eigenvalues=positive_values,
        corrected_distance=corrected,
        lingoes_constant=constant,
        variance_retained=retained_variance,
        axes_retained=retain,
    )


def qr_projection(design):
    a=np.asarray(design,float)
    if not np.all(np.isfinite(a)): raise ValueError('Nonfinite design matrix')
    u,s,_=np.linalg.svd(a,full_matrices=False)
    tol=max(a.shape)*np.finfo(float).eps*(s[0] if len(s) else 1.)
    rank=int(np.sum(s>tol))
    return u[:,:rank],rank


def fitted_by_q(q: np.ndarray, response: np.ndarray) -> np.ndarray:
    return q @ (q.T @ response) if q.shape[1] else np.zeros_like(response)


def rss_by_q(q: np.ndarray, response: np.ndarray) -> float:
    total = float(np.sum(response * response))
    explained = float(np.sum(np.square(q.T @ response))) if q.shape[1] else 0.0
    return max(total - explained, 0.0)


def restricted_permutation_indices(
    groups: np.ndarray | None, rng: np.random.Generator, n: int
) -> np.ndarray:
    if groups is None:
        return rng.permutation(n)
    groups = np.asarray(groups)
    result = np.arange(n)
    for group in pd.unique(groups):
        indexes = np.flatnonzero(groups == group)
        if len(indexes) > 1:
            result[indexes] = rng.permutation(indexes)
    return result


def multivariate_freedman_lane(y,full_design,reduced_design,groups,permutations,rng,permutation_list=None):
    y=np.asarray(y,float)
    qf,rf=qr_projection(full_design);qr,rr=qr_projection(reduced_design)
    if not np.allclose(reduced_design,fitted_by_q(qf,reduced_design),atol=1e-7):
        raise ValueError('Reduced design is not nested in full design')
    df=rf-rr; resid_df=len(y)-rf
    if df<=0 or resid_df<=0: raise ValueError('Unidentifiable term or no residual degrees of freedom')
    total=float(np.square(y-y.mean(axis=0)).sum())
    rssf=rss_by_q(qf,y);rssr=rss_by_q(qr,y);ss=max(0.,rssr-rssf)
    if rssf<=np.finfo(float).eps*max(total,1.): raise ValueError('Degenerate residual SS')
    observed=(ss/df)/(rssf/resid_df)
    fitted=fitted_by_q(qr,y);resid=y-fitted
    # Reduced fitted values cancel from both RSS. Eliminate invariant station
    # projections for efficiency while retaining the exact full-axis statistic.
    def effective_q(q):
        if groups is None:return q
        centered=q.copy()
        for level in pd.unique(groups):
            ix=np.flatnonzero(np.asarray(groups)==level)
            if not np.allclose(resid[ix].sum(axis=0),0,atol=1e-7):return q
            centered[ix]-=centered[ix].mean(axis=0)
        u,s,_=np.linalg.svd(centered,full_matrices=False)
        keep=s>1e-9
        return u[:,keep]*s[keep]
    qfp=effective_q(qf);qrp=effective_q(qr)
    residual_norm=float(np.square(resid).sum())
    indexes=(permutation_list if permutation_list is not None else
             [restricted_permutation_indices(groups,rng,len(y)) for _ in range(permutations)])
    exceed=0;valid=0
    for ix in indexes:
        perm=resid[ix]
        fss=max(0.,residual_norm-float(np.square(qfp.T@perm).sum()))
        rss=max(0.,residual_norm-float(np.square(qrp.T@perm).sum()))
        f=((max(0.,rss-fss)/df)/(fss/resid_df)) if fss>0 else np.nan
        if np.isfinite(f):valid+=1;exceed+=int(f>=observed-1e-12)
    # Exact synchronized cyclic list includes identity; random permutations use +1.
    p=(exceed/valid if permutation_list is not None else (exceed+1)/(valid+1)) if valid else np.nan
    return dict(F=observed,df_term=df,df_residual=resid_df,TermSS=ss,TotalSS=total,
                ResidualSS_Full=rssf,ResidualSS_Reduced=rssr,TermFractionTotal=ss/total,
                PartialR2_Conditional=ss/rssr,P_value=p,ValidPermutations=valid,
                DesignRankFull=rf,DesignRankReduced=rr,Status='OK' if valid==len(indexes) else 'PARTIAL_PERMUTATIONS')


def standardized_linear_design(
    frame: pd.DataFrame, variables: Sequence[str]
) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    columns = [np.ones(len(frame), dtype=float)]
    scaling: dict[str, tuple[float, float]] = {}
    for variable in variables:
        mean = float(frame[variable].mean())
        sd = safe_sd(frame[variable])
        columns.append(((frame[variable].to_numpy(float) - mean) / sd))
        scaling[variable] = (mean, sd)
    return np.column_stack(columns), scaling


def fit_dbrda(
    response_coordinates: np.ndarray,
    design: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    q, rank = qr_projection(design)
    fitted = fitted_by_q(q, response_coordinates)
    fitted -= fitted.mean(axis=0, keepdims=True)
    u, singular, _ = np.linalg.svd(fitted, full_matrices=False)
    usable = min(max(rank - 1, 1), len(singular))
    scores = u[:, :usable] * singular[:usable]
    eigenvalues = np.square(singular[:usable])
    total = float(np.sum(np.square(response_coordinates - response_coordinates.mean(axis=0))))
    constrained_r2 = float(eigenvalues.sum() / total) if total > 0 else np.nan
    return scores, eigenvalues, constrained_r2


def correlation_vector(values: np.ndarray, scores: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = []
    for axis in range(scores.shape[1]):
        x = scores[:, axis]
        if np.std(values) <= 1e-12 or np.std(x) <= 1e-12:
            result.append(np.nan)
        else:
            result.append(float(np.corrcoef(values, x)[0, 1]))
    return np.asarray(result)


def species_metadata(taxonomy: pd.DataFrame) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for _, row in taxonomy.iterrows():
        code = str(row["Species_Code"])
        result[f"B_{code}_kg_km2"] = {
            "Species_Code": code,
            "Species_CN": str(row.get("Species_CN", "")),
            "Species_Latin": str(row.get("Species_Latin", code)),
        }
    return result


def ordination_for_season(
    positive: pd.DataFrame,
    taxonomy: pd.DataFrame,
    month: int,
    rng: np.random.Generator,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    season = SEASON_LABELS[month]
    subset = positive.loc[positive["Month"].astype(int) == month].copy()
    biomass_columns = [f"B_Species_{i:02d}_kg_km2" for i in range(1, 26)]
    subset = subset.dropna(
        subset=[
            "Depth_m",
            "SST_Anomaly",
            "SSS_Anomaly",
            "log1p_Chla_0125_Anomaly",
            *biomass_columns,
        ]
    ).copy()
    abundance = np.sqrt(subset[biomass_columns].to_numpy(float))
    if np.any(abundance.sum(axis=1) <= 0):
        raise AssertionError("Positive-event ordination contains an all-zero row")
    distance = squareform(pdist(abundance, metric="braycurtis"))
    ordination_distance = np.sqrt(distance) if SQRT_BRAY_FOR_ORDINATION else distance
    pcoa = lingoes_pcoa(ordination_distance)
    coordinates = pcoa.coordinates

    temporal_variables = [
        "SST_Anomaly",
        "SSS_Anomaly",
        "log1p_Chla_0125_Anomaly",
    ]
    permutation_rows = temporal_dbrda_tests(subset, coordinates, temporal_variables, pcoa, month, rng)
    permutation_rows += balanced_cyclic_dbrda(subset, temporal_variables, biomass_columns, month, rng)

    # Station-level depth test: one mean community per LocationID, unrestricted.
    station_biomass = subset.groupby("LocationID", observed=True)[biomass_columns].mean()
    station_depth = subset.groupby("LocationID", observed=True)["Depth_m"].median()
    station_abundance = np.sqrt(station_biomass.to_numpy(float))
    station_distance = squareform(pdist(station_abundance, metric="braycurtis"))
    station_ordination_distance = (
        np.sqrt(station_distance) if SQRT_BRAY_FOR_ORDINATION else station_distance
    )
    station_pcoa = lingoes_pcoa(station_ordination_distance)
    station_frame = pd.DataFrame({"Depth_m": station_depth.loc[station_biomass.index]})
    depth_design, _ = standardized_linear_design(station_frame, ["Depth_m"])
    depth_test = multivariate_freedman_lane(
        station_pcoa.coordinates,
        depth_design,
        np.ones((len(station_frame), 1)),
        groups=None,
        permutations=PERMUTATIONS,
        rng=rng,
    )
    permutation_rows.append(
        {
            "Analysis": "Spatial_depth_station_mean_community",
            "Month": month,
            "Season": season,
            "Term": "Depth_m",
            "N": len(station_frame),
            "N_LocationID": len(station_frame),
            "PermutationRestriction": "Unrestricted permutations among station means",
            "ConditionTerms": "Intercept only; station-scale depth association",
            "ExchangeabilityAssumption": "spatial station residuals exchangeable; not established",
            "InferenceStatus": "EXPLORATORY_SPATIAL_AUTOCORRELATION_NOT_RESOLVED",
            "RequestedPermutations": PERMUTATIONS,
            **depth_test,
            "PCoAAxes": station_pcoa.axes_retained,
            "PCoAVarianceRetained": station_pcoa.variance_retained,
            "LingoesConstant": station_pcoa.lingoes_constant,
        }
    )

    # Descriptive dbRDA contains depth and all temporal anomalies. Depth's formal
    # p-value remains the separate station-level test above.
    full_variables = ["Depth_m", *temporal_variables]
    full_design, _ = standardized_linear_design(subset, full_variables)
    site_scores, constrained_eigen, constrained_r2 = fit_dbrda(
        coordinates, full_design
    )
    axis_count = min(site_scores.shape[1], 4)
    site_rows: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(subset.iterrows()):
        record: dict[str, Any] = {
            "EventID": row["EventID"],
            "Year": int(row["Year"]),
            "Month": month,
            "Season": season,
            "LocationID": row["LocationID"],
        }
        for axis in range(axis_count):
            record[f"dbRDA{axis + 1}"] = site_scores[position, axis]
        site_rows.append(record)

    eigen_rows: list[dict[str, Any]] = []
    for axis, value in enumerate(constrained_eigen[:axis_count], start=1):
        eigen_rows.append(
            {
                "Month": month,
                "Season": season,
                "Axis": f"dbRDA{axis}",
                "ConstrainedEigenvalue": value,
                "ShareOfConstrained": value / constrained_eigen.sum(),
                "ConstrainedR2_SelectedPCoASpace": constrained_r2,
                "PCoAAxes": pcoa.axes_retained,
                "PCoAVarianceRetained": pcoa.variance_retained,
            }
        )

    biplot_rows: list[dict[str, Any]] = []
    for variable in full_variables:
        vector = correlation_vector(subset[variable].to_numpy(float), site_scores[:, :axis_count])
        record = {
            "Month": month,
            "Season": season,
            "VectorType": "Environment",
            "Variable": variable,
            "Label": CURVE_LABELS.get(variable, (variable, ""))[0],
        }
        for axis, value in enumerate(vector, start=1):
            record[f"dbRDA{axis}"] = value
        biplot_rows.append(record)

    metadata = species_metadata(taxonomy)
    for column_index, column in enumerate(biomass_columns):
        vector = correlation_vector(abundance[:, column_index], site_scores[:, :axis_count])
        species = metadata.get(column, {"Species_Code": column, "Species_CN": "", "Species_Latin": column})
        record = {
            "Month": month,
            "Season": season,
            "VectorType": "Species",
            "Variable": column,
            "Label": species["Species_Latin"],
            **species,
        }
        for axis, value in enumerate(vector, start=1):
            record[f"dbRDA{axis}"] = value
        biplot_rows.append(record)

    pcoa_rows: list[dict[str, Any]] = []
    for axis, value in enumerate(pcoa.eigenvalues, start=1):
        pcoa_rows.append(
            {
                "Month": month,
                "Season": season,
                "Axis": f"PCoA{axis}",
                "Eigenvalue": value,
                "ShareOfAllPositiveEigenvalues": value / pcoa.all_positive_eigenvalues.sum(),
                "CumulativeShare": pcoa.eigenvalues[:axis].sum() / pcoa.all_positive_eigenvalues.sum(),
            }
        )

    diagnostic = {
        "Month": month,
        "Season": season,
        "N_PositiveEvents": len(subset),
        "N_LocationID": subset["LocationID"].nunique(),
        "N_Species": len(biomass_columns),
        "PCoAAxes": pcoa.axes_retained,
        "PCoAVarianceRetained": pcoa.variance_retained,
        "LingoesConstant": pcoa.lingoes_constant,
        "ConstrainedR2_SelectedPCoASpace": constrained_r2,
        "Distance": (
            "sqrt(Bray-Curtis) on square-root biomass; Lingoes fallback"
            if SQRT_BRAY_FOR_ORDINATION
            else "Bray-Curtis on square-root biomass; Lingoes correction"
        ),
    }
    return (
        permutation_rows,
        eigen_rows,
        site_rows,
        biplot_rows,
        pcoa_rows,
        diagnostic,
    )


def plot_dbrda(
    sites: pd.DataFrame,
    biplot: pd.DataFrame,
    eigenvalues: pd.DataFrame,
    output_dir: Path,
) -> None:
    if sites.empty or "dbRDA2" not in sites.columns:
        return
    for season, site_data in sites.groupby("Season", observed=True):
        vectors = biplot.loc[biplot["Season"] == season].copy()
        eig = eigenvalues.loc[eigenvalues["Season"] == season]
        if site_data.empty or vectors.empty or len(eig) < 2:
            continue
        fig, ax = plt.subplots(figsize=(9.2, 7.3))
        scatter = ax.scatter(
            site_data["dbRDA1"],
            site_data["dbRDA2"],
            c=site_data["Year"],
            cmap="viridis",
            s=18,
            alpha=0.58,
            linewidths=0,
        )
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        colorbar.set_label("Year")

        environment = vectors.loc[vectors["VectorType"] == "Environment"].copy()
        species = vectors.loc[vectors["VectorType"] == "Species"].copy()
        species["Norm2"] = np.sqrt(species["dbRDA1"] ** 2 + species["dbRDA2"] ** 2)
        species = species.nlargest(8, "Norm2")

        x_span = max(float(np.ptp(site_data["dbRDA1"])), 1e-6)
        y_span = max(float(np.ptp(site_data["dbRDA2"])), 1e-6)
        arrow_scale = 0.25 * min(x_span, y_span)
        for _, row in pd.concat([environment, species], ignore_index=True).iterrows():
            color = "#A23E48" if row["VectorType"] == "Environment" else "#24527A"
            x = float(row["dbRDA1"]) * arrow_scale
            y = float(row["dbRDA2"]) * arrow_scale
            ax.annotate(
                "",
                xy=(x, y),
                xytext=(0, 0),
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.3},
            )
            ax.text(x * 1.05, y * 1.05, str(row["Label"]), color=color, fontsize=8)

        axis1 = float(eig.iloc[0]["ShareOfConstrained"]) * 100
        axis2 = float(eig.iloc[1]["ShareOfConstrained"]) * 100
        ax.set_xlabel(f"dbRDA1 ({axis1:.1f}% constrained)")
        ax.set_ylabel(f"dbRDA2 ({axis2:.1f}% constrained)")
        ax.axhline(0, color="0.8", linewidth=0.8)
        ax.axvline(0, color="0.8", linewidth=0.8)
        ax.set_title(f"{season}: environmental dbRDA of positive-event assemblages")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"figure_environment_dbrda_{season.lower()}.png",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
        plt.close(fig)


def run_environment_ordination(
    strict_anomaly: pd.DataFrame,
    taxonomy: pd.DataFrame,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    if not RUN_ORDINATION:
        return {
            "permanova": pd.DataFrame(),
            "eigen": pd.DataFrame(),
            "sites": pd.DataFrame(),
            "biplot": pd.DataFrame(),
            "pcoa": pd.DataFrame(),
            "diagnostics": pd.DataFrame(),
        }
    positive = strict_anomaly.loc[strict_anomaly["Cephalopod_Presence"] == 1].copy()
    all_outputs: dict[str, list[Any]] = {
        "permanova": [],
        "eigen": [],
        "sites": [],
        "biplot": [],
        "pcoa": [],
        "diagnostics": [],
    }
    rng = np.random.default_rng(RANDOM_SEED + 6100)
    for month, season in SEASON_LABELS.items():
        log(
            f"Environmental Bray-Curtis/dbRDA and restricted permutations: "
            f"{season} ({PERMUTATIONS} permutations)"
        )
        outputs = ordination_for_season(positive, taxonomy, month, rng)
        for key, value in zip(all_outputs, outputs):
            all_outputs[key].extend(value if isinstance(value, list) else [value])

    result = {key: pd.DataFrame(values) for key, values in all_outputs.items()}
    if not result["permanova"].empty:
        result["permanova"]["P_FDR_within_analysis_season"] = np.nan
        groups = result["permanova"].groupby(
            ["Analysis", "Season"], observed=True
        ).groups
        for indexes in groups.values():
            idx = list(indexes)
            result["permanova"].loc[idx, "P_FDR_within_analysis_season"] = bh_fdr(
                result["permanova"].loc[idx, "P_value"].to_numpy()
            )
        result["permanova"]["Significant_FDR_0.05"] = result["permanova"][
            "P_FDR_within_analysis_season"
        ].lt(0.05)
    plot_dbrda(result["sites"], result["biplot"], result["eigen"], output_dir)
    return result


# ---------------------------------------------------------------------------
# Output documentation, QC and orchestration
# ---------------------------------------------------------------------------


def build_sample_flow(
    strict_anomaly: pd.DataFrame,
    no_chla_anomaly: pd.DataFrame,
    chla_min3: pd.DataFrame,
) -> pd.DataFrame:
    datasets = {
        "Strict_Main": strict_anomaly,
        "ChlaMin3_Sensitivity": chla_min3,
        "NoChla_2100_Sensitivity": no_chla_anomaly,
    }
    rows: list[dict[str, Any]] = []
    for dataset, frame in datasets.items():
        for month_value, season in [("All", "All"), *SEASON_LABELS.items()]:
            subset = (
                frame
                if month_value == "All"
                else frame.loc[frame["Month"].astype(int) == int(month_value)]
            )
            rows.append(
                {
                    "Dataset": dataset,
                    "Month": month_value,
                    "Season": season,
                    "N_Events": len(subset),
                    "N_LocationID": subset["LocationID"].nunique(),
                    "N_Positive": int((subset["Cephalopod_Presence"] == 1).sum()),
                    "N_Zero": int((subset["Cephalopod_Presence"] == 0).sum()),
                    "Years": ",".join(map(str, sorted(subset["Year"].dropna().astype(int).unique()))),
                }
            )
    return pd.DataFrame(rows)


def build_variable_dictionary() -> pd.DataFrame:
    rows = [
        ("Depth_m", "Station-scale water depth", "m", "Spatial/static gradient"),
        ("SST", "Sea-surface temperature", "deg C", "Raw monthly environment"),
        ("SSS", "Sea-surface salinity", "salinity units", "Raw monthly environment"),
        ("log1p_Chla_0125", "log(1 + 0.125-degree chlorophyll-a)", "dimensionless", "Raw monthly environment"),
        ("SST_Climatology", "LocationID-by-month mean SST across available years", "deg C", "Station-season baseline"),
        ("SST_Anomaly", "SST minus LocationID-by-month SST climatology", "deg C", "Interannual anomaly"),
        ("SSS_Climatology", "LocationID-by-month mean SSS across available years", "salinity units", "Station-season baseline"),
        ("SSS_Anomaly", "SSS minus LocationID-by-month SSS climatology", "salinity units", "Interannual anomaly"),
        ("log1p_Chla_0125_Climatology", "LocationID-by-month mean log1p Chla", "dimensionless", "Station-season baseline"),
        ("log1p_Chla_0125_Anomaly", "log1p Chla minus LocationID-by-month climatology", "dimensionless", "Interannual anomaly"),
        ("Cephalopod_Presence", "Any cephalopod biomass >0", "0/1", "Binomial response"),
        ("Total_Biomass_kg_km2", "All-25-taxa total biomass including zeros", "kg km-2", "Tweedie/Gamma response"),
        ("Species_Richness", "Number of survey taxa present among all 25", "taxa", "Negative-binomial response"),
        ("Year_rel", "Year minus first study year", "year", "Temporal sensitivity predictor"),
    ]
    return pd.DataFrame(rows, columns=["Variable", "Definition", "Unit", "Role"])


def build_metadata(
    input_zip: Path,
    dependency_versions: dict[str, str],
    selected_tweedie: dict[int, float],
) -> pd.DataFrame:
    settings: dict[str, Any] = {
        "ScriptVersion": SCRIPT_VERSION,
        "InputZIP": str(input_zip),
        "InputZIP_SHA256": sha256_file(input_zip),
        "Python": platform.python_version(),
        "Platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": __import__("scipy").__version__,
        "scikit-learn": __import__("sklearn").__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": sns.__version__,
        **dependency_versions,
        "RandomSeed": RANDOM_SEED,
        "SplineDF": SPLINE_DF,
        "YearSplineDF": YEAR_SPLINE_DF,
        "GEEMaxIterations": GEE_MAXITER,
        "GEEConvergenceTolerance": GEE_CTOL,
        "CovarianceStructure": "Exchangeable within LocationID; robust sandwich SE",
        "CVEnabled": RUN_GROUPED_CV,
        "CVFolds": CV_FOLDS,
        "CVGrouping": "Year (anomaly, known-location retrospective); LocationID (raw covariates)",
        "TweediePowerGrid": ",".join(map(str, TWEEDIE_POWER_GRID)),
        "TweediePowerApril": selected_tweedie.get(4, np.nan),
        "TweediePowerNovember": selected_tweedie.get(11, np.nan),
        "OrdinationEnabled": RUN_ORDINATION,
        "Permutations": PERMUTATIONS,
        "PermutationTemporal": "Freedman-Lane residual permutations within LocationID",
        "PermutationDepth": "Unrestricted among LocationID station-mean communities",
        "PCoAVarianceTarget": 1.0,
        "MaximumPCoAAxes": "all numerically positive axes",
        "OrdinationTransformation": "sqrt all-25-taxa biomass",
        "OrdinationDistance": "Bray-Curtis",
        "SqrtBrayEuclidification": SQRT_BRAY_FOR_ORDINATION,
        "NegativeEigenvalueCorrection": "sqrt(distance), with Lingoes fallback",
        "PrepOnly": PREP_ONLY,
    }
    return pd.DataFrame(
        [{"Setting": key, "Value": value} for key, value in settings.items()]
    )


def make_manifest(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "21_output_manifest.csv":
            rows.append(
                {
                    "RelativePath": str(path.relative_to(output_dir)),
                    "SizeBytes": path.stat().st_size,
                    "SHA256": sha256_file(path),
                }
            )
    return pd.DataFrame(rows)


def write_readme(output_dir, input_zip, selected_tweedie, sample_flow):
    (output_dir/'README.txt').write_text(
        'Revised explanatory spline GEE and conditional dbRDA.\n'
        'Natural cubic splines with training-derived knots and linear tails replace bounded B splines.\n'
        'Final fits use all available station-month covariates; validation rebuilds anomalies from training years only.\n'
        'Anomaly models: nested grouped-Year retrospective validation at known locations.\n'
        'Raw-covariate sensitivity: nested grouped-LocationID validation. Neither is future forecasting validation.\n'
        'Tweedie selection uses pooled held-out MSE, not deviance across different powers.\n'
        'NB2 alpha is a conditional Poisson-pilot moment working parameter; not an MLE. Nonpositive excess variance uses the Poisson limit (alpha=0, theta=inf).\n'
        'dbRDA conditions temporal tests on LocationID and Year indicators; retains all positive axes.\n'
        'TermFractionTotal = TermSS/TotalSS; PartialR2_Conditional = TermSS/ResidualSS_Reduced.\n'
        'Temporal permutations assume exchangeability; balanced 2017-2024 positive-panel cyclic shifts are an additional sensitivity.\n'
        'Cyclic tests with 8 years have at most 8 rotations and minimum p=1/8. Spatial depth p values are exploratory.\n'
        'See audit JSONL, design matrices, split membership, calibration, environment freeze and RUN_STATUS.txt.\n', encoding='utf-8')


def run_analysis(output_dir: Path) -> None:
    global AUDIT_DIR
    AUDIT_DIR = output_dir
    FINAL_POWER_ROWS.clear(); NESTED_PREDICTIONS.clear(); NESTED_TUNING.clear(); CV_SPLITS.clear()
    atomic_write_text("RUNNING\n", output_dir / "RUN_STATUS.txt")
    input_zip = discover_step3_zip()
    log(f"Step-3 input: {input_zip}")
    frames = read_step3_tables(input_zip)
    strict, surveyed, taxonomy, _, qc_rows = validate_and_prepare_inputs(frames)
    log(
        f"Input QC passed: strict={len(strict)}, no-Chla={len(surveyed)}, "
        f"LocationID={strict['LocationID'].nunique()}"
    )

    strict_anomaly = add_environment_anomalies(strict, include_chla=True)
    no_chla_anomaly = add_environment_anomalies(surveyed, include_chla=False)
    chla_min3 = strict_anomaly.loc[
        strict_anomaly["Use_ChlaMin3_Sensitivity"]
    ].copy()
    scenario_frames = {
        "strict": strict_anomaly,
        "chla_min3": chla_min3,
        "no_chla": no_chla_anomaly,
    }

    sample_flow = build_sample_flow(strict_anomaly, no_chla_anomaly, chla_min3)
    atomic_to_csv(sample_flow, output_dir / "01_sample_flow.csv")

    model_columns = [
        "EventID",
        "Year",
        "Month",
        "LocationID",
        "Lon",
        "Lat",
        "Depth_m",
        "SST",
        "SSS",
        "log1p_Chla_0125",
        "SST_Climatology",
        "SST_Anomaly",
        "SSS_Climatology",
        "SSS_Anomaly",
        "log1p_Chla_0125_Climatology",
        "log1p_Chla_0125_Anomaly",
        "Cephalopod_Presence",
        "Total_Biomass_kg_km2",
        "Species_Richness",
        "Year_rel",
        "Use_ChlaMin3_Sensitivity",
        *[f"B_Species_{i:02d}_kg_km2" for i in range(1, 26)],
    ]
    atomic_to_csv(
        strict_anomaly[[column for column in model_columns if column in strict_anomaly]],
        output_dir / "02_analysis_data_strict_with_anomalies.csv",
    )
    no_chla_columns = [
        "EventID",
        "Year",
        "Month",
        "LocationID",
        "Lon",
        "Lat",
        "Depth_m",
        "SST",
        "SSS",
        "SST_Climatology",
        "SST_Anomaly",
        "SSS_Climatology",
        "SSS_Anomaly",
        "Cephalopod_Presence",
        "Total_Biomass_kg_km2",
        "Species_Richness",
        "Year_rel",
    ]
    atomic_to_csv(
        no_chla_anomaly[
            [column for column in no_chla_columns if column in no_chla_anomaly]
        ],
        output_dir / "03_analysis_data_noChla_with_anomalies.csv",
    )

    # Correlations and VIF are calculated before model fitting and saved even in
    # PREP_ONLY mode.
    raw_correlations = pairwise_correlations(
        strict_anomaly, RAW_ENV, "Raw_environment"
    )
    anomaly_variables = [
        "Depth_m",
        "SST_Anomaly",
        "SSS_Anomaly",
        "log1p_Chla_0125_Anomaly",
    ]
    anomaly_correlations = pairwise_correlations(
        strict_anomaly, anomaly_variables, "Depth_plus_environment_anomalies"
    )
    atomic_to_csv(raw_correlations, output_dir / "04_environment_correlations_raw.csv")
    atomic_to_csv(
        anomaly_correlations,
        output_dir / "05_environment_correlations_anomaly.csv",
    )

    vif_tables: list[pd.DataFrame] = []
    for month, season in SEASON_LABELS.items():
        subset = strict_anomaly.loc[strict_anomaly["Month"].astype(int) == month]
        raw_vif = compute_vif(subset, RAW_ENV, "RawFull4")
        anomaly_vif = compute_vif(subset, anomaly_variables, "Main_Anomaly")
        year_vif = compute_vif(
            subset, [*anomaly_variables, "Year_rel"], "Main_AnomalyPlusYear"
        )
        for table in (raw_vif, anomaly_vif, year_vif):
            table.insert(1, "Month", month)
            table.insert(2, "Season", season)
            vif_tables.append(table)
    vif = pd.concat(vif_tables, ignore_index=True)
    atomic_to_csv(vif, output_dir / "06_environment_vif.csv")
    plot_correlation_heatmaps(strict_anomaly, RAW_ENV, "Raw", output_dir)
    plot_correlation_heatmaps(
        strict_anomaly, anomaly_variables, "Anomaly", output_dir
    )

    scaler_tables: dict[tuple[str, int], pd.DataFrame] = {}
    for scenario in SCENARIOS:
        for month in SEASON_LABELS:
            _, scaler = standardize_scenario_frame(
                scenario_frames[scenario.source], scenario, month
            )
            scaler_tables[(scenario.name, month)] = scaler
    scaler_all = pd.concat(scaler_tables.values(), ignore_index=True)
    atomic_to_csv(scaler_all, output_dir / "07_predictor_scaling.csv")
    atomic_to_csv(build_variable_dictionary(), output_dir / "08_variable_dictionary.csv")

    if PREP_ONLY:
        selected_tweedie = {month: DEFAULT_TWEEDIE_POWER for month in SEASON_LABELS}
        dependency_versions: dict[str, str] = {}
        model_summary = coefficients = wald = curves = cv_all = pd.DataFrame()
        ordination = {
            key: pd.DataFrame()
            for key in ["permanova", "eigen", "sites", "biplot", "pcoa", "diagnostics"]
        }
        model_failures: list[str] = []
    else:
        dependency_versions = require_model_stack()
        main_scenario = next(x for x in SCENARIOS if x.name == "Main_Anomaly")
        selected_tweedie, power_cv = select_tweedie_power(
            strict_anomaly, main_scenario
        )
        bundles, model_summary, coefficients, wald, model_failures = (
            fit_all_gee_models(
                scenario_frames,
                scaler_tables,
                selected_tweedie,
            )
        )
        curve_tables = [response_curve_rows(bundle) for bundle in bundles]
        curve_tables = [table for table in curve_tables if not table.empty]
        curves = (
            pd.concat(curve_tables, ignore_index=True)
            if curve_tables
            else pd.DataFrame()
        )
        plot_response_curves(curves, output_dir)
        primary_cv = run_primary_grouped_cv(
            strict_anomaly, main_scenario, selected_tweedie
        )
        cv_all = pd.concat(
            [table for table in [power_cv, primary_cv] if not table.empty],
            ignore_index=True,
        ) if (not power_cv.empty or not primary_cv.empty) else pd.DataFrame()
        ordination = run_environment_ordination(strict_anomaly, taxonomy, output_dir)

    atomic_to_csv(
        pd.DataFrame(
            [
                {
                    "Month": month,
                    "Season": SEASON_LABELS[month],
                    "SelectedTweediePower": selected_tweedie[month],
                    "SelectionMethod": (
                        "Year-grouped CV; training-only anomalies; pooled MSE; nested outer validation separately"
                        if RUN_GROUPED_CV and not PREP_ONLY
                        else "Configured default"
                    ),
                }
                for month in SEASON_LABELS
            ]
        ),
        output_dir / "09_tweedie_power_selection.csv",
    )
    atomic_to_csv(model_summary, output_dir / "10_gee_model_summary.csv")
    atomic_to_csv(coefficients, output_dir / "11_gee_coefficients.csv")
    atomic_to_csv(wald, output_dir / "12_gee_joint_wald_tests.csv")
    if not wald.empty:
        sensitivity = wald.loc[wald["Term"] != "AllPredictors"].copy()
        sensitivity.insert(
            2,
            "IsPrimary",
            sensitivity["Scenario"].eq("Main_Anomaly"),
        )
        sensitivity["PredictorScale"] = np.where(
            sensitivity["Scenario"].eq("RawFull4_Association"),
            "Raw season-standardized value",
            "Depth or within-LocationID season anomaly",
        )
    else:
        sensitivity = pd.DataFrame()
    atomic_to_csv(
        sensitivity,
        output_dir / "24_sensitivity_joint_test_comparison.csv",
    )
    atomic_to_csv(curves, output_dir / "13_response_curves.csv")
    atomic_to_csv(cv_all, output_dir / "14_grouped_cv_metrics.csv")
    atomic_to_csv(ordination["permanova"], output_dir / "15_environment_permutation_tests.csv")
    atomic_to_csv(ordination["eigen"], output_dir / "16_dbrda_eigenvalues.csv")
    atomic_to_csv(ordination["sites"], output_dir / "17_dbrda_site_scores.csv")
    atomic_to_csv(ordination["biplot"], output_dir / "18_dbrda_biplot_scores.csv")
    atomic_to_csv(ordination["pcoa"], output_dir / "19_pcoa_eigenvalues.csv")
    atomic_to_csv(ordination["diagnostics"], output_dir / "20_ordination_diagnostics.csv")

    write_revision_diagnostics(output_dir)
    if not ordination['permanova'].empty:
        atomic_to_csv(ordination['permanova'], output_dir/'Table_4b_recomputed.csv')
    if not wald.empty:
        atomic_to_csv(wald.loc[wald.Scenario.eq('Main_Anomaly')], output_dir/'Table_4a_recomputed.csv')
    if not model_summary.empty:
        acceptance = {'FinalFitCounts':model_summary.groupby('Status').size().to_dict(),
                      'NB_PoissonBoundary_Fits':int(model_summary.get('NB_Alpha_AtBoundary',pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                      'RejectedInnerCandidateFolds':int(sum(r.get('Status')!='OK' for r in NESTED_TUNING)),
                      'Interpretation':'No forecasting claim. Inspect boundary dispersion, held-out baseline comparison, and permutation assumptions.'}
        atomic_write_text(json.dumps(acceptance,indent=2),output_dir/'33_model_acceptance_diagnostics.json')
    # Final QC records.
    main_summary = (
        model_summary.loc[model_summary["Scenario"] == "Main_Anomaly"]
        if not model_summary.empty
        else pd.DataFrame()
    )
    expected_main_models = len(SEASON_LABELS) * len(RESPONSE_SPECS)
    qc_rows.extend(
        [
            {
                "Check": "Main GEE models completed",
                "Pass": PREP_ONLY
                or (
                    len(main_summary) == expected_main_models
                    and main_summary["Status"].eq("OK").all()
                ),
                "Observed": (
                    "PREP_ONLY"
                    if PREP_ONLY
                    else int(main_summary["Status"].eq("OK").sum())
                ),
                "Expected": expected_main_models,
                "Severity": "FATAL",
            },
            {
                "Check": "Primary response curves complete",
                "Pass": PREP_ONLY or len(curves) == expected_main_models * 4 * 100,
                "Observed": "PREP_ONLY" if PREP_ONLY else len(curves),
                "Expected": expected_main_models * 4 * 100,
                "Severity": "FATAL",
            },
            {
                "Check": "Grouped CV folds completed",
                "Pass": PREP_ONLY
                or (not RUN_GROUPED_CV)
                or cv_all.empty
                or not cv_all["Status"].eq("FAILED").any(),
                "Observed": (
                    "DISABLED/PREP_ONLY"
                    if PREP_ONLY or not RUN_GROUPED_CV
                    else int(cv_all["Status"].eq("FAILED").sum())
                ),
                "Expected": 0,
                "Severity": "WARNING",
            },
            {
                "Check": "Environmental ordination completed for both seasons",
                "Pass": PREP_ONLY
                or (not RUN_ORDINATION)
                or len(ordination["diagnostics"]) == 2,
                "Observed": (
                    "DISABLED/PREP_ONLY"
                    if PREP_ONLY or not RUN_ORDINATION
                    else len(ordination["diagnostics"])
                ),
                "Expected": 2,
                "Severity": "FATAL",
            },
        ]
    )
    qc = pd.DataFrame(qc_rows)
    atomic_to_csv(qc, output_dir / "22_qc_checks.csv")

    summary_rows = [
        ("Script version", SCRIPT_VERSION),
        ("Input Step3 ZIP", str(input_zip)),
        ("Strict events", len(strict_anomaly)),
        ("Strict positive events", int(strict_anomaly["Cephalopod_Presence"].sum())),
        ("Strict zero events", int((strict_anomaly["Cephalopod_Presence"] == 0).sum())),
        ("No-Chla sensitivity events", len(no_chla_anomaly)),
        ("ChlaMin3 sensitivity events", len(chla_min3)),
        ("LocationID", strict_anomaly["LocationID"].nunique()),
        ("Main GEE models OK", "PREP_ONLY" if PREP_ONLY else int(main_summary["Status"].eq("OK").sum())),
        ("Model warnings/failures", len(model_failures)),
        ("Grouped CV enabled", RUN_GROUPED_CV),
        ("Ordination enabled", RUN_ORDINATION),
        ("Permutations", PERMUTATIONS),
        ("Tweedie power April", selected_tweedie[4]),
        ("Tweedie power November", selected_tweedie[11]),
    ]
    atomic_to_csv(
        pd.DataFrame(summary_rows, columns=["Metric", "Value"]),
        output_dir / "00_step6_summary.csv",
    )
    atomic_to_csv(
        build_metadata(input_zip, dependency_versions, selected_tweedie),
        output_dir / "23_method_metadata.csv",
    )
    write_readme(output_dir, input_zip, selected_tweedie, sample_flow)

    script_path = Path(__file__).resolve() if "__file__" in globals() else None
    if script_path is not None and script_path.is_file():
        shutil.copy2(script_path, output_dir / script_path.name)

    fatal_failures = qc.loc[(qc["Severity"] == "FATAL") & (~qc["Pass"].astype(bool))]
    if not fatal_failures.empty:
        atomic_write_text(
            "FAILED_QC\n" + fatal_failures.to_string(index=False) + "\n",
            output_dir / "RUN_STATUS.txt",
        )
        raise AssertionError(
            "Step 6 finished calculations but failed fatal QC:\n"
            + fatal_failures.to_string(index=False)
        )

    status = "PREP_ONLY_SUCCESS" if PREP_ONLY else ("COMPLETED_WITH_WARNINGS" if ((~qc["Pass"].astype(bool)).any() or model_failures) else "SUCCESS")
    atomic_write_text(
        f"{status}\nElapsed: {elapsed()}\n",
        output_dir / "RUN_STATUS.txt",
    )
    manifest = make_manifest(output_dir)
    atomic_to_csv(manifest, output_dir / "21_output_manifest.csv")


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    temp_output = ensure_empty_output_dir(FINAL_OUTPUT_DIR)
    log(f"Temporary output: {temp_output}")
    try:
        run_analysis(temp_output)
        publish_output_dir(temp_output, FINAL_OUTPUT_DIR)
        build_zip(FINAL_OUTPUT_DIR, FINAL_ZIP)
    except Exception:
        error_text = traceback.format_exc()
        try:
            atomic_write_text("FAILED\n" + error_text, temp_output / "RUN_STATUS.txt")
            diagnostic = FINAL_OUTPUT_DIR.with_name(FINAL_OUTPUT_DIR.name + "_failed")
            if diagnostic.exists():
                shutil.rmtree(diagnostic)
            os.replace(temp_output, diagnostic)
            log(f"Diagnostic outputs retained at: {diagnostic}")
        except Exception:
            pass
        raise

    log("Step 6 completed successfully")
    print(f"Output directory: {FINAL_OUTPUT_DIR}")
    print(f"ZIP file: {FINAL_ZIP}")
    print("Please download: cephalopod_step6_environment_response.zip")




AUDIT_DIR = None


FIT_CONTEXT = {}


FINAL_POWER_ROWS = []


NESTED_PREDICTIONS = []


NESTED_TUNING = []


CV_SPLITS = []


def audit_record(record):
    if AUDIT_DIR is None:
        return
    def default(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        return str(x)
    with (AUDIT_DIR / '25_fit_attempts.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({**FIT_CONTEXT, **record}, default=default, ensure_ascii=False) + '\n')


def cv_group_column(scenario):
    # Held-out years, with station baselines estimated using training years only.
    # Raw-covariate sensitivity separately tests transfer to held-out locations.
    return 'Year' if any(x.endswith('_Anomaly') for x in scenario.raw_predictors) else 'LocationID'


def cv_source(source, scenario, month, response_key):
    d=source.loc[source.Month.astype(int)==month].copy()
    spec=RESPONSE_SPECS[response_key]
    # Retain zero-catch covariates for fold-specific environmental baselines.
    raw=[x[:-8] if x.endswith('_Anomaly') else x for x in scenario.raw_predictors]
    raw=[x for x in raw if x!='Year_rel']
    return d.dropna(subset=[spec['column'],'LocationID','Year',*raw]).reset_index(drop=True)


def cv_evaluate(train, test, scenario, month, response_key, power, context):
    FIT_CONTEXT.clear(); FIT_CONTEXT.update(context, Month=month, Scenario=scenario.name)
    train, test=scale_train_and_test(train,test,scenario)
    known=test[list(scenario.model_predictors)].notna().all(axis=1)
    eligible=(test[RESPONSE_SPECS[response_key]['column']]>0) if RESPONSE_SPECS[response_key]['positive_only'] else pd.Series(True,index=test.index)
    evaluable=known & eligible
    usable=test.loc[evaluable].copy()
    record=dict(Scenario=scenario.name, Month=month, Season=SEASON_LABELS[month],
                Response=response_key, Family=RESPONSE_SPECS[response_key]['family'],
                TweediePower=power if response_key=='TotalBiomass' else np.nan,
                CVGrouping=cv_group_column(scenario), N_Train=len(train), N_Test=len(test),
                N_Evaluated=len(usable), N_UnseenLocationExcluded=int((~known & eligible).sum()), N_IneligibleResponse=int((~eligible).sum()),
                Groups_Train=train.LocationID.nunique(),Groups_Test=test.LocationID.nunique(), **context)
    try:
        if usable.empty: raise ValueError('No evaluable held-out known locations')
        fit,_,_,warn,_=fit_gee_result(train,response_key,scenario.model_predictors,power)
        pred=np.asarray(fit.predict(usable),float)
        obs=usable[RESPONSE_SPECS[response_key]['column']].to_numpy(float)
        metrics=prediction_metrics(response_key,obs,pred,power)
        predictions=usable[['EventID','Year','Month','LocationID']].copy()
        predictions['Observed']=obs;predictions['Predicted']=pred
        for key,val in context.items(): predictions[key]=val
        predictions['Scenario']=scenario.name; predictions['Response']=response_key
        predictions['TweediePower']=power if response_key=='TotalBiomass' else np.nan
        null_y=train[RESPONSE_SPECS[response_key]['column']]
        if RESPONSE_SPECS[response_key]['positive_only']:null_y=null_y.loc[null_y>0]
        predictions['TrainingNullMean']=float(null_y.mean())
        record.update(Status='OK',Message=warn)
        return record,metrics,predictions
    except Exception as exc:
        record.update(Status='FAILED',Message=f'{type(exc).__name__}: {exc}')
        audit_record(record)
        return record,{},pd.DataFrame()


def tune_power(data,scenario,month,context):
    grouping=cv_group_column(scenario)
    folds=min(CV_FOLDS,data[grouping].nunique())
    if folds<2: raise ValueError('At least two training groups required for inner tuning')
    splits=list(GroupKFold(folds).split(data,groups=data[grouping]))
    rows=[]; scores={}
    for power in TWEEDIE_POWER_GRID:
        weighted_error=0.; n=0; all_ok=True
        for f,(a,b) in enumerate(splits,1):
            rec,met,_=cv_evaluate(data.iloc[a],data.iloc[b],scenario,month,'TotalBiomass',power,
                                  {**context,'Stage':'InnerTuning','InnerFold':f})
            rows.append({**rec,'Metric':'MSE','Value':met.get('MSE',np.nan),'CVPurpose':'TweediePowerSelection'})
            if rec['Status']!='OK': all_ok=False
            else: weighted_error+=met['MSE']*rec['N_Evaluated'];n+=rec['N_Evaluated']
        if all_ok and n: scores[power]=weighted_error/n
    NESTED_TUNING.extend(rows)
    if not scores: raise RuntimeError('Every candidate failed one or more tuning folds; no silent default power')
    power=min(scores,key=lambda p:(scores[p],p))
    return float(power),pd.DataFrame(rows)


def write_revision_diagnostics(output_dir):
    pd.DataFrame(FINAL_POWER_ROWS).to_csv(output_dir/'26_final_power_by_scenario_month.csv',index=False)
    pd.DataFrame(NESTED_TUNING).to_csv(output_dir/'27_nested_inner_tuning.csv',index=False)
    pd.DataFrame(CV_SPLITS).to_csv(output_dir/'28_outer_split_membership.csv',index=False)
    if NESTED_PREDICTIONS:
        pred=pd.concat(NESTED_PREDICTIONS,ignore_index=True)
        pred.to_csv(output_dir/'29_outer_heldout_predictions.csv',index=False)
        calibration=[];summaries=[]
        for keys,d in pred.groupby(['Scenario','Month','Response'],sort=False):
            scenario,month,response=keys
            y=d.Observed.to_numpy(float);mu=d.Predicted.to_numpy(float)
            metric=prediction_metrics(response,y,mu,DEFAULT_TWEEDIE_POWER)
            metric['TrainingNull_RMSE']=float(np.sqrt(np.mean((y-d.TrainingNullMean)**2)))
            rec=dict(Scenario=scenario,Month=month,Response=response,N=len(d),ObservedMean=y.mean(),PredictedMean=mu.mean(),**metric)
            # Descriptive calibration diagnostics; no iid Wald p-values for correlated OOF values.
            if response=='Occurrence':
                lp=np.log(np.clip(mu,1e-8,1-1e-8)/np.clip(1-mu,1e-8,1))
                try:
                    model=sm.GLM(y,sm.add_constant(lp),family=sm.families.Binomial()).fit()
                    rec.update(CalibrationIntercept=float(model.params[0]),CalibrationSlope=float(model.params[1]))
                except Exception as exc: rec['CalibrationError']=str(exc)
            else:
                rec['ObservedToPredictedRatio']=float(y.sum()/mu.sum())
                coef=np.linalg.lstsq(np.column_stack([np.ones(len(mu)),mu]),y,rcond=None)[0]
                rec.update(DescriptiveLinearCalibrationIntercept=coef[0],DescriptiveLinearCalibrationSlope=coef[1])
            summaries.append(rec)
            d=d.copy();d['Bin']=pd.qcut(d.Predicted,q=min(10,len(d)),duplicates='drop')
            for bin_,g in d.groupby('Bin',observed=True):
                calibration.append(dict(Scenario=scenario,Month=month,Response=response,Bin=str(bin_),N=len(g),
                                        ObservedMean=g.Observed.mean(),PredictedMean=g.Predicted.mean()))
        pd.DataFrame(summaries).to_csv(output_dir/'30_outer_validation_summary.csv',index=False)
        pd.DataFrame(calibration).to_csv(output_dir/'31_heldout_calibration_bins.csv',index=False)
    import subprocess
    (output_dir/'32_environment_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')


def temporal_dbrda_tests(subset,coordinates,variables,pcoa,month,rng,analysis_suffix='',permutation_list=None):
    env,_=standardized_linear_design(subset,variables)
    station=pd.get_dummies(subset.LocationID.astype(str),prefix='LocationID',drop_first=True,dtype=float)
    year=pd.get_dummies(subset.Year.astype(str),prefix='Year',drop_first=True,dtype=float)
    nuisance=np.column_stack([np.ones(len(subset)),station.to_numpy(),year.to_numpy()])
    full=np.column_stack([nuisance,env[:,1:]])
    names=['Intercept',*station.columns,*year.columns,*variables]
    if AUDIT_DIR is not None:
        d=pd.DataFrame(full,columns=names);d.insert(0,'EventID',subset.EventID.to_numpy())
        d.to_csv(AUDIT_DIR/f'design_dbRDA_{month}{analysis_suffix}.csv',index=False)
    tests=[('AllTemporalEnvironment',nuisance)]
    tests += [(v,np.delete(full,nuisance.shape[1]+i,axis=1)) for i,v in enumerate(variables)]
    rows=[]
    for term,reduced in tests:
        result=multivariate_freedman_lane(coordinates,full,reduced,subset.LocationID.astype(str).to_numpy(),
                                         PERMUTATIONS,rng,permutation_list)
        q,_=qr_projection(reduced);res=coordinates-fitted_by_q(q,coordinates)
        # Multivariate adjacent-year residual correlation diagnostic, not an exchangeability proof.
        earlier=[];later=[]
        for _,ix in subset.reset_index(drop=True).groupby('LocationID',observed=True).groups.items():
            ix=np.array(sorted(ix,key=lambda i:subset.iloc[i].Year))
            for a,b in zip(ix[:-1],ix[1:]):
                if subset.iloc[b].Year-subset.iloc[a].Year==1: earlier.append(res[a]);later.append(res[b])
        a=np.asarray(earlier);b=np.asarray(later)
        corr=float(np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b))) if len(a) and np.sum(a*a)*np.sum(b*b)>0 else np.nan
        rows.append(dict(Analysis='Conditional_LocationID_Year'+analysis_suffix,Month=month,Season=SEASON_LABELS[month],
                         Term=term,N=len(subset),N_LocationID=subset.LocationID.nunique(),
                         ConditionTerms='LocationID fixed indicators + Year fixed indicators',
                         PermutationRestriction=('synchronized cyclic shifts across years; exact rotations including identity'
                                                 if permutation_list is not None else 'Freedman-Lane residuals permuted within LocationID'),
                         ExchangeabilityAssumption=('residual process invariant to circular year shifts' if permutation_list is not None else
                                                     'residual years exchangeable within each location after conditioning'),
                         InferenceStatus='EXPLORATORY_ASSUMPTION_DEPENDENT',ResidualLag1InnerProductCorrelation=corr,
                         ResidualLag1Pairs=len(earlier),RequestedPermutations=len(permutation_list) if permutation_list is not None else PERMUTATIONS,
                         **result,PCoAAxes=pcoa.axes_retained,PCoAVarianceRetained=pcoa.variance_retained,LingoesConstant=pcoa.lingoes_constant))
    return rows


def balanced_cyclic_dbrda(subset,variables,biomass_columns,month,rng):
    years=list(range(2017,2025));d=subset.loc[subset.Year.isin(years)].copy()
    count=d.groupby('LocationID',observed=True).Year.nunique()
    ids=count.index[count.eq(len(years))]
    d=d.loc[d.LocationID.isin(ids)].sort_values(['LocationID','Year']).reset_index(drop=True)
    if len(ids)<3:
        return [dict(Analysis='Conditional_LocationID_Year_BALANCED_CYCLIC_2017_2024',Month=month,
                     Status='NOT_ESTIMABLE',Reason='Fewer than 3 locations positive and complete in all 8 years',N=len(d))]
    if d.duplicated(['LocationID','Year']).any(): raise ValueError('Duplicate station-year in cyclic panel')
    distance=squareform(pdist(np.sqrt(d[biomass_columns].to_numpy(float)),metric='braycurtis'))
    p=lingoes_pcoa(np.sqrt(distance) if SQRT_BRAY_FOR_ORDINATION else distance)
    indexes=np.arange(len(d)).reshape(len(ids),len(years))
    rotations=[np.roll(indexes,k,axis=1).ravel() for k in range(len(years))]
    return temporal_dbrda_tests(d,p.coordinates,variables,p,month,rng,'_BALANCED_CYCLIC_2017_2024',rotations)


if __name__ == "__main__":
    main()
