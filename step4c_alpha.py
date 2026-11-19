#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4C — Community alpha diversity for thesis Section 3.2.

Primary questions
-----------------
1. How do April and November differ?
2. How do alpha-diversity metrics vary among years within each month?
3. What are the patterns in richness, Shannon diversity, Pielou evenness,
   and dominance?

Inputs
------
1. cephalopod_step3_analysis_master.zip
2. cephalopod_step4_resource_iri_alpha.zip

Core rules
----------
- All 2,100 surveyed events are retained for encounter and richness.
- A surveyed zero catch has richness=0; composition-based indices are NA.
- Shannon is defined for positive catches, including single-species catches
  (H'=0 when S=1).
- Pielou J'=H'/ln(S) is defined only when S>=2. It is NOT set to zero when
  S=1, because ln(1)=0.
- Berger-Parker d=max(p_i) is the primary dominance index.
- Simpson D=sum(p_i^2) is retained as a dominance sensitivity index.
- Density-based indices are primary; biomass-based indices are retained as
  sensitivity fields in the event-level output.
- Repeated LocationID is respected by restricted/cluster permutation.

Typical Kaggle call
-------------------
from pathlib import Path
import os, runpy

os.environ["CEPHALOPOD_STEP3_ZIP"] = str(next(
    Path("/kaggle/input").rglob("cephalopod_step3_analysis_master.zip")
))
os.environ["CEPHALOPOD_STEP4_ZIP"] = str(next(
    Path("/kaggle/input").rglob("cephalopod_step4_resource_iri_alpha.zip")
))
runpy.run_path(
    "/kaggle/input/<code-dataset>/kaggle_step4c_community_diversity.py",
    run_name="__main__",
)
"""

from __future__ import annotations

import hashlib
import math
import os
import platform
import shutil
import tempfile
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import pandas as pd
import scipy
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_VERSION = "4C.1.0"
RANDOM_SEED = int(os.environ.get("CEPHALOPOD_STEP4C_SEED", "20260902"))
N_PERMUTATIONS = int(
    os.environ.get("CEPHALOPOD_STEP4C_PERMUTATIONS", "4999")
)
N_BOOTSTRAP = int(os.environ.get("CEPHALOPOD_STEP4C_BOOTSTRAP", "1999"))
FIGURE_DPI = int(os.environ.get("CEPHALOPOD_STEP4C_FIGURE_DPI", "600"))

OUTPUT_ROOT = Path(
    os.environ.get("CEPHALOPOD_STEP4C_OUTPUT_ROOT", "/kaggle/working")
).expanduser()
if not OUTPUT_ROOT.exists() and str(OUTPUT_ROOT).startswith("/kaggle/"):
    OUTPUT_ROOT = Path.cwd()

FINAL_DIR = OUTPUT_ROOT / "cephalopod_step4c_community_diversity"
FINAL_ZIP = OUTPUT_ROOT / "cephalopod_step4c_community_diversity.zip"
START_TIME = time.time()

MONTH_LABELS = {4: "April", 11: "November"}
EXPECTED_EVENTS = 2100
EXPECTED_POSITIVE = 1532
EXPECTED_ZERO = 568
EXPECTED_SPECIES = 25

STEP3_MEMBERS = {
    "metadata": "02_analysis_master_v1_surveyed_all25.csv",
    "biomass": "08_Y_biomass_all25_surveyed.csv",
    "density": "09_Y_density_all25_surveyed.csv",
}
STEP4_MEMBERS = {
    "alpha_event": "15_alpha_diversity_event_level.csv",
    "alpha_year_month": "16_alpha_summary_year_month.csv",
    "alpha_month": "18_alpha_summary_month.csv",
    "alpha_balanced": "19_alpha_annual_season_balanced.csv",
    "qc": "21_qc_checks.csv",
}

METRICS = [
    {
        "Metric": "Encounter",
        "Label": "Encounter probability",
        "Short_Label": "Encounter",
        "Basis": "All surveyed events",
        "Role": "Supporting",
        "Direction": "Higher values indicate more frequent cephalopod occurrence",
        "Bounded_01": True,
    },
    {
        "Metric": "Species_Richness",
        "Label": "Taxon richness (S)",
        "Short_Label": "Richness",
        "Basis": "All surveyed events; zero catch has S=0",
        "Role": "Primary",
        "Direction": "Higher values indicate more species",
        "Bounded_01": False,
    },
    {
        "Metric": "Shannon_H_Density",
        "Label": "Shannon diversity (H')",
        "Short_Label": "Shannon H'",
        "Basis": "Positive catches; numerical-density proportions",
        "Role": "Primary",
        "Direction": "Higher values indicate greater diversity",
        "Bounded_01": False,
    },
    {
        "Metric": "Pielou_J_Density",
        "Label": "Pielou evenness (J')",
        "Short_Label": "Pielou J'",
        "Basis": "Multi-species catches only (S>=2)",
        "Role": "Primary",
        "Direction": "Higher values indicate greater evenness",
        "Bounded_01": True,
    },
    {
        "Metric": "Berger_Parker_Density",
        "Label": "Berger-Parker dominance (d)",
        "Short_Label": "Berger-Parker d",
        "Basis": "Positive catches; numerical-density proportions",
        "Role": "Primary",
        "Direction": "Higher values indicate stronger single-species dominance",
        "Bounded_01": True,
    },
    {
        "Metric": "Simpson_D_Density",
        "Label": "Simpson dominance (D)",
        "Short_Label": "Simpson D",
        "Basis": "Positive catches; numerical-density proportions",
        "Role": "Sensitivity",
        "Direction": "Higher values indicate stronger dominance",
        "Bounded_01": True,
    },
]
METRIC_INFO = {item["Metric"]: item for item in METRICS}
PRIMARY_FIGURE_METRICS = [
    "Species_Richness",
    "Shannon_H_Density",
    "Pielou_J_Density",
    "Berger_Parker_Density",
]


def elapsed() -> str:
    return f"{(time.time() - START_TIME) / 60:.1f} min"


def log(message: str) -> None:
    print(f"[{elapsed()}] {message}", flush=True)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.shape[1] == 0:
        frame = pd.DataFrame({"Status": pd.Series(dtype="string")})
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh_fdr(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    result = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if valid.size == 0:
        return result
    ordered = valid[np.argsort(p[valid])]
    ranked = p[ordered]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[ordered] = np.minimum(adjusted, 1.0)
    return result


def prepare_temp_dir(final_dir: Path) -> Path:
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(prefix=f".{final_dir.name}_tmp_", dir=str(final_dir.parent))
    )


def publish_dir(temp_dir: Path, final_dir: Path) -> None:
    previous = final_dir.with_name(final_dir.name + "_previous")
    if previous.exists():
        shutil.rmtree(previous)
    if final_dir.exists():
        os.replace(final_dir, previous)
    os.replace(temp_dir, final_dir)


def build_zip(source_dir: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(source_dir))
    with zipfile.ZipFile(temporary) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Output ZIP CRC failure: {bad}")
    os.replace(temporary, destination)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    for suffix in ["png", "pdf", "svg"]:
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix == "png":
            kwargs["dpi"] = FIGURE_DPI
        fig.savefig(stem.with_suffix("." + suffix), **kwargs)
    plt.close(fig)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Noto Sans CJK SC",
                "Noto Sans CJK JP",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _validate_zip(path: Path, required: dict[str, str]) -> tuple[bool, str]:
    try:
        if not path.is_file() or not zipfile.is_zipfile(path):
            return False, "not a ZIP file"
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            missing = sorted(set(required.values()) - names)
            if missing:
                return False, "missing " + ", ".join(missing)
            bad = archive.testzip()
            if bad is not None:
                return False, f"CRC failure: {bad}"
        return True, "OK"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def discover_zip(env_name: str, pattern: str, required: dict[str, str]) -> Path:
    explicit = os.environ.get(env_name, "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        valid, reason = _validate_zip(path, required)
        if not valid:
            raise FileNotFoundError(f"Invalid {env_name}: {path}\n{reason}")
        return path

    candidates: list[Path] = []
    for root in [Path.cwd(), OUTPUT_ROOT, Path("/kaggle/input")]:
        if not root.exists():
            continue
        iterator = root.rglob(pattern) if str(root) == "/kaggle/input" else root.glob(pattern)
        for path in iterator:
            valid, _ = _validate_zip(path, required)
            if valid:
                candidates.append(path.resolve())
    candidates = sorted(
        set(candidates), key=lambda item: (item.stat().st_mtime, str(item)), reverse=True
    )
    if not candidates:
        raise FileNotFoundError(
            f"未找到{pattern}。请加入Kaggle Input，或设置os.environ['{env_name}']='ZIP完整路径'。"
        )
    if len(candidates) > 1:
        log(f"Multiple valid {pattern} files found; selected {candidates[0]}")
    return candidates[0]


def read_zip_tables(path: Path, members: dict[str, str]) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(path) as archive:
        for key, member in members.items():
            result[key] = pd.read_csv(archive.open(member), low_memory=False)
    return result


def align_and_validate_inputs(
    step3: dict[str, pd.DataFrame], step4: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    metadata = step3["metadata"].copy()
    density = step3["density"].copy()
    biomass = step3["biomass"].copy()
    qc_rows: list[dict[str, Any]] = []

    def add(check: str, passed: bool, observed: Any, expected: Any) -> None:
        qc_rows.append(
            {
                "Check": check,
                "Pass": bool(passed),
                "Observed": observed,
                "Expected": expected,
                "Severity": "FATAL",
            }
        )

    add("Step4 QC all passed", bool(step4["qc"]["Pass"].astype(bool).all()), int((~step4["qc"]["Pass"].astype(bool)).sum()), 0)
    add("Surveyed events", len(metadata) == EXPECTED_EVENTS, len(metadata), EXPECTED_EVENTS)
    add("Unique EventID", metadata["EventID"].is_unique, metadata["EventID"].nunique(), EXPECTED_EVENTS)
    add("Positive events", int(pd.to_numeric(metadata["Cephalopod_Presence"]).sum()) == EXPECTED_POSITIVE, int(pd.to_numeric(metadata["Cephalopod_Presence"]).sum()), EXPECTED_POSITIVE)
    add("Zero events", int((pd.to_numeric(metadata["Cephalopod_Presence"]) == 0).sum()) == EXPECTED_ZERO, int((pd.to_numeric(metadata["Cephalopod_Presence"]) == 0).sum()), EXPECTED_ZERO)
    add("Study years", set(metadata["Year"].astype(int)) == set(range(2014, 2025)), sorted(metadata["Year"].astype(int).unique()), "2014-2024")
    add("Study months", set(metadata["Month"].astype(int)) == {4, 11}, sorted(metadata["Month"].astype(int).unique()), [4, 11])
    add("LocationID complete", metadata["LocationID"].notna().all(), int(metadata["LocationID"].isna().sum()), 0)

    event_ids = metadata["EventID"].astype(str).tolist()
    for name, frame in [("Density", density), ("Biomass", biomass)]:
        add(f"{name} EventID unique", frame["EventID"].is_unique, frame["EventID"].nunique(), EXPECTED_EVENTS)
        add(f"{name} EventID set aligned", set(frame["EventID"].astype(str)) == set(event_ids), len(set(frame["EventID"].astype(str)) ^ set(event_ids)), 0)
        species_columns = [column for column in frame if column.startswith("Species_")]
        add(f"{name} has 25 species", len(species_columns) == EXPECTED_SPECIES, len(species_columns), EXPECTED_SPECIES)
        values = frame[species_columns].apply(pd.to_numeric, errors="coerce")
        add(f"{name} contains no NA", not values.isna().any().any(), int(values.isna().sum().sum()), 0)
        add(f"{name} contains no negative values", bool((values >= 0).all().all()), int((values < 0).sum().sum()), 0)

    density = density.set_index("EventID").loc[event_ids].reset_index()
    biomass = biomass.set_index("EventID").loc[event_ids].reset_index()
    density_sum = density.drop(columns="EventID").sum(axis=1).to_numpy(dtype=float)
    biomass_sum = biomass.drop(columns="EventID").sum(axis=1).to_numpy(dtype=float)
    density_error = float(np.max(np.abs(density_sum - pd.to_numeric(metadata["Total_Density_1000ind_km2"]).to_numpy(dtype=float))))
    biomass_error = float(np.max(np.abs(biomass_sum - pd.to_numeric(metadata["Total_Biomass_kg_km2"]).to_numpy(dtype=float))))
    add("Density matrix reconciles to event totals", density_error < 1e-8, density_error, 0)
    add("Biomass matrix reconciles to event totals", biomass_error < 1e-8, biomass_error, 0)
    add("Step4 alpha EventID aligned", set(step4["alpha_event"]["EventID"].astype(str)) == set(event_ids), len(set(step4["alpha_event"]["EventID"].astype(str)) ^ set(event_ids)), 0)

    failures = [row for row in qc_rows if not row["Pass"]]
    if failures:
        raise AssertionError("Input QC failed:\n" + pd.DataFrame(failures).to_string(index=False))
    metadata = metadata.sort_values(["Year", "Month", "LocationID", "EventID"], kind="stable").reset_index(drop=True)
    event_ids = metadata["EventID"].astype(str).tolist()
    density = density.set_index("EventID").loc[event_ids].reset_index()
    biomass = biomass.set_index("EventID").loc[event_ids].reset_index()
    return metadata, density, biomass, qc_rows


def diversity_from_matrix(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
    species_columns = [column for column in frame if column.startswith("Species_")]
    values = frame[species_columns].to_numpy(dtype=float)
    totals = values.sum(axis=1)
    richness = (values > 0).sum(axis=1).astype(int)
    proportions = np.divide(
        values,
        totals[:, None],
        out=np.zeros_like(values),
        where=totals[:, None] > 0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        shannon = -np.sum(
            np.where(proportions > 0, proportions * np.log(proportions), 0.0),
            axis=1,
        )
    simpson_d = np.sum(proportions**2, axis=1)
    berger_parker = np.max(proportions, axis=1)
    pielou = np.full(len(frame), np.nan, dtype=float)
    multi_species = richness >= 2
    pielou[multi_species] = shannon[multi_species] / np.log(richness[multi_species])
    no_catch = totals <= 0
    for array in [shannon, simpson_d, berger_parker]:
        array[no_catch] = np.nan
    result = pd.DataFrame({"EventID": frame["EventID"].astype(str)})
    result[f"Richness_{suffix}"] = richness
    result[f"Total_{suffix}"] = totals
    result[f"Shannon_H_{suffix}"] = shannon
    result[f"Pielou_J_{suffix}"] = pielou
    result[f"Simpson_D_{suffix}"] = simpson_d
    result[f"Simpson_1minusD_{suffix}"] = 1.0 - simpson_d
    result.loc[no_catch, f"Simpson_1minusD_{suffix}"] = np.nan
    result[f"Berger_Parker_{suffix}"] = berger_parker
    return result


def build_event_table(
    metadata: pd.DataFrame, density: pd.DataFrame, biomass: pd.DataFrame
) -> pd.DataFrame:
    density_alpha = diversity_from_matrix(density, "Density")
    biomass_alpha = diversity_from_matrix(biomass, "Biomass")
    keep = [
        "EventID",
        "Year",
        "Month",
        "Period",
        "Season",
        "LocationID",
        "Station_Raw",
        "Station_Canonical",
        "Lon",
        "Lat",
        "SurveyStatus",
        "Cephalopod_Presence",
        "Total_Biomass_kg_km2",
        "Total_Density_1000ind_km2",
        "Species_Richness",
    ]
    keep = [column for column in keep if column in metadata]
    result = metadata[keep].copy()
    result["EventID"] = result["EventID"].astype(str)
    result = result.merge(density_alpha, on="EventID", validate="one_to_one")
    result = result.merge(biomass_alpha, on="EventID", validate="one_to_one")
    result["Encounter"] = pd.to_numeric(result["Cephalopod_Presence"]).astype(int)
    result["Zero_Catch"] = result["Encounter"].eq(0)
    result["Single_Species_Catch"] = result["Richness_Density"].eq(1)
    result["Multi_Species_Catch"] = result["Richness_Density"].ge(2)
    richness_error = float(
        np.max(
            np.abs(
                pd.to_numeric(result["Species_Richness"]).to_numpy(dtype=float)
                - result["Richness_Density"].to_numpy(dtype=float)
            )
        )
    )
    if richness_error > 1e-9:
        raise AssertionError(f"Richness reconciliation failed: {richness_error}")
    result["Species_Richness"] = result["Richness_Density"].astype(int)
    return result.sort_values(["Year", "Month", "LocationID", "EventID"]).reset_index(drop=True)


def summarize_metrics(event: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if group_columns:
        iterator = event.groupby(group_columns, observed=True, sort=True, dropna=False)
    else:
        iterator = [((), event)]
    for key, subset in iterator:
        if not isinstance(key, tuple):
            key = (key,)
        group_values = dict(zip(group_columns, key))
        for definition in METRICS:
            metric = definition["Metric"]
            values = pd.to_numeric(subset[metric], errors="coerce").dropna()
            n = len(values)
            mean = float(values.mean()) if n else np.nan
            median = float(values.median()) if n else np.nan
            standard_deviation = float(values.std(ddof=1)) if n > 1 else np.nan
            standard_error = standard_deviation / math.sqrt(n) if n > 1 else np.nan
            critical = float(stats.t.ppf(0.975, n - 1)) if n > 1 else np.nan
            lower = mean - critical * standard_error if n > 1 else np.nan
            upper = mean + critical * standard_error if n > 1 else np.nan
            if definition["Bounded_01"]:
                lower = max(0.0, lower) if np.isfinite(lower) else lower
                upper = min(1.0, upper) if np.isfinite(upper) else upper
            if metric == "Species_Richness":
                lower = max(0.0, lower) if np.isfinite(lower) else lower
            rows.append(
                {
                    **group_values,
                    "Metric": metric,
                    "Metric_Label": definition["Label"],
                    "Role": definition["Role"],
                    "Sample_Basis": definition["Basis"],
                    "N_Surveyed": len(subset),
                    "N_Defined": n,
                    "N_Undefined": len(subset) - n,
                    "Mean": mean,
                    "Median": median,
                    "SD": standard_deviation,
                    "SE": standard_error,
                    "Q25": float(values.quantile(0.25)) if n else np.nan,
                    "Q75": float(values.quantile(0.75)) if n else np.nan,
                    "CI95_Lower": lower,
                    "CI95_Upper": upper,
                    "N_Positive": int(subset["Encounter"].sum()),
                    "N_Zero": int((subset["Encounter"] == 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def season_balanced_annual(year_month_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (year, metric), subset in year_month_summary.groupby(
        ["Year", "Metric"], observed=True
    ):
        subset = subset.sort_values("Month")
        n_months = subset["Month"].nunique()
        mean = float(subset["Mean"].mean())
        se_values = subset["SE"].to_numpy(dtype=float)
        se = float(np.sqrt(np.nansum(se_values**2)) / n_months)
        definition = METRIC_INFO[metric]
        lower = mean - 1.96 * se
        upper = mean + 1.96 * se
        if definition["Bounded_01"]:
            lower, upper = max(0.0, lower), min(1.0, upper)
        if metric == "Species_Richness":
            lower = max(0.0, lower)
        rows.append(
            {
                "Year": int(year),
                "Metric": metric,
                "Metric_Label": definition["Label"],
                "Role": definition["Role"],
                "Season_Balancing": "Equal mean of April and November event-level means",
                "N_Months": int(n_months),
                "N_Surveyed_Total": int(subset["N_Surveyed"].sum()),
                "N_Defined_Total": int(subset["N_Defined"].sum()),
                "Balanced_Mean": mean,
                "Approx_SE": se,
                "Approx_CI95_Lower": lower,
                "Approx_CI95_Upper": upper,
            }
        )
    return pd.DataFrame(rows).sort_values(["Year", "Metric"]).reset_index(drop=True)


def _cluster_sign_flip(
    differences: pd.DataFrame,
    value_column: str,
    cluster_column: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    clean = differences[[value_column, cluster_column]].dropna().copy()
    cluster = (
        clean.groupby(cluster_column, observed=True)[value_column]
        .agg(["sum", "count", "mean"])
        .reset_index()
    )
    sums = cluster["sum"].to_numpy(dtype=float)
    counts = cluster["count"].to_numpy(dtype=float)
    n_clusters = len(cluster)
    n_pairs = int(counts.sum())
    if n_clusters < 2 or n_pairs < 2:
        return {
            "N_Pairs": n_pairs,
            "N_Clusters": n_clusters,
            "Mean_Difference": np.nan,
            "Median_Difference": np.nan,
            "Cluster_Mean_Median": np.nan,
            "CI95_Lower": np.nan,
            "CI95_Upper": np.nan,
            "P_Raw": np.nan,
        }
    observed = float(sums.sum() / counts.sum())
    more_extreme = 0
    batch_size = 500
    completed = 0
    while completed < N_PERMUTATIONS:
        batch = min(batch_size, N_PERMUTATIONS - completed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(batch, n_clusters))
        permuted = np.abs(signs @ sums / counts.sum())
        more_extreme += int(np.sum(permuted >= abs(observed) - 1e-15))
        completed += batch
    p_value = (more_extreme + 1) / (N_PERMUTATIONS + 1)

    bootstrap_values = np.empty(N_BOOTSTRAP, dtype=float)
    for start in range(0, N_BOOTSTRAP, batch_size):
        batch = min(batch_size, N_BOOTSTRAP - start)
        sampled = rng.integers(0, n_clusters, size=(batch, n_clusters))
        bootstrap_values[start : start + batch] = (
            sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
        )
    lower, upper = np.quantile(bootstrap_values, [0.025, 0.975])
    return {
        "N_Pairs": n_pairs,
        "N_Clusters": n_clusters,
        "Mean_Difference": observed,
        "Median_Difference": float(clean[value_column].median()),
        "Cluster_Mean_Median": float(cluster["mean"].median()),
        "CI95_Lower": float(lower),
        "CI95_Upper": float(upper),
        "P_Raw": float(p_value),
    }


def season_paired_tests(
    event: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for definition in METRICS:
        metric = definition["Metric"]
        pivot = event.pivot_table(
            index=["Year", "LocationID"],
            columns="Month",
            values=metric,
            aggfunc="first",
        )
        if 4 not in pivot.columns or 11 not in pivot.columns:
            continue
        paired = pivot[[4, 11]].dropna().reset_index()
        paired["Difference"] = paired[11] - paired[4]
        result = _cluster_sign_flip(paired, "Difference", "LocationID", rng)
        difference_sd = float(paired["Difference"].std(ddof=1)) if len(paired) > 1 else np.nan
        result.update(
            {
                "Metric": metric,
                "Metric_Label": definition["Label"],
                "Role": definition["Role"],
                "Contrast": "November minus April",
                "April_Mean_Paired": float(paired[4].mean()) if len(paired) else np.nan,
                "November_Mean_Paired": float(paired[11].mean()) if len(paired) else np.nan,
                "Paired_Difference_SD": difference_sd,
                "Standardized_Mean_Difference": result["Mean_Difference"] / difference_sd if difference_sd > 0 else np.nan,
                "Standardized_CI95_Lower": result["CI95_Lower"] / difference_sd if difference_sd > 0 else np.nan,
                "Standardized_CI95_Upper": result["CI95_Upper"] / difference_sd if difference_sd > 0 else np.nan,
                "Permutation_Scheme": "One sign per LocationID applied to all year-specific paired differences",
                "N_Permutations": N_PERMUTATIONS,
                "N_Bootstrap": N_BOOTSTRAP,
            }
        )
        rows.append(result)
    output = pd.DataFrame(rows)
    output["P_FDR_Across_Metrics"] = bh_fdr(output["P_Raw"].to_numpy(dtype=float))
    output["FDR_Significant"] = output["P_FDR_Across_Metrics"] < 0.05
    return output


def _center_within_group(values: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    if values.ndim == 1:
        sums = np.zeros(n_groups, dtype=float)
        counts = np.bincount(codes, minlength=n_groups).astype(float)
        np.add.at(sums, codes, values)
        return values - sums[codes] / counts[codes]
    sums = np.zeros((n_groups, values.shape[1]), dtype=float)
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    np.add.at(sums, codes, values)
    return values - sums[codes] / counts[codes, None]


def _permuted_residual_matrix(
    residuals: np.ndarray,
    group_indices: list[np.ndarray],
    batch: int,
    rng: np.random.Generator,
) -> np.ndarray:
    matrix = np.empty((len(residuals), batch), dtype=float)
    for indices in group_indices:
        local = residuals[indices]
        for column in range(batch):
            matrix[indices, column] = rng.permutation(local)
    return matrix


def global_year_permutation_test(
    event: pd.DataFrame,
    metric: str,
    month: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    data = event.loc[event["Month"] == month, ["LocationID", "Year", metric]].dropna()
    counts = data.groupby("LocationID", observed=True).size()
    eligible = counts[counts >= 2].index
    data = data.loc[data["LocationID"].isin(eligible)].copy()
    codes, locations = pd.factorize(data["LocationID"], sort=True)
    n_groups = len(locations)
    y = data[metric].to_numpy(dtype=float)
    years = sorted(data["Year"].astype(int).unique())
    year_matrix = np.column_stack(
        [(data["Year"].astype(int).to_numpy() == year).astype(float) for year in years[1:]]
    )
    y_residual = _center_within_group(y, codes, n_groups)
    x_residual = _center_within_group(year_matrix, codes, n_groups)
    u, singular_values, _ = np.linalg.svd(x_residual, full_matrices=False)
    tolerance = singular_values.max() * max(x_residual.shape) * np.finfo(float).eps
    rank = int(np.sum(singular_values > tolerance))
    q = u[:, :rank]
    sse_reduced = float(y_residual @ y_residual)
    explained = float(np.sum((q.T @ y_residual) ** 2))
    sse_full = max(sse_reduced - explained, 0.0)
    df_residual = len(data) - n_groups - rank
    observed_f = (explained / rank) / (sse_full / df_residual) if rank > 0 and sse_full > 0 and df_residual > 0 else np.nan
    group_indices = [np.flatnonzero(codes == group) for group in range(n_groups)]
    more_extreme = 0
    completed = 0
    batch_size = 100
    while completed < N_PERMUTATIONS:
        batch = min(batch_size, N_PERMUTATIONS - completed)
        permuted = _permuted_residual_matrix(y_residual, group_indices, batch, rng)
        explained_permuted = np.sum((q.T @ permuted) ** 2, axis=0)
        sse_full_permuted = np.maximum(sse_reduced - explained_permuted, 1e-15)
        f_permuted = (explained_permuted / rank) / (sse_full_permuted / df_residual)
        more_extreme += int(np.sum(f_permuted >= observed_f - 1e-15))
        completed += batch
    p_value = (more_extreme + 1) / (N_PERMUTATIONS + 1)
    return {
        "Metric": metric,
        "Metric_Label": METRIC_INFO[metric]["Label"],
        "Role": METRIC_INFO[metric]["Role"],
        "Month": month,
        "Period": MONTH_LABELS[month],
        "N_Events_Used": len(data),
        "N_LocationID_Used": n_groups,
        "N_Years": len(years),
        "F_Partial": observed_f,
        "DF_Year": rank,
        "DF_Residual": df_residual,
        "Partial_R2": explained / sse_reduced if sse_reduced > 0 else np.nan,
        "P_Raw": p_value,
        "N_Permutations": N_PERMUTATIONS,
        "Permutation_Scheme": "Freedman-Lane reduced residuals permuted within LocationID",
    }


def global_year_tests(event: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = [
        global_year_permutation_test(event, definition["Metric"], month, rng)
        for definition in METRICS
        for month in [4, 11]
    ]
    output = pd.DataFrame(rows)
    output["P_FDR_Within_Month"] = output.groupby("Month", observed=True)["P_Raw"].transform(
        lambda values: bh_fdr(values.to_numpy(dtype=float))
    )
    output["FDR_Significant"] = output["P_FDR_Within_Month"] < 0.05
    return output


def adjacent_year_tests(event: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for definition in METRICS:
        metric = definition["Metric"]
        for month in [4, 11]:
            subset = event.loc[event["Month"] == month, ["LocationID", "Year", metric]]
            pivot = subset.pivot_table(index="LocationID", columns="Year", values=metric, aggfunc="first")
            for earlier in range(2014, 2024):
                later = earlier + 1
                if earlier not in pivot.columns or later not in pivot.columns:
                    continue
                paired = pivot[[earlier, later]].dropna().reset_index()
                paired["Difference"] = paired[later] - paired[earlier]
                result = _cluster_sign_flip(paired, "Difference", "LocationID", rng)
                result.update(
                    {
                        "Metric": metric,
                        "Metric_Label": definition["Label"],
                        "Role": definition["Role"],
                        "Month": month,
                        "Period": MONTH_LABELS[month],
                        "From_Year": earlier,
                        "To_Year": later,
                        "Contrast": f"{later} minus {earlier}",
                        "Earlier_Mean_Paired": float(paired[earlier].mean()) if len(paired) else np.nan,
                        "Later_Mean_Paired": float(paired[later].mean()) if len(paired) else np.nan,
                        "Permutation_Scheme": "Paired sign flip across matched LocationID",
                        "N_Permutations": N_PERMUTATIONS,
                        "N_Bootstrap": N_BOOTSTRAP,
                    }
                )
                rows.append(result)
    output = pd.DataFrame(rows)
    output["P_FDR_Within_Metric_Month"] = output.groupby(
        ["Metric", "Month"], observed=True
    )["P_Raw"].transform(lambda values: bh_fdr(values.to_numpy(dtype=float)))
    output["FDR_Significant"] = output["P_FDR_Within_Metric_Month"] < 0.05
    return output


def trend_test(
    event: pd.DataFrame,
    metric: str,
    month: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    data = event.loc[event["Month"] == month, ["LocationID", "Year", metric]].dropna()
    counts = data.groupby("LocationID", observed=True).size()
    data = data.loc[data["LocationID"].isin(counts[counts >= 2].index)].copy()
    codes, locations = pd.factorize(data["LocationID"], sort=True)
    n_groups = len(locations)
    x = data["Year"].to_numpy(dtype=float) - 2014.0
    y = data[metric].to_numpy(dtype=float)
    x_residual = _center_within_group(x, codes, n_groups)
    y_residual = _center_within_group(y, codes, n_groups)
    denominator = float(x_residual @ x_residual)
    slope = float(x_residual @ y_residual / denominator)
    residual = y_residual - slope * x_residual
    group_score = np.array(
        [
            float(np.sum(x_residual[codes == group] * residual[codes == group]))
            for group in range(n_groups)
        ]
    )
    correction = (n_groups / (n_groups - 1)) * ((len(data) - 1) / (len(data) - n_groups - 1))
    robust_variance = correction * np.sum(group_score**2) / (denominator**2)
    robust_se = math.sqrt(max(robust_variance, 0.0))
    group_indices = [np.flatnonzero(codes == group) for group in range(n_groups)]
    more_extreme = 0
    completed = 0
    batch_size = 100
    while completed < N_PERMUTATIONS:
        batch = min(batch_size, N_PERMUTATIONS - completed)
        permuted = _permuted_residual_matrix(y_residual, group_indices, batch, rng)
        slopes = x_residual @ permuted / denominator
        more_extreme += int(np.sum(np.abs(slopes) >= abs(slope) - 1e-15))
        completed += batch
    p_value = (more_extreme + 1) / (N_PERMUTATIONS + 1)
    return {
        "Metric": metric,
        "Metric_Label": METRIC_INFO[metric]["Label"],
        "Role": METRIC_INFO[metric]["Role"],
        "Month": month,
        "Period": MONTH_LABELS[month],
        "N_Events_Used": len(data),
        "N_LocationID_Used": n_groups,
        "Slope_Per_Year": slope,
        "Cluster_Robust_SE": robust_se,
        "CI95_Lower": slope - 1.96 * robust_se,
        "CI95_Upper": slope + 1.96 * robust_se,
        "P_Raw": p_value,
        "N_Permutations": N_PERMUTATIONS,
        "Permutation_Scheme": "Within-LocationID residual permutation after LocationID centering",
    }


def trend_tests(event: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    output = pd.DataFrame(
        [
            trend_test(event, definition["Metric"], month, rng)
            for definition in METRICS
            for month in [4, 11]
        ]
    )
    output["P_FDR_Within_Month"] = output.groupby("Month", observed=True)["P_Raw"].transform(
        lambda values: bh_fdr(values.to_numpy(dtype=float))
    )
    output["FDR_Significant"] = output["P_FDR_Within_Month"] < 0.05
    return output


def reconcile_with_step4(event: pd.DataFrame, step4_event: pd.DataFrame) -> pd.DataFrame:
    previous = step4_event.set_index("EventID").loc[event["EventID"]]
    current = event.set_index("EventID")
    rows: list[dict[str, Any]] = []

    def compare(name: str, previous_column: str, current_column: str, mask: np.ndarray, decision: str) -> None:
        a = pd.to_numeric(previous.loc[mask, previous_column], errors="coerce")
        b = pd.to_numeric(current.loc[mask, current_column], errors="coerce")
        finite = a.notna() & b.notna()
        rows.append(
            {
                "Metric": name,
                "Step4_Column": previous_column,
                "Step4C_Column": current_column,
                "N_Compared": int(finite.sum()),
                "Max_Absolute_Difference": float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else np.nan,
                "Decision": decision,
            }
        )

    all_mask = np.ones(len(event), dtype=bool)
    positive_mask = event["Encounter"].eq(1).to_numpy()
    multi_mask = event["Species_Richness"].ge(2).to_numpy()
    compare("Species richness", "Species_Richness", "Species_Richness", all_mask, "Directly reusable; identical")
    compare("Shannon H' density", "Shannon_H_Density", "Shannon_H_Density", positive_mask, "Directly reusable; identical")
    compare("Simpson 1-D density", "Simpson_1minusD_Density", "Simpson_1minusD_Density", positive_mask, "Directly reusable; identical")
    compare("Pielou J' for S>=2", "Pielou_J_Density", "Pielou_J_Density", multi_mask, "Reusable only for multi-species catches")
    single_previous = pd.to_numeric(previous.loc[event["Species_Richness"].eq(1).to_numpy(), "Pielou_J_Density"], errors="coerce")
    rows.append(
        {
            "Metric": "Pielou J' for S=1",
            "Step4_Column": "Pielou_J_Density",
            "Step4C_Column": "Pielou_J_Density",
            "N_Compared": int(event["Species_Richness"].eq(1).sum()),
            "Max_Absolute_Difference": np.nan,
            "Decision": f"Recomputed: Step4 stored {int(single_previous.eq(0).sum())} values as 0; Step4C uses NA because ln(1)=0",
        }
    )
    rows.append(
        {
            "Metric": "Berger-Parker dominance",
            "Step4_Column": "Not available",
            "Step4C_Column": "Berger_Parker_Density",
            "N_Compared": 0,
            "Max_Absolute_Difference": np.nan,
            "Decision": "New calculation required and supplied",
        }
    )
    return pd.DataFrame(rows)


def build_qc(
    input_qc: list[dict[str, Any]],
    event: pd.DataFrame,
    by_year_month: pd.DataFrame,
    balanced: pd.DataFrame,
    season_tests: pd.DataFrame,
    global_tests: pd.DataFrame,
    adjacent_tests: pd.DataFrame,
    trends: pd.DataFrame,
    reconciliation: pd.DataFrame,
) -> pd.DataFrame:
    rows = list(input_qc)

    def add(check: str, passed: bool, observed: Any, expected: Any, severity: str = "FATAL") -> None:
        rows.append({"Check": check, "Pass": bool(passed), "Observed": observed, "Expected": expected, "Severity": severity})

    zero = event["Zero_Catch"]
    single = event["Single_Species_Catch"]
    multi = event["Multi_Species_Catch"]
    composition_columns = ["Shannon_H_Density", "Pielou_J_Density", "Simpson_D_Density", "Berger_Parker_Density"]
    add("Zero catches retain richness zero", bool(event.loc[zero, "Species_Richness"].eq(0).all()), int(event.loc[zero, "Species_Richness"].ne(0).sum()), 0)
    add("Zero catches have undefined composition indices", bool(event.loc[zero, composition_columns].isna().all().all()), int(event.loc[zero, composition_columns].notna().sum().sum()), 0)
    add("Single-species Shannon equals zero", bool(event.loc[single, "Shannon_H_Density"].eq(0).all()), int(event.loc[single, "Shannon_H_Density"].ne(0).sum()), 0)
    add("Single-species Pielou is undefined", bool(event.loc[single, "Pielou_J_Density"].isna().all()), int(event.loc[single, "Pielou_J_Density"].notna().sum()), 0)
    add("Multi-species Pielou is defined", bool(event.loc[multi, "Pielou_J_Density"].notna().all()), int(event.loc[multi, "Pielou_J_Density"].isna().sum()), 0)
    add("Positive Berger-Parker in valid range", bool(event.loc[~zero, "Berger_Parker_Density"].between(0, 1).all()), int((~event.loc[~zero, "Berger_Parker_Density"].between(0, 1)).sum()), 0)
    add("Positive Simpson D in valid range", bool(event.loc[~zero, "Simpson_D_Density"].between(0, 1).all()), int((~event.loc[~zero, "Simpson_D_Density"].between(0, 1)).sum()), 0)
    add("Year-month summary rows", len(by_year_month) == 22 * len(METRICS), len(by_year_month), 22 * len(METRICS))
    add("Season-balanced annual rows", len(balanced) == 11 * len(METRICS), len(balanced), 11 * len(METRICS))
    add("All balanced annual estimates use two months", bool(balanced["N_Months"].eq(2).all()), sorted(balanced["N_Months"].unique()), [2])
    add("Season paired tests completed", len(season_tests) == len(METRICS), len(season_tests), len(METRICS))
    add("Global year tests completed", len(global_tests) == 2 * len(METRICS), len(global_tests), 2 * len(METRICS))
    add("Adjacent-year tests completed", len(adjacent_tests) == 20 * len(METRICS), len(adjacent_tests), 20 * len(METRICS))
    add("Trend tests completed", len(trends) == 2 * len(METRICS), len(trends), 2 * len(METRICS))
    p_columns = [season_tests["P_Raw"], global_tests["P_Raw"], adjacent_tests["P_Raw"], trends["P_Raw"]]
    all_p = pd.concat(p_columns, ignore_index=True)
    add("All inferential P values valid", bool(all_p.dropna().between(0, 1).all()), int((~all_p.dropna().between(0, 1)).sum()), 0)
    max_reconciliation = reconciliation.loc[reconciliation["N_Compared"] > 0, "Max_Absolute_Difference"].dropna().max()
    add("Recalculated indices reconcile with reusable Step4 values", bool(max_reconciliation < 1e-10), max_reconciliation, 0)
    result = pd.DataFrame(rows)
    result["Pass"] = result["Pass"].astype(bool)
    return result


def plot_year_month(by_year_month: pd.DataFrame, figure_dir: Path) -> None:
    colors = {4: "#0072B2", 11: "#D55E00"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for ax, metric in zip(axes.flat, PRIMARY_FIGURE_METRICS):
        subset = by_year_month.loc[by_year_month["Metric"] == metric]
        for month in [4, 11]:
            line = subset.loc[subset["Month"] == month].sort_values("Year")
            x = line["Year"].to_numpy(dtype=float)
            mean = line["Mean"].to_numpy(dtype=float)
            lower = line["CI95_Lower"].to_numpy(dtype=float)
            upper = line["CI95_Upper"].to_numpy(dtype=float)
            ax.plot(x, mean, color=colors[month], marker="o", linewidth=2, label=MONTH_LABELS[month])
            ax.fill_between(x, lower, upper, color=colors[month], alpha=0.16)
        ax.set_title(METRIC_INFO[metric]["Label"])
        ax.set_xticks(range(2014, 2025))
        ax.tick_params(axis="x", rotation=45)
        ax.grid(color="#D9D9D9", linewidth=0.6, alpha=0.7)
        ax.set_ylabel("Event-level mean (95% CI)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Seasonal and interannual variation in alpha diversity", fontsize=14, y=1.01)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, figure_dir / "Fig01_alpha_diversity_year_month")


def plot_month_distributions(event: pd.DataFrame, figure_dir: Path) -> None:
    colors = ["#56B4E9", "#E69F00"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, metric in zip(axes.flat, PRIMARY_FIGURE_METRICS):
        values = [
            pd.to_numeric(event.loc[event["Month"] == month, metric], errors="coerce").dropna().to_numpy(dtype=float)
            for month in [4, 11]
        ]
        box = ax.boxplot(values, patch_artist=True, showfliers=False, widths=0.58)
        ax.set_xticks([1, 2], ["April", "November"])
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for median in box["medians"]:
            median.set_color("#222222")
            median.set_linewidth(1.5)
        means = [np.mean(value) for value in values]
        ax.scatter([1, 2], means, color="#111111", marker="D", s=28, zorder=3, label="Mean")
        ax.set_title(METRIC_INFO[metric]["Label"])
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Event-level distributions in April and November", fontsize=14, y=1.01)
    fig.tight_layout()
    save_figure(fig, figure_dir / "Fig02_alpha_diversity_month_distributions")


def plot_season_effects(season_tests: pd.DataFrame, figure_dir: Path) -> None:
    subset = season_tests.loc[season_tests["Metric"].isin(["Encounter", *PRIMARY_FIGURE_METRICS])].copy()
    subset = subset.set_index("Metric").loc[["Encounter", *PRIMARY_FIGURE_METRICS]].reset_index()
    y = np.arange(len(subset))
    estimates = subset["Standardized_Mean_Difference"].to_numpy(dtype=float)
    lower = subset["Standardized_CI95_Lower"].to_numpy(dtype=float)
    upper = subset["Standardized_CI95_Upper"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.errorbar(
        estimates,
        y,
        xerr=np.vstack([estimates - lower, upper - estimates]),
        fmt="o",
        color="#0072B2",
        ecolor="#666666",
        capsize=4,
        markersize=6,
    )
    ax.axvline(0, color="#222222", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([METRIC_INFO[metric]["Short_Label"] for metric in subset["Metric"]])
    ax.invert_yaxis()
    ax.set_xlabel("Standardized paired difference (November minus April)")
    ax.set_title("Paired seasonal effects with LocationID-restricted inference")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    for index, row in subset.iterrows():
        label = f"FDR p={row['P_FDR_Across_Metrics']:.3g}"
        ax.text(upper[index] + 0.04, index, label, va="center", fontsize=8)
    fig.tight_layout()
    save_figure(fig, figure_dir / "Fig03_paired_season_effects")


def metadata_table(step3_zip: Path, step4_zip: Path) -> pd.DataFrame:
    rows = [
        ("Script_Version", SCRIPT_VERSION),
        ("Random_Seed", RANDOM_SEED),
        ("Python", platform.python_version()),
        ("NumPy", np.__version__),
        ("Pandas", pd.__version__),
        ("SciPy", scipy.__version__),
        ("Matplotlib", matplotlib.__version__),
        ("Step3_ZIP", str(step3_zip)),
        ("Step3_SHA256", sha256_file(step3_zip)),
        ("Step4_ZIP", str(step4_zip)),
        ("Step4_SHA256", sha256_file(step4_zip)),
        ("Surveyed_Events", EXPECTED_EVENTS),
        ("Positive_Events", EXPECTED_POSITIVE),
        ("Zero_Events", EXPECTED_ZERO),
        ("Species", EXPECTED_SPECIES),
        ("Primary_Weight", "Numerical density across all 25 species"),
        ("Zero_Catch_Rule", "Richness=0; composition-based indices=NA"),
        ("Single_Species_Pielou_Rule", "NA because J'=H'/ln(S) and ln(1)=0"),
        ("Dominance_Primary", "Berger-Parker d=max(p_i)"),
        ("Dominance_Sensitivity", "Simpson D=sum(p_i^2)"),
        ("Season_Test", "Exact Year+LocationID pairing; cluster sign flip by LocationID"),
        ("Global_Year_Test", "Freedman-Lane residual permutation within LocationID"),
        ("Adjacent_Year_Test", "Matched LocationID paired sign flip"),
        ("P_Adjustment", "Benjamini-Hochberg FDR within prespecified test family"),
        ("Permutations", N_PERMUTATIONS),
        ("Cluster_Bootstrap_Replicates", N_BOOTSTRAP),
    ]
    return pd.DataFrame(rows, columns=["Parameter", "Value"])


def summary_table(
    step3_zip: Path,
    step4_zip: Path,
    event: pd.DataFrame,
    by_month: pd.DataFrame,
    season_tests: pd.DataFrame,
    global_tests: pd.DataFrame,
    trends: pd.DataFrame,
    qc: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[tuple[str, Any]] = [
        ("Step", "Step 4C community alpha diversity"),
        ("Script version", SCRIPT_VERSION),
        ("Step3 input", str(step3_zip)),
        ("Step4 input", str(step4_zip)),
        ("Surveyed events", len(event)),
        ("Positive events", int(event["Encounter"].sum())),
        ("Zero catches", int(event["Zero_Catch"].sum())),
        ("Single-species positive catches", int(event["Single_Species_Catch"].sum())),
        ("Multi-species positive catches", int(event["Multi_Species_Catch"].sum())),
        ("Permutations", N_PERMUTATIONS),
        ("Bootstrap replicates", N_BOOTSTRAP),
        ("Fatal QC failures", int(((qc["Severity"] == "FATAL") & ~qc["Pass"]).sum())),
    ]
    for metric in ["Encounter", *PRIMARY_FIGURE_METRICS, "Simpson_D_Density"]:
        for month in [4, 11]:
            record = by_month.loc[(by_month["Metric"] == metric) & (by_month["Month"] == month)].iloc[0]
            rows.append((f"{METRIC_INFO[metric]['Short_Label']} mean - {MONTH_LABELS[month]}", record["Mean"]))
        test = season_tests.loc[season_tests["Metric"] == metric].iloc[0]
        rows.extend(
            [
                (f"{METRIC_INFO[metric]['Short_Label']} paired Nov-Apr difference", test["Mean_Difference"]),
                (f"{METRIC_INFO[metric]['Short_Label']} season FDR P", test["P_FDR_Across_Metrics"]),
            ]
        )
    for month in [4, 11]:
        rows.append((f"Global year tests FDR-significant - {MONTH_LABELS[month]}", int(global_tests.loc[(global_tests["Month"] == month) & global_tests["FDR_Significant"], "Metric"].nunique())))
        rows.append((f"Linear trends FDR-significant - {MONTH_LABELS[month]}", int(trends.loc[(trends["Month"] == month) & trends["FDR_Significant"], "Metric"].nunique())))
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def build_readme(
    by_month: pd.DataFrame,
    season_tests: pd.DataFrame,
    qc: pd.DataFrame,
) -> str:
    def mean(metric: str, month: int) -> float:
        return float(by_month.loc[(by_month["Metric"] == metric) & (by_month["Month"] == month), "Mean"].iloc[0])

    return f"""# Step 4C：群落α多样性（论文3.2）

## 数据与定义

- 使用2,100个已调查事件和全部25种：1,532个正捕获、568个零捕获。
- 丰富度使用全部已调查事件，零捕获记为0。
- Shannon基于数值密度比例，仅零捕获为NA；单物种事件H'=0。
- Pielou均匀度只对S>=2事件计算。S=1时`ln(S)=0`，因此记为NA，不再沿用第四步的0。
- Berger-Parker `d=max(p_i)`为主要优势度；Simpson `D=sum(p_i^2)`为敏感性优势度。两者越大，优势集中越强。
- 主要结果使用数值密度，事件表同时保留生物量加权结果供敏感性比较。

## 当前描述性结果

| 指标 | 4月均值 | 11月均值 |
|---|---:|---:|
| 丰富度S | {mean('Species_Richness', 4):.3f} | {mean('Species_Richness', 11):.3f} |
| Shannon H' | {mean('Shannon_H_Density', 4):.3f} | {mean('Shannon_H_Density', 11):.3f} |
| Pielou J'（S>=2） | {mean('Pielou_J_Density', 4):.3f} | {mean('Pielou_J_Density', 11):.3f} |
| Berger-Parker d | {mean('Berger_Parker_Density', 4):.3f} | {mean('Berger_Parker_Density', 11):.3f} |

季节显著性以`05_season_paired_cluster_permutation.csv`为准，不能仅依据均值或置信区间是否重叠判断。

## 推断框架

- 4月—11月：同一Year+LocationID配对；同一LocationID跨年份的差值整体翻转，保留重复测量结构。
- 年份总体效应：在每个月内先控制LocationID固定差异，再将约化模型残差限制在LocationID内部置换。
- 相邻年份：同一月份、同一LocationID配对，检验后按指标×月份进行BH-FDR。
- 线性趋势：先消除LocationID均值，输出每年斜率、聚类稳健SE及限制置换P值。

## 主要文件

- `01_alpha_diversity_event_recalculated.csv`：事件级完整指数。
- `02`—`04`：月份、年—月及季节均衡年度描述性结果。
- `05`：正式4月—11月配对检验。
- `06`：各月份年份总体效应。
- `07`：相邻年份配对比较。
- `08`：各月份线性趋势。
- `09`：与第四步原结果的逐项核对及Pielou修正说明。
- `11_qc_checks.csv`：全部质控。
- `figures/`：PNG（{FIGURE_DPI} dpi）、PDF和SVG。

## 质控

- 致命失败：{int(((qc['Severity'] == 'FATAL') & ~qc['Pass']).sum())}
- 只有`Severity=FATAL`且`Pass=False`时不得用于论文。
"""


def manifest_table(directory: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "12_output_manifest.csv":
            rows.append(
                {
                    "Relative_Path": str(path.relative_to(directory)),
                    "Size_Bytes": path.stat().st_size,
                    "SHA256": sha256_file(path),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    configure_plot_style()
    rng = np.random.default_rng(RANDOM_SEED)
    step3_zip = discover_zip(
        "CEPHALOPOD_STEP3_ZIP",
        "cephalopod_step3_analysis_master*.zip",
        STEP3_MEMBERS,
    )
    step4_zip = discover_zip(
        "CEPHALOPOD_STEP4_ZIP",
        "cephalopod_step4_resource_iri_alpha*.zip",
        STEP4_MEMBERS,
    )
    log(f"Step3 input: {step3_zip}")
    log(f"Step4 input: {step4_zip}")
    step3 = read_zip_tables(step3_zip, STEP3_MEMBERS)
    step4 = read_zip_tables(step4_zip, STEP4_MEMBERS)
    metadata, density, biomass, input_qc = align_and_validate_inputs(step3, step4)
    event = build_event_table(metadata, density, biomass)
    log("Event-level indices recalculated; Pielou S=1 correction applied")

    by_month = summarize_metrics(event, ["Month", "Period"])
    by_year_month = summarize_metrics(event, ["Year", "Month", "Period"])
    balanced = season_balanced_annual(by_year_month)
    season_tests = season_paired_tests(event, rng)
    log("Paired April-November cluster permutation completed")
    global_tests = global_year_tests(event, rng)
    log("Within-LocationID global year permutation completed")
    adjacent_tests = adjacent_year_tests(event, rng)
    trends = trend_tests(event, rng)
    log("Adjacent-year and linear-trend tests completed")

    reconciliation = reconcile_with_step4(event, step4["alpha_event"])
    qc = build_qc(
        input_qc,
        event,
        by_year_month,
        balanced,
        season_tests,
        global_tests,
        adjacent_tests,
        trends,
        reconciliation,
    )
    fatal = qc.loc[(qc["Severity"] == "FATAL") & ~qc["Pass"]]
    if not fatal.empty:
        raise AssertionError("Fatal QC failures:\n" + fatal.to_string(index=False))

    temp_dir = prepare_temp_dir(FINAL_DIR)
    try:
        figure_dir = temp_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        atomic_text("RUNNING\n", temp_dir / "RUN_STATUS.txt")
        outputs = [
            ("00_step4c_summary.csv", summary_table(step3_zip, step4_zip, event, by_month, season_tests, global_tests, trends, qc)),
            ("01_alpha_diversity_event_recalculated.csv", event),
            ("02_alpha_summary_by_month.csv", by_month),
            ("03_alpha_summary_by_year_month.csv", by_year_month),
            ("04_alpha_annual_season_balanced.csv", balanced),
            ("05_season_paired_cluster_permutation.csv", season_tests),
            ("06_year_global_within_location_permutation.csv", global_tests),
            ("07_adjacent_year_paired_permutation.csv", adjacent_tests),
            ("08_linear_trend_within_location_permutation.csv", trends),
            ("09_step4_reconciliation_and_pielou_correction.csv", reconciliation),
            ("10_method_metadata.csv", metadata_table(step3_zip, step4_zip)),
            ("11_qc_checks.csv", qc),
        ]
        for filename, frame in outputs:
            atomic_csv(frame, temp_dir / filename)

        plot_year_month(by_year_month, figure_dir)
        plot_month_distributions(event, figure_dir)
        plot_season_effects(season_tests, figure_dir)
        atomic_text(build_readme(by_month, season_tests, qc), temp_dir / "README_step4c.md")
        source = Path(globals().get("__file__", ""))
        if source.is_file():
            shutil.copy2(source, temp_dir / source.name)
        atomic_csv(manifest_table(temp_dir), temp_dir / "12_output_manifest.csv")
        atomic_text(
            f"SUCCESS\nScript_Version={SCRIPT_VERSION}\nElapsed={elapsed()}\n",
            temp_dir / "RUN_STATUS.txt",
        )
        publish_dir(temp_dir, FINAL_DIR)
        build_zip(FINAL_DIR, FINAL_ZIP)
    except Exception:
        atomic_text("FAILED\n" + traceback.format_exc(), temp_dir / "RUN_STATUS.txt")
        raise

    log(f"SUCCESS: {FINAL_ZIP}")
    print("请下载：cephalopod_step4c_community_diversity.zip", flush=True)


if __name__ == "__main__":
    main()