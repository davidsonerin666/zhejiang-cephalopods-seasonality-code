#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4B — Major-group composition, core dominant species and turnover.

Input
-----
cephalopod_step4_resource_iri_alpha.zip

Main outputs
------------
1. Auditable 25-species to three-major-group mapping.
2. Biomass and numerical-density composition by major group:
   overall, month, year-month and season-balanced year.
3. Overall/month-specific IRI summaries.
4. Core-species stability and dominant-species turnover metrics.
5. Optional temporal GEE tests for occurrence, total biomass and density.
6. Publication figures in PNG/PDF/SVG and a verified output ZIP.

Important taxonomy note
-----------------------
The three groups used here are analysis groups, not a claim that all accepted
modern cephalopod taxonomy has exactly three orders. The embedded legacy labels
are provided only for compatibility with traditional Chinese fisheries usage.
The existing numeric Species_Order field in Step 4 is a display sequence (1-25)
and is never used as a taxonomic order.

Typical Kaggle use
------------------
import os, runpy
os.environ["CEPHALOPOD_STEP4_ZIP"] = (
    "/kaggle/input/<dataset>/cephalopod_step4_resource_iri_alpha.zip"
)
runpy.run_path(
    "/kaggle/input/<code-dataset>/kaggle_step4b_species_composition_turnover.py",
    run_name="__main__",
)
"""

from __future__ import annotations

import hashlib
import itertools
import math
import os
import json
import platform
import shutil
import sys
import tempfile
import time
import traceback
import warnings
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import scipy
from packaging.version import Version
from scipy import stats
from scipy.special import expit

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


SCRIPT_VERSION = "4B.1.0"
RANDOM_SEED = int(os.environ.get("CEPHALOPOD_STEP4B_SEED", "20260902"))
CORE_TOP5_PROPORTION = float(
    os.environ.get("CEPHALOPOD_STEP4B_CORE_TOP5_PROPORTION", "0.75")
)
RUN_TEMPORAL_GEE = os.environ.get("CEPHALOPOD_STEP4B_RUN_GEE", "0") == "1"
GEE_MAXITER = int(os.environ.get("CEPHALOPOD_STEP4B_GEE_MAXITER", "200"))
GEE_CTOL = float(os.environ.get("CEPHALOPOD_STEP4B_GEE_CTOL", "1e-6"))
TWEEDIE_POWER = float(os.environ.get("CEPHALOPOD_STEP4B_TWEEDIE_POWER", "1.5"))
FIGURE_DPI = int(os.environ.get("CEPHALOPOD_STEP4B_FIGURE_DPI", "600"))

OUTPUT_ROOT = Path(
    os.environ.get("CEPHALOPOD_STEP4B_OUTPUT_ROOT", "/kaggle/working")
).expanduser()
if not OUTPUT_ROOT.exists() and str(OUTPUT_ROOT).startswith("/kaggle/"):
    OUTPUT_ROOT = Path.cwd()

FINAL_DIR = OUTPUT_ROOT / "cephalopod_step4b_composition_turnover"
FINAL_ZIP = OUTPUT_ROOT / "cephalopod_step4b_composition_turnover.zip"
START_TIME = time.time()

MONTH_LABELS = {4: "April", 11: "November"}
GROUP_ORDER = ["Squids", "Cuttlefishes_and_bobtail_squids", "Octopuses"]
GROUP_LABELS = {
    "Squids": {
        "Group_CN": "枪乌贼类",
        "Group_EN": "Squids",
        "Legacy_Order_CN": "枪形目",
        "Legacy_Order_EN": "Teuthida (legacy broad grouping)",
        "Color": "#0072B2",
    },
    "Cuttlefishes_and_bobtail_squids": {
        "Group_CN": "乌贼及耳乌贼类",
        "Group_EN": "Cuttlefishes and bobtail squids",
        "Legacy_Order_CN": "乌贼目",
        "Legacy_Order_EN": "Sepiida (legacy broad grouping)",
        "Color": "#E69F00",
    },
    "Octopuses": {
        "Group_CN": "蛸类",
        "Group_EN": "Octopuses",
        "Legacy_Order_CN": "八腕目",
        "Legacy_Order_EN": "Octopoda",
        "Color": "#009E73",
    },
}

# Analysis-group mapping. This deliberately uses Species_Code as the stable key.
SPECIES_GROUP = {
    "Species_01": "Squids",   # 中国枪乌贼
    "Species_02": "Squids",   # 剑尖枪乌贼
    "Species_03": "Octopuses",  # 卵蛸
    "Species_04": "Cuttlefishes_and_bobtail_squids",  # 双喙耳乌贼
    "Species_05": "Cuttlefishes_and_bobtail_squids",  # 四盘耳乌贼
    "Species_06": "Squids",   # 多钩钩腕乌贼
    "Species_07": "Squids",   # 太平洋褶柔鱼
    "Species_08": "Squids",   # 尤氏枪乌贼
    "Species_09": "Squids",   # 日本枪乌贼
    "Species_10": "Cuttlefishes_and_bobtail_squids",  # 曼氏无针乌贼
    "Species_11": "Squids",   # 杜氏枪乌贼
    "Species_12": "Octopuses",  # 条纹蛸
    "Species_13": "Squids",   # 枪乌贼
    "Species_14": "Cuttlefishes_and_bobtail_squids",  # 柏氏四盘耳乌贼
    "Species_15": "Squids",   # 火枪乌贼
    "Species_16": "Octopuses",  # 真蛸
    "Species_17": "Octopuses",  # 短蛸
    "Species_18": "Squids",   # 神户枪乌贼
    "Species_19": "Octopuses",  # 紫水孔蛸
    "Species_20": "Cuttlefishes_and_bobtail_squids",  # 耳乌贼
    "Species_21": "Octopuses",  # 船蛸
    "Species_22": "Cuttlefishes_and_bobtail_squids",  # 虎斑乌贼
    "Species_23": "Cuttlefishes_and_bobtail_squids",  # 金乌贼
    "Species_24": "Octopuses",  # 长蛸
    "Species_25": "Squids",   # 鸢乌贼
}

REQUIRED_MEMBERS = {
    "summary": "00_step4_summary.csv",
    "events": "01_resource_event_level.csv",
    "species_overall": "06_species_resource_overall.csv",
    "species_month": "07_species_resource_by_month.csv",
    "species_year_month": "09_species_resource_by_year_month.csv",
    "iri_overall": "10_IRI_overall.csv",
    "iri_month": "11_IRI_by_month.csv",
    "iri_year_month": "13_IRI_by_year_month.csv",
    "step4_qc": "21_qc_checks.csv",
}


def elapsed() -> str:
    return f"{(time.time() - START_TIME) / 60:.1f} min"


def log(message: str) -> None:
    print(f"[{elapsed()}] {message}", flush=True)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.shape[1] == 0:
        frame = pd.DataFrame({"Status": pd.Series(dtype="string")})
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def atomic_text(text: str, path: Path) -> None:
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


def save_figure(fig: plt.Figure, stem: Path) -> None:
    for suffix in ["png", "pdf", "svg"]:
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = FIGURE_DPI
        fig.savefig(stem.with_suffix("." + suffix), **kwargs)
    plt.close(fig)


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
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(source_dir))
    with zipfile.ZipFile(tmp) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Output ZIP failed CRC validation")
    os.replace(tmp, destination)


def valid_step4_zip(path: Path) -> tuple[bool, str]:
    try:
        if not path.is_file() or not zipfile.is_zipfile(path):
            return False, "not a ZIP file"
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            missing = sorted(set(REQUIRED_MEMBERS.values()) - names)
            if missing:
                return False, "missing " + ", ".join(missing)
            bad = archive.testzip()
            if bad is not None:
                return False, f"CRC failure: {bad}"
        return True, "OK"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def discover_step4_zip() -> Path:
    explicit = os.environ.get("CEPHALOPOD_STEP4_ZIP", "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        valid, reason = valid_step4_zip(path)
        if not valid:
            raise FileNotFoundError(f"Invalid CEPHALOPOD_STEP4_ZIP: {path}\n{reason}")
        return path

    candidates: list[Path] = []
    roots = [Path.cwd(), OUTPUT_ROOT, Path("/kaggle/input")]
    for root in roots:
        if not root.exists():
            continue
        iterator = (
            root.rglob("cephalopod_step4_resource_iri_alpha*.zip")
            if str(root) == "/kaggle/input"
            else root.glob("cephalopod_step4_resource_iri_alpha*.zip")
        )
        for path in iterator:
            valid, _ = valid_step4_zip(path)
            if valid:
                candidates.append(path.resolve())
    candidates = sorted(set(candidates), key=lambda p: (p.stat().st_size, str(p)), reverse=True)
    if not candidates:
        raise FileNotFoundError(
            "未找到cephalopod_step4_resource_iri_alpha.zip。请加入Kaggle Input，"
            "或设置os.environ['CEPHALOPOD_STEP4_ZIP']='ZIP完整路径'。"
        )
    if len(candidates) > 1:
        log(f"Multiple valid Step-4 ZIPs found; selected {candidates[0]}")
    return candidates[0]


def read_inputs(path: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(path) as archive:
        for key, member in REQUIRED_MEMBERS.items():
            frames[key] = pd.read_csv(archive.open(member), low_memory=False)
    return frames


def mapping_table(species: pd.DataFrame) -> pd.DataFrame:
    base = species[["Species_Code", "Species_CN", "Species_Latin"]].drop_duplicates()
    if len(base) != 25 or not base["Species_Code"].is_unique:
        raise AssertionError("Expected exactly 25 unique Species_Code values")
    expected = set(SPECIES_GROUP)
    observed = set(base["Species_Code"].astype(str))
    if observed != expected:
        raise AssertionError(
            f"Species mapping mismatch. Missing={sorted(observed - expected)}; "
            f"extra mapping={sorted(expected - observed)}"
        )
    base["Analysis_Group"] = base["Species_Code"].map(SPECIES_GROUP)
    base["Group_CN"] = base["Analysis_Group"].map(
        {key: value["Group_CN"] for key, value in GROUP_LABELS.items()}
    )
    base["Group_EN"] = base["Analysis_Group"].map(
        {key: value["Group_EN"] for key, value in GROUP_LABELS.items()}
    )
    base["Legacy_Order_CN"] = base["Analysis_Group"].map(
        {key: value["Legacy_Order_CN"] for key, value in GROUP_LABELS.items()}
    )
    base["Legacy_Order_EN"] = base["Analysis_Group"].map(
        {key: value["Legacy_Order_EN"] for key, value in GROUP_LABELS.items()}
    )
    base["Mapping_Type"] = "Analysis group; not asserted as a unified modern order"
    base["Mapping_Review_Required"] = False
    order = {code: i for i, code in enumerate(SPECIES_GROUP, start=1)}
    base["Display_Order"] = base["Species_Code"].map(order)
    return base.sort_values("Display_Order").reset_index(drop=True)


def validate_inputs(
    frames: dict[str, pd.DataFrame], mapping: pd.DataFrame
) -> list[dict[str, Any]]:
    qc: list[dict[str, Any]] = []

    def add(check: str, passed: bool, observed: Any, expected: Any, severity: str = "FATAL") -> None:
        qc.append(
            {
                "Check": check,
                "Pass": bool(passed),
                "Observed": observed,
                "Expected": expected,
                "Severity": severity,
            }
        )

    events = frames["events"]
    iriym = frames["iri_year_month"]
    species_ym = frames["species_year_month"]
    step4_qc = frames["step4_qc"]
    add("Step4 QC all passed", bool(step4_qc["Pass"].astype(bool).all()), int((~step4_qc["Pass"].astype(bool)).sum()), 0)
    add("Surveyed events", len(events) == 2100, len(events), 2100)
    add("EventID unique", events["EventID"].is_unique, events["EventID"].nunique(), 2100)
    add("Positive events", int(events["Cephalopod_Presence"].sum()) == 1532, int(events["Cephalopod_Presence"].sum()), 1532)
    add("Species mapping rows", len(mapping) == 25, len(mapping), 25)
    add("Three analysis groups", mapping["Analysis_Group"].nunique() == 3, mapping["Analysis_Group"].nunique(), 3)
    add("Year-month species rows", len(species_ym) == 550, len(species_ym), 550)
    add("Year-month IRI rows", len(iriym) == 550, len(iriym), 550)
    add("Study years", set(events["Year"].astype(int)) == set(range(2014, 2025)), sorted(events["Year"].unique()), "2014-2024")
    add("Study months", set(events["Month"].astype(int)) == {4, 11}, sorted(events["Month"].unique()), [4, 11])
    add("Nonnegative total biomass", bool((events["Total_Biomass_kg_km2"] >= 0).all()), int((events["Total_Biomass_kg_km2"] < 0).sum()), 0)
    add("Nonnegative total density", bool((events["Total_Density_1000ind_km2"] >= 0).all()), int((events["Total_Density_1000ind_km2"] < 0).sum()), 0)
    iri_sums = iriym.groupby(["Year", "Month"], observed=True)["IRI_Percent"].sum()
    add("Year-month %IRI sums to 100", float(np.max(np.abs(iri_sums - 100))) < 1e-8, float(np.max(np.abs(iri_sums - 100))), 0)
    fatal = [row for row in qc if row["Severity"] == "FATAL" and not row["Pass"]]
    if fatal:
        raise AssertionError("Input QC failed:\n" + pd.DataFrame(fatal).to_string(index=False))
    return qc


def add_group_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["Group_CN"] = result["Analysis_Group"].map(
        {key: value["Group_CN"] for key, value in GROUP_LABELS.items()}
    )
    result["Group_EN"] = result["Analysis_Group"].map(
        {key: value["Group_EN"] for key, value in GROUP_LABELS.items()}
    )
    result["Group_Order"] = result["Analysis_Group"].map(
        {group: index for index, group in enumerate(GROUP_ORDER, start=1)}
    )
    return result


def aggregate_major_groups(
    species_table: pd.DataFrame,
    mapping: pd.DataFrame,
    grouping: Sequence[str],
    scope: str,
) -> pd.DataFrame:
    data = species_table.drop(columns=["Species_Order"], errors="ignore").merge(
        mapping[["Species_Code", "Analysis_Group"]],
        on="Species_Code",
        how="left",
        validate="many_to_one",
    )
    if data["Analysis_Group"].isna().any():
        raise AssertionError("Unmapped species found during major-group aggregation")

    keys = [*grouping, "Analysis_Group"]
    aggregated = (
        data.groupby(keys, observed=True, sort=True)
        .agg(
            N_Surveyed=("N_Surveyed", "first"),
            N_Species_Recorded=("N_Occurrence", lambda x: int((pd.to_numeric(x, errors="coerce").fillna(0) > 0).sum())),
            Sum_Biomass_kg_km2=("Sum_Biomass_kg_km2", "sum"),
            Mean_Biomass_AllSurveyed_kg_km2=("Mean_Biomass_AllSurveyed_kg_km2", "sum"),
            Sum_Density_1000ind_km2=("Sum_Density_1000ind_km2", "sum"),
            Mean_Density_AllSurveyed_1000ind_km2=("Mean_Density_AllSurveyed_1000ind_km2", "sum"),
        )
        .reset_index()
    )
    group_sizes = mapping.groupby("Analysis_Group", observed=True).size()
    aggregated["N_Species_Assigned"] = aggregated["Analysis_Group"].map(group_sizes)

    denominator_keys = list(grouping)
    if denominator_keys:
        total_biomass = aggregated.groupby(denominator_keys, observed=True)[
            "Sum_Biomass_kg_km2"
        ].transform("sum")
        total_density = aggregated.groupby(denominator_keys, observed=True)[
            "Sum_Density_1000ind_km2"
        ].transform("sum")
    else:
        total_biomass = pd.Series(
            aggregated["Sum_Biomass_kg_km2"].sum(), index=aggregated.index
        )
        total_density = pd.Series(
            aggregated["Sum_Density_1000ind_km2"].sum(), index=aggregated.index
        )
    aggregated["Biomass_Share_Percent"] = np.where(
        total_biomass > 0,
        aggregated["Sum_Biomass_kg_km2"] / total_biomass * 100,
        np.nan,
    )
    aggregated["Density_Share_Percent"] = np.where(
        total_density > 0,
        aggregated["Sum_Density_1000ind_km2"] / total_density * 100,
        np.nan,
    )
    aggregated.insert(0, "Scope", scope)
    aggregated = add_group_labels(aggregated)
    sort_columns = [*grouping, "Group_Order"]
    return aggregated.sort_values(sort_columns).reset_index(drop=True)


def season_balanced_annual(group_year_month: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (year, group), subset in group_year_month.groupby(
        ["Year", "Analysis_Group"], observed=True
    ):
        months = sorted(subset["Month"].astype(int).unique())
        rows.append(
            {
                "Year": int(year),
                "Analysis_Group": group,
                "N_Months": len(months),
                "Months": ",".join(map(str, months)),
                "Season_Balancing": "Equal mean of April and November event means",
                "Balanced_Mean_Biomass_kg_km2": float(
                    subset["Mean_Biomass_AllSurveyed_kg_km2"].mean()
                ),
                "Balanced_Mean_Density_1000ind_km2": float(
                    subset["Mean_Density_AllSurveyed_1000ind_km2"].mean()
                ),
                "Mean_Monthly_Biomass_Share_Percent": float(
                    subset["Biomass_Share_Percent"].mean()
                ),
                "Mean_Monthly_Density_Share_Percent": float(
                    subset["Density_Share_Percent"].mean()
                ),
            }
        )
    result = add_group_labels(pd.DataFrame(rows))
    result["Balanced_Biomass_Share_Percent"] = (
        result["Balanced_Mean_Biomass_kg_km2"]
        / result.groupby("Year", observed=True)[
            "Balanced_Mean_Biomass_kg_km2"
        ].transform("sum")
        * 100
    )
    result["Balanced_Density_Share_Percent"] = (
        result["Balanced_Mean_Density_1000ind_km2"]
        / result.groupby("Year", observed=True)[
            "Balanced_Mean_Density_1000ind_km2"
        ].transform("sum")
        * 100
    )
    return result.sort_values(["Year", "Group_Order"]).reset_index(drop=True)


def iri_reporting_table(
    overall: pd.DataFrame, by_month: pd.DataFrame
) -> pd.DataFrame:
    overall_table = overall.copy()
    overall_table.insert(0, "Reporting_Scope", "Overall")
    overall_table["Reporting_Label"] = "Overall"
    month_table = by_month.copy()
    month_table.insert(0, "Reporting_Scope", "Month")
    month_table["Reporting_Label"] = month_table["Period"]
    keep = [
        "Reporting_Scope",
        "Reporting_Label",
        "Month",
        "Species_Code",
        "Species_CN",
        "Species_Latin",
        "Frequency_Percent",
        "Abundance_Percent_N",
        "Biomass_Percent_W",
        "IRI",
        "IRI_Percent",
        "IRI_Rank",
        "IRI_Class_Absolute",
        "IRI_Class_Relative",
        "Dominant_By_Absolute_IRI",
        "Dominant_By_Relative_IRI",
    ]
    for frame in [overall_table, month_table]:
        if "Month" not in frame:
            frame["Month"] = np.nan
    result = pd.concat(
        [overall_table[keep], month_table[keep]], ignore_index=True
    )
    return result.sort_values(
        ["Reporting_Scope", "Month", "IRI_Rank"], na_position="first"
    ).reset_index(drop=True)


def stability_summary(
    iri_year_month: pd.DataFrame,
    iri_overall: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_lookup = iri_overall.set_index("Species_Code")
    rows: list[dict[str, Any]] = []
    n_periods = iri_year_month[["Year", "Month"]].drop_duplicates().shape[0]
    for (code, cn, latin), subset in iri_year_month.groupby(
        ["Species_Code", "Species_CN", "Species_Latin"], observed=True
    ):
        top1 = int((subset["IRI_Rank"] <= 1).sum())
        top3 = int((subset["IRI_Rank"] <= 3).sum())
        top5 = int((subset["IRI_Rank"] <= 5).sum())
        overall_rank = int(overall_lookup.loc[code, "IRI_Rank"])
        top5_prop = top5 / n_periods
        is_core = overall_rank <= 5 and top5_prop >= CORE_TOP5_PROPORTION
        if is_core:
            core_class = "Stable_core"
        elif top5_prop >= 0.50:
            core_class = "Frequent_secondary"
        elif top5 > 0:
            core_class = "Episodic_top5"
        else:
            core_class = "Non_top5"
        rows.append(
            {
                "Species_Code": code,
                "Species_CN": cn,
                "Species_Latin": latin,
                "N_YearMonth": n_periods,
                "N_Top1": top1,
                "Proportion_Top1": top1 / n_periods,
                "N_Top3": top3,
                "Proportion_Top3": top3 / n_periods,
                "N_Top5": top5,
                "Proportion_Top5": top5_prop,
                "N_RelativeDominant": int(
                    subset["Dominant_By_Relative_IRI"].astype(bool).sum()
                ),
                "N_AbsoluteDominant": int(
                    subset["Dominant_By_Absolute_IRI"].astype(bool).sum()
                ),
                "Mean_IRI_Rank": float(subset["IRI_Rank"].mean()),
                "Median_IRI_Rank": float(subset["IRI_Rank"].median()),
                "SD_IRI_Rank": float(subset["IRI_Rank"].std(ddof=1)),
                "Mean_IRI_Percent": float(subset["IRI_Percent"].mean()),
                "Median_IRI_Percent": float(subset["IRI_Percent"].median()),
                "Max_IRI_Percent": float(subset["IRI_Percent"].max()),
                "Overall_IRI_Rank": overall_rank,
                "Overall_IRI_Percent": float(
                    overall_lookup.loc[code, "IRI_Percent"]
                ),
                "Core_Criterion": (
                    f"Overall IRI rank <=5 and Top5 proportion >="
                    f"{CORE_TOP5_PROPORTION:.2f}"
                ),
                "Is_Stable_Core": is_core,
                "Core_Class": core_class,
            }
        )
    overall_stability = pd.DataFrame(rows).sort_values(
        ["Is_Stable_Core", "Proportion_Top5", "Overall_IRI_Rank"],
        ascending=[False, False, True],
    )

    month_rows: list[dict[str, Any]] = []
    for (month, code, cn, latin), subset in iri_year_month.groupby(
        ["Month", "Species_Code", "Species_CN", "Species_Latin"], observed=True
    ):
        n = subset["Year"].nunique()
        month_rows.append(
            {
                "Month": int(month),
                "Period": MONTH_LABELS[int(month)],
                "Species_Code": code,
                "Species_CN": cn,
                "Species_Latin": latin,
                "N_Years": n,
                "N_Top1": int((subset["IRI_Rank"] <= 1).sum()),
                "N_Top3": int((subset["IRI_Rank"] <= 3).sum()),
                "N_Top5": int((subset["IRI_Rank"] <= 5).sum()),
                "Proportion_Top5": float((subset["IRI_Rank"] <= 5).mean()),
                "N_RelativeDominant": int(
                    subset["Dominant_By_Relative_IRI"].astype(bool).sum()
                ),
                "N_AbsoluteDominant": int(
                    subset["Dominant_By_Absolute_IRI"].astype(bool).sum()
                ),
                "Mean_IRI_Rank": float(subset["IRI_Rank"].mean()),
                "Mean_IRI_Percent": float(subset["IRI_Percent"].mean()),
            }
        )
    month_stability = pd.DataFrame(month_rows).sort_values(
        ["Month", "Proportion_Top5", "Mean_IRI_Rank"],
        ascending=[True, False, True],
    )
    return overall_stability.reset_index(drop=True), month_stability.reset_index(drop=True)


def top_species_and_turnover(
    iri_year_month: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    top1 = (
        iri_year_month.loc[iri_year_month["IRI_Rank"] == 1]
        .sort_values(["Year", "Month"])
        [[
            "Year",
            "Month",
            "Period",
            "Species_Code",
            "Species_CN",
            "Species_Latin",
            "IRI",
            "IRI_Percent",
        ]]
        .reset_index(drop=True)
    )
    top1["Previous_Top1_SameMonth"] = top1.groupby("Month", observed=True)[
        "Species_CN"
    ].shift(1)
    top1["Top1_Changed_From_PreviousYear"] = (
        top1["Previous_Top1_SameMonth"].notna()
        & top1["Species_CN"].ne(top1["Previous_Top1_SameMonth"])
    )

    consecutive_rows: list[dict[str, Any]] = []
    for month, subset in iri_year_month.groupby("Month", observed=True):
        sets = {
            int(year): set(group.loc[group["IRI_Rank"] <= 5, "Species_CN"])
            for year, group in subset.groupby("Year", observed=True)
        }
        years = sorted(sets)
        for previous, current in zip(years[:-1], years[1:]):
            a, b = sets[previous], sets[current]
            union = a | b
            consecutive_rows.append(
                {
                    "Month": int(month),
                    "Period": MONTH_LABELS[int(month)],
                    "From_Year": previous,
                    "To_Year": current,
                    "N_Shared_Top5": len(a & b),
                    "Top5_Jaccard": len(a & b) / len(union) if union else np.nan,
                    "N_Exited": len(a - b),
                    "Exited_Species": ";".join(sorted(a - b)),
                    "N_Entered": len(b - a),
                    "Entered_Species": ";".join(sorted(b - a)),
                }
            )
    consecutive = pd.DataFrame(consecutive_rows)

    between_month_rows: list[dict[str, Any]] = []
    for year, subset in iri_year_month.groupby("Year", observed=True):
        sets = {
            int(month): set(group.loc[group["IRI_Rank"] <= 5, "Species_CN"])
            for month, group in subset.groupby("Month", observed=True)
        }
        april, november = sets.get(4, set()), sets.get(11, set())
        union = april | november
        april_ranks = subset.loc[subset["Month"] == 4].set_index("Species_Code")[
            "IRI_Rank"
        ]
        november_ranks = subset.loc[subset["Month"] == 11].set_index("Species_Code")[
            "IRI_Rank"
        ]
        common_codes = april_ranks.index.intersection(november_ranks.index)
        rho = stats.spearmanr(
            april_ranks.loc[common_codes], november_ranks.loc[common_codes]
        )
        between_month_rows.append(
            {
                "Year": int(year),
                "N_Shared_Top5": len(april & november),
                "Top5_Jaccard_April_vs_November": (
                    len(april & november) / len(union) if union else np.nan
                ),
                "April_Only_Top5": ";".join(sorted(april - november)),
                "November_Only_Top5": ";".join(sorted(november - april)),
                "Spearman_All25_IRI_Rank": float(rho.statistic),
                "Spearman_P": float(rho.pvalue),
            }
        )
    between_month = pd.DataFrame(between_month_rows)
    return top1, consecutive, between_month


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
            "savefig.facecolor": "white",
        }
    )


def plot_group_composition(
    group_year_month: pd.DataFrame, figure_dir: Path
) -> None:
    colors = {group: GROUP_LABELS[group]["Color"] for group in GROUP_ORDER}
    metrics = [
        ("Biomass_Share_Percent", "Biomass composition (%)"),
        ("Density_Share_Percent", "Numerical-density composition (%)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    for row_index, (metric, ylabel) in enumerate(metrics):
        for column_index, month in enumerate([4, 11]):
            ax = axes[row_index, column_index]
            subset = group_year_month.loc[group_year_month["Month"] == month]
            pivot = (
                subset.pivot(
                    index="Year", columns="Analysis_Group", values=metric
                )
                .reindex(columns=GROUP_ORDER)
                .sort_index()
            )
            bottom = np.zeros(len(pivot), dtype=float)
            for group in GROUP_ORDER:
                values = pivot[group].to_numpy(dtype=float)
                ax.bar(
                    pivot.index.astype(str),
                    values,
                    bottom=bottom,
                    color=colors[group],
                    width=0.78,
                    label=GROUP_LABELS[group]["Group_EN"],
                )
                bottom += values
            ax.set_ylim(0, 100)
            ax.set_title(MONTH_LABELS[month])
            ax.set_ylabel(ylabel if column_index == 0 else "")
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
            ax.tick_params(axis="x", rotation=45)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Temporal composition of three cephalopod analysis groups",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, figure_dir / "Fig01_three_group_composition")


def plot_group_absolute_resource(
    group_year_month: pd.DataFrame, figure_dir: Path
) -> None:
    colors = {group: GROUP_LABELS[group]["Color"] for group in GROUP_ORDER}
    metrics = [
        (
            "Mean_Biomass_AllSurveyed_kg_km2",
            r"Mean biomass (kg km$^{-2}$)",
        ),
        (
            "Mean_Density_AllSurveyed_1000ind_km2",
            r"Mean density ($10^3$ ind km$^{-2}$)",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for row_index, (metric, ylabel) in enumerate(metrics):
        for column_index, month in enumerate([4, 11]):
            ax = axes[row_index, column_index]
            subset = group_year_month.loc[group_year_month["Month"] == month]
            for group in GROUP_ORDER:
                line = subset.loc[subset["Analysis_Group"] == group].sort_values(
                    "Year"
                )
                ax.plot(
                    line["Year"],
                    line[metric],
                    marker="o",
                    markersize=4,
                    linewidth=1.8,
                    color=colors[group],
                    label=GROUP_LABELS[group]["Group_EN"],
                )
            ax.set_title(MONTH_LABELS[month])
            ax.set_ylabel(ylabel if column_index == 0 else "")
            ax.set_xticks(range(2014, 2025))
            ax.tick_params(axis="x", rotation=45)
            ax.grid(color="#D9D9D9", linewidth=0.6, alpha=0.7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Absolute resource level by analysis group", fontsize=14, y=1.01)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, figure_dir / "Fig02_three_group_absolute_resource")


def plot_iri_profiles(
    iri_overall: pd.DataFrame,
    iri_month: pd.DataFrame,
    figure_dir: Path,
) -> None:
    panels = [
        ("Overall", iri_overall),
        ("April", iri_month.loc[iri_month["Month"] == 4]),
        ("November", iri_month.loc[iri_month["Month"] == 11]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    for ax, (title, frame) in zip(axes, panels):
        top = frame.nsmallest(8, "IRI_Rank").sort_values("IRI_Percent")
        colors = ["#CC79A7" if rank <= 5 else "#9E9E9E" for rank in top["IRI_Rank"]]
        ax.barh(top["Species_Latin"], top["IRI_Percent"], color=colors)
        for y, value in enumerate(top["IRI_Percent"]):
            ax.text(value + max(top["IRI_Percent"].max() * 0.015, 0.15), y, f"{value:.1f}", va="center", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("%IRI")
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.7)
        ax.set_xlim(0, top["IRI_Percent"].max() * 1.17)
    fig.suptitle("Dominant-species profiles based on IRI", fontsize=14, y=1.01)
    fig.tight_layout()
    save_figure(fig, figure_dir / "Fig03_iri_overall_and_month")


def plot_iri_heatmap(
    iri_year_month: pd.DataFrame,
    iri_overall: pd.DataFrame,
    figure_dir: Path,
) -> None:
    top_codes = iri_overall.nsmallest(12, "IRI_Rank")["Species_Code"].tolist()
    names = iri_overall.set_index("Species_Code")["Species_Latin"].to_dict()
    data = iri_year_month.loc[iri_year_month["Species_Code"].isin(top_codes)].copy()
    data["YearMonth_Label"] = data.apply(
        lambda row: f"{int(row['Year'])}-{int(row['Month']):02d}", axis=1
    )
    ordered_periods = (
        data[["Year", "Month", "YearMonth_Label"]]
        .drop_duplicates()
        .sort_values(["Year", "Month"])["YearMonth_Label"]
        .tolist()
    )
    pivot = data.pivot(
        index="Species_Code", columns="YearMonth_Label", values="IRI_Percent"
    ).reindex(index=top_codes, columns=ordered_periods)
    transformed = np.log1p(pivot.to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(17, 7.5))
    image = ax.imshow(transformed, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(ordered_periods)))
    ax.set_xticklabels(ordered_periods, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(top_codes)))
    ax.set_yticklabels([names[code] for code in top_codes])
    ax.set_xlabel("Survey period")
    ax.set_title("Temporal reorganization of leading species (%IRI)")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("log(1 + %IRI)")
    rank1 = data.loc[data["IRI_Rank"] == 1]
    row_lookup = {code: i for i, code in enumerate(top_codes)}
    column_lookup = {period: i for i, period in enumerate(ordered_periods)}
    for row in rank1.itertuples(index=False):
        if row.Species_Code in row_lookup:
            ax.scatter(
                column_lookup[row.YearMonth_Label],
                row_lookup[row.Species_Code],
                s=48,
                facecolors="none",
                edgecolors="white",
                linewidths=1.3,
            )
    fig.tight_layout()
    save_figure(fig, figure_dir / "Fig04_iri_year_month_heatmap")


def plot_core_stability(
    stability: pd.DataFrame, figure_dir: Path
) -> None:
    top = stability.sort_values(
        ["Proportion_Top5", "Overall_IRI_Rank"], ascending=[False, True]
    ).head(12)
    top = top.sort_values("Proportion_Top5")
    colors = np.where(top["Is_Stable_Core"].astype(bool), "#D55E00", "#9E9E9E")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["Species_Latin"], top["Proportion_Top5"] * 100, color=colors)
    ax.axvline(
        CORE_TOP5_PROPORTION * 100,
        color="#333333",
        linestyle="--",
        linewidth=1.2,
        label=f"Core threshold ({CORE_TOP5_PROPORTION:.0%})",
    )
    ax.set_xlim(0, 104)
    ax.set_xlabel("Proportion of 22 survey periods ranked in top five (%)")
    ax.set_title("Stability of dominant species")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, figure_dir / "Fig05_core_species_stability")


def plot_turnover(consecutive: pd.DataFrame, figure_dir: Path) -> None:
    colors = {4: "#0072B2", 11: "#D55E00"}
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for month in [4, 11]:
        line = consecutive.loc[consecutive["Month"] == month].sort_values(
            "To_Year"
        )
        ax.plot(
            line["To_Year"],
            line["Top5_Jaccard"],
            color=colors[month],
            marker="o",
            linewidth=2,
            label=MONTH_LABELS[month],
        )
    ax.set_xticks(range(2015, 2025))
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("Later year in consecutive-year comparison")
    ax.set_ylabel("Top-five Jaccard similarity")
    ax.set_title("Interannual turnover of dominant species")
    ax.grid(color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, figure_dir / "Fig06_top5_turnover")


def _empty_gee_tables(status: str, message: str) -> tuple[pd.DataFrame, ...]:
    models = pd.DataFrame(
        [
            {
                "Response": response,
                "Status": status,
                "Message": message,
                "Coefficient": np.nan,
                "Estimate_Link": np.nan,
                "Robust_SE": np.nan,
                "Z": np.nan,
                "P": np.nan,
            }
            for response in ["Occurrence", "Biomass", "Density"]
        ]
    )
    joint = pd.DataFrame(
        columns=["Response", "Test", "ChiSquare", "DF", "P", "Status"]
    )
    means = pd.DataFrame(
        columns=[
            "Response",
            "Year",
            "Month",
            "Period",
            "Estimate",
            "CI_Lower",
            "CI_Upper",
            "Scale",
            "Status",
        ]
    )
    pairwise = pd.DataFrame(
        columns=[
            "Response",
            "Pairwise_Type",
            "Reference",
            "Comparison",
            "Contrast_Link",
            "SE_Link",
            "Z",
            "P",
            "P_FDR",
            "FDR_Significant",
            "Effect_Ratio",
            "Effect_Ratio_CI_Lower",
            "Effect_Ratio_CI_Upper",
            "Status",
        ]
    )
    return models, joint, means, pairwise


def _design_matrix(result: Any, grid: pd.DataFrame, patsy_module: Any) -> pd.DataFrame:
    try:
        design_info = result.model.data.design_info
        matrix = patsy_module.build_design_matrices(
            [design_info], grid, return_type="dataframe"
        )[0]
    except Exception:
        matrix = patsy_module.dmatrix(
            "C(Year) * C(Month)", grid, return_type="dataframe"
        )
    return matrix.reindex(columns=result.params.index, fill_value=0.0)


def _link_inverse(response: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if response == "Occurrence":
        return expit(values)
    return np.exp(values)


def run_temporal_gee(
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not RUN_TEMPORAL_GEE:
        return _empty_gee_tables(
            "SKIPPED_CONFIG",
            "Set CEPHALOPOD_STEP4B_RUN_GEE=1 to enable optional temporal GEE tests",
        )
    try:
        import patsy
        import statsmodels
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        from statsmodels.genmod.cov_struct import Exchangeable, Independence
    except Exception as exc:
        return _empty_gee_tables(
            "SKIPPED_DEPENDENCY",
            f"statsmodels/patsy unavailable: {type(exc).__name__}: {exc}",
        )
    if Version(statsmodels.__version__) < Version("0.14"):
        return _empty_gee_tables(
            "SKIPPED_DEPENDENCY",
            f"statsmodels>=0.14 required; found {statsmodels.__version__}",
        )

    data = events.copy()
    data["Year"] = data["Year"].astype(int)
    data["Month"] = data["Month"].astype(int)
    data["LocationID"] = data["LocationID"].astype(str)
    specs = [
        (
            "Occurrence",
            "Cephalopod_Presence",
            sm.families.Binomial(link=sm.families.links.Logit()),
            "Probability; contrasts are odds ratios",
        ),
        (
            "Biomass",
            "Total_Biomass_kg_km2",
            sm.families.Tweedie(
                var_power=TWEEDIE_POWER, link=sm.families.links.Log()
            ),
            "kg km^-2; contrasts are mean ratios",
        ),
        (
            "Density",
            "Total_Density_1000ind_km2",
            sm.families.Tweedie(
                var_power=TWEEDIE_POWER, link=sm.families.links.Log()
            ),
            "10^3 ind km^-2; contrasts are mean ratios",
        ),
    ]
    coefficient_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    marginal_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    grid = pd.DataFrame(
        [(year, month) for year in range(2014, 2025) for month in [4, 11]],
        columns=["Year", "Month"],
    )

    for response, outcome, family, scale_label in specs:
        formula = f"{outcome} ~ C(Year) * C(Month)"
        fitted = None
        covariance_name = ""
        fit_messages: list[str] = []
        for covariance, name in [
            (Exchangeable(), "Exchangeable"),
            (Independence(), "Independence_fallback"),
        ]:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model = smf.gee(
                        formula,
                        groups="LocationID",
                        data=data,
                        family=family,
                        cov_struct=covariance,
                    )
                    candidate = model.fit(
                        maxiter=GEE_MAXITER,
                        ctol=GEE_CTOL,
                        cov_type="robust",
                    )
                warning_text = " | ".join(str(item.message) for item in caught)
                with (MODEL_AUDIT_DIR/'20_optional_temporal_GEE_fit_history.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps({'Response':response,'Formula':formula,'N':len(data),
                        'N_LocationID':data.LocationID.nunique(),'Covariance':name,'Warnings':warning_text,
                        'Converged':bool(getattr(candidate,'converged',False)),
                        'History':candidate.fit_history,'RandomSeed':RANDOM_SEED},default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v))+'\n')
                if not bool(getattr(candidate, "converged", False)):
                    raise RuntimeError("GEE did not converge")
                if not np.isfinite(candidate.params.to_numpy(dtype=float)).all():
                    raise RuntimeError("GEE returned non-finite coefficients")
                fitted = candidate
                covariance_name = name
                if warning_text:
                    fit_messages.append(warning_text)
                break
            except Exception as exc:
                fit_messages.append(f"{name}: {type(exc).__name__}: {exc}")
        if fitted is None:
            coefficient_rows.append(
                {
                    "Response": response,
                    "Status": "FAILED",
                    "Message": " | ".join(fit_messages),
                    "Coefficient": np.nan,
                    "Estimate_Link": np.nan,
                    "Robust_SE": np.nan,
                    "Z": np.nan,
                    "P": np.nan,
                    "Covariance_Structure": np.nan,
                    "Converged": False,
                }
            )
            continue

        confidence = fitted.conf_int()
        for coefficient in fitted.params.index:
            coefficient_rows.append(
                {
                    "Response": response,
                    "Status": "OK",
                    "Message": " | ".join(fit_messages),
                    "Coefficient": coefficient,
                    "Estimate_Link": float(fitted.params[coefficient]),
                    "Robust_SE": float(fitted.bse[coefficient]),
                    "Z": float(fitted.tvalues[coefficient]),
                    "P": float(fitted.pvalues[coefficient]),
                    "CI_Lower_Link": float(confidence.loc[coefficient, 0]),
                    "CI_Upper_Link": float(confidence.loc[coefficient, 1]),
                    "Covariance_Structure": covariance_name,
                    "Converged": bool(getattr(fitted, "converged", False)),
                    "N_Events": int(fitted.nobs),
                    "N_LocationID": int(data["LocationID"].nunique()),
                    "Family": family.__class__.__name__,
                    "Link": family.link.__class__.__name__,
                    "Tweedie_Variance_Power": TWEEDIE_POWER if response != "Occurrence" else np.nan,
                }
            )

        covariance = fitted.cov_params().loc[fitted.params.index, fitted.params.index]
        parameter_names = list(fitted.params.index)
        tests = {
            "Year_overall_including_interaction": [
                name for name in parameter_names if "C(Year)" in name
            ],
            "Month_overall_including_interaction": [
                name for name in parameter_names if "C(Month)" in name
            ],
            "Year_by_Month_interaction": [
                name for name in parameter_names if ":" in name
            ],
        }
        for test_name, names in tests.items():
            beta = fitted.params.loc[names].to_numpy(dtype=float)
            vcov = covariance.loc[names, names].to_numpy(dtype=float)
            degrees = int(np.linalg.matrix_rank(vcov))
            statistic = float(beta @ np.linalg.pinv(vcov) @ beta)
            joint_rows.append(
                {
                    "Response": response,
                    "Test": test_name,
                    "ChiSquare": statistic,
                    "DF": degrees,
                    "P": float(stats.chi2.sf(statistic, degrees)),
                    "Status": "OK",
                    "Covariance_Structure": covariance_name,
                }
            )

        xgrid = _design_matrix(fitted, grid, patsy)
        beta_all = fitted.params.to_numpy(dtype=float)
        vcov_all = covariance.to_numpy(dtype=float)
        eta = xgrid.to_numpy(dtype=float) @ beta_all
        eta_se = np.sqrt(
            np.clip(
                np.einsum("ij,jk,ik->i", xgrid, vcov_all, xgrid),
                a_min=0,
                a_max=None,
            )
        )
        estimates = _link_inverse(response, eta)
        lower = _link_inverse(response, eta - 1.96 * eta_se)
        upper = _link_inverse(response, eta + 1.96 * eta_se)
        for index, row in grid.iterrows():
            marginal_rows.append(
                {
                    "Response": response,
                    "Year": int(row["Year"]),
                    "Month": int(row["Month"]),
                    "Period": MONTH_LABELS[int(row["Month"])],
                    "Estimate": float(estimates[index]),
                    "CI_Lower": float(lower[index]),
                    "CI_Upper": float(upper[index]),
                    "SE_Link": float(eta_se[index]),
                    "Scale": scale_label,
                    "Status": "OK",
                }
            )

        def add_contrast(
            pairwise_type: str,
            reference_label: str,
            comparison_label: str,
            reference_index: int,
            comparison_index: int,
            fdr_family: str,
        ) -> None:
            contrast = (
                xgrid.iloc[comparison_index].to_numpy(dtype=float)
                - xgrid.iloc[reference_index].to_numpy(dtype=float)
            )
            estimate_link = float(contrast @ beta_all)
            se_link = float(
                np.sqrt(max(float(contrast @ vcov_all @ contrast), 0.0))
            )
            z_value = estimate_link / se_link if se_link > 0 else np.nan
            p_value = (
                float(2 * stats.norm.sf(abs(z_value)))
                if np.isfinite(z_value)
                else np.nan
            )
            pairwise_rows.append(
                {
                    "Response": response,
                    "Pairwise_Type": pairwise_type,
                    "Reference": reference_label,
                    "Comparison": comparison_label,
                    "Contrast_Link": estimate_link,
                    "SE_Link": se_link,
                    "Z": z_value,
                    "P": p_value,
                    "FDR_Family": fdr_family,
                    "Effect_Ratio": float(np.exp(estimate_link)),
                    "Effect_Ratio_CI_Lower": float(np.exp(estimate_link - 1.96 * se_link)),
                    "Effect_Ratio_CI_Upper": float(np.exp(estimate_link + 1.96 * se_link)),
                    "Effect_Interpretation": "comparison/reference",
                    "Status": "OK",
                }
            )

        index_lookup = {
            (int(row.Year), int(row.Month)): int(index)
            for index, row in grid.iterrows()
        }
        for month in [4, 11]:
            for earlier, later in itertools.combinations(range(2014, 2025), 2):
                add_contrast(
                    "Year_within_month",
                    f"{earlier}-{month:02d}",
                    f"{later}-{month:02d}",
                    index_lookup[(earlier, month)],
                    index_lookup[(later, month)],
                    f"{response}_{MONTH_LABELS[month]}_all_year_pairs",
                )
        for year in range(2014, 2025):
            add_contrast(
                "November_vs_April_within_year",
                f"{year}-04",
                f"{year}-11",
                index_lookup[(year, 4)],
                index_lookup[(year, 11)],
                f"{response}_November_vs_April_all_years",
            )

    coefficients = pd.DataFrame(coefficient_rows)
    joint = pd.DataFrame(joint_rows)
    marginal = pd.DataFrame(marginal_rows)
    pairwise = pd.DataFrame(pairwise_rows)
    if not pairwise.empty:
        pairwise["P_FDR"] = pairwise.groupby("FDR_Family", observed=True)[
            "P"
        ].transform(lambda values: bh_fdr(values.to_numpy(dtype=float)))
        pairwise["FDR_Significant"] = pairwise["P_FDR"] < 0.05
    return coefficients, joint, marginal, pairwise


def build_qc(
    input_qc: list[dict[str, Any]],
    mapping: pd.DataFrame,
    group_overall: pd.DataFrame,
    group_year_month: pd.DataFrame,
    balanced: pd.DataFrame,
    species_overall: pd.DataFrame,
    species_year_month: pd.DataFrame,
    stability: pd.DataFrame,
    top1: pd.DataFrame,
    consecutive: pd.DataFrame,
    between_month: pd.DataFrame,
    gee_coefficients: pd.DataFrame,
) -> pd.DataFrame:
    rows = list(input_qc)

    def add(
        check: str,
        passed: bool,
        observed: Any,
        expected: Any,
        severity: str = "FATAL",
    ) -> None:
        rows.append(
            {
                "Check": check,
                "Pass": bool(passed),
                "Observed": observed,
                "Expected": expected,
                "Severity": severity,
            }
        )

    group_counts = mapping["Analysis_Group"].value_counts().to_dict()
    add(
        "Species counts in three groups",
        group_counts == {
            "Squids": 11,
            "Cuttlefishes_and_bobtail_squids": 7,
            "Octopuses": 7,
        },
        group_counts,
        "Squids=11; Cuttlefishes/bobtail=7; Octopuses=7",
    )
    add(
        "Overall biomass reconciles to 25 species",
        math.isclose(
            group_overall["Sum_Biomass_kg_km2"].sum(),
            species_overall["Sum_Biomass_kg_km2"].sum(),
            rel_tol=0,
            abs_tol=1e-7,
        ),
        group_overall["Sum_Biomass_kg_km2"].sum(),
        species_overall["Sum_Biomass_kg_km2"].sum(),
    )
    add(
        "Overall density reconciles to 25 species",
        math.isclose(
            group_overall["Sum_Density_1000ind_km2"].sum(),
            species_overall["Sum_Density_1000ind_km2"].sum(),
            rel_tol=0,
            abs_tol=1e-7,
        ),
        group_overall["Sum_Density_1000ind_km2"].sum(),
        species_overall["Sum_Density_1000ind_km2"].sum(),
    )
    add(
        "Overall biomass shares sum to 100",
        abs(group_overall["Biomass_Share_Percent"].sum() - 100) < 1e-8,
        group_overall["Biomass_Share_Percent"].sum(),
        100,
    )
    add(
        "Overall density shares sum to 100",
        abs(group_overall["Density_Share_Percent"].sum() - 100) < 1e-8,
        group_overall["Density_Share_Percent"].sum(),
        100,
    )
    biomass_sums = group_year_month.groupby(["Year", "Month"], observed=True)[
        "Biomass_Share_Percent"
    ].sum()
    density_sums = group_year_month.groupby(["Year", "Month"], observed=True)[
        "Density_Share_Percent"
    ].sum()
    add(
        "Every year-month biomass shares sum to 100",
        float(np.max(np.abs(biomass_sums - 100))) < 1e-8,
        float(np.max(np.abs(biomass_sums - 100))),
        0,
    )
    add(
        "Every year-month density shares sum to 100",
        float(np.max(np.abs(density_sums - 100))) < 1e-8,
        float(np.max(np.abs(density_sums - 100))),
        0,
    )
    merged_reconciliation = (
        group_year_month.groupby(["Year", "Month"], observed=True)[
            ["Sum_Biomass_kg_km2", "Sum_Density_1000ind_km2"]
        ]
        .sum()
        .join(
            species_year_month.groupby(["Year", "Month"], observed=True)[
                ["Sum_Biomass_kg_km2", "Sum_Density_1000ind_km2"]
            ].sum(),
            lsuffix="_group",
            rsuffix="_species",
        )
    )
    maximum_reconciliation_error = float(
        np.max(
            np.abs(
                merged_reconciliation.filter(like="_group").to_numpy()
                - merged_reconciliation.filter(like="_species").to_numpy()
            )
        )
    )
    add(
        "Year-month group totals reconcile to species totals",
        maximum_reconciliation_error < 1e-7,
        maximum_reconciliation_error,
        0,
    )
    add(
        "All annual estimates contain both months",
        bool((balanced["N_Months"] == 2).all()),
        sorted(balanced["N_Months"].unique()),
        [2],
    )
    add("Top-ranked species rows", len(top1) == 22, len(top1), 22)
    add(
        "Consecutive-year top-five comparisons",
        len(consecutive) == 20,
        len(consecutive),
        20,
    )
    add(
        "April-November top-five comparisons",
        len(between_month) == 11,
        len(between_month),
        11,
    )
    observed_core = sorted(
        stability.loc[stability["Is_Stable_Core"], "Species_CN"].tolist()
    )
    expected_core = sorted(["多钩钩腕乌贼", "杜氏枪乌贼", "长蛸"])
    add(
        "Stable-core species match expected Step-4 pattern",
        observed_core == expected_core,
        ";".join(observed_core),
        ";".join(expected_core),
        severity="WARNING",
    )
    if RUN_TEMPORAL_GEE:
        model_status = (
            gee_coefficients.groupby("Response", observed=True)["Status"]
            .apply(lambda values: "OK" if values.eq("OK").any() else values.iloc[0])
            .to_dict()
        )
        add(
            "Optional temporal GEE completed",
            set(model_status) == {"Occurrence", "Biomass", "Density"}
            and all(value == "OK" for value in model_status.values()),
            model_status,
            "All three responses OK",
            severity="WARNING",
        )
    result = pd.DataFrame(rows)
    result["Pass"] = result["Pass"].astype(bool)
    return result


def summary_table(
    input_zip: Path,
    events: pd.DataFrame,
    mapping: pd.DataFrame,
    group_overall: pd.DataFrame,
    stability: pd.DataFrame,
    top1: pd.DataFrame,
    consecutive: pd.DataFrame,
    between_month: pd.DataFrame,
    qc: pd.DataFrame,
    gee_coefficients: pd.DataFrame,
) -> pd.DataFrame:
    core_names = "; ".join(
        stability.loc[stability["Is_Stable_Core"], "Species_CN"].tolist()
    )
    top1_counts = (
        top1["Species_CN"].value_counts().rename_axis("Species").to_dict()
    )
    gee_status = (
        gee_coefficients.groupby("Response", observed=True)["Status"]
        .apply(lambda values: "OK" if values.eq("OK").any() else values.iloc[0])
        .to_dict()
    )
    rows = [
        ("Step", "Step 4B species composition and turnover"),
        ("Script version", SCRIPT_VERSION),
        ("Input ZIP", str(input_zip)),
        ("Input SHA256", sha256_file(input_zip)),
        ("Surveyed events", len(events)),
        ("Positive events", int(events["Cephalopod_Presence"].sum())),
        ("Zero-catch events", int((events["Cephalopod_Presence"] == 0).sum())),
        ("Species", len(mapping)),
        ("Analysis groups", mapping["Analysis_Group"].nunique()),
        ("Year-month periods", top1[["Year", "Month"]].drop_duplicates().shape[0]),
        ("Core top-five threshold", CORE_TOP5_PROPORTION),
        ("Stable core species", core_names),
        ("Top-ranked species frequency", str(top1_counts)),
        ("Mean consecutive-year top5 Jaccard", consecutive["Top5_Jaccard"].mean()),
        (
            "Mean April-November top5 Jaccard",
            between_month["Top5_Jaccard_April_vs_November"].mean(),
        ),
        ("Temporal GEE requested", RUN_TEMPORAL_GEE),
        ("Temporal GEE status", str(gee_status)),
        (
            "Fatal QC failures",
            int(((qc["Severity"] == "FATAL") & ~qc["Pass"]).sum()),
        ),
        (
            "QC warnings",
            int(((qc["Severity"] == "WARNING") & ~qc["Pass"]).sum()),
        ),
    ]
    for row in group_overall.itertuples(index=False):
        rows.extend(
            [
                (
                    f"Overall biomass share - {row.Group_EN}",
                    row.Biomass_Share_Percent,
                ),
                (
                    f"Overall density share - {row.Group_EN}",
                    row.Density_Share_Percent,
                ),
            ]
        )
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def metadata_table(input_zip: Path) -> pd.DataFrame:
    rows = [
        ("Script_Version", SCRIPT_VERSION),
        ("Random_Seed", RANDOM_SEED),
        ("Python", platform.python_version()),
        ("NumPy", np.__version__),
        ("Pandas", pd.__version__),
        ("SciPy", scipy.__version__),
        ("Matplotlib", matplotlib.__version__),
        ("Input_ZIP", str(input_zip)),
        ("Input_SHA256", sha256_file(input_zip)),
        ("Analysis_Unit", "Survey event for temporal GEE; species-period summaries for composition and IRI"),
        ("Surveyed_Events", 2100),
        ("Species", 25),
        ("Months", "April and November"),
        ("Three_Group_Note", "Analysis groups; not a claim of exactly three accepted modern orders"),
        ("Annual_Estimate", "Equal mean of April and November event means"),
        ("Core_Criterion", f"Overall IRI rank <=5 and top-five occurrence proportion >={CORE_TOP5_PROPORTION:.2f}"),
        ("Turnover_Metric", "Jaccard similarity of top-five species sets"),
        ("Temporal_GEE_Requested", RUN_TEMPORAL_GEE),
        ("GEE_Formula", "response ~ C(Year) * C(Month)"),
        ("GEE_Cluster", "LocationID"),
        ("GEE_Correlation", "Exchangeable; independence fallback"),
        ("GEE_Continuous_Family", f"Tweedie log link, variance power={TWEEDIE_POWER}"),
        ("GEE_Multiple_Testing", "Benjamini-Hochberg FDR within predefined comparison family"),
    ]
    return pd.DataFrame(rows, columns=["Parameter", "Value"])


def build_readme(
    input_zip: Path,
    stability: pd.DataFrame,
    qc: pd.DataFrame,
    gee_coefficients: pd.DataFrame,
) -> str:
    core_names = "、".join(
        stability.loc[stability["Is_Stable_Core"], "Species_CN"].tolist()
    )
    gee_status = (
        gee_coefficients.groupby("Response", observed=True)["Status"]
        .apply(lambda values: "OK" if values.eq("OK").any() else values.iloc[0])
        .to_dict()
    )
    warning_count = int(((qc["Severity"] == "WARNING") & ~qc["Pass"]).sum())
    return f"""# Step 4B：三大类群组成、核心优势种与优势种更替

## 分析范围

- 输入：`{input_zip.name}`
- 使用第四步已核验的2,100个已调查事件（1,532个正捕获、568个零捕获）和全部25种。
- 组成与IRI基于第四步汇总结果；可选GEE回到2,100个事件层面。
- 三个类别为论文中的生态分析类群：枪乌贼类、乌贼及耳乌贼类、蛸类。它们不是“现代分类学只有三个有效目”的声明；完整物种映射见`01_species_three_group_mapping.csv`。
- 年度结果使用4月与11月事件均值的等权平均，防止两季样本量不同造成年度构成偏移。

## 核心定义

- 稳定核心优势种：总体IRI排名前5，且在22个年—月组合中进入前5的比例不低于{CORE_TOP5_PROPORTION:.0%}。
- 本次识别为：{core_names}。
- 优势种更替：相邻年份同月前5物种集合的Jaccard相似度；数值越低表示更替越强。
- 4月—11月重组：同一年两季前5集合Jaccard，并附全部25种IRI排序的Spearman相关。

## 主要输出

- `02`—`05`：三类群总体、月份、年—月及季节均衡年度组成。
- `06`—`07`：带类群字段的物种资源贡献及可直接引用的IRI表。
- `08`—`12`：核心种稳定性、第一优势种、相邻年更替和季节差异。
- `13`—`16`：可选时间GEE系数、联合检验、边际均值和FDR校正成对比较。
- `18_qc_checks.csv`：输入、回算和输出质控。
- `figures/`：PNG（{FIGURE_DPI} dpi）、PDF和SVG三种格式。

## 可选GEE

当前请求状态：`{RUN_TEMPORAL_GEE}`；运行状态：`{gee_status}`。

默认关闭是为了让组成与优势种结果不受Kaggle依赖包影响。如需正式检验总体资源量的年份、月份及交互效应，在运行脚本前设置：

```python
import os
os.environ["CEPHALOPOD_STEP4B_RUN_GEE"] = "1"
```

模型为`response ~ C(Year) * C(Month)`，以`LocationID`为重复测量聚类单元；出现率使用Binomial-logit，生物量与密度使用Tweedie-log。成对比较在预先定义的检验族内进行BH-FDR校正。

若输出显示`SKIPPED_DEPENDENCY`，请先在单独单元安装并重启Kaggle会话：

```python
!pip install -q --upgrade "statsmodels>=0.14,<0.15" "patsy>=0.5"
```

## 质控结论

- 致命质控失败：{int(((qc['Severity'] == 'FATAL') & ~qc['Pass']).sum())}
- 警告：{warning_count}
- 只有`Severity=FATAL`且`Pass=False`时结果不可用于论文。
"""


def manifest_table(directory: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name not in {"19_output_manifest.csv"}:
            rows.append(
                {
                    "Relative_Path": str(path.relative_to(directory)),
                    "Size_Bytes": path.stat().st_size,
                    "SHA256": sha256_file(path),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    global MODEL_AUDIT_DIR
    np.random.seed(RANDOM_SEED)
    configure_plot_style()
    input_zip = discover_step4_zip()
    log(f"Input: {input_zip}")
    frames = read_inputs(input_zip)
    mapping = mapping_table(frames["species_overall"])
    input_qc = validate_inputs(frames, mapping)
    log("Input and 25-species mapping QC passed")

    group_overall = aggregate_major_groups(
        frames["species_overall"], mapping, [], "Overall"
    )
    group_month = aggregate_major_groups(
        frames["species_month"], mapping, ["Month", "Period"], "Month"
    )
    group_year_month = aggregate_major_groups(
        frames["species_year_month"],
        mapping,
        ["Year", "Month", "Period"],
        "YearMonth",
    )
    balanced = season_balanced_annual(group_year_month)
    species_contributions = frames["species_year_month"].drop(
        columns=["Species_Order"], errors="ignore"
    ).merge(
        mapping[
            [
                "Species_Code",
                "Analysis_Group",
                "Group_CN",
                "Group_EN",
                "Legacy_Order_CN",
                "Legacy_Order_EN",
            ]
        ],
        on="Species_Code",
        how="left",
        validate="many_to_one",
    )
    iri_reporting = iri_reporting_table(
        frames["iri_overall"], frames["iri_month"]
    )
    stability, stability_month = stability_summary(
        frames["iri_year_month"], frames["iri_overall"]
    )
    top1, consecutive, between_month = top_species_and_turnover(
        frames["iri_year_month"]
    )
    log("Composition, IRI stability and turnover tables completed")

    gee_coefficients, gee_joint, gee_marginal, gee_pairwise = run_temporal_gee(
        frames["events"]
    )
    if RUN_TEMPORAL_GEE:
        gee_ok = gee_coefficients["Status"].eq("OK").any()
        log(f"Optional temporal GEE finished; any successful model={gee_ok}")
    else:
        log("Optional temporal GEE skipped by configuration")

    qc = build_qc(
        input_qc,
        mapping,
        group_overall,
        group_year_month,
        balanced,
        frames["species_overall"],
        frames["species_year_month"],
        stability,
        top1,
        consecutive,
        between_month,
        gee_coefficients,
    )
    fatal_failures = qc.loc[(qc["Severity"] == "FATAL") & ~qc["Pass"]]
    if not fatal_failures.empty:
        raise AssertionError(
            "Fatal output QC failures:\n" + fatal_failures.to_string(index=False)
        )

    temp_dir = prepare_temp_dir(FINAL_DIR)
    MODEL_AUDIT_DIR = temp_dir
    try:
        figure_dir = temp_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        atomic_text("RUNNING\n", temp_dir / "RUN_STATUS.txt")
        outputs = [
            ("00_step4b_summary.csv", summary_table(input_zip, frames["events"], mapping, group_overall, stability, top1, consecutive, between_month, qc, gee_coefficients)),
            ("01_species_three_group_mapping.csv", mapping),
            ("02_major_group_overall.csv", group_overall),
            ("03_major_group_by_month.csv", group_month),
            ("04_major_group_by_year_month.csv", group_year_month),
            ("05_major_group_annual_season_balanced.csv", balanced),
            ("06_species_contributions_with_group.csv", species_contributions),
            ("07_iri_reporting_overall_month.csv", iri_reporting),
            ("08_core_species_stability_overall.csv", stability),
            ("09_core_species_stability_by_month.csv", stability_month),
            ("10_rank1_species_by_year_month.csv", top1),
            ("11_top5_turnover_consecutive_years.csv", consecutive),
            ("12_april_november_top5_similarity.csv", between_month),
            ("13_temporal_gee_coefficients.csv", gee_coefficients),
            ("14_temporal_gee_joint_wald.csv", gee_joint),
            ("15_temporal_gee_marginal_means.csv", gee_marginal),
            ("16_temporal_gee_pairwise.csv", gee_pairwise),
            ("17_method_metadata.csv", metadata_table(input_zip)),
            ("18_qc_checks.csv", qc),
        ]
        for filename, frame in outputs:
            atomic_csv(frame, temp_dir / filename)

        plot_group_composition(group_year_month, figure_dir)
        plot_group_absolute_resource(group_year_month, figure_dir)
        plot_iri_profiles(
            frames["iri_overall"], frames["iri_month"], figure_dir
        )
        plot_iri_heatmap(
            frames["iri_year_month"], frames["iri_overall"], figure_dir
        )
        plot_core_stability(stability, figure_dir)
        plot_turnover(consecutive, figure_dir)
        log("Six publication figures written in PNG/PDF/SVG")

        atomic_text(
            build_readme(input_zip, stability, qc, gee_coefficients),
            temp_dir / "README_step4b.md",
        )
        source = Path(globals().get("__file__", ""))
        if source.is_file():
            shutil.copy2(source, temp_dir / source.name)
        atomic_csv(manifest_table(temp_dir), temp_dir / "19_output_manifest.csv")
        atomic_text(
            f"SUCCESS\nScript_Version={SCRIPT_VERSION}\nElapsed={elapsed()}\n",
            temp_dir / "RUN_STATUS.txt",
        )
        publish_dir(temp_dir, FINAL_DIR)
        build_zip(FINAL_DIR, FINAL_ZIP)
    except Exception:
        atomic_text(
            "FAILED\n" + traceback.format_exc(), temp_dir / "RUN_STATUS.txt"
        )
        raise

    log(f"SUCCESS: {FINAL_ZIP}")
    print("请下载：cephalopod_step4b_composition_turnover.zip", flush=True)


if __name__ == "__main__":
    main()
