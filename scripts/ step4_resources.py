# -*- coding: utf-8 -*-
"""
浙江近岸头足类数据分析——第四步：资源量、IRI与α多样性

运行顺序
--------
1. 运行最新版第一步：渔业调查数据清洗；
2. 运行最新版第二步：环境数据匹配；
3. 运行最新版第三步：构建 Master v1.0；
4. 再运行本代码。

本步骤分析对象
------------
- 仅使用第三步中2,100个已调查事件；
- 同时保留1,532个正捕获事件和568个真实零捕获事件；
- 保留全部25种头足类，不进行JSDM稀有物种筛选；
- 未调查事件不进入本步骤；
- 生物量和尾数密度均使用原始值，不做log1p转换。

核心方法
--------
1. 资源量：
   - 使用事件总生物量 kg/km² 和总尾数密度 千尾/km²；
   - 年月比较以“全部已调查事件（含零捕获）的均值”为主；
   - 总和只作为数据核对字段，不作为不同年月的资源量比较指标；
   - 由于各年月调查事件数不同，同时输出4月/11月等权的年度指数。

2. 相对重要性指数 IRI：
       IRI_i = (N_i% + W_i%) * F_i%
   其中：
   - N_i%：物种尾数密度加总占该组全部物种尾数密度加总的百分比；
   - W_i%：物种生物量加总占该组全部物种生物量加总的百分比；
   - F_i%：物种出现事件数 / 该组全部已调查事件数 * 100；
   - 频率分母包括零捕获事件；
   - 同时输出 %IRI = IRI_i / sum(IRI) * 100。

3. α多样性：
   - 主结果按尾数密度构建物种比例p_i；
   - 输出物种丰富度S、Shannon H'、Pielou J'、Simpson 1-D、
     Hill N1、Hill N2和Margalef指数；
   - 同时输出生物量比例计算的Shannon/Pielou/Simpson/Hill指数作敏感性；
   - 零捕获事件：S=0，但H'、J'、Simpson、Hill和Margalef均记为NA，
     避免把“没有群落”错误等同于“单一物种群落”；
   - 单物种事件：H'=0、J'=0、Simpson=0、Margalef=0。

4. 置信区间：
   - 年月均值使用事件级非参数bootstrap百分位95%区间；
   - 默认2,000次，固定随机种子，结果可复现；
   - 本步骤只做描述性分析，不进行时间效应显著性检验；
     后续时间效应应使用LocationID重复测量结构的混合模型/GAM。

默认输入目录
------------
/kaggle/working/cephalopod_step3_analysis_master

若目录不存在，代码会尝试解压：
/kaggle/working/cephalopod_step3_analysis_master.zip
或 /kaggle/input 下同名ZIP。

默认输出目录
------------
/kaggle/working/cephalopod_step4_resource_iri_alpha

主要输出
--------
00_step4_summary.csv
01_resource_event_level.csv
02_resource_summary_year_month.csv
03_resource_summary_year_event_weighted.csv
04_resource_summary_month.csv
05_resource_annual_season_balanced.csv
06_species_resource_overall.csv
07_species_resource_by_month.csv
08_species_resource_by_year.csv
09_species_resource_by_year_month.csv
10_IRI_overall.csv
11_IRI_by_month.csv
12_IRI_by_year.csv
13_IRI_by_year_month.csv
14_IRI_top5_and_dominant.csv
15_alpha_diversity_event_level.csv
16_alpha_summary_year_month.csv
17_alpha_summary_year_event_weighted.csv
18_alpha_summary_month.csv
19_alpha_annual_season_balanced.csv
20_method_metadata.csv
21_qc_checks.csv
22_output_manifest.csv
README_step4.txt
Step4_resource_IRI_alpha_summary.xlsx
figures/Fig01...Fig04（PNG 600 dpi、PDF、SVG）
cephalopod_step4_resource_iri_alpha.zip
"""

from __future__ import annotations

import importlib.util
import math
import os
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# ============================================================
# 0. 参数设置
# ============================================================

STEP3_DIR = Path(
    os.environ.get(
        "CEPHALOPOD_STEP3_DIR",
        "/kaggle/working/cephalopod_step3_analysis_master",
    )
)
STEP3_ZIP = Path(
    os.environ.get(
        "CEPHALOPOD_STEP3_ZIP",
        "/kaggle/working/cephalopod_step3_analysis_master.zip",
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        "CEPHALOPOD_STEP4_OUTPUT_DIR",
        "/kaggle/working/cephalopod_step4_resource_iri_alpha",
    )
)
FIGURE_DIR = OUTPUT_DIR / "figures"

SURVEYED_MASTER_NAME = "02_analysis_master_v1_surveyed_all25.csv"
LONG_MASTER_NAME = "04_analysis_master_v1_long_surveyed_all25.csv"
SPECIES_FILTER_NAME = "05_species_filter_summary.csv"

EXPECTED_EVENTS = int(os.environ.get("CEPHALOPOD_EXPECTED_SURVEYED_EVENTS", "2100"))
EXPECTED_POSITIVE = int(os.environ.get("CEPHALOPOD_EXPECTED_POSITIVE_EVENTS", "1532"))
EXPECTED_ZERO = int(os.environ.get("CEPHALOPOD_EXPECTED_ZERO_EVENTS", "568"))
EXPECTED_SPECIES = int(os.environ.get("CEPHALOPOD_EXPECTED_SPECIES_COUNT", "25"))
EXPECTED_LONG_ROWS = EXPECTED_EVENTS * EXPECTED_SPECIES

BOOTSTRAP_ITERATIONS = int(os.environ.get("CEPHALOPOD_BOOTSTRAP_ITERATIONS", "2000"))
RANDOM_SEED = int(os.environ.get("CEPHALOPOD_RANDOM_SEED", "20260827"))
FIGURE_DPI = int(os.environ.get("CEPHALOPOD_FIGURE_DPI", "600"))

# 常见IRI绝对值分级；可在Kaggle环境变量中修改。
IRI_DOMINANT_THRESHOLD = float(os.environ.get("CEPHALOPOD_IRI_DOMINANT", "1000"))
IRI_IMPORTANT_THRESHOLD = float(os.environ.get("CEPHALOPOD_IRI_IMPORTANT", "100"))
IRI_COMMON_THRESHOLD = float(os.environ.get("CEPHALOPOD_IRI_COMMON", "10"))

# %IRI分级仅作描述性辅助。
REL_IRI_DOMINANT_THRESHOLD = float(
    os.environ.get("CEPHALOPOD_REL_IRI_DOMINANT", "10")
)
REL_IRI_IMPORTANT_THRESHOLD = float(
    os.environ.get("CEPHALOPOD_REL_IRI_IMPORTANT", "1")
)
REL_IRI_COMMON_THRESHOLD = float(
    os.environ.get("CEPHALOPOD_REL_IRI_COMMON", "0.1")
)


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "n", "off"}


STRICT_EXPECTED_COUNTS = env_bool("CEPHALOPOD_STRICT_EXPECTED_COUNTS", True)

STATUS_ZERO = "SURVEYED_ZERO"
STATUS_POS = "SURVEYED_POSITIVE"
PERIOD_ORDER = ["April", "November"]
MONTH_COLORS = {4: "#0072B2", 11: "#D55E00"}
MONTH_LABELS = {4: "April", 11: "November"}


# ============================================================
# 1. 输入、输出和通用函数
# ============================================================

def locate_step3_dir() -> Path:
    required = [SURVEYED_MASTER_NAME, LONG_MASTER_NAME, SPECIES_FILTER_NAME]
    if all((STEP3_DIR / name).exists() for name in required):
        return STEP3_DIR

    zip_candidates = []
    if STEP3_ZIP.exists():
        zip_candidates.append(STEP3_ZIP)

    for base in [Path("/kaggle/input"), Path("/kaggle/working")]:
        if base.exists():
            for candidate in base.rglob("cephalopod_step3_analysis_master.zip"):
                if candidate not in zip_candidates:
                    zip_candidates.append(candidate)

    if len(zip_candidates) == 0:
        raise FileNotFoundError(
            "未找到第三步输出目录或cephalopod_step3_analysis_master.zip。\n"
            "请先运行第三步，或设置CEPHALOPOD_STEP3_DIR/CEPHALOPOD_STEP3_ZIP。"
        )
    if len(zip_candidates) > 1:
        raise FileNotFoundError(
            "发现多个同名第三步ZIP，无法安全自动选择：\n"
            + "\n".join(str(p) for p in zip_candidates)
            + "\n请通过CEPHALOPOD_STEP3_ZIP指定唯一文件。"
        )

    extract_dir = Path("/kaggle/working/cephalopod_step3_analysis_master_input")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_candidates[0], "r") as archive:
        archive.extractall(extract_dir)
    if not all((extract_dir / name).exists() for name in required):
        raise FileNotFoundError("第三步ZIP解压成功，但缺少Master v1.0必要文件。")
    return extract_dir


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def write_csv(df: pd.DataFrame, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{label}缺少字段：{missing}")


def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
        .fillna(False)
    )


def bootstrap_mean_ci(
    values: pd.Series | np.ndarray,
    rng: np.random.Generator,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> tuple[float, float]:
    array = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if len(array) == 0:
        return np.nan, np.nan
    if len(array) == 1 or np.allclose(array, array[0]):
        return float(array.mean()), float(array.mean())

    # 分块抽样，避免一次构造过大的数组。
    means = np.empty(iterations, dtype=float)
    block = 250
    start = 0
    while start < iterations:
        stop = min(start + block, iterations)
        indices = rng.integers(0, len(array), size=(stop - start, len(array)))
        means[start:stop] = array[indices].mean(axis=1)
        start = stop
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def describe_numeric(
    values: pd.Series,
    name: str,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    n = len(numeric)
    if n == 0:
        return {
            f"N_Defined_{name}": 0,
            f"Mean_{name}": np.nan,
            f"Median_{name}": np.nan,
            f"SD_{name}": np.nan,
            f"SE_{name}": np.nan,
            f"CI95_Lower_{name}": np.nan,
            f"CI95_Upper_{name}": np.nan,
        }
    sd = float(numeric.std(ddof=1)) if n > 1 else 0.0
    low, high = bootstrap_mean_ci(numeric, rng)
    return {
        f"N_Defined_{name}": n,
        f"Mean_{name}": float(numeric.mean()),
        f"Median_{name}": float(numeric.median()),
        f"SD_{name}": sd,
        f"SE_{name}": sd / math.sqrt(n) if n > 0 else np.nan,
        f"CI95_Lower_{name}": low,
        f"CI95_Upper_{name}": high,
    }


def stable_group_iterator(df: pd.DataFrame, group_cols: list[str]):
    if not group_cols:
        yield (), df
        return
    grouped = df.groupby(group_cols, sort=True, observed=True, dropna=False)
    for key, group in grouped:
        if len(group_cols) == 1:
            key = key if isinstance(key, tuple) else (key,)
        yield key, group


def summarize_event_table(
    df: pd.DataFrame,
    group_cols: list[str],
    metrics: list[str],
    seed_offset: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + seed_offset)
    rows = []
    for key, group in stable_group_iterator(df, group_cols):
        row = {col: value for col, value in zip(group_cols, key)}
        n = len(group)
        n_positive = int(group["SurveyStatus"].eq(STATUS_POS).sum())
        n_zero = int(group["SurveyStatus"].eq(STATUS_ZERO).sum())
        encounter = n_positive / n if n else np.nan
        row.update(
            {
                "N_Surveyed": n,
                "N_Positive": n_positive,
                "N_Zero": n_zero,
                "Encounter_Rate": encounter,
                "Encounter_Rate_Percent": encounter * 100 if n else np.nan,
                "Encounter_SE": (
                    math.sqrt(encounter * (1 - encounter) / n)
                    if n and np.isfinite(encounter)
                    else np.nan
                ),
            }
        )
        for metric in metrics:
            row.update(describe_numeric(group[metric], metric, rng))
        rows.append(row)
    return pd.DataFrame(rows)


def season_balanced_summary(
    year_month_summary: pd.DataFrame,
    metric_names: list[str],
) -> pd.DataFrame:
    rows = []
    for year, group in year_month_summary.groupby("Year", sort=True):
        group = group.loc[group["Month"].isin([4, 11])].copy()
        row = {
            "Year": int(year),
            "N_Months": int(group["Month"].nunique()),
            "N_Surveyed_Total": int(group["N_Surveyed"].sum()),
            "N_Positive_Total": int(group["N_Positive"].sum()),
            "N_Zero_Total": int(group["N_Zero"].sum()),
            "Season_Balancing": "Equal weight to April and November means",
        }
        for metric in metric_names:
            means = pd.to_numeric(group[f"Mean_{metric}"], errors="coerce").dropna()
            ses = pd.to_numeric(group[f"SE_{metric}"], errors="coerce").dropna()
            mean_value = float(means.mean()) if len(means) else np.nan
            se_value = (
                float(np.sqrt(np.sum(np.square(ses))) / len(ses))
                if len(ses)
                else np.nan
            )
            row[f"Mean_{metric}"] = mean_value
            row[f"SE_{metric}"] = se_value
            row[f"CI95_Lower_{metric}"] = (
                max(0.0, mean_value - 1.96 * se_value)
                if np.isfinite(mean_value) and np.isfinite(se_value)
                else np.nan
            )
            row[f"CI95_Upper_{metric}"] = (
                mean_value + 1.96 * se_value
                if np.isfinite(mean_value) and np.isfinite(se_value)
                else np.nan
            )

        encounter_means = pd.to_numeric(group["Encounter_Rate"], errors="coerce").dropna()
        encounter_ses = pd.to_numeric(group["Encounter_SE"], errors="coerce").dropna()
        encounter_mean = float(encounter_means.mean()) if len(encounter_means) else np.nan
        encounter_se = (
            float(np.sqrt(np.sum(np.square(encounter_ses))) / len(encounter_ses))
            if len(encounter_ses)
            else np.nan
        )
        row["Encounter_Rate"] = encounter_mean
        row["Encounter_Rate_Percent"] = encounter_mean * 100
        row["Encounter_SE"] = encounter_se
        row["Encounter_CI95_Lower"] = max(0.0, encounter_mean - 1.96 * encounter_se)
        row["Encounter_CI95_Upper"] = min(1.0, encounter_mean + 1.96 * encounter_se)
        rows.append(row)
    return pd.DataFrame(rows)


def iri_absolute_class(value: float) -> str:
    if not np.isfinite(value):
        return "Undefined"
    if value >= IRI_DOMINANT_THRESHOLD:
        return "Dominant"
    if value >= IRI_IMPORTANT_THRESHOLD:
        return "Important"
    if value >= IRI_COMMON_THRESHOLD:
        return "Common"
    return "Rare"


def iri_relative_class(value: float) -> str:
    if not np.isfinite(value):
        return "Undefined"
    if value >= REL_IRI_DOMINANT_THRESHOLD:
        return "Dominant"
    if value >= REL_IRI_IMPORTANT_THRESHOLD:
        return "Important"
    if value >= REL_IRI_COMMON_THRESHOLD:
        return "Common"
    return "Rare"


def species_resource_summary(
    long_df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    keys = group_cols + ["Species_Code", "Species_CN", "Species_Latin", "Species_Order"]
    grouped = long_df.groupby(keys, sort=True, observed=True, dropna=False)
    result = grouped.agg(
        N_Surveyed=("EventID", "nunique"),
        N_Occurrence=("Presence", "sum"),
        Sum_Biomass_kg_km2=("Biomass_kg_km2", "sum"),
        Mean_Biomass_AllSurveyed_kg_km2=("Biomass_kg_km2", "mean"),
        Median_Biomass_AllSurveyed_kg_km2=("Biomass_kg_km2", "median"),
        Max_Biomass_kg_km2=("Biomass_kg_km2", "max"),
        Sum_Density_1000ind_km2=("Density_1000ind_km2", "sum"),
        Mean_Density_AllSurveyed_1000ind_km2=("Density_1000ind_km2", "mean"),
        Median_Density_AllSurveyed_1000ind_km2=("Density_1000ind_km2", "median"),
        Max_Density_1000ind_km2=("Density_1000ind_km2", "max"),
    ).reset_index()

    positive_means = grouped.agg(
        Mean_Biomass_WhenPresent_kg_km2=(
            "Biomass_kg_km2",
            lambda s: pd.to_numeric(s[s > 0], errors="coerce").mean(),
        ),
        Mean_Density_WhenPresent_1000ind_km2=(
            "Density_1000ind_km2",
            lambda s: pd.to_numeric(s[s > 0], errors="coerce").mean(),
        ),
    ).reset_index()
    result = result.merge(positive_means, on=keys, how="left", validate="one_to_one")

    result["Frequency_Percent"] = 100 * result["N_Occurrence"] / result["N_Surveyed"]
    if group_cols:
        result["Biomass_Share_Percent"] = 100 * result["Sum_Biomass_kg_km2"] / result.groupby(
            group_cols, observed=True
        )["Sum_Biomass_kg_km2"].transform("sum")
        result["Density_Share_Percent"] = 100 * result["Sum_Density_1000ind_km2"] / result.groupby(
            group_cols, observed=True
        )["Sum_Density_1000ind_km2"].transform("sum")
        result["Biomass_Rank"] = result.groupby(group_cols, observed=True)[
            "Sum_Biomass_kg_km2"
        ].rank(method="min", ascending=False).astype(int)
        result["Density_Rank"] = result.groupby(group_cols, observed=True)[
            "Sum_Density_1000ind_km2"
        ].rank(method="min", ascending=False).astype(int)
    else:
        result["Biomass_Share_Percent"] = (
            100 * result["Sum_Biomass_kg_km2"] / result["Sum_Biomass_kg_km2"].sum()
        )
        result["Density_Share_Percent"] = (
            100 * result["Sum_Density_1000ind_km2"] / result["Sum_Density_1000ind_km2"].sum()
        )
        result["Biomass_Rank"] = result["Sum_Biomass_kg_km2"].rank(
            method="min", ascending=False
        ).astype(int)
        result["Density_Rank"] = result["Sum_Density_1000ind_km2"].rank(
            method="min", ascending=False
        ).astype(int)
    return result.sort_values(group_cols + ["Species_Order"] if group_cols else ["Species_Order"])


def calculate_iri(
    resource_table: pd.DataFrame,
    group_cols: list[str],
    scope_name: str,
) -> pd.DataFrame:
    result = resource_table.copy()
    result.insert(0, "Scope", scope_name)

    if group_cols:
        density_total = result.groupby(group_cols, observed=True)[
            "Sum_Density_1000ind_km2"
        ].transform("sum")
        biomass_total = result.groupby(group_cols, observed=True)[
            "Sum_Biomass_kg_km2"
        ].transform("sum")
    else:
        density_total = pd.Series(
            result["Sum_Density_1000ind_km2"].sum(), index=result.index
        )
        biomass_total = pd.Series(result["Sum_Biomass_kg_km2"].sum(), index=result.index)

    result["Abundance_Percent_N"] = np.where(
        density_total > 0,
        100 * result["Sum_Density_1000ind_km2"] / density_total,
        np.nan,
    )
    result["Biomass_Percent_W"] = np.where(
        biomass_total > 0,
        100 * result["Sum_Biomass_kg_km2"] / biomass_total,
        np.nan,
    )
    result["IRI"] = (
        result["Abundance_Percent_N"] + result["Biomass_Percent_W"]
    ) * result["Frequency_Percent"]

    if group_cols:
        iri_total = result.groupby(group_cols, observed=True)["IRI"].transform("sum")
        result["IRI_Percent"] = np.where(iri_total > 0, 100 * result["IRI"] / iri_total, np.nan)
        result["IRI_Rank"] = result.groupby(group_cols, observed=True)["IRI"].rank(
            method="min", ascending=False
        ).astype("Int64")
    else:
        iri_total_value = result["IRI"].sum()
        result["IRI_Percent"] = (
            100 * result["IRI"] / iri_total_value if iri_total_value > 0 else np.nan
        )
        result["IRI_Rank"] = result["IRI"].rank(method="min", ascending=False).astype("Int64")

    result["IRI_Class_Absolute"] = result["IRI"].map(iri_absolute_class)
    result["IRI_Class_Relative"] = result["IRI_Percent"].map(iri_relative_class)
    result["Dominant_By_Absolute_IRI"] = result["IRI"].ge(IRI_DOMINANT_THRESHOLD)
    result["Dominant_By_Relative_IRI"] = result["IRI_Percent"].ge(
        REL_IRI_DOMINANT_THRESHOLD
    )
    sort_cols = group_cols + ["IRI_Rank"] if group_cols else ["IRI_Rank"]
    return result.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def diversity_from_matrix(
    matrix: pd.DataFrame,
    prefix: str,
    include_margalef: bool,
) -> pd.DataFrame:
    values = matrix.to_numpy(dtype=float)
    if np.isnan(values).any() or (values < 0).any():
        raise ValueError(f"{prefix}矩阵含NA或负值。")

    totals = values.sum(axis=1)
    richness = (values > 0).sum(axis=1).astype(int)
    p = np.divide(values, totals[:, None], out=np.zeros_like(values), where=totals[:, None] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log(p), 0.0)
    shannon = -np.sum(np.where(p > 0, p * logp, 0.0), axis=1)
    sum_p2 = np.sum(p**2, axis=1)
    simpson = 1.0 - sum_p2
    hill_n1 = np.exp(shannon)
    hill_n2 = np.divide(1.0, sum_p2, out=np.full_like(sum_p2, np.nan), where=sum_p2 > 0)

    pielou = np.full(len(matrix), np.nan, dtype=float)
    multi = richness > 1
    single = richness == 1
    pielou[multi] = shannon[multi] / np.log(richness[multi])
    pielou[single] = 0.0

    no_community = richness == 0
    for array in [shannon, simpson, hill_n1, hill_n2]:
        array[no_community] = np.nan
    shannon[single] = 0.0
    simpson[single] = 0.0
    hill_n1[single] = 1.0
    hill_n2[single] = 1.0

    result = pd.DataFrame(index=matrix.index)
    result[f"Richness_{prefix}"] = richness
    result[f"Total_{prefix}"] = totals
    result[f"Shannon_H_{prefix}"] = shannon
    result[f"Pielou_J_{prefix}"] = pielou
    result[f"Simpson_1minusD_{prefix}"] = simpson
    result[f"Hill_N1_{prefix}"] = hill_n1
    result[f"Hill_N2_{prefix}"] = hill_n2

    if include_margalef:
        # 原始单位为千尾/km²，乘1000转为尾/km²后代入ln(N)。
        abundance_ind_km2 = totals * 1000.0
        margalef = np.full(len(matrix), np.nan, dtype=float)
        valid_multi = (richness > 1) & (abundance_ind_km2 > 1)
        margalef[valid_multi] = (richness[valid_multi] - 1) / np.log(
            abundance_ind_km2[valid_multi]
        )
        margalef[single] = 0.0
        result[f"Margalef_D_{prefix}"] = margalef
    return result


def make_period_column(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "Period" not in result.columns:
        result["Period"] = result["Month"].map({4: "April", 11: "November"})
    return result


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    paths = []
    for ext in ["png", "pdf", "svg"]:
        path = FIGURE_DIR / f"{stem}.{ext}"
        temporary_path = FIGURE_DIR / f".{stem}.{ext}.tmp"
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if ext == "png":
            kwargs["dpi"] = FIGURE_DPI
        fig.savefig(temporary_path, format=ext, **kwargs)
        temporary_path.replace(path)
        paths.append(path)
    plt.close(fig)
    return paths


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": FIGURE_DPI,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot_year_month_metric(
    ax: plt.Axes,
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    for month in [4, 11]:
        subset = summary.loc[summary["Month"].eq(month)].sort_values("Year")
        x = subset["Year"].to_numpy()
        y = subset[f"Mean_{metric}"].to_numpy(float)
        low = subset[f"CI95_Lower_{metric}"].to_numpy(float)
        high = subset[f"CI95_Upper_{metric}"].to_numpy(float)
        yerr = np.vstack([np.maximum(0, y - low), np.maximum(0, high - y)])
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            markersize=3.5,
            linewidth=1.3,
            capsize=2.2,
            color=MONTH_COLORS[month],
            label=MONTH_LABELS[month],
        )
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(summary["Year"].unique()))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
    ax.legend(frameon=False)


def write_excel_summary(
    tables: OrderedDict[str, pd.DataFrame],
    path: Path,
) -> bool:
    engine = None
    if importlib.util.find_spec("xlsxwriter") is not None:
        engine = "xlsxwriter"
    elif importlib.util.find_spec("openpyxl") is not None:
        engine = "openpyxl"
    if engine is None:
        print("警告：未检测到xlsxwriter/openpyxl，跳过Excel摘要。")
        return False

    try:
        with pd.ExcelWriter(path, engine=engine) as writer:
            for sheet_name, table in tables.items():
                safe_name = sheet_name[:31]
                table.to_excel(writer, sheet_name=safe_name, index=False)
                if engine == "xlsxwriter":
                    workbook = writer.book
                    worksheet = writer.sheets[safe_name]
                    header = workbook.add_format(
                        {
                            "bold": True,
                            "font_color": "#FFFFFF",
                            "bg_color": "#1F4E78",
                            "align": "center",
                            "valign": "vcenter",
                        }
                    )
                    for idx, col in enumerate(table.columns):
                        worksheet.write(0, idx, col, header)
                        sample = table[col].astype("string").head(300)
                        max_len = max(
                            [len(str(col))]
                            + [len(str(v)) for v in sample if pd.notna(v)]
                        )
                        worksheet.set_column(idx, idx, min(max(max_len + 2, 10), 32))
                    worksheet.freeze_panes(1, 1)
                    worksheet.autofilter(
                        0,
                        0,
                        max(len(table), 1),
                        max(len(table.columns) - 1, 0),
                    )
                else:
                    worksheet = writer.sheets[safe_name]
                    worksheet.freeze_panes = "B2"
                    worksheet.auto_filter.ref = worksheet.dimensions
                    from openpyxl.styles import Font, PatternFill

                    for cell in worksheet[1]:
                        cell.font = Font(bold=True, color="FFFFFF")
                        cell.fill = PatternFill("solid", fgColor="1F4E78")
        return True
    except Exception as exc:
        print(f"警告：Excel摘要生成失败，不影响CSV和图件：{type(exc).__name__}: {exc}")
        if path.exists():
            path.unlink()
        return False


# ============================================================
# 2. 读取Master v1.0并严格验收
# ============================================================

INPUT_DIR = locate_step3_dir()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

surveyed = read_csv(INPUT_DIR / SURVEYED_MASTER_NAME)
long_df = read_csv(INPUT_DIR / LONG_MASTER_NAME)
species_filter = read_csv(INPUT_DIR / SPECIES_FILTER_NAME)

surveyed_required = [
    "EventID",
    "Year",
    "Month",
    "Period",
    "LocationID",
    "Lon",
    "Lat",
    "SurveyStatus",
    "Cephalopod_Presence",
    "Total_Biomass_kg_km2",
    "Total_Density_1000ind_km2",
    "Species_Richness",
]
long_required = [
    "EventID",
    "Year",
    "Month",
    "Period",
    "LocationID",
    "SurveyStatus",
    "Species_Code",
    "Species_CN",
    "Species_Latin",
    "Species_Order",
    "Presence",
    "Biomass_kg_km2",
    "Density_1000ind_km2",
]
require_columns(surveyed, surveyed_required, "已调查事件主表")
require_columns(long_df, long_required, "事件×物种长表")

if not surveyed["EventID"].is_unique:
    raise ValueError("已调查事件主表EventID不唯一。")
if not surveyed["SurveyStatus"].isin([STATUS_POS, STATUS_ZERO]).all():
    raise ValueError("已调查主表中出现未调查或未知状态。")

for col in ["Year", "Month", "Total_Biomass_kg_km2", "Total_Density_1000ind_km2", "Species_Richness"]:
    surveyed[col] = pd.to_numeric(surveyed[col], errors="raise")
for col in ["Year", "Month", "Species_Order", "Presence", "Biomass_kg_km2", "Density_1000ind_km2"]:
    long_df[col] = pd.to_numeric(long_df[col], errors="raise")

surveyed["Year"] = surveyed["Year"].astype(int)
surveyed["Month"] = surveyed["Month"].astype(int)
long_df["Year"] = long_df["Year"].astype(int)
long_df["Month"] = long_df["Month"].astype(int)
surveyed = make_period_column(surveyed)
long_df = make_period_column(long_df)

species_codes = (
    species_filter.sort_values("Species_Code")["Species_Code"].astype(str).tolist()
)
if len(species_codes) != len(set(species_codes)):
    raise ValueError("物种字典Species_Code不唯一。")
if set(long_df["Species_Code"].astype(str)) != set(species_codes):
    raise ValueError("长表物种集合与物种字典不一致。")

pair_duplicates = int(long_df[["EventID", "Species_Code"]].duplicated().sum())
events_per_row = long_df.groupby("EventID")["Species_Code"].nunique()
event_set_matches = set(surveyed["EventID"]) == set(long_df["EventID"])

if pair_duplicates:
    raise AssertionError("事件×物种长表存在重复EventID-Species_Code。")
if not event_set_matches:
    raise AssertionError("已调查主表和长表EventID集合不一致。")
if not events_per_row.eq(len(species_codes)).all():
    raise AssertionError("并非每个调查事件都有完整25个物种记录。")
if long_df[["Presence", "Biomass_kg_km2", "Density_1000ind_km2"]].isna().any().any():
    raise AssertionError("已调查事件长表的物种响应仍有NA。")
if not set(long_df["Presence"].unique()).issubset({0, 1, 0.0, 1.0}):
    raise AssertionError("Presence含0/1以外数值。")
if (long_df[["Biomass_kg_km2", "Density_1000ind_km2"]] < 0).any().any():
    raise AssertionError("物种生物量或尾数密度含负值。")

zero_rows = long_df["Presence"].eq(0)
positive_rows = long_df["Presence"].eq(1)
if (long_df.loc[zero_rows, ["Biomass_kg_km2", "Density_1000ind_km2"]] != 0).any().any():
    raise AssertionError("Presence=0的物种仍有非零生物量或密度。")
if (
    (long_df.loc[positive_rows, "Biomass_kg_km2"] <= 0)
    | (long_df.loc[positive_rows, "Density_1000ind_km2"] <= 0)
).any():
    raise AssertionError("Presence=1的物种存在非正生物量或密度。")

long_event_summary = long_df.groupby("EventID", sort=False).agg(
    Richness_Recomputed=("Presence", "sum"),
    Total_Biomass_Recomputed=("Biomass_kg_km2", "sum"),
    Total_Density_Recomputed=("Density_1000ind_km2", "sum"),
)
surveyed_indexed = surveyed.set_index("EventID")
long_event_summary = long_event_summary.loc[surveyed_indexed.index]

richness_diff = (
    long_event_summary["Richness_Recomputed"] - surveyed_indexed["Species_Richness"]
).abs().max()
biomass_diff = (
    long_event_summary["Total_Biomass_Recomputed"]
    - surveyed_indexed["Total_Biomass_kg_km2"]
).abs().max()
density_diff = (
    long_event_summary["Total_Density_Recomputed"]
    - surveyed_indexed["Total_Density_1000ind_km2"]
).abs().max()
if richness_diff > 1e-9 or biomass_diff > 1e-6 or density_diff > 1e-6:
    raise AssertionError("长表物种加总与事件主表总量不一致。")

observed_counts = {
    "Surveyed events": len(surveyed),
    "Positive events": int(surveyed["SurveyStatus"].eq(STATUS_POS).sum()),
    "Zero events": int(surveyed["SurveyStatus"].eq(STATUS_ZERO).sum()),
    "Species": len(species_codes),
    "Long rows": len(long_df),
}
expected_counts = {
    "Surveyed events": EXPECTED_EVENTS,
    "Positive events": EXPECTED_POSITIVE,
    "Zero events": EXPECTED_ZERO,
    "Species": EXPECTED_SPECIES,
    "Long rows": EXPECTED_LONG_ROWS,
}
if STRICT_EXPECTED_COUNTS:
    mismatches = [
        f"{key}: expected {expected_counts[key]}, observed {observed_counts[key]}"
        for key in expected_counts
        if expected_counts[key] != observed_counts[key]
    ]
    if mismatches:
        raise AssertionError(
            "冻结计数检查未通过：\n"
            + "\n".join(mismatches)
            + "\n如为正式数据更新，请先核查后设置CEPHALOPOD_STRICT_EXPECTED_COUNTS=0。"
        )

surveyed = surveyed.sort_values(["Year", "Month", "LocationID", "EventID"], kind="stable")
event_order = surveyed["EventID"].tolist()
long_df["_EventOrder"] = long_df["EventID"].map(
    {event_id: idx for idx, event_id in enumerate(event_order)}
)
long_df = (
    long_df.sort_values(["_EventOrder", "Species_Order"], kind="stable")
    .drop(columns="_EventOrder")
    .reset_index(drop=True)
)


# ============================================================
# 3. 资源量分析
# ============================================================

resource_event_columns = [
    "EventID",
    "EventID_Raw",
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
resource_event_columns = [c for c in resource_event_columns if c in surveyed.columns]
resource_event = surveyed[resource_event_columns].copy()
resource_event["Zero_Catch"] = resource_event["SurveyStatus"].eq(STATUS_ZERO)

resource_metrics = [
    "Total_Biomass_kg_km2",
    "Total_Density_1000ind_km2",
    "Species_Richness",
]
resource_year_month = summarize_event_table(
    resource_event, ["Year", "Month", "Period"], resource_metrics, seed_offset=10
)
resource_year = summarize_event_table(
    resource_event, ["Year"], resource_metrics, seed_offset=20
)
resource_month = summarize_event_table(
    resource_event, ["Month", "Period"], resource_metrics, seed_offset=30
)
resource_overall = summarize_event_table(
    resource_event, [], resource_metrics, seed_offset=40
)
resource_annual_balanced = season_balanced_summary(resource_year_month, resource_metrics)

species_resource_overall = species_resource_summary(long_df, [])
species_resource_month = species_resource_summary(long_df, ["Month", "Period"])
species_resource_year = species_resource_summary(long_df, ["Year"])
species_resource_year_month = species_resource_summary(
    long_df, ["Year", "Month", "Period"]
)


# ============================================================
# 4. IRI分析
# ============================================================

iri_overall = calculate_iri(species_resource_overall, [], "Overall")
iri_month = calculate_iri(species_resource_month, ["Month", "Period"], "Month")
iri_year = calculate_iri(species_resource_year, ["Year"], "Year")
iri_year_month = calculate_iri(
    species_resource_year_month,
    ["Year", "Month", "Period"],
    "YearMonth",
)
iri_year_month["YearMonth"] = (
    iri_year_month["Year"].astype(str)
    + "-"
    + iri_year_month["Month"].astype(int).astype(str).str.zfill(2)
)

iri_top_rows = []
for table, group_cols in [
    (iri_overall, []),
    (iri_month, ["Month", "Period"]),
    (iri_year, ["Year"]),
    (iri_year_month, ["Year", "Month", "Period", "YearMonth"]),
]:
    if not group_cols:
        group_iter = [((), table)]
    else:
        group_iter = stable_group_iterator(table, group_cols)
    for _, group in group_iter:
        selected = group.loc[
            group["IRI_Rank"].le(5) | group["Dominant_By_Absolute_IRI"]
        ].copy()
        selected["Selection_Reason"] = np.where(
            selected["Dominant_By_Absolute_IRI"] & selected["IRI_Rank"].le(5),
            "Top5_and_Dominant",
            np.where(selected["Dominant_By_Absolute_IRI"], "Dominant", "Top5"),
        )
        iri_top_rows.append(selected)
iri_top_dominant = pd.concat(iri_top_rows, ignore_index=True)


# ============================================================
# 5. α多样性分析
# ============================================================

density_matrix = (
    long_df.pivot(index="EventID", columns="Species_Code", values="Density_1000ind_km2")
    .loc[event_order, species_codes]
)
biomass_matrix = (
    long_df.pivot(index="EventID", columns="Species_Code", values="Biomass_kg_km2")
    .loc[event_order, species_codes]
)
presence_matrix = (
    long_df.pivot(index="EventID", columns="Species_Code", values="Presence")
    .loc[event_order, species_codes]
)

alpha_density = diversity_from_matrix(
    density_matrix,
    prefix="Density",
    include_margalef=True,
)
alpha_biomass = diversity_from_matrix(
    biomass_matrix,
    prefix="Biomass",
    include_margalef=False,
)

alpha_metadata_cols = [
    "EventID",
    "Year",
    "Month",
    "Period",
    "Season",
    "LocationID",
    "Lon",
    "Lat",
    "SurveyStatus",
    "Cephalopod_Presence",
    "Total_Biomass_kg_km2",
    "Total_Density_1000ind_km2",
    "Species_Richness",
]
alpha_metadata_cols = [c for c in alpha_metadata_cols if c in surveyed.columns]
alpha_event = surveyed.set_index("EventID")[alpha_metadata_cols[1:]].copy()
alpha_event = alpha_event.join(alpha_density, how="left", validate="one_to_one")
alpha_event = alpha_event.join(alpha_biomass, how="left", validate="one_to_one")
alpha_event = alpha_event.reset_index()
alpha_event["Zero_Catch"] = alpha_event["SurveyStatus"].eq(STATUS_ZERO)
alpha_event["Alpha_Diversity_Defined"] = alpha_event["Richness_Density"].gt(0)

alpha_metrics = [
    "Species_Richness",
    "Shannon_H_Density",
    "Pielou_J_Density",
    "Simpson_1minusD_Density",
    "Hill_N1_Density",
    "Hill_N2_Density",
    "Margalef_D_Density",
    "Shannon_H_Biomass",
    "Pielou_J_Biomass",
    "Simpson_1minusD_Biomass",
    "Hill_N1_Biomass",
    "Hill_N2_Biomass",
]
alpha_year_month = summarize_event_table(
    alpha_event, ["Year", "Month", "Period"], alpha_metrics, seed_offset=100
)
alpha_year = summarize_event_table(
    alpha_event, ["Year"], alpha_metrics, seed_offset=200
)
alpha_month = summarize_event_table(
    alpha_event, ["Month", "Period"], alpha_metrics, seed_offset=300
)
alpha_overall = summarize_event_table(alpha_event, [], alpha_metrics, seed_offset=400)
alpha_annual_balanced = season_balanced_summary(alpha_year_month, alpha_metrics)


# ============================================================
# 6. 方法、质控与摘要
# ============================================================

def max_group_sum_error(
    table: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    target: float = 100.0,
) -> float:
    if not group_cols:
        return float(abs(table[value_col].sum() - target))
    sums = table.groupby(group_cols, observed=True)[value_col].sum()
    return float((sums - target).abs().max())


iri_checks = {
    "Overall N% sum error": max_group_sum_error(iri_overall, [], "Abundance_Percent_N"),
    "Overall W% sum error": max_group_sum_error(iri_overall, [], "Biomass_Percent_W"),
    "Overall %IRI sum error": max_group_sum_error(iri_overall, [], "IRI_Percent"),
    "Month %IRI max sum error": max_group_sum_error(
        iri_month, ["Month", "Period"], "IRI_Percent"
    ),
    "Year %IRI max sum error": max_group_sum_error(iri_year, ["Year"], "IRI_Percent"),
    "YearMonth %IRI max sum error": max_group_sum_error(
        iri_year_month, ["Year", "Month", "Period"], "IRI_Percent"
    ),
}

zero_alpha_columns = [
    "Shannon_H_Density",
    "Pielou_J_Density",
    "Simpson_1minusD_Density",
    "Hill_N1_Density",
    "Hill_N2_Density",
    "Margalef_D_Density",
    "Shannon_H_Biomass",
    "Pielou_J_Biomass",
    "Simpson_1minusD_Biomass",
    "Hill_N1_Biomass",
    "Hill_N2_Biomass",
]

qc_rows = [
    ("Surveyed events", EXPECTED_EVENTS, len(surveyed), len(surveyed) == EXPECTED_EVENTS),
    ("Positive events", EXPECTED_POSITIVE, observed_counts["Positive events"], observed_counts["Positive events"] == EXPECTED_POSITIVE),
    ("Zero events", EXPECTED_ZERO, observed_counts["Zero events"], observed_counts["Zero events"] == EXPECTED_ZERO),
    ("Species", EXPECTED_SPECIES, len(species_codes), len(species_codes) == EXPECTED_SPECIES),
    ("Long rows", EXPECTED_LONG_ROWS, len(long_df), len(long_df) == EXPECTED_LONG_ROWS),
    ("Duplicate EventID-Species", 0, pair_duplicates, pair_duplicates == 0),
    ("Max richness reconciliation difference", 0.0, float(richness_diff), richness_diff <= 1e-9),
    ("Max biomass reconciliation difference", 0.0, float(biomass_diff), biomass_diff <= 1e-6),
    ("Max density reconciliation difference", 0.0, float(density_diff), density_diff <= 1e-6),
    ("Zero catches with defined alpha values", 0, int(alpha_event.loc[alpha_event["Zero_Catch"], zero_alpha_columns].notna().sum().sum()), int(alpha_event.loc[alpha_event["Zero_Catch"], zero_alpha_columns].notna().sum().sum()) == 0),
    ("Positive events with undefined density Shannon", 0, int(alpha_event.loc[~alpha_event["Zero_Catch"], "Shannon_H_Density"].isna().sum()), int(alpha_event.loc[~alpha_event["Zero_Catch"], "Shannon_H_Density"].isna().sum()) == 0),
    ("IRI overall N% sum error", 0.0, iri_checks["Overall N% sum error"], iri_checks["Overall N% sum error"] < 1e-8),
    ("IRI overall W% sum error", 0.0, iri_checks["Overall W% sum error"], iri_checks["Overall W% sum error"] < 1e-8),
    ("IRI overall % sum error", 0.0, iri_checks["Overall %IRI sum error"], iri_checks["Overall %IRI sum error"] < 1e-8),
    ("IRI YearMonth max % sum error", 0.0, iri_checks["YearMonth %IRI max sum error"], iri_checks["YearMonth %IRI max sum error"] < 1e-8),
    ("Raw biomass used; no log1p response", True, True, True),
]
qc_checks = pd.DataFrame(qc_rows, columns=["Check", "Expected", "Observed", "Pass"])
if not to_bool(qc_checks["Pass"]).all():
    raise AssertionError(
        "第四步质控未全部通过：\n"
        + qc_checks.loc[~to_bool(qc_checks["Pass"])].to_string(index=False)
    )

method_rows = [
    ("Analysis unit", "One surveyed EventID = Year + Month + LocationID"),
    ("Surveyed events", len(surveyed)),
    ("Positive events", observed_counts["Positive events"]),
    ("Zero-catch events", observed_counts["Zero events"]),
    ("Species retained", len(species_codes)),
    ("Biomass response", "Raw Biomass_kg_km2; no log1p"),
    ("Density response", "Raw Density_1000ind_km2; no log1p"),
    ("Resource comparison", "Mean across all surveyed events including zero catches"),
    ("Annual primary descriptive index", "Equal-weight mean of April and November means"),
    ("IRI formula", "(Abundance_Percent_N + Biomass_Percent_W) * Frequency_Percent"),
    ("IRI abundance component", "Sum of Density_1000ind_km2"),
    ("IRI biomass component", "Sum of Biomass_kg_km2"),
    ("IRI occurrence denominator", "All surveyed events in the group, including zero catches"),
    ("Absolute IRI thresholds", f"Dominant >= {IRI_DOMINANT_THRESHOLD}; Important >= {IRI_IMPORTANT_THRESHOLD}; Common >= {IRI_COMMON_THRESHOLD}; else Rare"),
    ("Relative IRI thresholds", f"Dominant >= {REL_IRI_DOMINANT_THRESHOLD}%; Important >= {REL_IRI_IMPORTANT_THRESHOLD}%; Common >= {REL_IRI_COMMON_THRESHOLD}%; else Rare"),
    ("Primary alpha proportions", "Species density proportions"),
    ("Shannon logarithm", "Natural logarithm"),
    ("Margalef abundance basis", "Total density converted from 1000 ind/km2 to ind/km2"),
    ("Zero-catch alpha handling", "Richness=0; H/J/Simpson/Hill/Margalef=NA"),
    ("Single-species alpha handling", "H=0; J=0; Simpson=0; Hill N1=N2=1; Margalef=0"),
    ("Confidence intervals", f"Event-level nonparametric bootstrap percentile 95% CI; B={BOOTSTRAP_ITERATIONS}"),
    ("Random seed", RANDOM_SEED),
    ("Inferential tests", "Not performed in Step4"),
]
method_metadata = pd.DataFrame(method_rows, columns=["Item", "Definition"])

overall_row = resource_overall.iloc[0]
alpha_overall_row = alpha_overall.iloc[0]
summary_rows = [
    ("Step", "Step4 resource, IRI and alpha diversity"),
    ("Surveyed events", len(surveyed)),
    ("Positive events", observed_counts["Positive events"]),
    ("Zero events", observed_counts["Zero events"]),
    ("Encounter rate percent", float(overall_row["Encounter_Rate_Percent"])),
    ("Species retained", len(species_codes)),
    ("Mean event biomass kg/km2", float(overall_row["Mean_Total_Biomass_kg_km2"])),
    ("Median event biomass kg/km2", float(overall_row["Median_Total_Biomass_kg_km2"])),
    ("Mean event density 1000ind/km2", float(overall_row["Mean_Total_Density_1000ind_km2"])),
    ("Mean richness including zero catches", float(overall_row["Mean_Species_Richness"])),
    ("Mean Shannon density conditional on positive catch", float(alpha_overall_row["Mean_Shannon_H_Density"])),
    ("Mean Pielou density conditional on positive catch", float(alpha_overall_row["Mean_Pielou_J_Density"])),
    ("Overall absolute-IRI dominant species", int(iri_overall["Dominant_By_Absolute_IRI"].sum())),
    ("Overall relative-IRI dominant species", int(iri_overall["Dominant_By_Relative_IRI"].sum())),
    ("Bootstrap iterations", BOOTSTRAP_ITERATIONS),
    ("Raw biomass used", True),
    ("Inferential tests performed", False),
]
step4_summary = pd.DataFrame(summary_rows, columns=["Metric", "Value"])


# ============================================================
# 7. 绘制论文用图
# ============================================================

configure_plot_style()
figure_paths: list[Path] = []

# Fig01 资源量年际变化
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1))
plot_year_month_metric(
    axes[0],
    resource_year_month,
    "Total_Biomass_kg_km2",
    "Mean biomass (kg km$^{-2}$)",
    "(a) Cephalopod biomass",
)
plot_year_month_metric(
    axes[1],
    resource_year_month,
    "Total_Density_1000ind_km2",
    "Mean density (10$^3$ ind. km$^{-2}$)",
    "(b) Cephalopod density",
)
fig.suptitle("Interannual variation in cephalopod resources", fontsize=11, y=1.02)
fig.tight_layout()
figure_paths.extend(save_figure(fig, "Fig01_resource_year_month"))

# Fig02 总体IRI排序
iri_plot = iri_overall.nsmallest(min(15, len(iri_overall)), "IRI_Rank").copy()
iri_plot = iri_plot.sort_values("IRI_Percent", ascending=True)
class_colors = {
    "Dominant": "#B2182B",
    "Important": "#EF8A62",
    "Common": "#67A9CF",
    "Rare": "#BDBDBD",
    "Undefined": "#BDBDBD",
}
fig, ax = plt.subplots(figsize=(8.2, 6.3))
ax.barh(
    np.arange(len(iri_plot)),
    iri_plot["IRI_Percent"],
    color=[class_colors.get(v, "#BDBDBD") for v in iri_plot["IRI_Class_Absolute"]],
    edgecolor="none",
)
ax.set_yticks(np.arange(len(iri_plot)))
ax.set_yticklabels(iri_plot["Species_Latin"])
for tick in ax.get_yticklabels():
    tick.set_fontstyle("italic")
ax.set_xlabel("Relative index of importance (%IRI)")
ax.set_title("Overall relative importance of cephalopod species")
ax.grid(axis="x", color="#D9D9D9", linewidth=0.5)
fig.tight_layout()
figure_paths.extend(save_figure(fig, "Fig02_IRI_overall_top15"))

# Fig03 年月IRI热图（总体IRI前12种）
top_codes = iri_overall.nsmallest(min(12, len(iri_overall)), "IRI_Rank")[
    "Species_Code"
].tolist()
heat_source = iri_year_month.loc[iri_year_month["Species_Code"].isin(top_codes)].copy()
heat_order = (
    iri_overall.loc[iri_overall["Species_Code"].isin(top_codes)]
    .sort_values("IRI_Rank")["Species_Code"]
    .tolist()
)
time_order = (
    iri_year_month[["Year", "Month", "YearMonth"]]
    .drop_duplicates()
    .sort_values(["Year", "Month"])["YearMonth"]
    .tolist()
)
heat_matrix = (
    heat_source.pivot(index="Species_Code", columns="YearMonth", values="IRI_Percent")
    .reindex(index=heat_order, columns=time_order)
)
latin_lookup = iri_overall.set_index("Species_Code")["Species_Latin"].to_dict()
positive_heat = heat_matrix.to_numpy(float)
positive_values = positive_heat[np.isfinite(positive_heat) & (positive_heat > 0)]
vmin = max(0.01, float(np.nanmin(positive_values))) if len(positive_values) else 0.01
vmax = float(np.nanmax(positive_values)) if len(positive_values) else 1.0
masked = np.ma.masked_less_equal(positive_heat, 0)
fig, ax = plt.subplots(figsize=(13.2, 6.0))
image = ax.imshow(
    masked,
    aspect="auto",
    interpolation="nearest",
    cmap="YlOrRd",
    norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin * 1.01)),
)
ax.set_xticks(np.arange(len(time_order)))
ax.set_xticklabels(time_order, rotation=45, ha="right")
ax.set_yticks(np.arange(len(heat_order)))
ax.set_yticklabels([latin_lookup[c] for c in heat_order])
for tick in ax.get_yticklabels():
    tick.set_fontstyle("italic")
ax.set_xlabel("Survey period")
ax.set_ylabel("Species")
ax.set_title("Temporal variation in %IRI of the 12 most important species")
cbar = fig.colorbar(image, ax=ax, pad=0.02)
cbar.set_label("%IRI (log colour scale)")
fig.tight_layout()
figure_paths.extend(save_figure(fig, "Fig03_IRI_year_month_heatmap_top12"))

# Fig04 α多样性年际变化
fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.5), sharex=True)
alpha_panels = [
    ("Species_Richness", "Mean richness", "(a) Species richness (all surveyed events)"),
    ("Shannon_H_Density", "Shannon H′", "(b) Shannon diversity"),
    ("Pielou_J_Density", "Pielou J′", "(c) Pielou evenness"),
    ("Simpson_1minusD_Density", "Simpson 1 − D", "(d) Simpson diversity"),
]
for ax, (metric, ylabel, title) in zip(axes.ravel(), alpha_panels):
    plot_year_month_metric(ax, alpha_year_month, metric, ylabel, title)
for ax in axes[0, :]:
    ax.set_xlabel("")
fig.suptitle(
    "Interannual variation in cephalopod alpha diversity",
    fontsize=11,
    y=1.01,
)
fig.tight_layout()
figure_paths.extend(save_figure(fig, "Fig04_alpha_diversity_year_month"))


# ============================================================
# 8. 输出CSV、Excel、README和ZIP
# ============================================================

output_tables: OrderedDict[str, pd.DataFrame] = OrderedDict(
    [
        ("00_step4_summary.csv", step4_summary),
        ("01_resource_event_level.csv", resource_event),
        ("02_resource_summary_year_month.csv", resource_year_month),
        ("03_resource_summary_year_event_weighted.csv", resource_year),
        ("04_resource_summary_month.csv", resource_month),
        ("05_resource_annual_season_balanced.csv", resource_annual_balanced),
        ("06_species_resource_overall.csv", species_resource_overall),
        ("07_species_resource_by_month.csv", species_resource_month),
        ("08_species_resource_by_year.csv", species_resource_year),
        ("09_species_resource_by_year_month.csv", species_resource_year_month),
        ("10_IRI_overall.csv", iri_overall),
        ("11_IRI_by_month.csv", iri_month),
        ("12_IRI_by_year.csv", iri_year),
        ("13_IRI_by_year_month.csv", iri_year_month),
        ("14_IRI_top5_and_dominant.csv", iri_top_dominant),
        ("15_alpha_diversity_event_level.csv", alpha_event),
        ("16_alpha_summary_year_month.csv", alpha_year_month),
        ("17_alpha_summary_year_event_weighted.csv", alpha_year),
        ("18_alpha_summary_month.csv", alpha_month),
        ("19_alpha_annual_season_balanced.csv", alpha_annual_balanced),
        ("20_method_metadata.csv", method_metadata),
        ("21_qc_checks.csv", qc_checks),
    ]
)

output_paths: list[Path] = []
for filename, table in output_tables.items():
    output_paths.append(write_csv(table, filename))

purpose_map = {
    "01_resource_event_level.csv": "事件级资源量；含真实零捕获",
    "02_resource_summary_year_month.csv": "4月/11月分别的年际资源量主表",
    "05_resource_annual_season_balanced.csv": "4月和11月等权年度资源指数",
    "06_species_resource_overall.csv": "25种总体资源组成",
    "09_species_resource_by_year_month.csv": "25种逐年逐月资源组成",
    "10_IRI_overall.csv": "25种总体IRI与%IRI",
    "11_IRI_by_month.csv": "4月和11月IRI",
    "13_IRI_by_year_month.csv": "逐年逐月IRI",
    "15_alpha_diversity_event_level.csv": "事件级α多样性；零捕获多样性为NA",
    "16_alpha_summary_year_month.csv": "4月/11月分别的α多样性年际主表",
    "19_alpha_annual_season_balanced.csv": "4月和11月等权年度α多样性",
}
manifest_rows = []
for filename, table in output_tables.items():
    manifest_rows.append(
        {
            "File": filename,
            "Rows": len(table),
            "Columns": len(table.columns),
            "Purpose": purpose_map.get(filename, "辅助汇总、方法或质量检查"),
        }
    )
for path in figure_paths:
    manifest_rows.append(
        {
            "File": str(path.relative_to(OUTPUT_DIR)),
            "Rows": pd.NA,
            "Columns": pd.NA,
            "Purpose": "论文用图件",
        }
    )
output_manifest = pd.DataFrame(manifest_rows)
manifest_path = write_csv(output_manifest, "22_output_manifest.csv")
output_paths.append(manifest_path)

readme = f"""浙江近岸头足类第四步：资源量、IRI与α多样性
============================================================

分析样本
--------
- 已调查事件：{len(surveyed)}
- 正捕获事件：{observed_counts['Positive events']}
- 零捕获事件：{observed_counts['Zero events']}
- 物种：{len(species_codes)}
- 未调查事件：不进入本步骤

资源量
------
- 使用原始生物量kg/km²和原始尾数密度千尾/km²，不做log1p；
- 年月主表：02_resource_summary_year_month.csv；
- 年际比较优先分别展示4月和11月；
- 若需要单一年度指数，使用05_resource_annual_season_balanced.csv，
  该表给予4月和11月相同权重；
- Sum字段只用于数据核对，不能在调查样本数不同时直接比较。

IRI
---
IRI = (N% + W%) × F%
- N%基于尾数密度加总；
- W%基于生物量加总；
- F%的分母为该组全部已调查事件，包括零捕获；
- 同时输出IRI和%IRI；
- 总体、月份、年份、年份×月份结果分别见10—13号文件；
- IRI等级阈值属于可配置描述性标准，论文中必须明确写出所用阈值。

α多样性
--------
- 主结果按尾数密度比例计算；
- 物种丰富度包含零捕获事件（S=0）；
- Shannon、Pielou、Simpson、Hill和Margalef对零捕获事件记为NA，
  因此这些指标的组均值代表“有头足类事件中的条件多样性”；
- 同时输出生物量比例版本作为敏感性结果；
- Margalef使用尾数密度换算为尾/km²后的总量，数值受丰度单位影响，
  论文中应明确计算基础。

置信区间与统计推断
------------------
- 年月均值95%区间为事件级bootstrap，B={BOOTSTRAP_ITERATIONS}；
- 本步骤不进行显著性检验；
- 下一步时间效应模型应考虑LocationID重复测量和零膨胀/偏态分布。

图件
----
- Fig01：4月/11月资源量年际变化；
- Fig02：总体%IRI前15种；
- Fig03：总体IRI前12种的年份×月份热图；
- Fig04：α多样性年际变化；
- 每图均输出PNG 600 dpi、PDF和SVG。
"""
readme_path = OUTPUT_DIR / "README_step4.txt"
readme_path.write_text(readme, encoding="utf-8")
output_paths.append(readme_path)

excel_tables: OrderedDict[str, pd.DataFrame] = OrderedDict(
    [
        ("Summary", step4_summary),
        ("Resource_year_month", resource_year_month),
        ("Resource_balanced", resource_annual_balanced),
        ("IRI_overall", iri_overall),
        ("IRI_month", iri_month),
        ("Alpha_year_month", alpha_year_month),
        ("Alpha_balanced", alpha_annual_balanced),
        ("Methods", method_metadata),
        ("QC", qc_checks),
        ("Manifest", output_manifest),
    ]
)
excel_path = OUTPUT_DIR / "Step4_resource_IRI_alpha_summary.xlsx"
if write_excel_summary(excel_tables, excel_path):
    output_paths.append(excel_path)

output_paths.extend(figure_paths)
zip_path = Path(str(OUTPUT_DIR) + ".zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in output_paths:
        archive.write(path, arcname=str(path.relative_to(OUTPUT_DIR)))


# ============================================================
# 9. Kaggle控制台摘要
# ============================================================

print("=" * 76)
print("第四步完成：浙江近岸头足类资源量、IRI与α多样性")
print("=" * 76)
print(f"已调查事件：{len(surveyed):,}")
print(f"正捕获事件：{observed_counts['Positive events']:,}")
print(f"零捕获事件：{observed_counts['Zero events']:,}")
print(f"物种数：{len(species_codes)}")
print(f"总体出现率：{float(overall_row['Encounter_Rate_Percent']):.2f}%")
print(f"总体平均生物量：{float(overall_row['Mean_Total_Biomass_kg_km2']):.4f} kg/km²")
print(f"总体平均密度：{float(overall_row['Mean_Total_Density_1000ind_km2']):.4f} 千尾/km²")
print(f"绝对IRI标准下总体优势种：{int(iri_overall['Dominant_By_Absolute_IRI'].sum())}种")
print(f"质控通过：{bool(to_bool(qc_checks['Pass']).all())}")
print(f"输出目录：{OUTPUT_DIR}")
print(f"ZIP文件：{zip_path}")
print("请下载：cephalopod_step4_resource_iri_alpha.zip")
