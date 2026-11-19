# -*- coding: utf-8 -*-
"""


运行顺序
--------
1. 运行最新版第一步：kaggle_step1_cephalopod_fishery_cleaning.py
2. 运行最新版第二步：kaggle_step2_environment_matching.py
3. 再运行本代码。

负责：
1. 以 EventID（Year + Month + LocationID）为唯一主键，合并调查事件、
   环境变量、25个物种的出现/生物量/尾数密度矩阵；
2. 形成后续所有分析统一读取的 Master v1.0；
3. 区分：
   - 全部设计事件（预期2659）；
   - 已调查事件（预期2100，包括有头足类和调查为0）；
   - 严格环境匹配事件（预期2077）；
   - 不含Chla敏感性样本；
   - Chla至少3个有效4 km源像元的敏感性样本；
4. 25种全部保留用于IRI、物种组成、多样性、NMDS和PERMANOVA；
5. 根据严格环境样本中的出现次数，自动生成：
   - 出现 >= 20次：JSDM/共现主分析物种；
   - 出现 >= 10次：敏感性分析物种；
6. 分别形成4月和11月的环境模型矩阵，降低Month与SST强相关造成的解释混淆；
7. 所有连续变量保留原始尺度。本步骤绝不做全数据Z-score；标准化必须在
   后续交叉验证的训练折内完成，避免信息泄漏。

原则
--------
- 渔业事件不聚合到0.125°；同一环境格网内的不同站位仍保留为不同行。
- 坐标是空间分析元数据；坐标缺失不应删除非空间群落分析样本。
- NOT_SURVEYED不是0，物种值必须保持NA，不能进入群落或环境模型。
- SURVEYED_ZERO是真实调查结果，25个物种均为0，必须保留。
- Chla正式变量为log1p_Chla_0125；Chla_4km_Nearest仅为对照字段。
- Year_rel只是长期未观测年际效应的代理变量，不应解释为直接生态驱动。

输入目录
------------
/kaggle/working/cephalopod_step1_cleaned
/kaggle/working/cephalopod_step2_environment_matched

输出目录
------------
/kaggle/working/cephalopod_step3_analysis_master

主要输出
--------
00_step3_summary.csv
01_analysis_master_v1_all_events_wide.csv
02_analysis_master_v1_surveyed_all25.csv
03_analysis_master_v1_environment_strict_all25.csv
04_analysis_master_v1_long_surveyed_all25.csv
05_species_filter_summary.csv
06_taxonomy_analysis_dictionary.csv
07_Y_presence_all25_surveyed.csv
08_Y_biomass_all25_surveyed.csv
09_Y_density_all25_surveyed.csv
10_X_environment_strict_raw.csv
11_Y_presence_main_GE20_strict.csv
12_Y_biomass_main_GE20_strict.csv
13_Y_presence_sensitivity_GE10_strict.csv
14_Y_biomass_sensitivity_GE10_strict.csv
15_X_environment_noChla_sensitivity_raw.csv
16_Y_presence_main_GE20_noChla.csv
17_Y_biomass_main_GE20_noChla.csv
18_X_environment_ChlaMin3_sensitivity_raw.csv
19_Y_presence_main_GE20_ChlaMin3.csv
20_Y_biomass_main_GE20_ChlaMin3.csv
21_X_environment_april_strict_raw.csv
22_Y_presence_april_GE20.csv
23_Y_biomass_april_GE20.csv
24_X_environment_november_strict_raw.csv
25_Y_presence_november_GE20.csv
26_Y_biomass_november_GE20.csv
27_sample_flow_by_year_month.csv
28_environment_excluded_surveyed_events.csv
29_matrix_alignment_qc.csv
30_predictor_vif.csv
31_predictor_correlations.csv
32_analysis_variable_dictionary.csv
33_output_manifest.csv
README_step3.txt
Step3_analysis_master.xlsx
cephalopod_step3_analysis_master.zip
"""

from __future__ import annotations

import importlib.util
import os
import re
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 0. 参数设置
# ============================================================

STEP1_DIR = Path(
    os.environ.get(
        "CEPHALOPOD_STEP1_DIR",
        "/kaggle/working/cephalopod_step1_cleaned",
    )
)
STEP2_DIR = Path(
    os.environ.get(
        "CEPHALOPOD_STEP2_DIR",
        "/kaggle/working/cephalopod_step2_environment_matched",
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        "CEPHALOPOD_STEP3_OUTPUT_DIR",
        "/kaggle/working/cephalopod_step3_analysis_master",
    )
)

EVENT_ENV_CSV = STEP2_DIR / "01_event_environment_matched.csv"
BIOMASS_MATRIX_CSV = STEP1_DIR / "03_biomass_matrix.csv"
DENSITY_MATRIX_CSV = STEP1_DIR / "04_density_matrix.csv"
PRESENCE_MATRIX_CSV = STEP1_DIR / "05_presence_matrix.csv"
TAXONOMY_CSV = STEP1_DIR / "06_taxonomy_dictionary.csv"

EXPECTED_YEARS = set(range(2014, 2025))
EXPECTED_MONTHS = {4, 11}
EXPECTED_TOTAL_EVENTS = int(os.environ.get("CEPHALOPOD_EXPECTED_TOTAL_EVENTS", "2659"))
EXPECTED_SURVEYED_EVENTS = int(os.environ.get("CEPHALOPOD_EXPECTED_SURVEYED_EVENTS", "2100"))
EXPECTED_STRICT_EVENTS = int(os.environ.get("CEPHALOPOD_EXPECTED_STRICT_EVENTS", "2077"))
EXPECTED_SPECIES_COUNT = int(os.environ.get("CEPHALOPOD_EXPECTED_SPECIES_COUNT", "25"))
EXPECTED_LOCATION_COUNT = int(os.environ.get("CEPHALOPOD_EXPECTED_LOCATION_COUNT", "124"))
EXPECTED_MAIN_SPECIES_COUNT = int(os.environ.get("CEPHALOPOD_EXPECTED_MAIN_SPECIES", "16"))
EXPECTED_SENS_SPECIES_COUNT = int(os.environ.get("CEPHALOPOD_EXPECTED_SENS_SPECIES", "20"))

MAIN_OCCURRENCE_THRESHOLD = int(
    os.environ.get("CEPHALOPOD_MAIN_OCCURRENCE_THRESHOLD", "20")
)
SENS_OCCURRENCE_THRESHOLD = int(
    os.environ.get("CEPHALOPOD_SENS_OCCURRENCE_THRESHOLD", "10")
)
CHLA_MIN_PIXELS_SENSITIVITY = int(
    os.environ.get("CEPHALOPOD_CHLA_MIN_PIXELS_SENSITIVITY", "3")
)


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "n", "off"}


# 当前数据已冻结时建议保持True，可防止误读旧版第一/二步结果。
# 仅在模拟联调或数据确有正式更新时通过环境变量设为0。
STRICT_EXPECTED_COUNTS = env_bool("CEPHALOPOD_STRICT_EXPECTED_COUNTS", True)

STATUS_NOT = "NOT_SURVEYED"
STATUS_ZERO = "SURVEYED_ZERO"
STATUS_POS = "SURVEYED_POSITIVE"
SURVEYED_STATUSES = {STATUS_ZERO, STATUS_POS}

MASTER_VERSION = "Master v1.0"


# ============================================================
# 1. 通用函数
# ============================================================

def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"缺少输入文件：{path}\n"
            "请确认已经在同一个Kaggle会话中依次运行最新版第一步和第二步，"
            "或通过环境变量CEPHALOPOD_STEP1_DIR/CEPHALOPOD_STEP2_DIR指定目录。"
        )
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def write_csv(df: pd.DataFrame, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"{label}缺少字段：{missing}\n"
            "这通常说明读取了旧版输出。请重新运行最新版第一步和第二步。"
        )


def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    true_values = {"true", "1", "yes", "y", "t"}
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin(true_values)
        .fillna(False)
        .astype(bool)
    )


def species_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if re.fullmatch(r"Species_\d+", str(c))]
    return sorted(cols, key=lambda x: int(str(x).split("_")[1]))


def safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return np.where(denominator.gt(0), numerator / denominator, np.nan)


def assert_close_series(
    observed: pd.Series,
    expected: pd.Series,
    label: str,
    atol: float = 1e-7,
) -> None:
    left = pd.to_numeric(observed, errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(expected, errors="coerce").to_numpy(dtype=float)
    if not np.allclose(left, right, equal_nan=True, rtol=1e-7, atol=atol):
        bad = ~np.isclose(left, right, equal_nan=True, rtol=1e-7, atol=atol)
        raise AssertionError(f"{label}不一致，异常行数：{int(bad.sum())}")


def make_y(
    matrix_indexed: pd.DataFrame,
    event_ids: list[str],
    selected_species: list[str],
) -> pd.DataFrame:
    if not selected_species:
        raise ValueError("物种筛选结果为空，无法形成Y矩阵。")
    result = matrix_indexed.loc[event_ids, selected_species].reset_index()
    if result["EventID"].tolist() != event_ids:
        raise AssertionError("Y矩阵EventID顺序与目标事件顺序不一致。")
    return result


def calculate_vif(df: pd.DataFrame, variables: list[str], scenario: str) -> pd.DataFrame:
    data = df[variables].apply(pd.to_numeric, errors="coerce").dropna()
    rows = []
    if len(data) < 3:
        return pd.DataFrame(
            [{
                "Scenario": scenario,
                "Variable": pd.NA,
                "VIF": np.nan,
                "N_Complete": len(data),
                "Status": "INSUFFICIENT_ROWS",
            }]
        )

    for variable in variables:
        y = data[variable].to_numpy(dtype=float)
        others = [c for c in variables if c != variable]
        if np.nanstd(y, ddof=0) <= 1e-12:
            vif = np.inf
            status = "ZERO_VARIANCE"
        elif not others:
            vif = 1.0
            status = "OK"
        else:
            x = data[others].to_numpy(dtype=float)
            x = np.column_stack([np.ones(len(x)), x])
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            fitted = x @ beta
            ss_res = float(np.sum((y - fitted) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
            if r2 >= 1.0 - 1e-12:
                vif = np.inf
            else:
                vif = 1.0 / max(1.0 - r2, 1e-12)
            status = "HIGH" if vif >= 5 else "OK"
        rows.append(
            {
                "Scenario": scenario,
                "Variable": variable,
                "VIF": vif,
                "N_Complete": len(data),
                "Status": status,
            }
        )
    return pd.DataFrame(rows)


def correlation_long(
    df: pd.DataFrame,
    variables: list[str],
    scenario: str,
) -> pd.DataFrame:
    data = df[variables].apply(pd.to_numeric, errors="coerce")
    rows = []
    for i, var1 in enumerate(variables):
        for var2 in variables[i + 1 :]:
            pair = data[[var1, var2]].dropna()
            variable_is_constant = (
                len(pair) < 3
                or pair[var1].nunique(dropna=True) < 2
                or pair[var2].nunique(dropna=True) < 2
            )
            rows.append(
                {
                    "Scenario": scenario,
                    "Variable_1": var1,
                    "Variable_2": var2,
                    "N_Pairs": len(pair),
                    "Pearson_r": (
                        pair[var1].corr(pair[var2], method="pearson")
                        if not variable_is_constant
                        else np.nan
                    ),
                    "Spearman_rho": (
                        pair[var1].corr(pair[var2], method="spearman")
                        if not variable_is_constant
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def dataframe_to_excel(
    tables: OrderedDict[str, pd.DataFrame],
    output_path: Path,
) -> bool:
    """输出便于人工检查的摘要工作簿；大主表仍以CSV为正式机器输入。"""
    engine = None
    if importlib.util.find_spec("xlsxwriter") is not None:
        engine = "xlsxwriter"
    elif importlib.util.find_spec("openpyxl") is not None:
        engine = "openpyxl"
    if engine is None:
        print("警告：未检测到xlsxwriter/openpyxl，跳过Excel摘要工作簿。")
        return False

    try:
        with pd.ExcelWriter(output_path, engine=engine) as writer:
            for sheet_name, table in tables.items():
                safe_sheet = sheet_name[:31]
                table.to_excel(writer, sheet_name=safe_sheet, index=False)

                if engine == "xlsxwriter":
                    workbook = writer.book
                    worksheet = writer.sheets[safe_sheet]
                    header_format = workbook.add_format(
                        {
                            "bold": True,
                            "font_color": "#FFFFFF",
                            "bg_color": "#1F4E78",
                            "align": "center",
                            "valign": "vcenter",
                            "border": 0,
                        }
                    )
                    for col_idx, column in enumerate(table.columns):
                        worksheet.write(0, col_idx, column, header_format)
                        sample_values = table[column].astype("string").head(300)
                        max_len = max(
                            [len(str(column))]
                            + [len(str(v)) for v in sample_values if pd.notna(v)]
                        )
                        worksheet.set_column(col_idx, col_idx, min(max(max_len + 2, 10), 32))
                    worksheet.freeze_panes(1, 1)
                    worksheet.autofilter(0, 0, max(len(table), 1), max(len(table.columns) - 1, 0))
                else:
                    worksheet = writer.sheets[safe_sheet]
                    worksheet.freeze_panes = "B2"
                    worksheet.auto_filter.ref = worksheet.dimensions
                    from openpyxl.styles import Font, PatternFill

                    for cell in worksheet[1]:
                        cell.font = Font(bold=True, color="FFFFFF")
                        cell.fill = PatternFill("solid", fgColor="1F4E78")
                    for column_cells in worksheet.columns:
                        letter = column_cells[0].column_letter
                        max_len = max(
                            len(str(cell.value)) if cell.value is not None else 0
                            for cell in list(column_cells)[:301]
                        )
                        worksheet.column_dimensions[letter].width = min(max(max_len + 2, 10), 32)
        return True
    except Exception as exc:
        print(f"警告：Excel摘要工作簿生成失败，不影响CSV主表：{type(exc).__name__}: {exc}")
        if output_path.exists():
            output_path.unlink()
        return False


# ============================================================
# 2. 读取与版本检查
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

events = read_csv(EVENT_ENV_CSV)
biomass_matrix = read_csv(BIOMASS_MATRIX_CSV)
density_matrix = read_csv(DENSITY_MATRIX_CSV)
presence_matrix = read_csv(PRESENCE_MATRIX_CSV)
taxonomy = read_csv(TAXONOMY_CSV)

required_event_columns = [
    "EventID",
    "EventID_Raw",
    "Year",
    "Month",
    "Station_Raw",
    "Station_Canonical",
    "LocationID",
    "Lon",
    "Lat",
    "SurveyStatus",
    "UseInCommunityAnalysis",
    "Total_Biomass_kg_km2",
    "Total_Density_1000ind_km2",
    "Species_Richness",
    "SST",
    "SSS",
    "Chla_0125",
    "log1p_Chla_0125",
    "Chla_0125_ValidPixels",
    "Chla_0125_ValidFraction",
    "Chla_0125_MatchStatus",
    "Depth_m",
    "EnvironmentMatchOK",
    "EnvGridID_0125",
]
require_columns(events, required_event_columns, "第二步事件—环境主表")
require_columns(taxonomy, ["Species_Code", "Species_CN", "Species_Latin"], "分类字典")

for label, matrix in (
    ("生物量矩阵", biomass_matrix),
    ("尾数密度矩阵", density_matrix),
    ("出现矩阵", presence_matrix),
):
    require_columns(matrix, ["EventID"], label)
    if not matrix["EventID"].is_unique:
        raise ValueError(f"{label}的EventID不唯一。")

if not events["EventID"].is_unique:
    raise ValueError("第二步事件—环境主表的EventID不唯一。")
if events["EventID"].isna().any():
    raise ValueError("第二步事件—环境主表存在缺失EventID。")

for flag in [
    "UseInCommunityAnalysis",
    "EnvironmentMatchOK",
    "UseInEnvironmentalCommunityModel",
    "UseInEnvironmentalBiomassModel",
]:
    if flag in events.columns:
        events[flag] = to_bool(events[flag])

events["Year"] = pd.to_numeric(events["Year"], errors="raise").astype(int)
events["Month"] = pd.to_numeric(events["Month"], errors="raise").astype(int)

observed_years = set(events["Year"])
observed_months = set(events["Month"])
if observed_years - EXPECTED_YEARS:
    raise ValueError(f"存在2014—2024之外年份：{sorted(observed_years - EXPECTED_YEARS)}")
if observed_months - EXPECTED_MONTHS:
    raise ValueError(f"存在4月、11月之外月份：{sorted(observed_months - EXPECTED_MONTHS)}")
if STRICT_EXPECTED_COUNTS and observed_years != EXPECTED_YEARS:
    raise ValueError(
        f"年份集合不是完整的2014—2024：{sorted(observed_years)}"
    )
if STRICT_EXPECTED_COUNTS and observed_months != EXPECTED_MONTHS:
    raise ValueError(f"月份集合不是完整的4月和11月：{sorted(observed_months)}")

presence_species = species_columns(presence_matrix)
biomass_species = species_columns(biomass_matrix)
density_species = species_columns(density_matrix)

if not presence_species:
    raise ValueError("出现矩阵中未识别到Species_XX物种列。")
if presence_species != biomass_species or presence_species != density_species:
    raise ValueError("出现、生物量和尾数密度矩阵的物种列不完全一致。")
if set(taxonomy["Species_Code"].astype(str)) != set(presence_species):
    raise ValueError("分类字典中的Species_Code与三个物种矩阵不一致。")

event_id_set = set(events["EventID"].astype(str))
for label, matrix in (
    ("生物量矩阵", biomass_matrix),
    ("尾数密度矩阵", density_matrix),
    ("出现矩阵", presence_matrix),
):
    if set(matrix["EventID"].astype(str)) != event_id_set:
        missing = event_id_set - set(matrix["EventID"].astype(str))
        extra = set(matrix["EventID"].astype(str)) - event_id_set
        raise ValueError(
            f"{label}与第二步主表EventID集合不一致；"
            f"缺少{len(missing)}，多出{len(extra)}。"
        )


# ============================================================
# 3. 固定事件顺序、形成分析标记
# ============================================================

events = events.sort_values(
    ["Year", "Month", "LocationID", "EventID"], kind="stable"
).reset_index(drop=True)
event_order = events["EventID"].astype(str).tolist()

presence_indexed = presence_matrix.set_index("EventID").loc[event_order]
biomass_indexed = biomass_matrix.set_index("EventID").loc[event_order]
density_indexed = density_matrix.set_index("EventID").loc[event_order]

status_surveyed = events["SurveyStatus"].isin(SURVEYED_STATUSES)
flag_surveyed = events["UseInCommunityAnalysis"].fillna(False)
if not status_surveyed.equals(flag_surveyed):
    mismatch = int((status_surveyed != flag_surveyed).sum())
    raise AssertionError(
        f"调查状态与UseInCommunityAnalysis不一致，异常事件数：{mismatch}"
    )

events["Surveyed_Event"] = status_surveyed
events["Use_Descriptive_Community"] = status_surveyed
events["Use_Environment_Strict"] = status_surveyed & events["EnvironmentMatchOK"]

if "UseInEnvironmentalCommunityModel" in events.columns:
    if not events["Use_Environment_Strict"].equals(
        events["UseInEnvironmentalCommunityModel"]
    ):
        raise AssertionError(
            "重新计算的严格环境样本与第二步UseInEnvironmentalCommunityModel不一致。"
        )

# 不含Chla敏感性：仍严格要求SST、SSS和水深有可靠值。
no_chla_ok = events[["SST", "SSS", "Depth_m"]].notna().all(axis=1)
for variable in ["SST", "SSS", "Depth_m"]:
    status_col = f"{variable}_MatchStatus"
    plausible_col = f"{variable}_Plausible"
    if status_col in events.columns:
        no_chla_ok &= events[status_col].eq("OK")
    if plausible_col in events.columns:
        no_chla_ok &= to_bool(events[plausible_col])

events["EnvironmentNoChlaOK"] = no_chla_ok
events["Use_NoChla_Sensitivity"] = status_surveyed & no_chla_ok
events["Use_ChlaMin3_Sensitivity"] = (
    events["Use_Environment_Strict"]
    & pd.to_numeric(events["Chla_0125_ValidPixels"], errors="coerce").ge(
        CHLA_MIN_PIXELS_SENSITIVITY
    )
)

events["Year_rel"] = events["Year"] - min(EXPECTED_YEARS)
events["Year_Factor"] = "Y" + events["Year"].astype(str)
events["Month_Factor"] = events["Month"].map({4: "M04", 11: "M11"})
events["Month_Binary_November"] = events["Month"].eq(11).astype(int)
events["Period"] = events["Month"].map({4: "April", 11: "November"})
events["Season"] = events["Month"].map({4: "Spring", 11: "Autumn"})
events["YearMonth"] = (
    events["Year"].astype(str) + "-" + events["Month"].astype(str).str.zfill(2)
)
events["SampleKey"] = events["EventID"]
events["SpatialMetadataComplete"] = events[["Lon", "Lat"]].notna().all(axis=1)

valid_pixels = pd.to_numeric(events["Chla_0125_ValidPixels"], errors="coerce")
chla_status = events["Chla_0125_MatchStatus"].astype("string")
events["Chla_Quality_Class"] = np.select(
    [
        chla_status.eq("OK") & valid_pixels.ge(CHLA_MIN_PIXELS_SENSITIVITY),
        chla_status.eq("OK_ADJACENT") & valid_pixels.ge(CHLA_MIN_PIXELS_SENSITIVITY),
        chla_status.eq("OK") & valid_pixels.lt(CHLA_MIN_PIXELS_SENSITIVITY),
        chla_status.eq("OK_ADJACENT") & valid_pixels.lt(CHLA_MIN_PIXELS_SENSITIVITY),
    ],
    [
        "DIRECT_GE3",
        "ADJACENT_GE3",
        "DIRECT_LT3",
        "ADJACENT_LT3",
    ],
    default="UNAVAILABLE",
)


# ============================================================
# 4. 核查NA/0、总量、丰富度与矩阵一致性
# ============================================================

surveyed_ids = events.loc[events["Surveyed_Event"], "EventID"].astype(str).tolist()
not_surveyed_ids = events.loc[~events["Surveyed_Event"], "EventID"].astype(str).tolist()
strict_ids = events.loc[events["Use_Environment_Strict"], "EventID"].astype(str).tolist()
no_chla_ids = events.loc[events["Use_NoChla_Sensitivity"], "EventID"].astype(str).tolist()
chla_min3_ids = events.loc[events["Use_ChlaMin3_Sensitivity"], "EventID"].astype(str).tolist()

presence_surveyed = presence_indexed.loc[surveyed_ids, presence_species].apply(
    pd.to_numeric, errors="coerce"
)
biomass_surveyed = biomass_indexed.loc[surveyed_ids, presence_species].apply(
    pd.to_numeric, errors="coerce"
)
density_surveyed = density_indexed.loc[surveyed_ids, presence_species].apply(
    pd.to_numeric, errors="coerce"
)

if presence_surveyed.isna().any().any():
    raise AssertionError("已调查事件的出现矩阵仍有NA。")
if biomass_surveyed.isna().any().any():
    raise AssertionError("已调查事件的生物量矩阵仍有NA。")
if density_surveyed.isna().any().any():
    raise AssertionError("已调查事件的尾数密度矩阵仍有NA。")

presence_values = set(np.unique(presence_surveyed.to_numpy(dtype=float)))
if not presence_values.issubset({0.0, 1.0}):
    raise AssertionError(f"出现矩阵含0/1以外数值：{sorted(presence_values)}")
if (biomass_surveyed < 0).any().any():
    raise AssertionError("已调查事件的生物量矩阵含负值。")
if (density_surveyed < 0).any().any():
    raise AssertionError("已调查事件的尾数密度矩阵含负值。")

for label, indexed in (
    ("出现矩阵", presence_indexed),
    ("生物量矩阵", biomass_indexed),
    ("尾数密度矩阵", density_indexed),
):
    if not_surveyed_ids and indexed.loc[not_surveyed_ids, presence_species].notna().any().any():
        raise AssertionError(f"未调查事件在{label}中存在非NA值，不能把未调查误作0。")

surveyed_event_rows = events.set_index("EventID").loc[surveyed_ids]
assert_close_series(
    presence_surveyed.sum(axis=1),
    surveyed_event_rows["Species_Richness"],
    "出现矩阵物种数与Species_Richness",
)
assert_close_series(
    biomass_surveyed.sum(axis=1),
    surveyed_event_rows["Total_Biomass_kg_km2"],
    "物种生物量加总与事件总生物量",
    atol=1e-6,
)
assert_close_series(
    density_surveyed.sum(axis=1),
    surveyed_event_rows["Total_Density_1000ind_km2"],
    "物种尾数密度加总与事件总尾数密度",
    atol=1e-6,
)

zero_ids = events.loc[events["SurveyStatus"].eq(STATUS_ZERO), "EventID"].astype(str)
if len(zero_ids):
    if (presence_indexed.loc[zero_ids, presence_species].fillna(-1) != 0).any().any():
        raise AssertionError("SURVEYED_ZERO事件的出现矩阵并非全0。")
    if (biomass_indexed.loc[zero_ids, presence_species].fillna(-1) != 0).any().any():
        raise AssertionError("SURVEYED_ZERO事件的生物量矩阵并非全0。")
    if (density_indexed.loc[zero_ids, presence_species].fillna(-1) != 0).any().any():
        raise AssertionError("SURVEYED_ZERO事件的尾数密度矩阵并非全0。")


# ============================================================
# 5. 物种出现频次与筛选规则
# ============================================================

strict_presence = presence_indexed.loc[strict_ids, presence_species].apply(
    pd.to_numeric, errors="coerce"
)
strict_biomass = biomass_indexed.loc[strict_ids, presence_species].apply(
    pd.to_numeric, errors="coerce"
)

april_strict_ids = events.loc[
    events["Use_Environment_Strict"] & events["Month"].eq(4), "EventID"
].astype(str).tolist()
november_strict_ids = events.loc[
    events["Use_Environment_Strict"] & events["Month"].eq(11), "EventID"
].astype(str).tolist()

april_presence = presence_indexed.loc[april_strict_ids, presence_species].apply(
    pd.to_numeric, errors="coerce"
)
november_presence = presence_indexed.loc[november_strict_ids, presence_species].apply(
    pd.to_numeric, errors="coerce"
)

taxonomy_ordered = (
    taxonomy.assign(Species_Code=taxonomy["Species_Code"].astype(str))
    .set_index("Species_Code")
    .loc[presence_species]
    .reset_index()
)

species_summary = taxonomy_ordered.copy()
species_summary["N_Occurrence_Surveyed"] = [
    int(presence_surveyed[c].sum()) for c in presence_species
]
species_summary["OccurrenceRate_Surveyed"] = [
    float(presence_surveyed[c].mean()) for c in presence_species
]
species_summary["N_Occurrence_Strict"] = [
    int(strict_presence[c].sum()) for c in presence_species
]
species_summary["OccurrenceRate_Strict"] = [
    float(strict_presence[c].mean()) for c in presence_species
]
species_summary["N_Occurrence_April_Strict"] = [
    int(april_presence[c].sum()) for c in presence_species
]
species_summary["N_Occurrence_November_Strict"] = [
    int(november_presence[c].sum()) for c in presence_species
]
species_summary["Total_Biomass_Surveyed_kg_km2"] = [
    float(biomass_surveyed[c].sum()) for c in presence_species
]
species_summary["Total_Biomass_Strict_kg_km2"] = [
    float(strict_biomass[c].sum()) for c in presence_species
]

species_summary["Keep_Community_All25"] = True
species_summary["Keep_Cooccurrence_Main_GE20_Surveyed"] = (
    species_summary["N_Occurrence_Surveyed"] >= MAIN_OCCURRENCE_THRESHOLD
)
species_summary["Keep_JSDM_Main_GE20_Strict"] = (
    species_summary["N_Occurrence_Strict"] >= MAIN_OCCURRENCE_THRESHOLD
)
species_summary["Keep_JSDM_Sensitivity_GE10_Strict"] = (
    species_summary["N_Occurrence_Strict"] >= SENS_OCCURRENCE_THRESHOLD
)
species_summary["Keep_April_GE20_Strict"] = (
    species_summary["N_Occurrence_April_Strict"] >= MAIN_OCCURRENCE_THRESHOLD
)
species_summary["Keep_November_GE20_Strict"] = (
    species_summary["N_Occurrence_November_Strict"] >= MAIN_OCCURRENCE_THRESHOLD
)
species_summary["Keep_BothSeasons_GE20_Strict"] = (
    species_summary["Keep_April_GE20_Strict"]
    & species_summary["Keep_November_GE20_Strict"]
)
species_summary["Rarity_Class_Strict"] = np.select(
    [
        species_summary["N_Occurrence_Strict"] >= MAIN_OCCURRENCE_THRESHOLD,
        species_summary["N_Occurrence_Strict"] >= SENS_OCCURRENCE_THRESHOLD,
    ],
    ["MAIN_GE20", "SENSITIVITY_10_TO_19"],
    default="DESCRIPTIVE_LT10",
)

main_species = species_summary.loc[
    species_summary["Keep_JSDM_Main_GE20_Strict"], "Species_Code"
].tolist()
sens_species = species_summary.loc[
    species_summary["Keep_JSDM_Sensitivity_GE10_Strict"], "Species_Code"
].tolist()
april_species = species_summary.loc[
    species_summary["Keep_April_GE20_Strict"], "Species_Code"
].tolist()
november_species = species_summary.loc[
    species_summary["Keep_November_GE20_Strict"], "Species_Code"
].tolist()

if not main_species:
    raise ValueError("严格环境样本中没有物种达到主分析出现次数阈值。")
if not sens_species:
    raise ValueError("严格环境样本中没有物种达到敏感性分析出现次数阈值。")
if not april_species or not november_species:
    raise ValueError("4月或11月没有物种达到季节模型出现次数阈值。")


# ============================================================
# 6. 构建Master v1.0宽表和长表
# ============================================================

presence_prefixed = presence_indexed[presence_species].rename(
    columns={c: f"PA_{c}" for c in presence_species}
)
biomass_prefixed = biomass_indexed[presence_species].rename(
    columns={c: f"B_{c}_kg_km2" for c in presence_species}
)
density_prefixed = density_indexed[presence_species].rename(
    columns={c: f"D_{c}_1000ind_km2" for c in presence_species}
)

master_all = (
    events.set_index("EventID")
    .join(presence_prefixed, how="left", validate="one_to_one")
    .join(biomass_prefixed, how="left", validate="one_to_one")
    .join(density_prefixed, how="left", validate="one_to_one")
    .reset_index()
)

master_surveyed = master_all.loc[master_all["Surveyed_Event"]].copy()
master_strict = master_all.loc[master_all["Use_Environment_Strict"]].copy()

long_metadata_cols = [
    "EventID",
    "EventID_Raw",
    "SampleKey",
    "Year",
    "Year_rel",
    "Year_Factor",
    "Month",
    "Month_Factor",
    "Period",
    "Season",
    "Station_Raw",
    "Station_Canonical",
    "LocationID",
    "Lon",
    "Lat",
    "EnvGridID_0125",
    "SurveyStatus",
    "Cephalopod_Presence",
    "Total_Biomass_kg_km2",
    "Total_Density_1000ind_km2",
    "Species_Richness",
    "SST",
    "SSS",
    "Chla_0125",
    "log1p_Chla_0125",
    "Chla_0125_ValidPixels",
    "Chla_0125_ValidFraction",
    "Chla_0125_MatchStatus",
    "Depth_m",
    "Use_Environment_Strict",
    "Use_NoChla_Sensitivity",
    "Use_ChlaMin3_Sensitivity",
]
long_metadata_cols = [c for c in long_metadata_cols if c in master_surveyed.columns]

presence_long = (
    presence_surveyed.rename_axis(index="EventID", columns="Species_Code")
    .reset_index()
    .melt(
        id_vars="EventID",
        value_vars=presence_species,
        var_name="Species_Code",
        value_name="Presence",
    )
)
biomass_long = (
    biomass_surveyed.rename_axis(index="EventID", columns="Species_Code")
    .reset_index()
    .melt(
        id_vars="EventID",
        value_vars=presence_species,
        var_name="Species_Code",
        value_name="Biomass_kg_km2",
    )
)
density_long = (
    density_surveyed.rename_axis(index="EventID", columns="Species_Code")
    .reset_index()
    .melt(
        id_vars="EventID",
        value_vars=presence_species,
        var_name="Species_Code",
        value_name="Density_1000ind_km2",
    )
)

response_long = (
    presence_long.merge(
        biomass_long,
        on=["EventID", "Species_Code"],
        how="inner",
        validate="one_to_one",
    )
    .merge(
        density_long,
        on=["EventID", "Species_Code"],
        how="inner",
        validate="one_to_one",
    )
)

species_rank = {code: idx + 1 for idx, code in enumerate(presence_species)}
master_long_surveyed = (
    master_surveyed[long_metadata_cols]
    .merge(response_long, on="EventID", how="inner", validate="one_to_many")
    .merge(
        taxonomy_ordered[["Species_Code", "Species_CN", "Species_Latin"]],
        on="Species_Code",
        how="left",
        validate="many_to_one",
    )
)
master_long_surveyed["Species_Order"] = master_long_surveyed["Species_Code"].map(
    species_rank
)
event_rank = {event_id: idx for idx, event_id in enumerate(surveyed_ids)}
master_long_surveyed["_EventOrder"] = master_long_surveyed["EventID"].map(event_rank)
master_long_surveyed = (
    master_long_surveyed.sort_values(["_EventOrder", "Species_Order"], kind="stable")
    .drop(columns="_EventOrder")
    .reset_index(drop=True)
)
expected_long_rows = len(surveyed_ids) * len(presence_species)
if len(master_long_surveyed) != expected_long_rows:
    raise AssertionError(
        "调查事件×物种长表行数错误："
        f"应为{expected_long_rows}，实际为{len(master_long_surveyed)}。"
    )
if master_long_surveyed[["EventID", "Species_Code"]].duplicated().any():
    raise AssertionError("调查事件×物种长表存在重复EventID-Species_Code。")


# ============================================================
# 7. 构建各分析分支的X/Y矩阵
# ============================================================

base_x_columns = [
    "EventID",
    "Year",
    "Year_rel",
    "Year_Factor",
    "Month",
    "Month_Factor",
    "Month_Binary_November",
    "Period",
    "Season",
    "LocationID",
    "Lon",
    "Lat",
    "EnvGridID_0125",
    "Depth_m",
    "SST",
    "SSS",
    "log1p_Chla_0125",
]

x_strict = master_all.loc[
    master_all["Use_Environment_Strict"], base_x_columns
].copy()

no_chla_x_columns = [c for c in base_x_columns if c != "log1p_Chla_0125"]
x_no_chla = master_all.loc[
    master_all["Use_NoChla_Sensitivity"], no_chla_x_columns
].copy()

x_chla_min3 = master_all.loc[
    master_all["Use_ChlaMin3_Sensitivity"], base_x_columns
].copy()

season_x_columns = [
    "EventID",
    "Year",
    "Year_rel",
    "Year_Factor",
    "LocationID",
    "Lon",
    "Lat",
    "EnvGridID_0125",
    "Depth_m",
    "SST",
    "SSS",
    "log1p_Chla_0125",
]
x_april = master_all.loc[
    master_all["Use_Environment_Strict"] & master_all["Month"].eq(4),
    season_x_columns,
].copy()
x_november = master_all.loc[
    master_all["Use_Environment_Strict"] & master_all["Month"].eq(11),
    season_x_columns,
].copy()

y_presence_all25 = make_y(presence_indexed, surveyed_ids, presence_species)
y_biomass_all25 = make_y(biomass_indexed, surveyed_ids, presence_species)
y_density_all25 = make_y(density_indexed, surveyed_ids, presence_species)

y_presence_main_strict = make_y(presence_indexed, strict_ids, main_species)
y_biomass_main_strict = make_y(biomass_indexed, strict_ids, main_species)
y_presence_sens_strict = make_y(presence_indexed, strict_ids, sens_species)
y_biomass_sens_strict = make_y(biomass_indexed, strict_ids, sens_species)

y_presence_main_no_chla = make_y(presence_indexed, no_chla_ids, main_species)
y_biomass_main_no_chla = make_y(biomass_indexed, no_chla_ids, main_species)
y_presence_main_chla_min3 = make_y(presence_indexed, chla_min3_ids, main_species)
y_biomass_main_chla_min3 = make_y(biomass_indexed, chla_min3_ids, main_species)

y_presence_april = make_y(presence_indexed, april_strict_ids, april_species)
y_biomass_april = make_y(biomass_indexed, april_strict_ids, april_species)
y_presence_november = make_y(
    presence_indexed, november_strict_ids, november_species
)
y_biomass_november = make_y(
    biomass_indexed, november_strict_ids, november_species
)

alignment_pairs = [
    ("Strict main presence", x_strict, y_presence_main_strict),
    ("Strict main biomass", x_strict, y_biomass_main_strict),
    ("Strict sensitivity presence", x_strict, y_presence_sens_strict),
    ("Strict sensitivity biomass", x_strict, y_biomass_sens_strict),
    ("No-Chla main presence", x_no_chla, y_presence_main_no_chla),
    ("No-Chla main biomass", x_no_chla, y_biomass_main_no_chla),
    ("ChlaMin3 main presence", x_chla_min3, y_presence_main_chla_min3),
    ("ChlaMin3 main biomass", x_chla_min3, y_biomass_main_chla_min3),
    ("April main presence", x_april, y_presence_april),
    ("April main biomass", x_april, y_biomass_april),
    ("November main presence", x_november, y_presence_november),
    ("November main biomass", x_november, y_biomass_november),
]

alignment_rows = []
for label, x_table, y_table in alignment_pairs:
    same_order = x_table["EventID"].tolist() == y_table["EventID"].tolist()
    y_values = y_table.drop(columns="EventID")
    alignment_rows.append(
        {
            "Check": label,
            "X_Rows": len(x_table),
            "Y_Rows": len(y_table),
            "Y_Species": y_values.shape[1],
            "EventID_Order_Identical": same_order,
            "Y_Missing_Values": int(y_values.isna().sum().sum()),
            "Pass": bool(
                same_order
                and len(x_table) == len(y_table)
                and not y_values.isna().any().any()
            ),
        }
    )
matrix_alignment_qc = pd.DataFrame(alignment_rows)
if not matrix_alignment_qc["Pass"].all():
    raise AssertionError("至少一组X/Y矩阵未通过行序或缺失值检查。")


# ============================================================
# 8. 样本流程、排除事件与共线性诊断
# ============================================================

flow_source = master_all.assign(
    _Positive=master_all["SurveyStatus"].eq(STATUS_POS).astype(int),
    _Zero=master_all["SurveyStatus"].eq(STATUS_ZERO).astype(int),
    _NotSurveyed=master_all["SurveyStatus"].eq(STATUS_NOT).astype(int),
    _Surveyed=master_all["Surveyed_Event"].astype(int),
    _Strict=master_all["Use_Environment_Strict"].astype(int),
    _NoChla=master_all["Use_NoChla_Sensitivity"].astype(int),
    _ChlaMin3=master_all["Use_ChlaMin3_Sensitivity"].astype(int),
)
sample_flow = (
    flow_source.groupby(["Year", "Month", "Period"], sort=True)
    .agg(
        N_Designed_Events=("EventID", "size"),
        N_Surveyed=("_Surveyed", "sum"),
        N_Surveyed_Positive=("_Positive", "sum"),
        N_Surveyed_Zero=("_Zero", "sum"),
        N_NotSurveyed=("_NotSurveyed", "sum"),
        N_Environment_Strict=("_Strict", "sum"),
        N_NoChla_Sensitivity=("_NoChla", "sum"),
        N_ChlaMin3_Sensitivity=("_ChlaMin3", "sum"),
    )
    .reset_index()
)
sample_flow["N_Excluded_From_Strict"] = (
    sample_flow["N_Surveyed"] - sample_flow["N_Environment_Strict"]
)
sample_flow["Strict_Retention_Rate"] = safe_rate(
    sample_flow["N_Environment_Strict"], sample_flow["N_Surveyed"]
)
sample_flow["NoChla_Retention_Rate"] = safe_rate(
    sample_flow["N_NoChla_Sensitivity"], sample_flow["N_Surveyed"]
)
sample_flow["ChlaMin3_Retention_Rate"] = safe_rate(
    sample_flow["N_ChlaMin3_Sensitivity"], sample_flow["N_Surveyed"]
)

excluded_columns = [
    "EventID",
    "EventID_Raw",
    "Year",
    "Month",
    "Period",
    "Station_Raw",
    "Station_Canonical",
    "LocationID",
    "Lon",
    "Lat",
    "SurveyStatus",
    "SST",
    "SSS",
    "Chla_0125",
    "Chla_0125_Distance_km",
    "Chla_0125_ValidPixels",
    "Chla_0125_ValidFraction",
    "Chla_0125_MatchStatus",
    "Depth_m",
    "EnvironmentComplete",
    "EnvironmentMatchOK",
]
excluded_columns = [c for c in excluded_columns if c in master_all.columns]
environment_excluded = master_all.loc[
    master_all["Surveyed_Event"] & ~master_all["Use_Environment_Strict"],
    excluded_columns,
].copy()

strict_for_diag = master_all.loc[master_all["Use_Environment_Strict"]].copy()
april_for_diag = strict_for_diag.loc[strict_for_diag["Month"].eq(4)].copy()
november_for_diag = strict_for_diag.loc[strict_for_diag["Month"].eq(11)].copy()
no_chla_for_diag = master_all.loc[master_all["Use_NoChla_Sensitivity"]].copy()

diagnostic_scenarios = [
    (
        "Strict_all_months",
        strict_for_diag,
        [
            "Year_rel",
            "Month_Binary_November",
            "Depth_m",
            "SST",
            "SSS",
            "log1p_Chla_0125",
        ],
    ),
    (
        "Strict_April",
        april_for_diag,
        ["Year_rel", "Depth_m", "SST", "SSS", "log1p_Chla_0125"],
    ),
    (
        "Strict_November",
        november_for_diag,
        ["Year_rel", "Depth_m", "SST", "SSS", "log1p_Chla_0125"],
    ),
    (
        "NoChla_all_months",
        no_chla_for_diag,
        ["Year_rel", "Month_Binary_November", "Depth_m", "SST", "SSS"],
    ),
]

vif_table = pd.concat(
    [calculate_vif(df, variables, scenario) for scenario, df, variables in diagnostic_scenarios],
    ignore_index=True,
)
correlation_table = pd.concat(
    [
        correlation_long(df, variables, scenario)
        for scenario, df, variables in diagnostic_scenarios
    ],
    ignore_index=True,
)


# ============================================================
# 9. 冻结计数检查与摘要
# ============================================================

observed_counts = {
    "Total events": len(master_all),
    "Surveyed events": len(master_surveyed),
    "Strict environmental events": len(master_strict),
    "Species count": len(presence_species),
    "Unique LocationID": master_all["LocationID"].nunique(),
    "Main species GE20": len(main_species),
    "Sensitivity species GE10": len(sens_species),
}
expected_counts = {
    "Total events": EXPECTED_TOTAL_EVENTS,
    "Surveyed events": EXPECTED_SURVEYED_EVENTS,
    "Strict environmental events": EXPECTED_STRICT_EVENTS,
    "Species count": EXPECTED_SPECIES_COUNT,
    "Unique LocationID": EXPECTED_LOCATION_COUNT,
    "Main species GE20": EXPECTED_MAIN_SPECIES_COUNT,
    "Sensitivity species GE10": EXPECTED_SENS_SPECIES_COUNT,
}

count_check_rows = []
for check, observed in observed_counts.items():
    expected = expected_counts[check]
    count_check_rows.append(
        {
            "Check": check,
            "Expected": expected,
            "Observed": observed,
            "Pass": observed == expected,
        }
    )
count_check = pd.DataFrame(count_check_rows)

if STRICT_EXPECTED_COUNTS and not count_check["Pass"].all():
    details = count_check.loc[~count_check["Pass"]].to_string(index=False)
    raise AssertionError(
        "冻结计数检查未通过，可能读取了旧版输出或数据发生变化：\n"
        + details
        + "\n若这是经过确认的正式数据更新，可设置"
        "CEPHALOPOD_STRICT_EXPECTED_COUNTS=0后运行，并先核查变化原因。"
    )

matrix_alignment_qc = pd.concat(
    [
        pd.DataFrame(
            {
                "Check": "Frozen count: " + count_check["Check"],
                "X_Rows": count_check["Observed"],
                "Y_Rows": count_check["Expected"],
                "Y_Species": pd.NA,
                "EventID_Order_Identical": pd.NA,
                "Y_Missing_Values": pd.NA,
                "Pass": count_check["Pass"],
            }
        ),
        matrix_alignment_qc,
    ],
    ignore_index=True,
)

summary_rows = [
    ("Master version", MASTER_VERSION),
    ("Total designed events", len(master_all)),
    ("Surveyed events", len(master_surveyed)),
    ("Surveyed positive events", int(master_all["SurveyStatus"].eq(STATUS_POS).sum())),
    ("Surveyed zero events", int(master_all["SurveyStatus"].eq(STATUS_ZERO).sum())),
    ("Not surveyed events", int(master_all["SurveyStatus"].eq(STATUS_NOT).sum())),
    ("Strict environmental events", len(master_strict)),
    ("Surveyed events excluded from strict", len(environment_excluded)),
    ("No-Chla sensitivity events", len(no_chla_ids)),
    (f"Chla >= {CHLA_MIN_PIXELS_SENSITIVITY} source-pixel sensitivity events", len(chla_min3_ids)),
    ("All descriptive species", len(presence_species)),
    (f"Main species occurrence >= {MAIN_OCCURRENCE_THRESHOLD}", len(main_species)),
    (f"Sensitivity species occurrence >= {SENS_OCCURRENCE_THRESHOLD}", len(sens_species)),
    ("April strict events", len(april_strict_ids)),
    (f"April species occurrence >= {MAIN_OCCURRENCE_THRESHOLD}", len(april_species)),
    ("November strict events", len(november_strict_ids)),
    (f"November species occurrence >= {MAIN_OCCURRENCE_THRESHOLD}", len(november_species)),
    ("Unique LocationID", master_all["LocationID"].nunique()),
    ("Fisheries observations aggregated to 0.125 degree", False),
    ("Environment target resolution degree", 0.125),
    ("Global Z-score performed", False),
    ("ENSO included", False),
    ("Strict expected count enforcement", STRICT_EXPECTED_COUNTS),
]
step3_summary = pd.DataFrame(summary_rows, columns=["Metric", "Value"])


# ============================================================
# 10. 分类分析字典与变量字典
# ============================================================

taxonomy_analysis = taxonomy_ordered.merge(
    species_summary[
        [
            "Species_Code",
            "N_Occurrence_Surveyed",
            "N_Occurrence_Strict",
            "Keep_Community_All25",
            "Keep_JSDM_Main_GE20_Strict",
            "Keep_JSDM_Sensitivity_GE10_Strict",
            "Keep_April_GE20_Strict",
            "Keep_November_GE20_Strict",
            "Keep_BothSeasons_GE20_Strict",
            "Rarity_Class_Strict",
        ]
    ],
    on="Species_Code",
    how="left",
    validate="one_to_one",
)

variable_notes = {
    "EventID": ("Identifier", "年+月+标准LocationID形成的唯一事件主键", "text"),
    "EventID_Raw": ("Identifier", "使用原始站位编号形成的事件编号", "text"),
    "SampleKey": ("Identifier", "分析样本主键，等于EventID", "text"),
    "Year": ("Time", "调查年份", "year"),
    "Year_rel": ("Predictor", "Year-2014；长期未观测年际效应代理，不是直接生态驱动", "year"),
    "Year_Factor": ("Time factor", "年份分类标签", "factor"),
    "Month": ("Time", "调查月份，仅4或11", "month"),
    "Month_Factor": ("Time factor", "月份分类标签M04/M11", "factor"),
    "Month_Binary_November": ("Diagnostic", "11月=1、4月=0，仅用于共线性诊断", "0/1"),
    "Period": ("Time factor", "April或November", "factor"),
    "Season": ("Time factor", "Spring或Autumn", "factor"),
    "Station_Raw": ("Identifier", "原始站位编号；2014年11月可为D系列", "text"),
    "Station_Canonical": ("Identifier", "按坐标映射后的A/B标准站位", "text"),
    "LocationID": ("Identifier", "统一位置编号；空间重复测量单元", "text"),
    "Lon": ("Spatial metadata", "真实调查经度；渔业数据未聚合到0.125度", "degree_east"),
    "Lat": ("Spatial metadata", "真实调查纬度；渔业数据未聚合到0.125度", "degree_north"),
    "EnvGridID_0125": ("Environment grid", "环境变量0.125度目标格网编号", "text"),
    "SurveyStatus": ("Sampling", "NOT_SURVEYED/SURVEYED_ZERO/SURVEYED_POSITIVE", "factor"),
    "Surveyed_Event": ("Analysis flag", "该年-月-位置是否实际完成调查", "boolean"),
    "Use_Descriptive_Community": ("Analysis flag", "可用于IRI、组成、多样性、NMDS、PERMANOVA", "boolean"),
    "Use_Environment_Strict": ("Analysis flag", "已调查且SST/SSS/Chla/Depth均严格合格", "boolean"),
    "Use_NoChla_Sensitivity": ("Analysis flag", "已调查且SST/SSS/Depth合格，不要求Chla", "boolean"),
    "Use_ChlaMin3_Sensitivity": ("Analysis flag", "严格样本且Chla至少3个有效4 km源像元", "boolean"),
    "SST": ("Predictor", "0.125度格网海表温度", "degree_C"),
    "SSS": ("Predictor", "0.125度格网海表盐度", "practical_salinity"),
    "Chla_0125": ("Environment raw", "4 km Chla面积加权聚合至0.125度后的算术平均", "mg_m-3"),
    "log1p_Chla_0125": ("Predictor", "log(1+Chla_0125)，正式模型使用此变量", "log_mg_m-3"),
    "Depth_m": ("Predictor", "按真实站位坐标从高分辨率地形提取的正水深", "m"),
    "Total_Biomass_kg_km2": ("Response summary", "事件内25种头足类生物量合计", "kg_km-2"),
    "Total_Density_1000ind_km2": ("Response summary", "事件内25种尾数密度合计", "1000_ind_km-2"),
    "Species_Richness": ("Response summary", "事件内出现的头足类物种数", "count"),
}

dictionary_rows = []
species_name_lookup = taxonomy_ordered.set_index("Species_Code")
for column in master_all.columns:
    if column in variable_notes:
        role, definition, units = variable_notes[column]
    elif re.fullmatch(r"PA_Species_\d+", column):
        code = column.replace("PA_", "")
        role = "Species response"
        definition = f"{species_name_lookup.at[code, 'Species_CN']}出现(1)/未出现(0)；未调查为NA"
        units = "0/1"
    elif re.fullmatch(r"B_Species_\d+_kg_km2", column):
        code = re.search(r"Species_\d+", column).group(0)
        role = "Species response"
        definition = f"{species_name_lookup.at[code, 'Species_CN']}生物量；未调查为NA"
        units = "kg_km-2"
    elif re.fullmatch(r"D_Species_\d+_1000ind_km2", column):
        code = re.search(r"Species_\d+", column).group(0)
        role = "Species response"
        definition = f"{species_name_lookup.at[code, 'Species_CN']}尾数密度；未调查为NA"
        units = "1000_ind_km-2"
    else:
        role = "QC or metadata"
        definition = "沿用第一步或第二步字段；详见相应步骤代码"
        units = "as_recorded"
    dictionary_rows.append(
        {
            "Variable": column,
            "Role": role,
            "Definition": definition,
            "Units_or_Type": units,
            "Standardization_in_Step3": "No",
        }
    )
variable_dictionary = pd.DataFrame(dictionary_rows)


# ============================================================
# 11. 写出CSV、Excel摘要、README和ZIP
# ============================================================

output_tables: OrderedDict[str, pd.DataFrame] = OrderedDict(
    [
        ("00_step3_summary.csv", step3_summary),
        ("01_analysis_master_v1_all_events_wide.csv", master_all),
        ("02_analysis_master_v1_surveyed_all25.csv", master_surveyed),
        ("03_analysis_master_v1_environment_strict_all25.csv", master_strict),
        ("04_analysis_master_v1_long_surveyed_all25.csv", master_long_surveyed),
        ("05_species_filter_summary.csv", species_summary),
        ("06_taxonomy_analysis_dictionary.csv", taxonomy_analysis),
        ("07_Y_presence_all25_surveyed.csv", y_presence_all25),
        ("08_Y_biomass_all25_surveyed.csv", y_biomass_all25),
        ("09_Y_density_all25_surveyed.csv", y_density_all25),
        ("10_X_environment_strict_raw.csv", x_strict),
        ("11_Y_presence_main_GE20_strict.csv", y_presence_main_strict),
        ("12_Y_biomass_main_GE20_strict.csv", y_biomass_main_strict),
        ("13_Y_presence_sensitivity_GE10_strict.csv", y_presence_sens_strict),
        ("14_Y_biomass_sensitivity_GE10_strict.csv", y_biomass_sens_strict),
        ("15_X_environment_noChla_sensitivity_raw.csv", x_no_chla),
        ("16_Y_presence_main_GE20_noChla.csv", y_presence_main_no_chla),
        ("17_Y_biomass_main_GE20_noChla.csv", y_biomass_main_no_chla),
        ("18_X_environment_ChlaMin3_sensitivity_raw.csv", x_chla_min3),
        ("19_Y_presence_main_GE20_ChlaMin3.csv", y_presence_main_chla_min3),
        ("20_Y_biomass_main_GE20_ChlaMin3.csv", y_biomass_main_chla_min3),
        ("21_X_environment_april_strict_raw.csv", x_april),
        ("22_Y_presence_april_GE20.csv", y_presence_april),
        ("23_Y_biomass_april_GE20.csv", y_biomass_april),
        ("24_X_environment_november_strict_raw.csv", x_november),
        ("25_Y_presence_november_GE20.csv", y_presence_november),
        ("26_Y_biomass_november_GE20.csv", y_biomass_november),
        ("27_sample_flow_by_year_month.csv", sample_flow),
        ("28_environment_excluded_surveyed_events.csv", environment_excluded),
        ("29_matrix_alignment_qc.csv", matrix_alignment_qc),
        ("30_predictor_vif.csv", vif_table),
        ("31_predictor_correlations.csv", correlation_table),
        ("32_analysis_variable_dictionary.csv", variable_dictionary),
    ]
)

output_paths: list[Path] = []
for filename, table in output_tables.items():
    output_paths.append(write_csv(table, filename))

manifest_rows = []
purpose_map = {
    "01_analysis_master_v1_all_events_wide.csv": "Master v1.0唯一母表；保留未调查NA与全部QC字段",
    "02_analysis_master_v1_surveyed_all25.csv": "全部已调查事件；IRI、组成、多样性、NMDS、PERMANOVA",
    "03_analysis_master_v1_environment_strict_all25.csv": "含四个可靠环境变量的已调查事件",
    "04_analysis_master_v1_long_surveyed_all25.csv": "事件×物种长表；IRI、组成图、单物种分析",
    "10_X_environment_strict_raw.csv": "严格环境模型X；连续变量未全局标准化",
    "11_Y_presence_main_GE20_strict.csv": "JSDM主分析出现Y",
    "12_Y_biomass_main_GE20_strict.csv": "JSDM主分析生物量Y",
    "13_Y_presence_sensitivity_GE10_strict.csv": "稀有物种阈值敏感性出现Y",
    "14_Y_biomass_sensitivity_GE10_strict.csv": "稀有物种阈值敏感性生物量Y",
    "15_X_environment_noChla_sensitivity_raw.csv": "不含Chla敏感性X",
    "18_X_environment_ChlaMin3_sensitivity_raw.csv": "Chla至少3个源像元敏感性X",
    "21_X_environment_april_strict_raw.csv": "4月单独环境模型X",
    "24_X_environment_november_strict_raw.csv": "11月单独环境模型X",
}
for filename, table in output_tables.items():
    manifest_rows.append(
        {
            "File": filename,
            "Rows": len(table),
            "Columns": len(table.columns),
            "Purpose": purpose_map.get(filename, "辅助矩阵、字典或质量检查"),
        }
    )
output_manifest = pd.DataFrame(manifest_rows)
manifest_path = write_csv(output_manifest, "33_output_manifest.csv")
output_paths.append(manifest_path)

readme_text = f"""浙江近岸头足类第三步分析主表（{MASTER_VERSION}）
============================================================

正式母表
--------
01_analysis_master_v1_all_events_wide.csv
  - 一行一个EventID，保留全部设计事件；
  - 未调查事件的物种值仍为NA；
  - 含PA_/B_/D_三套25物种字段；
  - 后续不再回到原始Excel重复清洗。

分析样本
--------
- 全部事件：{len(master_all)}
- 已调查事件：{len(master_surveyed)}
- 严格环境事件：{len(master_strict)}
- 不含Chla敏感性事件：{len(no_chla_ids)}
- Chla有效源像元>={CHLA_MIN_PIXELS_SENSITIVITY}敏感性事件：{len(chla_min3_ids)}

物种规则
--------
- IRI、物种组成、多样性、NMDS、PERMANOVA：保留全部{len(presence_species)}种；
- JSDM主分析：严格样本中出现>={MAIN_OCCURRENCE_THRESHOLD}次，共{len(main_species)}种；
- 稀有物种敏感性：严格样本中出现>={SENS_OCCURRENCE_THRESHOLD}次，共{len(sens_species)}种；
- 具体物种及出现次数见05_species_filter_summary.csv。

环境模型规则
------------
- 正式Chla变量：log1p_Chla_0125；
- Chla_4km_Nearest只作质量对照；
- 4月与11月优先分别建模，避免Month与SST强相关；
- 本步骤没有进行全数据Z-score；标准化必须在每个交叉验证训练折内拟合，
  再应用到对应验证/测试折；
- 渔业生物数据没有聚合到0.125度，0.125度仅为环境匹配格网。

X/Y对齐
-------
所有配套X/Y矩阵均以EventID严格对齐，检查结果见29_matrix_alignment_qc.csv。
"""
readme_path = OUTPUT_DIR / "README_step3.txt"
readme_path.write_text(readme_text, encoding="utf-8")
output_paths.append(readme_path)

excel_tables: OrderedDict[str, pd.DataFrame] = OrderedDict(
    [
        ("Summary", step3_summary),
        ("Sample_flow", sample_flow),
        ("Species_filter", species_summary),
        ("Matrix_QC", matrix_alignment_qc),
        ("VIF", vif_table),
        ("Correlations", correlation_table),
        ("Excluded_events", environment_excluded),
        ("Variable_dictionary", variable_dictionary),
        ("Manifest", output_manifest),
    ]
)
excel_path = OUTPUT_DIR / "Step3_analysis_master.xlsx"
if dataframe_to_excel(excel_tables, excel_path):
    output_paths.append(excel_path)

zip_path = Path(str(OUTPUT_DIR) + ".zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in output_paths:
        archive.write(path, arcname=path.name)


# ============================================================
# 12. Kaggle控制台摘要
# ============================================================

print("=" * 72)
print("第三步完成：浙江近岸头足类分析主表 Master v1.0")
print("=" * 72)
print(f"全部事件：{len(master_all):,}")
print(f"已调查事件：{len(master_surveyed):,}")
print(f"严格环境匹配事件：{len(master_strict):,}")
print(f"不含Chla敏感性事件：{len(no_chla_ids):,}")
print(f"Chla>={CHLA_MIN_PIXELS_SENSITIVITY}源像元敏感性事件：{len(chla_min3_ids):,}")
print(f"描述性分析物种：{len(presence_species)}")
print(f"JSDM主分析物种（出现>={MAIN_OCCURRENCE_THRESHOLD}）：{len(main_species)}")
print(f"JSDM敏感性物种（出现>={SENS_OCCURRENCE_THRESHOLD}）：{len(sens_species)}")
print(f"4月严格样本：{len(april_strict_ids):,}；季节阈值物种：{len(april_species)}")
print(f"11月严格样本：{len(november_strict_ids):,}；季节阈值物种：{len(november_species)}")
print("全局Z-score：未执行（正确；后续应在交叉验证训练折内完成）")
print(f"输出目录：{OUTPUT_DIR}")
print(f"ZIP文件：{zip_path}")
print("请下载：cephalopod_step3_analysis_master.zip")