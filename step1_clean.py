# -*- coding: utf-8 -*-
"""
#适用输入：
    #/kaggle/input/private-fishery/fishery_long.csv

#主要输出：
    #01_species_long_clean.csv        完整的“调查事件 × 物种”长表
   # 02_event_master.csv              一行一个年-月-站位调查事件
    #03_biomass_matrix.csv            生物量矩阵
   # 04_density_matrix.csv            尾数密度矩阵
   # 05_presence_matrix.csv           出现/不出现矩阵
    #06_taxonomy_dictionary.csv       Species_01...物种代码映射
   # 07_observed_species_aggregated.csv  原始出现物种聚合表
   # 09_location_dictionary.csv       原始站位到统一位置编号的映射
   # qc_*.csv                         各类质量检查结果
    #Step1_cephalopod_outputs.xlsx    汇总工作簿
    #cephalopod_step1_cleaned.zip     全部结果压缩包

#重要规则：
#1. 未调查（NOT_SURVEYED）：所有物种数值保持 NA。
#2. 已调查但头足类为 0（SURVEYED_ZERO）：全部物种赋值 0。
#3. 已调查且有头足类（SURVEYED_POSITIVE）：记录物种保留数值，
   未记录物种赋值 0。
#4. 状态冲突或无法判断的调查事件不强行填 0，矩阵中保持 NA，
   并写入 QC 文件。
#5. 同一调查事件中的同种多行记录按数值求和，同时保留 SourceRowCount。
#6. 展示名按附件二更新，原标签保留 Taxon_raw；没有重新鉴定标本。
#7. 2014 年 11 月的 D 系列站位按相同经纬度映射到 A/B 系列标准站位；
   #保留 Station_Raw，不增加未设计的站位记录，也不改变 NA/0 规则。
"""

from __future__ import annotations

import os
import re
import shutil
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 0. 参数设置
# ============================================================

INPUT_CSV = Path(
    os.environ.get(
        "CEPHALOPOD_INPUT_CSV",
        "/kaggle/input/private-fishery/fishery_long.csv",
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        "CEPHALOPOD_OUTPUT_DIR",
        "/kaggle/working/cephalopod_step1_cleaned",
    )
)

EXPECTED_YEARS = set(range(2014, 2025))
EXPECTED_MONTHS = {4, 11}
EXPECTED_SPECIES_COUNT = 25
EXPECTED_LOCATION_COUNT = int(os.environ.get("CEPHALOPOD_EXPECTED_LOCATION_COUNT", "124"))

# 同一次调查事件中，经纬度差值超过该阈值即标记为坐标冲突。
COORD_TOLERANCE_DEG = 1e-4

# 是否删除完全相同的重复行。
# 默认 False：先保留并汇总，同时在 QC 文件中列出，避免静默删除真实分项记录。
DROP_EXACT_DUPLICATES = False

STATUS_NOT = "NOT_SURVEYED"
STATUS_ZERO = "SURVEYED_ZERO"
STATUS_POS = "SURVEYED_POSITIVE"
STATUS_CONFLICT = "CONFLICT"
STATUS_UNKNOWN = "UNKNOWN"

STATUS_CN = {
    STATUS_NOT: "未调查（NA）",
    STATUS_ZERO: "已调查-头足类为0",
    STATUS_POS: "已调查-有头足类",
    STATUS_CONFLICT: "状态冲突-待核查",
    STATUS_UNKNOWN: "状态未知-待核查",
}

NUMERIC_METRICS = [
    "Weight_1h_g",
    "Abundance_1h_ind",
    "Biomass_kg_km2",
    "Density_1000ind_km2",
]

CORE_METRICS = ["Biomass_kg_km2", "Density_1000ind_km2"]


# ============================================================
# 1. 通用函数
# ============================================================

def normalize_text(value) -> str | pd.NA:
    """统一 Unicode、首尾空格和连续空白；空字符串转为 NA。"""
    if pd.isna(value):
        return pd.NA
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"\s+", " ", text).strip()
    if text == "" or text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return pd.NA
    return text


def normalize_station(value) -> str | pd.NA:
    """站位代码去除全部空白，并统一英文字母为大写。"""
    text = normalize_text(value)
    if pd.isna(text):
        return pd.NA
    return re.sub(r"\s+", "", str(text)).upper()


def column_key(value: str) -> str:
    """生成用于自动识别列名的宽松键。"""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = text.replace("²", "2")
    text = re.sub(r"[\s_\-—–]+", "", text)
    return text


def read_csv_flexible(path: Path) -> tuple[pd.DataFrame, str]:
    """依次尝试常见中文 CSV 编码。"""
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False), encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeError("CSV 编码识别失败：\n" + "\n".join(errors))


def find_columns(columns) -> dict[str, str]:
    """把实际中文列名映射为统一英文列名。"""
    aliases = {
        "Year": ["年", "年份", "year"],
        "Month": ["月", "月份", "month"],
        "Station": ["站位", "站点", "station", "stationid"],
        "Lon": ["经度", "lon", "longitude"],
        "Lat": ["纬度", "lat", "latitude"],
        "Species_CN": ["渔获名称", "物种名称", "种名", "species", "speciescn"],
        "Species_Latin": ["拉丁名", "学名", "latin", "latinname", "scientificname"],
        "Weight_1h_g": [
            "拖速3节时的1h重量(g)",
            "拖速3节时的1h重量（g）",
            "1h重量(g)",
            "weight1hg",
        ],
        "Abundance_1h_ind": [
            "拖速3节时的1h尾数(ind)",
            "拖速3节时的1h尾数（ind）",
            "1h尾数(ind)",
            "abundance1hind",
        ],
        "Biomass_kg_km2": [
            "生物量(kg/km2)",
            "生物量（kg/km2）",
            "生物量(kg/km²)",
            "biomasskgkm2",
        ],
        "Density_1000ind_km2": [
            "尾数密度(千尾/km2)",
            "尾数密度（千尾/km2）",
            "尾数密度(千尾/km²)",
            "density1000indkm2",
        ],
        "SurveyStatus_Raw": ["调查状态", "调查情况", "surveystatus", "status"],
    }

    actual_by_key = {column_key(c): c for c in columns}
    rename_map = {}
    missing_required = []

    required = {
        "Year",
        "Month",
        "Station",
        "Lon",
        "Lat",
        "Species_CN",
        "Species_Latin",
        "Biomass_kg_km2",
        "Density_1000ind_km2",
        "SurveyStatus_Raw",
    }

    for target, candidates in aliases.items():
        found = None
        for candidate in candidates:
            if column_key(candidate) in actual_by_key:
                found = actual_by_key[column_key(candidate)]
                break
        if found is not None:
            rename_map[found] = target
        elif target in required:
            missing_required.append(target)

    if missing_required:
        raise KeyError(
            "缺少必要字段："
            + ", ".join(missing_required)
            + "\n实际列名："
            + str(list(columns))
        )
    return rename_map


def normalize_status(value) -> str:
    """把调查状态统一为五类英文代码。"""
    text = normalize_text(value)
    if pd.isna(text):
        return STATUS_UNKNOWN

    compact = re.sub(r"[\s\-—–_()（）]+", "", str(text)).lower()

    if "未调查" in compact or compact in {"na", "not surveyed", "notsurveyed"}:
        return STATUS_NOT
    if (
        "头足类为0" in compact
        or "头足类为零" in compact
        or "无头足类" in compact
        or compact in {"已调查0", "surveyedzero"}
    ):
        return STATUS_ZERO
    if (
        "有头足类" in compact
        or "调查到头足类" in compact
        or compact in {"surveyedpositive", "positive"}
    ):
        return STATUS_POS
    return STATUS_UNKNOWN


def sum_with_nan(series: pd.Series):
    """全部缺失时返回 NA，否则对非缺失值求和。"""
    return series.sum(min_count=1)


def joined_unique(series: pd.Series) -> str:
    values = sorted({str(v) for v in series.dropna() if str(v).strip()})
    return " | ".join(values)


def safe_excel_sheet_name(name: str) -> str:
    return re.sub(r"[\\/*?:\[\]]", "_", name)[:31]


def write_csv(df: pd.DataFrame, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ============================================================
# 2. 读取和基础标准化
# ============================================================

if not INPUT_CSV.exists():
    raise FileNotFoundError(f"未找到输入文件：{INPUT_CSV}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

raw, detected_encoding = read_csv_flexible(INPUT_CSV)
raw.columns = [str(c).replace("\ufeff", "").strip() for c in raw.columns]
rename_map = find_columns(raw.columns)
df = raw.rename(columns=rename_map).copy()
df.insert(0, "SourceRow", np.arange(2, len(df) + 2))

# 可选的 1 h 指标如果不存在，则创建为空列。
for optional_col in ("Weight_1h_g", "Abundance_1h_ind"):
    if optional_col not in df.columns:
        df[optional_col] = np.nan

keep_cols = [
    "SourceRow",
    "Year",
    "Month",
    "Station",
    "Lon",
    "Lat",
    "Species_CN",
    "Species_Latin",
    *NUMERIC_METRICS,
    "SurveyStatus_Raw",
]
df = df[keep_cols].copy()

df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
df["Month"] = pd.to_numeric(df["Month"], errors="coerce").astype("Int64")
df["Lon"] = pd.to_numeric(df["Lon"], errors="coerce")
df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
for col in NUMERIC_METRICS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Station"] = df["Station"].map(normalize_station).astype("string")
df["Species_CN_Raw"] = df["Species_CN"]
df["Species_Latin_Raw"] = df["Species_Latin"]
df["Species_CN"] = df["Species_CN"].map(normalize_text).astype("string")
df["Species_Latin"] = df["Species_Latin"].map(normalize_text).astype("string")
df["SurveyStatus_Raw"] = df["SurveyStatus_Raw"].map(normalize_text).astype("string")
df["SurveyStatus_Normalized"] = df["SurveyStatus_Raw"].map(normalize_status)


# ============================================================
# 3. 关键字段、范围、负值和完全重复检查
# ============================================================

invalid_key_mask = df["Year"].isna() | df["Month"].isna() | df["Station"].isna()
qc_invalid_key_rows = df.loc[invalid_key_mask].copy()

# 缺少 EventID 关键字段的行不能进入后续分析，但完整保存在 QC 文件中。
df = df.loc[~invalid_key_mask].copy()

df["EventID"] = (
    df["Year"].astype(int).astype(str)
    + "_"
    + df["Month"].astype(int).astype(str).str.zfill(2)
    + "_"
    + df["Station"].astype(str)
)

unexpected_time_mask = (
    ~df["Year"].isin(EXPECTED_YEARS) | ~df["Month"].isin(EXPECTED_MONTHS)
)
qc_unexpected_year_month = df.loc[unexpected_time_mask].copy()

invalid_coord_mask = (
    df["Lon"].isna()
    | df["Lat"].isna()
    | ~df["Lon"].between(-180, 180)
    | ~df["Lat"].between(-90, 90)
)
qc_invalid_coordinates = df.loc[invalid_coord_mask].copy()

study_area_mask = (
    df["Lon"].notna()
    & df["Lat"].notna()
    & (~df["Lon"].between(119.0, 125.0) | ~df["Lat"].between(26.0, 32.0))
)
qc_outside_expected_area = df.loc[study_area_mask].copy()

negative_mask = df[NUMERIC_METRICS].lt(0).any(axis=1)
qc_negative_values = df.loc[negative_mask].copy()

# 完全重复：除 SourceRow 外所有分析字段完全相同。
exact_subset = [c for c in keep_cols if c != "SourceRow"]
exact_dup_mask = df.duplicated(subset=exact_subset, keep=False)
qc_exact_duplicate_rows = df.loc[exact_dup_mask].sort_values(
    ["EventID", "Species_CN", "SourceRow"], na_position="last"
)

if DROP_EXACT_DUPLICATES:
    df = df.drop_duplicates(subset=exact_subset, keep="first").copy()


# ============================================================
# 4. 调查状态规则检查与未知状态推断
# ============================================================

has_species = df["Species_CN"].notna() | df["Species_Latin"].notna()
any_numeric = df[NUMERIC_METRICS].notna().any(axis=1)
any_positive = df[NUMERIC_METRICS].gt(0).any(axis=1)
all_numeric_zero = df[NUMERIC_METRICS].fillna(np.nan).eq(0).all(axis=1)

inferred_status = np.select(
    [
        has_species | any_positive,
        (~has_species) & all_numeric_zero,
        (~has_species) & (~any_numeric),
    ],
    [STATUS_POS, STATUS_ZERO, STATUS_NOT],
    default=STATUS_UNKNOWN,
)

unknown_source_mask = df["SurveyStatus_Normalized"].eq(STATUS_UNKNOWN)
df["StatusWasInferred"] = unknown_source_mask & pd.Series(inferred_status, index=df.index).ne(
    STATUS_UNKNOWN
)
df.loc[unknown_source_mask, "SurveyStatus_Normalized"] = pd.Series(
    inferred_status, index=df.index
)[unknown_source_mask]

# 规则违背原因。保留原始值，只在最终完整矩阵中按照事件状态赋 NA/0。
violation_reasons = []
critical_conflict = pd.Series(False, index=df.index)

for idx, row in df.iterrows():
    reasons = []
    status = row["SurveyStatus_Normalized"]
    row_has_species = pd.notna(row["Species_CN"]) or pd.notna(row["Species_Latin"])
    values = pd.to_numeric(row[NUMERIC_METRICS], errors="coerce")
    row_any_numeric = values.notna().any()
    row_any_positive = values.gt(0).any()
    row_any_nonzero = values.fillna(0).ne(0).any()
    row_has_negative = values.lt(0).any()

    if status == STATUS_NOT:
        if row_has_species:
            reasons.append("未调查行含物种名称")
        if row_any_numeric:
            reasons.append("未调查行数值不是NA")
        if row_has_species or row_any_positive:
            critical_conflict.loc[idx] = True

    elif status == STATUS_ZERO:
        if row_has_species:
            reasons.append("调查为0行含物种名称")
        if not row_any_numeric:
            reasons.append("调查为0行数值缺失；输出中将按规则填0")
        elif row_any_nonzero:
            reasons.append("调查为0行存在非零数值")
        if row_has_species or row_any_nonzero:
            critical_conflict.loc[idx] = True

    elif status == STATUS_POS:
        if not row_has_species:
            reasons.append("有头足类行缺少物种名称")
            critical_conflict.loc[idx] = True
        if row[CORE_METRICS].isna().any():
            reasons.append("有头足类行的生物量或尾数密度缺失")
        if not row_any_positive:
            reasons.append("有头足类行所有数值均非正值")

    else:
        reasons.append("调查状态无法识别")
        critical_conflict.loc[idx] = True

    if row_has_negative:
        reasons.append("存在负数渔获指标")
        critical_conflict.loc[idx] = True

    violation_reasons.append("；".join(reasons))

df["StatusRuleViolation"] = violation_reasons
df["CriticalRowConflict"] = critical_conflict
qc_status_rule_violations = df.loc[df["StatusRuleViolation"].ne("")].copy()

qc_status_mapping = (
    df.groupby(["SurveyStatus_Raw", "SurveyStatus_Normalized"], dropna=False)
    .size()
    .reset_index(name="N_Rows")
    .sort_values("N_Rows", ascending=False)
)


# ============================================================
# 5. 物种名与拉丁名内部一致性处理
# ============================================================

# 先用“唯一的拉丁名 -> 中文名”关系补齐少量中文名缺失。
pair_base = df.loc[
    df["Species_CN"].notna() & df["Species_Latin"].notna(),
    ["Species_CN", "Species_Latin"],
].copy()

latin_to_cn_unique = (
    pair_base.groupby("Species_Latin")["Species_CN"]
    .agg(lambda s: sorted(set(s.dropna())))
    .to_dict()
)
latin_to_cn_fill = {
    latin: names[0] for latin, names in latin_to_cn_unique.items() if len(names) == 1
}

fill_cn_mask = df["Species_CN"].isna() & df["Species_Latin"].isin(latin_to_cn_fill)
df.loc[fill_cn_mask, "Species_CN"] = df.loc[fill_cn_mask, "Species_Latin"].map(
    latin_to_cn_fill
)
df["SpeciesCNWasFilled"] = fill_cn_mask

# 按中文名统计拉丁名对应频数，并选取出现次数最多者作为内部统一值。
qc_taxon_pair_counts = (
    df.loc[df["Species_CN"].notna() & df["Species_Latin"].notna()]
    .groupby(["Species_CN", "Species_Latin"], dropna=False)
    .size()
    .reset_index(name="N_Rows")
    .sort_values(["Species_CN", "N_Rows", "Species_Latin"], ascending=[True, False, True])
)

canonical_latin = {}
for species_cn, group in qc_taxon_pair_counts.groupby("Species_CN", sort=True):
    ranked = group.sort_values(["N_Rows", "Species_Latin"], ascending=[False, True])
    canonical_latin[species_cn] = ranked.iloc[0]["Species_Latin"]

# EI revision: map presentation names by author-approved Chinese category; preserve original labels.
from taxonomy_revision import AUTHOR_NAMES, EXPECTED_CODE_NAMES
df['Taxon_raw'] = df['Species_Latin'].copy()
unknown = set(canonical_latin) - set(AUTHOR_NAMES)
if unknown:
    raise ValueError(f'Author dictionary is missing survey categories: {sorted(unknown)}')
canonical_latin.update(AUTHOR_NAMES)
df["Species_Latin_Clean"] = df["Species_CN"].map(canonical_latin).astype("string")
df["LatinWasFilled"] = df["Species_Latin"].isna() & df["Species_Latin_Clean"].notna()
df["LatinWasReplaced"] = (
    df["Species_Latin"].notna()
    & df["Species_Latin_Clean"].notna()
    & df["Species_Latin"].ne(df["Species_Latin_Clean"])
)

cn_to_latin_counts = (
    qc_taxon_pair_counts.groupby("Species_CN")["Species_Latin"]
    .nunique()
    .reset_index(name="N_Latin_Names")
)
qc_cn_to_latin_conflicts = cn_to_latin_counts.loc[
    cn_to_latin_counts["N_Latin_Names"] > 1
].merge(qc_taxon_pair_counts, on="Species_CN", how="left")

latin_to_cn_counts = (
    qc_taxon_pair_counts.groupby("Species_Latin")["Species_CN"]
    .nunique()
    .reset_index(name="N_Chinese_Names")
)
qc_latin_to_cn_conflicts = latin_to_cn_counts.loc[
    latin_to_cn_counts["N_Chinese_Names"] > 1
].merge(qc_taxon_pair_counts, on="Species_Latin", how="left")

qc_taxon_corrections = df.loc[
    df["SpeciesCNWasFilled"] | df["LatinWasFilled"] | df["LatinWasReplaced"],
    [
        "SourceRow",
        "EventID",
        "Species_CN_Raw",
        "Species_Latin_Raw",
        "Species_CN",
        "Species_Latin_Clean",
        "SpeciesCNWasFilled",
        "LatinWasFilled",
        "LatinWasReplaced",
    ],
].copy()

unresolved_taxon_mask = (
    df["SurveyStatus_Normalized"].eq(STATUS_POS) & df["Species_CN"].isna()
)
qc_unresolved_taxon_rows = df.loc[unresolved_taxon_mask].copy()
df.loc[unresolved_taxon_mask, "CriticalRowConflict"] = True


# ============================================================
# 6. 事件状态、坐标与重复物种记录检查
# ============================================================

def summarize_event_status(group: pd.DataFrame) -> pd.Series:
    statuses = sorted(set(group["SurveyStatus_Normalized"].dropna()))
    known = [s for s in statuses if s != STATUS_UNKNOWN]
    mixed = len(set(known)) > 1
    critical = bool(group["CriticalRowConflict"].any())

    if mixed or critical:
        event_status = STATUS_CONFLICT
    elif len(known) == 1:
        event_status = known[0]
    else:
        event_status = STATUS_UNKNOWN

    lon_values = group["Lon"].dropna()
    lat_values = group["Lat"].dropna()
    lon_spread = lon_values.max() - lon_values.min() if len(lon_values) else np.nan
    lat_spread = lat_values.max() - lat_values.min() if len(lat_values) else np.nan
    coord_conflict = (
        (pd.notna(lon_spread) and lon_spread > COORD_TOLERANCE_DEG)
        or (pd.notna(lat_spread) and lat_spread > COORD_TOLERANCE_DEG)
    )

    return pd.Series(
        {
            "Year": int(group["Year"].iloc[0]),
            "Month": int(group["Month"].iloc[0]),
            "Station": group["Station"].iloc[0],
            "Lon": lon_values.median() if len(lon_values) else np.nan,
            "Lat": lat_values.median() if len(lat_values) else np.nan,
            "SurveyStatus": event_status,
            "SurveyStatus_CN": STATUS_CN[event_status],
            "SourceRowCount": len(group),
            "SourceStatuses": " | ".join(statuses),
            "N_SourceStatuses": len(set(known)),
            "StatusConflict": mixed or critical,
            "AnyStatusWasInferred": bool(group["StatusWasInferred"].any()),
            "LonSpreadDeg": lon_spread,
            "LatSpreadDeg": lat_spread,
            "CoordinateConflict": bool(coord_conflict),
            "CoordinateMissing": bool(len(lon_values) == 0 or len(lat_values) == 0),
        }
    )


event_groups = df.groupby("EventID", sort=True, group_keys=False)
try:
    # pandas >= 2.2
    event_master = event_groups.apply(
        summarize_event_status, include_groups=False
    ).reset_index()
except TypeError:
    # 兼容 Kaggle 中可能存在的较旧 pandas。
    event_master = event_groups.apply(summarize_event_status).reset_index()

# 兼容较旧 pandas（Kaggle 环境若不支持 include_groups）。
if "EventID" not in event_master.columns:
    event_master = event_master.reset_index().rename(columns={"index": "EventID"})


# ============================================================
# 6A. 跨时期站位编号标准化：D 系列 -> 同坐标 A/B 系列
# ============================================================

# 保留原始站位和原始事件编号。统一编号仅用于后续重复测量、环境匹配和建模。
event_master = event_master.rename(
    columns={"EventID": "EventID_Raw", "Station": "Station_Raw"}
)
event_master["_Lon6"] = event_master["Lon"].round(6)
event_master["_Lat6"] = event_master["Lat"].round(6)

coord_station_frequency = (
    event_master.dropna(subset=["_Lon6", "_Lat6", "Station_Raw"])
    .groupby(["_Lon6", "_Lat6", "Station_Raw"])
    .size()
    .reset_index(name="N_Events_With_Code")
)

location_rows = []
for (lon6, lat6), group in coord_station_frequency.groupby(["_Lon6", "_Lat6"]):
    group = group.copy()
    group["Is_D_Code"] = group["Station_Raw"].astype(str).str.fullmatch(
        r"D\d+", case=False, na=False
    )
    standard = group.loc[~group["Is_D_Code"]].copy()
    candidates = standard if not standard.empty else group
    ranked = candidates.sort_values(
        ["N_Events_With_Code", "Station_Raw"], ascending=[False, True]
    )
    canonical = ranked.iloc[0]["Station_Raw"]

    if len(standard) == 1:
        mapping_status = "UNIQUE_STANDARD_CODE"
    elif len(standard) > 1:
        mapping_status = "MULTIPLE_STANDARD_CODES_USE_MOST_FREQUENT"
    else:
        mapping_status = "NO_STANDARD_CODE_USE_RAW"

    location_rows.append(
        {
            "_Lon6": lon6,
            "_Lat6": lat6,
            "Lon": group["_Lon6"].iloc[0],
            "Lat": group["_Lat6"].iloc[0],
            "Station_Canonical": canonical,
            "LocationID": canonical,
            "N_RawStationCodes": int(group["Station_Raw"].nunique()),
            "RawStationCodes": joined_unique(group["Station_Raw"]),
            "N_StandardCandidates": int(standard["Station_Raw"].nunique()),
            "MappingStatus": mapping_status,
            "N_Events": int(group["N_Events_With_Code"].sum()),
        }
    )

location_dictionary = pd.DataFrame(location_rows).sort_values(
    ["Lat", "Lon", "Station_Canonical"]
).reset_index(drop=True)

event_master = event_master.merge(
    location_dictionary[
        ["_Lon6", "_Lat6", "Station_Canonical", "LocationID", "MappingStatus"]
    ],
    on=["_Lon6", "_Lat6"],
    how="left",
    validate="many_to_one",
)

# 坐标缺失时无法跨期匹配，退回原始站位并保留质控标记。
event_master["Station_Canonical"] = event_master["Station_Canonical"].fillna(
    event_master["Station_Raw"]
)
event_master["LocationID"] = event_master["LocationID"].fillna(
    event_master["Station_Raw"]
)
event_master["MappingStatus"] = event_master["MappingStatus"].fillna(
    "MISSING_COORDINATE_USE_RAW"
)
event_master["Station"] = event_master["Station_Canonical"]
event_master["EventID"] = (
    event_master["Year"].astype(int).astype(str)
    + "_"
    + event_master["Month"].astype(int).astype(str).str.zfill(2)
    + "_"
    + event_master["LocationID"].astype(str)
)

qc_analysis_event_collisions = event_master.loc[
    event_master.duplicated("EventID", keep=False)
].sort_values("EventID")
if not qc_analysis_event_collisions.empty:
    write_csv(qc_analysis_event_collisions, "qc_20_analysis_event_collisions.csv")
    raise ValueError(
        "统一站位编号后出现重复 EventID；已输出 "
        "qc_20_analysis_event_collisions.csv，请人工核查。"
    )

qc_location_mapping_conflicts = location_dictionary.loc[
    location_dictionary["MappingStatus"].isin(
        ["MULTIPLE_STANDARD_CODES_USE_MOST_FREQUENT", "NO_STANDARD_CODE_USE_RAW"]
    )
].copy()

# 将统一后的站位字段和 EventID 同步回物种源记录。
event_id_map = event_master[
    [
        "EventID_Raw",
        "EventID",
        "Station_Raw",
        "Station_Canonical",
        "LocationID",
    ]
]
df = df.rename(columns={"EventID": "EventID_Raw", "Station": "Station_Raw"})
df = df.merge(
    event_id_map,
    on=["EventID_Raw", "Station_Raw"],
    how="left",
    validate="many_to_one",
)
df["Station"] = df["Station_Canonical"]

event_master = event_master.drop(columns=["_Lon6", "_Lat6"])

qc_event_status_conflicts = event_master.loc[
    event_master["SurveyStatus"].isin([STATUS_CONFLICT, STATUS_UNKNOWN])
].copy()

qc_event_coordinate_conflicts = event_master.loc[
    event_master["CoordinateConflict"] | event_master["CoordinateMissing"]
].copy()

# 同一 Station 在不同年份/月份的坐标变化，仅报告，不直接判为错误。
station_coordinate_variation = (
    event_master.groupby("Station", dropna=False)
    .agg(
        N_Events=("EventID", "nunique"),
        N_Unique_Lon=("Lon", "nunique"),
        N_Unique_Lat=("Lat", "nunique"),
        Lon_Min=("Lon", "min"),
        Lon_Max=("Lon", "max"),
        Lat_Min=("Lat", "min"),
        Lat_Max=("Lat", "max"),
    )
    .reset_index()
)
station_coordinate_variation["LonSpreadDeg"] = (
    station_coordinate_variation["Lon_Max"] - station_coordinate_variation["Lon_Min"]
)
station_coordinate_variation["LatSpreadDeg"] = (
    station_coordinate_variation["Lat_Max"] - station_coordinate_variation["Lat_Min"]
)
qc_station_coordinate_variation = station_coordinate_variation.loc[
    (station_coordinate_variation["N_Unique_Lon"] > 1)
    | (station_coordinate_variation["N_Unique_Lat"] > 1)
].copy()

# 同一年月同一坐标对应多个站位。
coord_station_counts = (
    event_master.dropna(subset=["Lon", "Lat"])
    .assign(Lon6=lambda x: x["Lon"].round(6), Lat6=lambda x: x["Lat"].round(6))
    .groupby(["Year", "Month", "Lon6", "Lat6"])["Station"]
    .agg(N_Stations="nunique", Stations=joined_unique)
    .reset_index()
)
qc_same_coordinate_multiple_stations = coord_station_counts.loc[
    coord_station_counts["N_Stations"] > 1
].copy()

# 只检查可明确为“有头足类”的事件-物种重复；随后按数值求和。
positive_event_ids = set(
    event_master.loc[event_master["SurveyStatus"].eq(STATUS_POS), "EventID"]
)
positive_rows = df.loc[
    df["EventID"].isin(positive_event_ids) & df["Species_CN"].notna()
].copy()

event_species_counts = (
    positive_rows.groupby(["EventID", "Species_CN"], dropna=False)
    .size()
    .reset_index(name="SourceRowCount")
)
qc_event_species_duplicates = event_species_counts.loc[
    event_species_counts["SourceRowCount"] > 1
].merge(
    positive_rows,
    on=["EventID", "Species_CN"],
    how="left",
).sort_values(["EventID", "Species_CN", "SourceRow"])


# ============================================================
# 7. 聚合真实记录，并建立物种字典
# ============================================================

if positive_rows.empty:
    raise ValueError("没有找到可识别的‘已调查-有头足类’物种记录，请检查调查状态列。")

observed_agg = (
    positive_rows.groupby(["EventID", "Species_CN"], dropna=False)
    .agg(
        Weight_1h_g=("Weight_1h_g", sum_with_nan),
        Abundance_1h_ind=("Abundance_1h_ind", sum_with_nan),
        Biomass_kg_km2=("Biomass_kg_km2", sum_with_nan),
        Density_1000ind_km2=("Density_1000ind_km2", sum_with_nan),
        SourceRowCount=("SourceRow", "count"),
        SourceRows=("SourceRow", lambda s: " | ".join(map(str, sorted(s)))),
    )
    .reset_index()
)

species_stats = (
    observed_agg.groupby("Species_CN")
    .agg(
        N_Events_Observed=("EventID", "nunique"),
        N_Source_Rows=("SourceRowCount", "sum"),
    )
    .reset_index()
    .sort_values("Species_CN")
    .reset_index(drop=True)
)
species_stats["Species_Code"] = [
    f"Species_{i:02d}" for i in range(1, len(species_stats) + 1)
]
species_stats["Species_Latin"] = species_stats["Species_CN"].map(canonical_latin)

latin_variant_table = (
    qc_taxon_pair_counts.groupby("Species_CN")
    .agg(
        N_Latin_Variants=("Species_Latin", "nunique"),
        Latin_Variants=("Species_Latin", joined_unique),
    )
    .reset_index()
)

taxonomy = species_stats.merge(latin_variant_table, on="Species_CN", how="left")
taxonomy["N_Latin_Variants"] = taxonomy["N_Latin_Variants"].fillna(0).astype(int)
taxonomy["Latin_Mapping_Status"] = np.select(
    [
        taxonomy["Species_Latin"].isna(),
        taxonomy["N_Latin_Variants"].eq(1),
        taxonomy["N_Latin_Variants"].gt(1),
    ],
    ["NO_LATIN_NAME", "ONE_TO_ONE_IN_DATA", "MULTIPLE_LATIN_USE_MOST_FREQUENT"],
    default="UNKNOWN",
)
taxonomy = taxonomy[
    [
        "Species_Code",
        "Species_CN",
        "Species_Latin",
        "Latin_Mapping_Status",
        "N_Latin_Variants",
        "Latin_Variants",
        "N_Events_Observed",
        "N_Source_Rows",
    ]
]

# Stable code alignment check prevents relabeling matrix columns accidentally.
actual = taxonomy.set_index('Species_Code')['Species_CN'].to_dict()
if actual != EXPECTED_CODE_NAMES:
    raise ValueError('Taxon codes differ from the original 25-category dictionary; review before analysis')
taxonomy['Taxon_raw'] = taxonomy['Latin_Variants']
taxonomy['Taxon_Rank'] = np.where(taxonomy.Species_Latin.str.endswith('sp.'), 'genus', 'species_label')
taxonomy['Name_Source'] = 'attachment_2_author_revision_2026-09-21'
taxonomy['Identification_Status'] = 'not_independently_verified'
taxonomy['Latin_Mapping_Status'] = np.where(taxonomy.Species_CN.eq('枪乌贼'),
    'AUTHOR_GENUS_LEVEL_REASSIGNMENT_NOT_SYNONYMY', 'AUTHOR_DICTIONARY_PRESENTATION_LABEL')

observed_agg = observed_agg.merge(
    taxonomy[["Species_Code", "Species_CN", "Species_Latin"]],
    on="Species_CN",
    how="left",
)


# ============================================================
# 8. 构建完整“事件 × 物种”长表，并严格执行 NA/0 规则
# ============================================================

all_pairs = pd.MultiIndex.from_product(
    [event_master["EventID"].tolist(), taxonomy["Species_Code"].tolist()],
    names=["EventID", "Species_Code"],
).to_frame(index=False)

species_lookup = taxonomy[["Species_Code", "Species_CN", "Species_Latin"]]
long_complete = all_pairs.merge(species_lookup, on="Species_Code", how="left")
long_complete = long_complete.merge(
    event_master[
        [
            "EventID",
            "EventID_Raw",
            "Year",
            "Month",
            "Station",
            "Station_Raw",
            "Station_Canonical",
            "LocationID",
            "Lon",
            "Lat",
            "SurveyStatus",
            "SurveyStatus_CN",
        ]
    ],
    on="EventID",
    how="left",
)

observed_values = observed_agg[
    [
        "EventID",
        "Species_Code",
        *NUMERIC_METRICS,
        "SourceRowCount",
        "SourceRows",
    ]
]
long_complete = long_complete.merge(
    observed_values,
    on=["EventID", "Species_Code"],
    how="left",
)
long_complete["ObservedSpeciesRow"] = long_complete["SourceRowCount"].notna()

is_not = long_complete["SurveyStatus"].eq(STATUS_NOT)
is_zero = long_complete["SurveyStatus"].eq(STATUS_ZERO)
is_pos = long_complete["SurveyStatus"].eq(STATUS_POS)
is_bad = long_complete["SurveyStatus"].isin([STATUS_CONFLICT, STATUS_UNKNOWN])
is_absent_in_positive = is_pos & ~long_complete["ObservedSpeciesRow"]

# 已调查为零、以及阳性事件中的未记录物种，赋值为 0。
for col in NUMERIC_METRICS:
    long_complete.loc[is_zero | is_absent_in_positive, col] = 0.0
    long_complete.loc[is_not | is_bad, col] = np.nan

long_complete["Presence"] = np.nan
long_complete.loc[is_zero | is_absent_in_positive, "Presence"] = 0
long_complete.loc[is_pos & long_complete["ObservedSpeciesRow"], "Presence"] = 1
long_complete["Presence"] = long_complete["Presence"].astype("Int64")

long_complete["SourceRowCount"] = long_complete["SourceRowCount"].fillna(0).astype(int)
long_complete["SourceRows"] = long_complete["SourceRows"].fillna("")

long_complete = long_complete[
    [
        "EventID",
        "EventID_Raw",
        "Year",
        "Month",
        "Station",
        "Station_Raw",
        "Station_Canonical",
        "LocationID",
        "Lon",
        "Lat",
        "SurveyStatus",
        "SurveyStatus_CN",
        "Species_Code",
        "Species_CN",
        "Species_Latin",
        "Weight_1h_g",
        "Abundance_1h_ind",
        "Biomass_kg_km2",
        "Density_1000ind_km2",
        "Presence",
        "ObservedSpeciesRow",
        "SourceRowCount",
        "SourceRows",
    ]
].sort_values(["Year", "Month", "LocationID", "Species_Code"])


# ============================================================
# 9. 为事件主表增加总量、丰富度和分析可用标记
# ============================================================

event_derived = (
    long_complete.groupby("EventID")
    .agg(
        Total_Biomass_kg_km2=("Biomass_kg_km2", sum_with_nan),
        Total_Density_1000ind_km2=("Density_1000ind_km2", sum_with_nan),
        Species_Richness=("Presence", sum_with_nan),
        N_Observed_Species_Rows=("ObservedSpeciesRow", "sum"),
    )
    .reset_index()
)

# 如果一个“有头足类”事件的已记录物种存在指标缺失，不把不完整和误作总量。
missing_observed = (
    long_complete.loc[long_complete["ObservedSpeciesRow"]]
    .groupby("EventID")
    .agg(
        N_Missing_Biomass=("Biomass_kg_km2", lambda s: int(s.isna().sum())),
        N_Missing_Density=("Density_1000ind_km2", lambda s: int(s.isna().sum())),
    )
    .reset_index()
)

event_master = event_master.merge(event_derived, on="EventID", how="left")
event_master = event_master.merge(missing_observed, on="EventID", how="left")
event_master[["N_Missing_Biomass", "N_Missing_Density"]] = event_master[
    ["N_Missing_Biomass", "N_Missing_Density"]
].fillna(0).astype(int)

event_master.loc[event_master["N_Missing_Biomass"] > 0, "Total_Biomass_kg_km2"] = np.nan
event_master.loc[event_master["N_Missing_Density"] > 0, "Total_Density_1000ind_km2"] = np.nan

surveyed_mask = event_master["SurveyStatus"].isin([STATUS_ZERO, STATUS_POS])
valid_coord = (
    event_master["Lon"].notna()
    & event_master["Lat"].notna()
    & ~event_master["CoordinateConflict"]
)
event_master["UseInCommunityAnalysis"] = surveyed_mask & valid_coord
event_master["UseInBiomassAnalysis"] = (
    event_master["UseInCommunityAnalysis"]
    & event_master["Total_Biomass_kg_km2"].notna()
)
event_master["UseInDensityAnalysis"] = (
    event_master["UseInCommunityAnalysis"]
    & event_master["Total_Density_1000ind_km2"].notna()
)

event_master = event_master.sort_values(["Year", "Month", "LocationID"]).reset_index(drop=True)


# ============================================================
# 10. 生成三套物种矩阵
# ============================================================

matrix_meta_cols = [
    "EventID",
    "EventID_Raw",
    "Year",
    "Month",
    "Station",
    "Station_Raw",
    "Station_Canonical",
    "LocationID",
    "Lon",
    "Lat",
    "SurveyStatus",
    "SurveyStatus_CN",
    "UseInCommunityAnalysis",
]
matrix_meta = event_master[matrix_meta_cols]
species_codes = taxonomy["Species_Code"].tolist()


def make_matrix(value_col: str) -> pd.DataFrame:
    wide = long_complete.pivot(index="EventID", columns="Species_Code", values=value_col)
    wide = wide.reindex(columns=species_codes).reset_index()
    wide.columns.name = None
    return matrix_meta.merge(wide, on="EventID", how="left")


biomass_matrix = make_matrix("Biomass_kg_km2")
density_matrix = make_matrix("Density_1000ind_km2")
presence_matrix = make_matrix("Presence")
for code in species_codes:
    presence_matrix[code] = presence_matrix[code].astype("Int64")


# ============================================================
# 11. 质量验证
# ============================================================

def all_species_na(matrix: pd.DataFrame) -> pd.Series:
    return matrix[species_codes].isna().all(axis=1)


def all_species_zero(matrix: pd.DataFrame) -> pd.Series:
    return matrix[species_codes].fillna(np.nan).eq(0).all(axis=1)


not_ids = set(event_master.loc[event_master["SurveyStatus"].eq(STATUS_NOT), "EventID"])
zero_ids = set(event_master.loc[event_master["SurveyStatus"].eq(STATUS_ZERO), "EventID"])
bad_ids = set(
    event_master.loc[
        event_master["SurveyStatus"].isin([STATUS_CONFLICT, STATUS_UNKNOWN]), "EventID"
    ]
)

checks = []


def add_check(
    name: str,
    passed: bool,
    observed,
    expected,
    note: str = "",
    critical: bool = True,
):
    checks.append(
        {
            "Check": name,
            "Passed": bool(passed),
            "Observed": observed,
            "Expected": expected,
            "Note": note,
            "Critical": critical,
        }
    )


add_check(
    "Event master EventID unique",
    event_master["EventID"].is_unique,
    event_master["EventID"].duplicated().sum(),
    0,
)
add_check(
    "Event master EventID_Raw unique",
    event_master["EventID_Raw"].is_unique,
    event_master["EventID_Raw"].duplicated().sum(),
    0,
)
add_check(
    "Canonical location count",
    event_master["LocationID"].nunique() == EXPECTED_LOCATION_COUNT,
    event_master["LocationID"].nunique(),
    EXPECTED_LOCATION_COUNT,
    "唯一调查位置数应等于 EXPECTED_LOCATION_COUNT 参数。",
)
raw_d_mask = event_master["Station_Raw"].astype(str).str.fullmatch(
    r"D\d+", case=False, na=False
)
canonical_d_mask = event_master["Station_Canonical"].astype(str).str.fullmatch(
    r"D\d+", case=False, na=False
)
add_check(
    "D-series station codes mapped to A/B canonical codes",
    not bool((raw_d_mask & canonical_d_mask).any()),
    int((raw_d_mask & canonical_d_mask).sum()),
    0,
)
add_check(
    "Long table one row per EventID-Species_Code",
    not long_complete.duplicated(["EventID", "Species_Code"]).any(),
    long_complete.duplicated(["EventID", "Species_Code"]).sum(),
    0,
)
add_check(
    "Species count",
    len(species_codes) == EXPECTED_SPECIES_COUNT,
    len(species_codes),
    EXPECTED_SPECIES_COUNT,
    "若不等于25，请先检查物种名称冲突或缺失。",
    critical=False,
)
add_check(
    "Biomass matrix rows equal events",
    len(biomass_matrix) == len(event_master),
    len(biomass_matrix),
    len(event_master),
)
add_check(
    "Density matrix rows equal events",
    len(density_matrix) == len(event_master),
    len(density_matrix),
    len(event_master),
)
add_check(
    "Presence matrix rows equal events",
    len(presence_matrix) == len(event_master),
    len(presence_matrix),
    len(event_master),
)

for label, matrix in (
    ("Biomass", biomass_matrix),
    ("Density", density_matrix),
    ("Presence", presence_matrix),
):
    indexed = matrix.set_index("EventID")
    not_ok = indexed.loc[indexed.index.intersection(not_ids), species_codes].isna().all(axis=1).all()
    bad_ok = indexed.loc[indexed.index.intersection(bad_ids), species_codes].isna().all(axis=1).all()
    zero_ok = indexed.loc[indexed.index.intersection(zero_ids), species_codes].eq(0).all(axis=1).all()
    add_check(f"{label}: NOT_SURVEYED all NA", not_ok, not_ok, True)
    add_check(f"{label}: CONFLICT/UNKNOWN all NA", bad_ok, bad_ok, True)
    add_check(f"{label}: SURVEYED_ZERO all 0", zero_ok, zero_ok, True)

qc_matrix_checks = pd.DataFrame(checks)

critical_failed = qc_matrix_checks.loc[
    (~qc_matrix_checks["Passed"]) & qc_matrix_checks["Critical"]
]
if not critical_failed.empty:
    failed = critical_failed
    raise AssertionError("矩阵规则验证失败：\n" + failed.to_string(index=False))


# ============================================================
# 12. 汇总表和文件输出
# ============================================================

event_summary_year_month = (
    event_master.groupby(["Year", "Month", "SurveyStatus", "SurveyStatus_CN"])
    .agg(
        N_Events=("EventID", "nunique"),
        N_Stations=("LocationID", "nunique"),
        N_Usable_Community=("UseInCommunityAnalysis", "sum"),
        N_Usable_Biomass=("UseInBiomassAnalysis", "sum"),
    )
    .reset_index()
    .sort_values(["Year", "Month", "SurveyStatus"])
)

qc_summary = pd.DataFrame(
    [
        ("Input CSV", str(INPUT_CSV)),
        ("Detected encoding", detected_encoding),
        ("Input rows", len(raw)),
        ("Rows with invalid EventID keys", len(qc_invalid_key_rows)),
        ("Rows retained after key check", len(df)),
        ("Unique survey events", event_master["EventID"].nunique()),
        ("Raw station codes", event_master["Station_Raw"].nunique()),
        ("Canonical locations", event_master["LocationID"].nunique()),
        ("Locations with multiple raw station codes", int((location_dictionary["N_RawStationCodes"] > 1).sum())),
        ("Location mapping conflicts", len(qc_location_mapping_conflicts)),
        ("Analysis EventID collisions", len(qc_analysis_event_collisions)),
        ("SURVEYED_POSITIVE events", int((event_master["SurveyStatus"] == STATUS_POS).sum())),
        ("SURVEYED_ZERO events", int((event_master["SurveyStatus"] == STATUS_ZERO).sum())),
        ("NOT_SURVEYED events", int((event_master["SurveyStatus"] == STATUS_NOT).sum())),
        ("CONFLICT events", int((event_master["SurveyStatus"] == STATUS_CONFLICT).sum())),
        ("UNKNOWN events", int((event_master["SurveyStatus"] == STATUS_UNKNOWN).sum())),
        ("Species in dictionary", len(taxonomy)),
        ("Exact duplicate source rows", len(qc_exact_duplicate_rows)),
        ("Duplicated EventID-Species groups", qc_event_species_duplicates[["EventID", "Species_CN"]].drop_duplicates().shape[0]),
        ("Status rule violation rows", len(qc_status_rule_violations)),
        ("Event status conflicts/unknown", len(qc_event_status_conflicts)),
        ("Event coordinate conflicts/missing", len(qc_event_coordinate_conflicts)),
        ("Stations with coordinate variation", len(qc_station_coordinate_variation)),
        ("Chinese names with multiple Latin variants", qc_cn_to_latin_conflicts["Species_CN"].nunique() if not qc_cn_to_latin_conflicts.empty else 0),
        ("Latin names linked to multiple Chinese names", qc_latin_to_cn_conflicts["Species_Latin"].nunique() if not qc_latin_to_cn_conflicts.empty else 0),
        ("Rows with internal taxon-name correction", len(qc_taxon_corrections)),
        ("Unresolved positive taxon rows", len(qc_unresolved_taxon_rows)),
        ("Negative-value rows", len(qc_negative_values)),
        ("Unexpected year/month rows", len(qc_unexpected_year_month)),
        ("Invalid-coordinate rows", len(qc_invalid_coordinates)),
        ("Rows outside expected study area", len(qc_outside_expected_area)),
    ],
    columns=["Metric", "Value"],
)

outputs = {
    "00_source_records_taxon_trace.csv": df,
    "00_qc_summary.csv": qc_summary,
    "01_species_long_clean.csv": long_complete,
    "02_event_master.csv": event_master,
    "03_biomass_matrix.csv": biomass_matrix,
    "04_density_matrix.csv": density_matrix,
    "05_presence_matrix.csv": presence_matrix,
    "06_taxonomy_dictionary.csv": taxonomy,
    "07_observed_species_aggregated.csv": observed_agg,
    "08_event_summary_year_month.csv": event_summary_year_month,
    "09_location_dictionary.csv": location_dictionary,
    "qc_01_status_mapping.csv": qc_status_mapping,
    "qc_02_status_rule_violations.csv": qc_status_rule_violations,
    "qc_03_event_status_conflicts.csv": qc_event_status_conflicts,
    "qc_04_event_species_duplicates.csv": qc_event_species_duplicates,
    "qc_05_exact_duplicate_rows.csv": qc_exact_duplicate_rows,
    "qc_06_event_coordinate_conflicts.csv": qc_event_coordinate_conflicts,
    "qc_07_station_coordinate_variation.csv": qc_station_coordinate_variation,
    "qc_08_same_coordinate_multiple_stations.csv": qc_same_coordinate_multiple_stations,
    "qc_09_taxon_pair_counts.csv": qc_taxon_pair_counts,
    "qc_10_cn_to_latin_conflicts.csv": qc_cn_to_latin_conflicts,
    "qc_11_latin_to_cn_conflicts.csv": qc_latin_to_cn_conflicts,
    "qc_12_taxon_corrections.csv": qc_taxon_corrections,
    "qc_13_unresolved_taxon_rows.csv": qc_unresolved_taxon_rows,
    "qc_14_negative_values.csv": qc_negative_values,
    "qc_15_invalid_key_rows.csv": qc_invalid_key_rows,
    "qc_16_unexpected_year_month.csv": qc_unexpected_year_month,
    "qc_17_invalid_coordinates.csv": qc_invalid_coordinates,
    "qc_18_outside_expected_area.csv": qc_outside_expected_area,
    "qc_19_matrix_checks.csv": qc_matrix_checks,
    "qc_20_analysis_event_collisions.csv": qc_analysis_event_collisions,
    "qc_21_location_mapping_conflicts.csv": qc_location_mapping_conflicts,
}

for filename, table in outputs.items():
    write_csv(table, filename)

# 汇总 Excel：即使某个 QC 表为空，也保留表头和工作表。
excel_path = OUTPUT_DIR / "Step1_cephalopod_outputs.xlsx"
try:
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        excel_tables = {
            "QC_summary": qc_summary,
            "Event_master": event_master,
            "Species_long": long_complete,
            "Biomass_matrix": biomass_matrix,
            "Density_matrix": density_matrix,
            "Presence_matrix": presence_matrix,
            "Taxonomy": taxonomy,
            "Year_month_summary": event_summary_year_month,
            "Location_dictionary": location_dictionary,
            "Status_violations": qc_status_rule_violations,
            "Status_conflicts": qc_event_status_conflicts,
            "Species_duplicates": qc_event_species_duplicates,
            "Coordinate_conflicts": qc_event_coordinate_conflicts,
            "CN_Latin_conflicts": qc_cn_to_latin_conflicts,
            "Latin_CN_conflicts": qc_latin_to_cn_conflicts,
            "Taxon_corrections": qc_taxon_corrections,
            "Matrix_checks": qc_matrix_checks,
            "Location_conflicts": qc_location_mapping_conflicts,
        }
        for sheet_name, table in excel_tables.items():
            table.to_excel(writer, sheet_name=safe_excel_sheet_name(sheet_name), index=False)
except ImportError:
    print("提示：当前环境缺少 openpyxl，CSV 已正常输出，未生成 Excel。")

# 压缩全部结果，便于从 Kaggle 下载。
zip_base = OUTPUT_DIR
zip_path = Path(shutil.make_archive(str(zip_base), "zip", root_dir=OUTPUT_DIR))


# ============================================================
# 13. 屏幕摘要
# ============================================================

print("\n" + "=" * 72)
print("第一步处理完成：头足类渔业调查数据已整理")
print("=" * 72)
print(qc_summary.to_string(index=False))
print("\n按年、月和调查状态统计：")
print(event_summary_year_month.to_string(index=False))
print("\n矩阵规则检查：")
print(qc_matrix_checks.to_string(index=False))
print("\n主要输出目录：", OUTPUT_DIR)
print("压缩包：", zip_path)

print("\n下一步运行前请优先检查以下文件：")
print("1. 00_qc_summary.csv")
print("2. qc_02_status_rule_violations.csv")
print("3. qc_03_event_status_conflicts.csv")
print("4. qc_04_event_species_duplicates.csv")
print("5. qc_06_event_coordinate_conflicts.csv")
print("6. qc_10_cn_to_latin_conflicts.csv")
print("7. qc_11_latin_to_cn_conflicts.csv")
print("8. qc_12_taxon_corrections.csv")
print("9. qc_19_matrix_checks.csv")
print("10. 09_location_dictionary.csv")
print("11. qc_21_location_mapping_conflicts.csv")