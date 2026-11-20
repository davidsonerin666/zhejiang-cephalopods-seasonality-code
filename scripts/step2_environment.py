# -*- coding: utf-8 -*-
"""


运行顺序：
1. 先运行《kaggle_step1_cephalopod_fishery_cleaning.py》修订版；
2. 再运行本代码。

本步骤匹配：
    SST   = Temperature NetCDF 的 0 m 层
    SSS   = Salinity NetCDF 的 0 m 层
    Chla_0125 = 原始约 4 km Chla 按 SST/SSS 格网面积加权聚合至 0.125°
    Chla_4km_Nearest = 原始约 4 km 最近有效像元，仅作为敏感性对照
    Depth = elevation < 0 的海洋像元转换为正水深

原则：
- 渔业调查事件不合并、不取平均；每个事件仍按真实 Lon/Lat 保留一行。
- Chla 先聚合到与 SST/SSS 完全对齐的 0.125°格网，再赋给调查事件。
- 0.125° Chla 使用格网内有效 4 km 像元的纬度余弦面积加权平均。
- 目标格网无有效 Chla 时，仅允许搜索相邻一个 0.125°格网且距离不超过15 km；
  不使用几十公里外的值。
- 年和月必须精确匹配，只使用 2014—2024 年的 4 月、11 月层。
- 原始4 km最近像元结果仍保留，但不参与 EnvironmentMatchOK 判定。
- 盐度原始值约 30—36，已经是实用盐度，不乘 0.001。
- 本步骤不读取、不匹配、不输出 ENSO。
- Chla 同时保留 Chla_0125、log1p_Chla_0125 和原4 km对照值；
  模型标准化应在训练折内完成，
  本步骤不预先做全数据 Z-score，避免交叉验证信息泄漏。

输入：
    /kaggle/working/cephalopod_step1_cleaned/02_event_master.csv
    /kaggle/working/cephalopod_step1_cleaned/03_biomass_matrix.csv
    /kaggle/working/cephalopod_step1_cleaned/05_presence_matrix.csv

环境文件：
    /kaggle/input/private-environment/

输出：
    01_event_environment_matched.csv
    02_environment_model_table_surveyed.csv
    03_biomass_model_table.csv
    04_presence_model_table.csv
    05_jsdm_X_environment.csv
    06_jsdm_Y_presence.csv
    07_jsdm_Y_biomass.csv
    08_environment_match_qc_events.csv
    09_environment_summary.csv
    10_environment_time_coverage.csv
    11_environment_variable_metadata.csv
    12_match_status_counts.csv
    13_environment_by_year_month_summary.csv
    14_chla_regridding_summary.csv
    15_chla_resolution_comparison.csv
    16_chla_4km_match_qc_events.csv
    cephalopod_step2_environment_matched.zip
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


# ============================================================
# 0. 参数设置
# ============================================================

STEP1_DIR = Path(
    os.environ.get(
        "CEPHALOPOD_STEP1_DIR",
        "/kaggle/working/cephalopod_step1_cleaned",
    )
)
EVENT_MASTER_CSV = STEP1_DIR / "02_event_master.csv"
BIOMASS_MATRIX_CSV = STEP1_DIR / "03_biomass_matrix.csv"
PRESENCE_MATRIX_CSV = STEP1_DIR / "05_presence_matrix.csv"

ENV_ROOT = Path(
    os.environ.get(
        "CEPHALOPOD_ENV_ROOT",
        "/kaggle/input/private-environment",
    )
)

TEMPERATURE_NC = Path(
    os.environ.get(
        "CEPHALOPOD_TEMPERATURE_NC",
        str(ENV_ROOT / "Temperature_0.125deg_April_November_2014_2024.nc"),
    )
)
SALINITY_NC = Path(
    os.environ.get(
        "CEPHALOPOD_SALINITY_NC",
        str(ENV_ROOT / "Salinity_0.125deg_April_November_2014_2024.nc"),
    )
)
CHLA_NC = Path(
    os.environ.get(
        "CEPHALOPOD_CHLA_NC",
        str(ENV_ROOT / "Chla_4km_April_November_2014_2024.nc"),
    )
)
DEPTH_NC = Path(
    os.environ.get(
        "CEPHALOPOD_DEPTH_NC",
        str(ENV_ROOT / "Depth_0.005°_2024.nc"),
    )
)

OUTPUT_DIR = Path(
    os.environ.get(
        "CEPHALOPOD_STEP2_OUTPUT_DIR",
        "/kaggle/working/cephalopod_step2_environment_matched",
    )
)

EXPECTED_YEARS = set(range(2014, 2025))
EXPECTED_MONTHS = {4, 11}
EXPECTED_TIME_KEYS = {(year, month) for year in EXPECTED_YEARS for month in EXPECTED_MONTHS}

# 距离阈值用于质控。
# SST/SSS/原4 km Chla 超阈值仍保留值并标记；0.125° Chla 超阈值直接不采用。
MAX_DISTANCE_KM = {
    "SST": 15.0,
    "SSS": 15.0,
    "Chla_0125": 15.0,
    "Chla_4km_Nearest": 10.0,
    "Depth_m": 5.0,
}

# 一个0.125°格网中至少需要多少个有效4 km像元才计算平均值。
# 默认1：最大限度保留真实近岸观测，同时输出像元数和有效比例供质控。
CHLA_MIN_VALID_PIXELS = int(
    os.environ.get("CEPHALOPOD_CHLA_MIN_VALID_PIXELS", "1")
)

# 仅用于识别明显单位/编码错误，不作为生态异常值删除阈值。
PLAUSIBLE_RANGES = {
    "SST": (-5.0, 40.0),
    "SSS": (0.0, 50.0),
    "Chla_0125": (0.0, 500.0),
    "Chla_4km_Nearest": (0.0, 500.0),
    "Depth_m": (0.0, 12000.0),
}

EARTH_RADIUS_KM = 6371.0088


# ============================================================
# 1. 通用函数
# ============================================================

def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def write_csv(df: pd.DataFrame, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def open_dataset_safe(path: Path) -> xr.Dataset:
    """按常用引擎依次打开 NetCDF。"""
    errors = []
    attempts = [None, "netcdf4", "h5netcdf", "scipy"]
    for engine in attempts:
        try:
            kwargs = {"decode_times": True, "mask_and_scale": True}
            if engine is not None:
                kwargs["engine"] = engine
            return xr.open_dataset(path, **kwargs)
        except Exception as exc:
            errors.append(f"engine={engine or 'default'}: {type(exc).__name__}: {exc}")
    raise OSError(f"无法打开 NetCDF：{path}\n" + "\n".join(errors))


def coord_name(ds: xr.Dataset, kind: str) -> str:
    candidates = {
        "time": ["time", "date", "datetime"],
        "depth": ["depth", "deptht", "lev", "level", "z"],
        "lat": ["lat", "latitude", "nav_lat", "y"],
        "lon": ["lon", "longitude", "nav_lon", "x"],
    }[kind]

    names = list(dict.fromkeys(list(ds.coords) + list(ds.dims)))
    lower_map = {str(name).lower(): str(name) for name in names}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    standard_name_targets = {
        "time": {"time"},
        "depth": {"depth"},
        "lat": {"latitude"},
        "lon": {"longitude"},
    }[kind]
    for name in ds.coords:
        standard_name = str(ds[name].attrs.get("standard_name", "")).lower()
        axis = str(ds[name].attrs.get("axis", "")).upper()
        if standard_name in standard_name_targets:
            return str(name)
        if kind == "time" and axis == "T":
            return str(name)
        if kind == "depth" and axis == "Z":
            return str(name)
        if kind == "lat" and axis == "Y":
            return str(name)
        if kind == "lon" and axis == "X":
            return str(name)
    raise KeyError(f"未识别到 {kind} 坐标；现有坐标/维度：{names}")


def optional_coord_name(ds: xr.Dataset, kind: str) -> str | None:
    try:
        return coord_name(ds, kind)
    except KeyError:
        return None


def choose_data_var(
    ds: xr.Dataset,
    preferred_names: list[str],
    lat_name: str,
    lon_name: str,
) -> str:
    lower_vars = {str(name).lower(): str(name) for name in ds.data_vars}
    for preferred in preferred_names:
        if preferred.lower() in lower_vars:
            return lower_vars[preferred.lower()]

    candidates = []
    for name, da in ds.data_vars.items():
        if lat_name in da.dims and lon_name in da.dims and np.issubdtype(da.dtype, np.number):
            candidates.append(str(name))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise KeyError(
            f"未找到同时含 {lat_name}/{lon_name} 的数值变量；变量：{list(ds.data_vars)}"
        )
    raise KeyError(
        "存在多个候选环境变量，无法安全自动选择："
        + str(candidates)
        + "。请把目标变量名加入 preferred_names。"
    )


def year_month_from_value(value) -> tuple[int, int]:
    if hasattr(value, "year") and hasattr(value, "month"):
        return int(value.year), int(value.month)
    stamp = pd.Timestamp(value)
    return int(stamp.year), int(stamp.month)


def time_index_lookup(time_values) -> dict[tuple[int, int], list[int]]:
    lookup: dict[tuple[int, int], list[int]] = {}
    for idx, value in enumerate(time_values):
        key = year_month_from_value(value)
        lookup.setdefault(key, []).append(idx)
    return lookup


def normalize_query_longitudes(query_lon: np.ndarray, grid_lon: np.ndarray) -> np.ndarray:
    query_lon = np.asarray(query_lon, dtype=float).copy()
    grid_lon = np.asarray(grid_lon, dtype=float)
    if np.nanmin(grid_lon) >= 0 and np.nanmax(grid_lon) > 180:
        query_lon = np.mod(query_lon, 360.0)
    elif np.nanmin(grid_lon) < 0:
        query_lon = ((query_lon + 180.0) % 360.0) - 180.0
    return query_lon


def lonlat_to_xyz(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    cos_lat = np.cos(lat)
    return np.column_stack(
        [cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)]
    )


def chord_to_km(chord_distance: np.ndarray) -> np.ndarray:
    chord = np.clip(np.asarray(chord_distance, dtype=float), 0.0, 2.0)
    angle = 2.0 * np.arcsin(chord / 2.0)
    return EARTH_RADIUS_KM * angle


def nearest_lon_index(grid_lon: np.ndarray, value: float) -> int:
    # 环形经度差，兼容 0—360 和 -180—180。
    diff = np.abs(((np.asarray(grid_lon, dtype=float) - value + 180.0) % 360.0) - 180.0)
    return int(np.nanargmin(diff))


def coordinate_cell_bounds(grid: np.ndarray) -> tuple[float, float]:
    """由格点中心估计覆盖边界，避免把最外侧半个像元误判为越界。"""
    values = np.asarray(grid, dtype=float)
    values = np.sort(np.unique(values[np.isfinite(values)]))
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    spacing = np.diff(values)
    spacing = spacing[spacing > 0]
    half_cell = float(np.median(spacing) / 2.0) if len(spacing) else 0.0
    return float(values[0] - half_cell), float(values[-1] + half_cell)


def ensure_2d_lat_lon(
    da: xr.DataArray,
    lat_name: str,
    lon_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回按 (lat, lon) 排列的二维数值和一维坐标。"""
    da = da.squeeze(drop=True)
    extra_dims = [dim for dim in da.dims if dim not in {lat_name, lon_name}]
    if extra_dims:
        raise ValueError(
            f"变量 {da.name} 在时间/深度选择后仍有额外维度 {extra_dims}，"
            "为避免错误匹配，代码停止。"
        )
    da = da.transpose(lat_name, lon_name)
    lat = np.asarray(da[lat_name].values, dtype=float)
    lon = np.asarray(da[lon_name].values, dtype=float)
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("当前代码要求规则的一维 lat/lon 网格。")
    values = np.asarray(da.values, dtype=float)
    return values, lat, lon


def grid_cell_edges(centers: np.ndarray) -> np.ndarray:
    """由严格递增的格点中心计算格网边界。"""
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or len(centers) < 2:
        raise ValueError("目标格网坐标必须是一维且至少包含2个格点。")
    differences = np.diff(centers)
    if not np.all(differences > 0):
        raise ValueError("目标格网坐标必须严格递增。")
    edges = np.empty(len(centers) + 1, dtype=float)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2.0
    edges[0] = centers[0] - differences[0] / 2.0
    edges[-1] = centers[-1] + differences[-1] / 2.0
    return edges


def load_reference_grid(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取SST/SSS的一维规则格网，作为0.125°共同目标格网。"""
    with open_dataset_safe(path) as ds:
        lat_name = coord_name(ds, "lat")
        lon_name = coord_name(ds, "lon")
        lat = np.asarray(ds[lat_name].values, dtype=float).copy()
        lon = np.asarray(ds[lon_name].values, dtype=float).copy()
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("SST/SSS目标格网必须使用一维lat/lon坐标。")
    if not np.all(np.diff(lat) > 0) or not np.all(np.diff(lon) > 0):
        raise ValueError("SST/SSS目标格网lat/lon必须严格递增。")
    return lat, lon


def aggregate_area_weighted_to_target_grid(
    source_values: np.ndarray,
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    min_valid_pixels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """把规则源格网按目标格网边界分箱，并计算纬度余弦面积加权平均。"""
    source_values = np.asarray(source_values, dtype=float)
    source_lat = np.asarray(source_lat, dtype=float)
    source_lon = normalize_query_longitudes(source_lon, target_lon)
    target_lat = np.asarray(target_lat, dtype=float)
    target_lon = np.asarray(target_lon, dtype=float)

    if source_values.shape != (len(source_lat), len(source_lon)):
        raise ValueError("Chla二维数组形状与源lat/lon长度不一致。")
    if min_valid_pixels < 1:
        raise ValueError("CHLA_MIN_VALID_PIXELS 必须大于等于1。")

    lat_edges = grid_cell_edges(target_lat)
    lon_edges = grid_cell_edges(target_lon)
    source_lon2d, source_lat2d = np.meshgrid(source_lon, source_lat)

    lat_index = np.searchsorted(lat_edges, source_lat2d, side="right") - 1
    lon_index = np.searchsorted(lon_edges, source_lon2d, side="right") - 1
    inside = (
        (lat_index >= 0)
        & (lat_index < len(target_lat))
        & (lon_index >= 0)
        & (lon_index < len(target_lon))
    )

    n_cells = len(target_lat) * len(target_lon)
    flat_cell = lat_index * len(target_lon) + lon_index
    total_pixels_flat = np.bincount(
        flat_cell[inside].astype(int), minlength=n_cells
    ).astype(int)

    finite = inside & np.isfinite(source_values)
    valid_pixels_flat = np.bincount(
        flat_cell[finite].astype(int), minlength=n_cells
    ).astype(int)

    # 经度方向像元面积近似一致，纬向面积按 cos(latitude) 修正。
    pixel_weights = np.cos(np.deg2rad(source_lat2d))
    weighted_sum = np.bincount(
        flat_cell[finite].astype(int),
        weights=(source_values[finite] * pixel_weights[finite]),
        minlength=n_cells,
    )
    weight_sum = np.bincount(
        flat_cell[finite].astype(int),
        weights=pixel_weights[finite],
        minlength=n_cells,
    )

    aggregated_flat = np.full(n_cells, np.nan, dtype=float)
    usable = (valid_pixels_flat >= min_valid_pixels) & (weight_sum > 0)
    aggregated_flat[usable] = weighted_sum[usable] / weight_sum[usable]

    valid_fraction_flat = np.full(n_cells, np.nan, dtype=float)
    has_source_pixels = total_pixels_flat > 0
    valid_fraction_flat[has_source_pixels] = (
        valid_pixels_flat[has_source_pixels] / total_pixels_flat[has_source_pixels]
    )

    target_shape = (len(target_lat), len(target_lon))
    return (
        aggregated_flat.reshape(target_shape),
        valid_pixels_flat.reshape(target_shape),
        total_pixels_flat.reshape(target_shape),
        valid_fraction_flat.reshape(target_shape),
    )


def point_distance_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    xyz1 = lonlat_to_xyz(np.asarray([lon1]), np.asarray([lat1]))[0]
    xyz2 = lonlat_to_xyz(np.asarray([lon2]), np.asarray([lat2]))[0]
    return float(chord_to_km(np.asarray([np.linalg.norm(xyz1 - xyz2)]))[0])


def query_chla_0125(
    values: np.ndarray,
    valid_pixels: np.ndarray,
    total_pixels: np.ndarray,
    valid_fraction: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    points: pd.DataFrame,
) -> pd.DataFrame:
    """按目标0.125°格网匹配；直接格为空时仅搜索一个相邻格网。"""
    value_name = "Chla_0125"
    output_columns = [
        "EventID",
        value_name,
        f"{value_name}_AssignedGridLon",
        f"{value_name}_AssignedGridLat",
        f"{value_name}_GridLon",
        f"{value_name}_GridLat",
        f"{value_name}_Distance_km",
        f"{value_name}_ValidPixels",
        f"{value_name}_TotalPixels",
        f"{value_name}_ValidFraction",
        f"{value_name}_DirectCellValid",
        f"{value_name}_UsedValidCellSearch",
        f"{value_name}_UsedAdjacentCell",
        f"{value_name}_OutsideGridBounds",
        f"{value_name}_MatchStatus",
    ]
    if points.empty:
        return pd.DataFrame(columns=output_columns)

    n_points = len(points)
    result = points[["EventID"]].copy()
    for column in output_columns[1:]:
        result[column] = np.nan

    result[f"{value_name}_DirectCellValid"] = False
    result[f"{value_name}_UsedValidCellSearch"] = False
    result[f"{value_name}_UsedAdjacentCell"] = False
    result[f"{value_name}_OutsideGridBounds"] = False
    result[f"{value_name}_MatchStatus"] = "POINT_COORDINATE_MISSING"

    point_lon_raw = pd.to_numeric(points["Lon"], errors="coerce").to_numpy(float)
    point_lat = pd.to_numeric(points["Lat"], errors="coerce").to_numpy(float)
    point_lon = normalize_query_longitudes(point_lon_raw, target_lon)
    lon_min, lon_max = coordinate_cell_bounds(target_lon)
    lat_min, lat_max = coordinate_cell_bounds(target_lat)
    max_distance = MAX_DISTANCE_KM[value_name]

    for point_i in range(n_points):
        if not np.isfinite(point_lon[point_i]) or not np.isfinite(point_lat[point_i]):
            continue

        lat_i = int(np.nanargmin(np.abs(target_lat - point_lat[point_i])))
        lon_i = nearest_lon_index(target_lon, point_lon[point_i])
        assigned_lon = float(target_lon[lon_i])
        assigned_lat = float(target_lat[lat_i])
        result.at[result.index[point_i], f"{value_name}_AssignedGridLon"] = assigned_lon
        result.at[result.index[point_i], f"{value_name}_AssignedGridLat"] = assigned_lat

        outside = bool(
            point_lat[point_i] < lat_min
            or point_lat[point_i] > lat_max
            or point_lon[point_i] < lon_min
            or point_lon[point_i] > lon_max
        )
        result.at[result.index[point_i], f"{value_name}_OutsideGridBounds"] = outside
        if outside:
            result.at[result.index[point_i], f"{value_name}_MatchStatus"] = (
                "OUTSIDE_GRID_BOUNDS"
            )
            continue

        direct_is_valid = bool(np.isfinite(values[lat_i, lon_i]))
        result.at[result.index[point_i], f"{value_name}_DirectCellValid"] = (
            direct_is_valid
        )

        candidate_lat_i = lat_i
        candidate_lon_i = lon_i
        used_adjacent = False

        if not direct_is_valid:
            result.at[
                result.index[point_i], f"{value_name}_UsedValidCellSearch"
            ] = True
            candidates = []
            for neighbor_lat_i in range(max(0, lat_i - 1), min(len(target_lat), lat_i + 2)):
                for neighbor_lon_i in range(max(0, lon_i - 1), min(len(target_lon), lon_i + 2)):
                    if neighbor_lat_i == lat_i and neighbor_lon_i == lon_i:
                        continue
                    if not np.isfinite(values[neighbor_lat_i, neighbor_lon_i]):
                        continue
                    distance = point_distance_km(
                        point_lon[point_i],
                        point_lat[point_i],
                        float(target_lon[neighbor_lon_i]),
                        float(target_lat[neighbor_lat_i]),
                    )
                    candidates.append((distance, neighbor_lat_i, neighbor_lon_i))

            if not candidates:
                result.at[result.index[point_i], f"{value_name}_GridLon"] = assigned_lon
                result.at[result.index[point_i], f"{value_name}_GridLat"] = assigned_lat
                result.at[result.index[point_i], f"{value_name}_Distance_km"] = (
                    point_distance_km(
                        point_lon[point_i], point_lat[point_i], assigned_lon, assigned_lat
                    )
                )
                result.at[result.index[point_i], f"{value_name}_ValidPixels"] = int(
                    valid_pixels[lat_i, lon_i]
                )
                result.at[result.index[point_i], f"{value_name}_TotalPixels"] = int(
                    total_pixels[lat_i, lon_i]
                )
                result.at[result.index[point_i], f"{value_name}_ValidFraction"] = (
                    valid_fraction[lat_i, lon_i]
                )
                result.at[result.index[point_i], f"{value_name}_MatchStatus"] = (
                    "NO_VALID_TARGET_OR_ADJACENT"
                )
                continue

            _, candidate_lat_i, candidate_lon_i = min(candidates, key=lambda item: item[0])
            used_adjacent = True

        matched_lon = float(target_lon[candidate_lon_i])
        matched_lat = float(target_lat[candidate_lat_i])
        distance = point_distance_km(
            point_lon[point_i], point_lat[point_i], matched_lon, matched_lat
        )
        result.at[result.index[point_i], f"{value_name}_GridLon"] = matched_lon
        result.at[result.index[point_i], f"{value_name}_GridLat"] = matched_lat
        result.at[result.index[point_i], f"{value_name}_Distance_km"] = distance
        result.at[result.index[point_i], f"{value_name}_ValidPixels"] = int(
            valid_pixels[candidate_lat_i, candidate_lon_i]
        )
        result.at[result.index[point_i], f"{value_name}_TotalPixels"] = int(
            total_pixels[candidate_lat_i, candidate_lon_i]
        )
        result.at[result.index[point_i], f"{value_name}_ValidFraction"] = (
            valid_fraction[candidate_lat_i, candidate_lon_i]
        )
        result.at[result.index[point_i], f"{value_name}_UsedAdjacentCell"] = (
            used_adjacent
        )

        if distance <= max_distance:
            result.at[result.index[point_i], value_name] = values[
                candidate_lat_i, candidate_lon_i
            ]
            result.at[result.index[point_i], f"{value_name}_MatchStatus"] = (
                "OK_ADJACENT" if used_adjacent else "OK"
            )
        else:
            result.at[result.index[point_i], f"{value_name}_MatchStatus"] = (
                "REJECTED_ADJACENT_DISTANCE"
                if used_adjacent
                else "REJECTED_DIRECT_DISTANCE"
            )

    return result[output_columns]


def query_nearest_valid(
    values: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    points: pd.DataFrame,
    key_col: str,
    value_name: str,
    max_distance_km: float,
) -> pd.DataFrame:
    """匹配最近有效网格，并记录直接格点、距离、边界和搜索标记。"""
    output_columns = [
        key_col,
        value_name,
        f"{value_name}_GridLon",
        f"{value_name}_GridLat",
        f"{value_name}_Distance_km",
        f"{value_name}_DirectCellValid",
        f"{value_name}_UsedValidCellSearch",
        f"{value_name}_OutsideGridBounds",
        f"{value_name}_MatchStatus",
    ]

    if points.empty:
        return pd.DataFrame(columns=output_columns)

    lon2d, lat2d = np.meshgrid(lon_grid, lat_grid)
    valid = np.isfinite(values)
    if not valid.any():
        result = points[[key_col]].copy()
        for col in output_columns[1:]:
            result[col] = np.nan
        result[f"{value_name}_DirectCellValid"] = False
        result[f"{value_name}_UsedValidCellSearch"] = False
        result[f"{value_name}_OutsideGridBounds"] = False
        result[f"{value_name}_MatchStatus"] = "NO_VALID_CELL_IN_LAYER"
        return result[output_columns]

    valid_lon = lon2d[valid]
    valid_lat = lat2d[valid]
    valid_values = values[valid]
    valid_xyz = lonlat_to_xyz(valid_lon, valid_lat)

    point_lon_original = pd.to_numeric(points["Lon"], errors="coerce").to_numpy(float)
    point_lat = pd.to_numeric(points["Lat"], errors="coerce").to_numpy(float)
    point_lon = normalize_query_longitudes(point_lon_original, lon_grid)
    point_valid = np.isfinite(point_lon) & np.isfinite(point_lat)

    matched_value = np.full(len(points), np.nan)
    matched_lon = np.full(len(points), np.nan)
    matched_lat = np.full(len(points), np.nan)
    distance_km = np.full(len(points), np.nan)
    direct_valid = np.zeros(len(points), dtype=bool)
    used_search = np.zeros(len(points), dtype=bool)
    outside = np.zeros(len(points), dtype=bool)
    status = np.full(len(points), "POINT_COORDINATE_MISSING", dtype=object)

    if point_valid.any():
        idx_valid_points = np.where(point_valid)[0]
        query_xyz = lonlat_to_xyz(point_lon[point_valid], point_lat[point_valid])

        if cKDTree is not None:
            tree = cKDTree(valid_xyz)
            chord, nearest_idx = tree.query(query_xyz, k=1)
        else:
            # SciPy 缺失时的安全回退；本研究数据量仍可运行，但速度较慢。
            nearest_idx = []
            chord = []
            for xyz in query_xyz:
                distances = np.sqrt(((valid_xyz - xyz) ** 2).sum(axis=1))
                idx = int(np.argmin(distances))
                nearest_idx.append(idx)
                chord.append(distances[idx])
            nearest_idx = np.asarray(nearest_idx, dtype=int)
            chord = np.asarray(chord, dtype=float)

        km = chord_to_km(chord)
        matched_value[idx_valid_points] = valid_values[nearest_idx]
        matched_lon[idx_valid_points] = valid_lon[nearest_idx]
        matched_lat[idx_valid_points] = valid_lat[nearest_idx]
        distance_km[idx_valid_points] = km

        lon_min, lon_max = coordinate_cell_bounds(lon_grid)
        lat_min, lat_max = coordinate_cell_bounds(lat_grid)

        for output_i, point_i in enumerate(idx_valid_points):
            lat_idx = int(np.nanargmin(np.abs(lat_grid - point_lat[point_i])))
            lon_idx = nearest_lon_index(lon_grid, point_lon[point_i])
            direct_valid[point_i] = bool(np.isfinite(values[lat_idx, lon_idx]))
            used_search[point_i] = not direct_valid[point_i]
            outside[point_i] = bool(
                point_lat[point_i] < lat_min
                or point_lat[point_i] > lat_max
                or point_lon[point_i] < lon_min
                or point_lon[point_i] > lon_max
            )
            status[point_i] = "OK" if km[output_i] <= max_distance_km else "WARN_DISTANCE"

    result = points[[key_col]].copy()
    result[value_name] = matched_value
    result[f"{value_name}_GridLon"] = matched_lon
    result[f"{value_name}_GridLat"] = matched_lat
    result[f"{value_name}_Distance_km"] = distance_km
    result[f"{value_name}_DirectCellValid"] = direct_valid
    result[f"{value_name}_UsedValidCellSearch"] = used_search
    result[f"{value_name}_OutsideGridBounds"] = outside
    result[f"{value_name}_MatchStatus"] = status
    return result[output_columns]


def select_surface_layer(
    da: xr.DataArray,
    depth_name: str | None,
) -> tuple[xr.DataArray, float | None]:
    if depth_name is None or depth_name not in da.dims:
        return da, None
    depth_values = np.asarray(da[depth_name].values, dtype=float)
    surface_idx = int(np.nanargmin(np.abs(depth_values)))
    return da.isel({depth_name: surface_idx}), float(depth_values[surface_idx])


def match_time_varying_file(
    path: Path,
    label: str,
    preferred_vars: list[str],
    events: pd.DataFrame,
    use_surface: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """对一个含时间的环境文件完成精确年月 + 最近有效格点匹配。"""
    with open_dataset_safe(path) as ds:
        lat_name = coord_name(ds, "lat")
        lon_name = coord_name(ds, "lon")
        time_name = coord_name(ds, "time")
        depth_name = optional_coord_name(ds, "depth")
        var_name = choose_data_var(ds, preferred_vars, lat_name, lon_name)
        da = ds[var_name]

        selected_depth = None
        if use_surface:
            da, selected_depth = select_surface_layer(da, depth_name)

        lookup = time_index_lookup(ds[time_name].values)
        coverage_rows = []
        for key, indices in sorted(lookup.items()):
            coverage_rows.append(
                {
                    "Dataset": label,
                    "Year": key[0],
                    "Month": key[1],
                    "N_SourceLayers": len(indices),
                    "ExpectedByStudy": key in EXPECTED_TIME_KEYS,
                }
            )

        result_parts = []
        for (year, month), group in events.groupby(["Year", "Month"], sort=True):
            key = (int(year), int(month))
            points = group[["EventID", "Lon", "Lat"]].copy()
            indices = lookup.get(key, [])
            if not indices:
                missing = points[["EventID"]].copy()
                missing[label] = np.nan
                missing[f"{label}_GridLon"] = np.nan
                missing[f"{label}_GridLat"] = np.nan
                missing[f"{label}_Distance_km"] = np.nan
                missing[f"{label}_DirectCellValid"] = False
                missing[f"{label}_UsedValidCellSearch"] = False
                missing[f"{label}_OutsideGridBounds"] = False
                missing[f"{label}_MatchStatus"] = "NO_TIME_LAYER"
                result_parts.append(missing)
                continue

            if len(indices) == 1:
                layer = da.isel({time_name: indices[0]})
            else:
                layer = da.isel({time_name: indices}).mean(dim=time_name, skipna=True)

            values, lat_grid, lon_grid = ensure_2d_lat_lon(layer, lat_name, lon_name)
            matched = query_nearest_valid(
                values=values,
                lat_grid=lat_grid,
                lon_grid=lon_grid,
                points=points,
                key_col="EventID",
                value_name=label,
                max_distance_km=MAX_DISTANCE_KM[label],
            )
            result_parts.append(matched)

        result = pd.concat(result_parts, ignore_index=True)
        metadata = {
            "Dataset": label,
            "File": str(path),
            "Variable": var_name,
            "Dimensions": " | ".join(map(str, ds[var_name].dims)),
            "Shape": " x ".join(map(str, ds[var_name].shape)),
            "Units": str(ds[var_name].attrs.get("units", "")),
            "LatitudeCoordinate": lat_name,
            "LongitudeCoordinate": lon_name,
            "TimeCoordinate": time_name,
            "DepthCoordinate": depth_name or "",
            "SelectedSurfaceDepth": selected_depth,
            "LatitudeMin": float(np.nanmin(ds[lat_name].values)),
            "LatitudeMax": float(np.nanmax(ds[lat_name].values)),
            "LongitudeMin": float(np.nanmin(ds[lon_name].values)),
            "LongitudeMax": float(np.nanmax(ds[lon_name].values)),
            "N_TimeKeys": len(lookup),
            "MaxAllowedDistance_km": MAX_DISTANCE_KM[label],
        }

    return result, pd.DataFrame(coverage_rows), metadata


def match_chla_0125_file(
    path: Path,
    events: pd.DataFrame,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """把每个Chla时间层聚合到SST/SSS格网后，再匹配调查事件。"""
    label = "Chla_0125"
    with open_dataset_safe(path) as ds:
        lat_name = coord_name(ds, "lat")
        lon_name = coord_name(ds, "lon")
        time_name = coord_name(ds, "time")
        var_name = choose_data_var(
            ds,
            ["Chla", "chla", "chlor_a", "chlorophyll", "chl"],
            lat_name,
            lon_name,
        )
        da = ds[var_name]
        lookup = time_index_lookup(ds[time_name].values)

        coverage_rows = [
            {
                "Dataset": label,
                "Year": key[0],
                "Month": key[1],
                "N_SourceLayers": len(indices),
                "ExpectedByStudy": key in EXPECTED_TIME_KEYS,
            }
            for key, indices in sorted(lookup.items())
        ]

        result_parts = []
        regridding_rows = []
        for (year, month), group in events.groupby(["Year", "Month"], sort=True):
            key = (int(year), int(month))
            points = group[["EventID", "Lon", "Lat"]].copy()
            indices = lookup.get(key, [])

            if not indices:
                empty_values = np.full((len(target_lat), len(target_lon)), np.nan)
                empty_count = np.zeros_like(empty_values, dtype=int)
                empty_fraction = np.full_like(empty_values, np.nan, dtype=float)
                missing = query_chla_0125(
                    empty_values,
                    empty_count,
                    empty_count,
                    empty_fraction,
                    target_lat,
                    target_lon,
                    points,
                )
                missing[f"{label}_MatchStatus"] = "NO_TIME_LAYER"
                result_parts.append(missing)
                regridding_rows.append(
                    {
                        "Year": int(year),
                        "Month": int(month),
                        "N_SourceLayers": 0,
                        "N_SourceValidPixels": 0,
                        "N_TargetCells": len(target_lat) * len(target_lon),
                        "N_TargetCellsWithSourcePixels": 0,
                        "N_TargetCellsWithValidChla": 0,
                        "MedianValidPixelsPerValidCell": np.nan,
                        "N_Events": len(group),
                        "N_EventDirectOK": 0,
                        "N_EventAdjacentOK": 0,
                        "N_EventUnusable": len(group),
                        "N_SurveyedEventUnusable": int(
                            group["UseInCommunityAnalysis"].fillna(False).sum()
                        ),
                    }
                )
                continue

            if len(indices) == 1:
                layer = da.isel({time_name: indices[0]})
            else:
                layer = da.isel({time_name: indices}).mean(
                    dim=time_name, skipna=True
                )

            source_values, source_lat, source_lon = ensure_2d_lat_lon(
                layer, lat_name, lon_name
            )
            (
                target_values,
                valid_pixels,
                total_pixels,
                valid_fraction,
            ) = aggregate_area_weighted_to_target_grid(
                source_values=source_values,
                source_lat=source_lat,
                source_lon=source_lon,
                target_lat=target_lat,
                target_lon=target_lon,
                min_valid_pixels=CHLA_MIN_VALID_PIXELS,
            )
            matched = query_chla_0125(
                values=target_values,
                valid_pixels=valid_pixels,
                total_pixels=total_pixels,
                valid_fraction=valid_fraction,
                target_lat=target_lat,
                target_lon=target_lon,
                points=points,
            )
            result_parts.append(matched)

            status = matched[f"{label}_MatchStatus"]
            unusable_ids = set(
                matched.loc[~status.isin(["OK", "OK_ADJACENT"]), "EventID"]
            )
            valid_cell_counts = valid_pixels[np.isfinite(target_values)]
            regridding_rows.append(
                {
                    "Year": int(year),
                    "Month": int(month),
                    "N_SourceLayers": len(indices),
                    "N_SourceValidPixels": int(np.isfinite(source_values).sum()),
                    "N_TargetCells": int(target_values.size),
                    "N_TargetCellsWithSourcePixels": int((total_pixels > 0).sum()),
                    "N_TargetCellsWithValidChla": int(np.isfinite(target_values).sum()),
                    "MedianValidPixelsPerValidCell": (
                        float(np.median(valid_cell_counts))
                        if len(valid_cell_counts)
                        else np.nan
                    ),
                    "N_Events": len(group),
                    "N_EventDirectOK": int(status.eq("OK").sum()),
                    "N_EventAdjacentOK": int(status.eq("OK_ADJACENT").sum()),
                    "N_EventUnusable": int(
                        (~status.isin(["OK", "OK_ADJACENT"])).sum()
                    ),
                    "N_SurveyedEventUnusable": int(
                        group.loc[
                            group["EventID"].isin(unusable_ids),
                            "UseInCommunityAnalysis",
                        ]
                        .fillna(False)
                        .sum()
                    ),
                }
            )

        result = pd.concat(result_parts, ignore_index=True)
        metadata = {
            "Dataset": label,
            "File": str(path),
            "Variable": var_name,
            "Dimensions": " | ".join(map(str, ds[var_name].dims)),
            "Shape": " x ".join(map(str, ds[var_name].shape)),
            "Units": str(ds[var_name].attrs.get("units", "")),
            "LatitudeCoordinate": lat_name,
            "LongitudeCoordinate": lon_name,
            "TimeCoordinate": time_name,
            "DepthCoordinate": "",
            "SelectedSurfaceDepth": np.nan,
            "LatitudeMin": float(np.nanmin(ds[lat_name].values)),
            "LatitudeMax": float(np.nanmax(ds[lat_name].values)),
            "LongitudeMin": float(np.nanmin(ds[lon_name].values)),
            "LongitudeMax": float(np.nanmax(ds[lon_name].values)),
            "N_TimeKeys": len(lookup),
            "MaxAllowedDistance_km": MAX_DISTANCE_KM[label],
            "Transformation": (
                "4 km valid pixels binned to SST/SSS cells; "
                "cos(latitude) area-weighted arithmetic mean"
            ),
            "TargetGridRows": len(target_lat),
            "TargetGridColumns": len(target_lon),
            "TargetLatitudeMin": float(np.nanmin(target_lat)),
            "TargetLatitudeMax": float(np.nanmax(target_lat)),
            "TargetLongitudeMin": float(np.nanmin(target_lon)),
            "TargetLongitudeMax": float(np.nanmax(target_lon)),
            "TargetLatitudeResolution_deg": float(np.median(np.diff(target_lat))),
            "TargetLongitudeResolution_deg": float(np.median(np.diff(target_lon))),
            "MinimumValidSourcePixels": CHLA_MIN_VALID_PIXELS,
            "AdjacentSearch": "one target-grid cell in each direction",
        }

    return (
        result,
        pd.DataFrame(coverage_rows),
        metadata,
        pd.DataFrame(regridding_rows),
    )


def match_depth_file(
    path: Path,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """把 GEBCO elevation<0 转换为正水深并匹配。"""
    with open_dataset_safe(path) as ds:
        lat_name = coord_name(ds, "lat")
        lon_name = coord_name(ds, "lon")
        var_name = choose_data_var(
            ds,
            ["elevation", "depth", "z", "bathymetry"],
            lat_name,
            lon_name,
        )
        elevation = ds[var_name]
        depth_da = xr.where(elevation < 0, -elevation, np.nan)
        values, lat_grid, lon_grid = ensure_2d_lat_lon(depth_da, lat_name, lon_name)

        # 水深不随时间变化，只对唯一 LocationID 匹配一次。
        unique_points = (
            events[["LocationID", "Lon", "Lat"]]
            .drop_duplicates("LocationID")
            .sort_values("LocationID")
        )
        matched_locations = query_nearest_valid(
            values=values,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            points=unique_points,
            key_col="LocationID",
            value_name="Depth_m",
            max_distance_km=MAX_DISTANCE_KM["Depth_m"],
        )
        result = events[["EventID", "LocationID"]].merge(
            matched_locations,
            on="LocationID",
            how="left",
            validate="many_to_one",
        )
        result = result.drop(columns="LocationID")

        ocean_values = values[np.isfinite(values)]
        metadata = {
            "Dataset": "Depth_m",
            "File": str(path),
            "Variable": var_name,
            "Transformation": "Depth_m = -elevation where elevation < 0; land = NA",
            "Dimensions": " | ".join(map(str, ds[var_name].dims)),
            "Shape": " x ".join(map(str, ds[var_name].shape)),
            "Units": str(ds[var_name].attrs.get("units", "")),
            "LatitudeCoordinate": lat_name,
            "LongitudeCoordinate": lon_name,
            "TimeCoordinate": "",
            "DepthCoordinate": "",
            "SelectedSurfaceDepth": np.nan,
            "LatitudeMin": float(np.nanmin(ds[lat_name].values)),
            "LatitudeMax": float(np.nanmax(ds[lat_name].values)),
            "LongitudeMin": float(np.nanmin(ds[lon_name].values)),
            "LongitudeMax": float(np.nanmax(ds[lon_name].values)),
            "N_TimeKeys": 0,
            "MaxAllowedDistance_km": MAX_DISTANCE_KM["Depth_m"],
            "OceanDepthMin": float(np.nanmin(ocean_values)),
            "OceanDepthMax": float(np.nanmax(ocean_values)),
        }

    return result, metadata


def summarize_environment(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    rows = []
    for variable in variables:
        values = pd.to_numeric(df[variable], errors="coerce")
        distances = pd.to_numeric(df[f"{variable}_Distance_km"], errors="coerce")
        rows.append(
            {
                "Variable": variable,
                "N_Events": len(df),
                "N_NonMissing": int(values.notna().sum()),
                "N_Missing": int(values.isna().sum()),
                "Min": values.min(),
                "Median": values.median(),
                "Mean": values.mean(),
                "Std": values.std(),
                "Max": values.max(),
                "DistanceMedian_km": distances.median(),
                "DistanceP95_km": distances.quantile(0.95),
                "DistanceMax_km": distances.max(),
                "N_UsedValidCellSearch": int(
                    df[f"{variable}_UsedValidCellSearch"].fillna(False).sum()
                ),
                "N_OutsideGridBounds": int(
                    df[f"{variable}_OutsideGridBounds"].fillna(False).sum()
                ),
                "N_WarnDistance": int(
                    df[f"{variable}_MatchStatus"].eq("WARN_DISTANCE").sum()
                ),
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# 2. 输入检查
# ============================================================

required_files = [
    EVENT_MASTER_CSV,
    BIOMASS_MATRIX_CSV,
    PRESENCE_MATRIX_CSV,
    TEMPERATURE_NC,
    SALINITY_NC,
    CHLA_NC,
    DEPTH_NC,
]
missing_files = [str(path) for path in required_files if not path.exists()]
if missing_files:
    raise FileNotFoundError("以下输入文件不存在：\n" + "\n".join(missing_files))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

events = read_csv(EVENT_MASTER_CSV)
biomass_matrix = read_csv(BIOMASS_MATRIX_CSV)
presence_matrix = read_csv(PRESENCE_MATRIX_CSV)

required_event_cols = {
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
    "Total_Biomass_kg_km2",
    "Total_Density_1000ind_km2",
    "Species_Richness",
    "UseInCommunityAnalysis",
    "UseInBiomassAnalysis",
}
missing_event_cols = sorted(required_event_cols - set(events.columns))
if missing_event_cols:
    raise KeyError(
        "第一步事件主表缺少字段："
        + ", ".join(missing_event_cols)
        + "。请先运行带 Station_Canonical/LocationID 的第一步修订版。"
    )

if not events["EventID"].is_unique:
    raise ValueError("第一步事件主表 EventID 不唯一，停止环境匹配。")
if events.duplicated(["Year", "Month", "LocationID"]).any():
    raise ValueError("存在重复 Year-Month-LocationID，停止环境匹配。")

location_coordinate_check = (
    events.assign(_Lon6=events["Lon"].round(6), _Lat6=events["Lat"].round(6))
    .groupby("LocationID", dropna=False)
    .agg(N_Lon=("_Lon6", "nunique"), N_Lat=("_Lat6", "nunique"))
)
if (
    (location_coordinate_check["N_Lon"] > 1)
    | (location_coordinate_check["N_Lat"] > 1)
).any():
    raise ValueError("同一 LocationID 对应多个经纬度，停止环境匹配。")
if set(events["Year"].dropna().astype(int)) - EXPECTED_YEARS:
    raise ValueError("事件主表含预期范围外年份。")
if set(events["Month"].dropna().astype(int)) - EXPECTED_MONTHS:
    raise ValueError("事件主表含4月、11月之外月份。")

for name, matrix in (("biomass", biomass_matrix), ("presence", presence_matrix)):
    if not matrix["EventID"].is_unique:
        raise ValueError(f"{name} 矩阵 EventID 不唯一。")
    if set(matrix["EventID"]) != set(events["EventID"]):
        raise ValueError(f"{name} 矩阵与事件主表 EventID 集合不一致。")

species_columns = [c for c in presence_matrix.columns if str(c).startswith("Species_")]
if not species_columns:
    raise ValueError("未在出现矩阵中识别到 Species_XX 列。")
biomass_species_columns = [
    c for c in biomass_matrix.columns if str(c).startswith("Species_")
]
if biomass_species_columns != species_columns:
    raise ValueError("生物量矩阵与出现矩阵的物种列不一致。")


# ============================================================
# 3. 匹配 SST、SSS、水深，并把Chla聚合到共同0.125°格网
# ============================================================

target_lat, target_lon = load_reference_grid(TEMPERATURE_NC)
sss_target_lat, sss_target_lon = load_reference_grid(SALINITY_NC)
if (
    target_lat.shape != sss_target_lat.shape
    or target_lon.shape != sss_target_lon.shape
    or not np.allclose(target_lat, sss_target_lat, atol=1e-7, rtol=0)
    or not np.allclose(target_lon, sss_target_lon, atol=1e-7, rtol=0)
):
    raise ValueError("Temperature与Salinity的lat/lon格网不完全一致，停止处理。")

target_lat_resolution = float(np.median(np.diff(target_lat)))
target_lon_resolution = float(np.median(np.diff(target_lon)))
if not (
    np.isclose(target_lat_resolution, 0.125, atol=1e-5)
    and np.isclose(target_lon_resolution, 0.125, atol=1e-5)
):
    raise ValueError(
        "SST/SSS目标格网并非0.125°："
        f"lat={target_lat_resolution}, lon={target_lon_resolution}。"
    )

sst_match, sst_coverage, sst_metadata = match_time_varying_file(
    TEMPERATURE_NC,
    label="SST",
    preferred_vars=["Temperature", "temperature", "thetao", "temp", "sst"],
    events=events,
    use_surface=True,
)

sss_match, sss_coverage, sss_metadata = match_time_varying_file(
    SALINITY_NC,
    label="SSS",
    preferred_vars=["Salinity", "salinity", "so", "salt", "sss"],
    events=events,
    use_surface=True,
)

(
    chla_0125_match,
    chla_coverage,
    chla_0125_metadata,
    chla_regridding_summary,
) = match_chla_0125_file(
    CHLA_NC,
    events=events,
    target_lat=target_lat,
    target_lon=target_lon,
)

chla_4km_match, _, chla_4km_metadata = match_time_varying_file(
    CHLA_NC,
    label="Chla_4km_Nearest",
    preferred_vars=["Chla", "chla", "chlor_a", "chlorophyll", "chl"],
    events=events,
    use_surface=False,
)

depth_match, depth_metadata = match_depth_file(DEPTH_NC, events)

matched = events.copy()
for table in (
    sst_match,
    sss_match,
    chla_0125_match,
    chla_4km_match,
    depth_match,
):
    if not table["EventID"].is_unique:
        raise ValueError("环境匹配中间表 EventID 不唯一。")
    matched = matched.merge(table, on="EventID", how="left", validate="one_to_one")


# ============================================================
# 4. 环境值质控和分析标记
# ============================================================

environment_variables = ["SST", "SSS", "Chla_0125", "Depth_m"]
comparison_variables = ["Chla_4km_Nearest"]

for variable in environment_variables + comparison_variables:
    lower, upper = PLAUSIBLE_RANGES[variable]
    matched[f"{variable}_Plausible"] = matched[variable].between(
        lower, upper, inclusive="both"
    )

matched["log1p_Chla_0125"] = np.where(
    matched["Chla_0125"].ge(0), np.log1p(matched["Chla_0125"]), np.nan
)
matched["log1p_Chla_4km_Nearest"] = np.where(
    matched["Chla_4km_Nearest"].ge(0),
    np.log1p(matched["Chla_4km_Nearest"]),
    np.nan,
)
matched["Chla_0125_minus_4km"] = (
    matched["Chla_0125"] - matched["Chla_4km_Nearest"]
)

# 兼容旧版下游代码；正式建模请优先使用名称明确的 Chla_0125/log1p_Chla_0125。
matched["Chla"] = matched["Chla_0125"]
matched["log1p_Chla"] = matched["log1p_Chla_0125"]
matched["EnvGridLon_0125"] = matched["Chla_0125_AssignedGridLon"]
matched["EnvGridLat_0125"] = matched["Chla_0125_AssignedGridLat"]
matched["EnvGridID_0125"] = pd.Series(pd.NA, index=matched.index, dtype="string")
grid_id_mask = (
    matched["EnvGridLon_0125"].notna() & matched["EnvGridLat_0125"].notna()
)
matched.loc[grid_id_mask, "EnvGridID_0125"] = [
    f"GRID_{lat:.4f}N_{lon:.4f}E"
    for lat, lon in zip(
        matched.loc[grid_id_mask, "EnvGridLat_0125"],
        matched.loc[grid_id_mask, "EnvGridLon_0125"],
    )
]

chla_0125_accepted = matched["Chla_0125_MatchStatus"].isin(
    ["OK", "OK_ADJACENT"]
)
if matched.loc[chla_0125_accepted, "Chla_0125"].isna().any():
    raise AssertionError("Chla_0125状态合格但数值缺失。")
if matched.loc[~chla_0125_accepted, "Chla_0125"].notna().any():
    raise AssertionError("Chla_0125状态不合格但仍保留了数值。")
if (
    matched["Chla_0125_MatchStatus"].eq("OK_ADJACENT")
    & ~matched["Chla_0125_UsedAdjacentCell"].fillna(False)
).any():
    raise AssertionError("Chla_0125相邻格网状态与搜索标记不一致。")

matched["EnvironmentComplete"] = matched[environment_variables].notna().all(axis=1)

status_ok = pd.Series(True, index=matched.index)
plausible_ok = pd.Series(True, index=matched.index)
for variable in environment_variables:
    accepted_statuses = (
        {"OK", "OK_ADJACENT"} if variable == "Chla_0125" else {"OK"}
    )
    status_ok &= matched[f"{variable}_MatchStatus"].isin(accepted_statuses)
    plausible_ok &= matched[f"{variable}_Plausible"].fillna(False)

matched["EnvironmentMatchOK"] = (
    matched["EnvironmentComplete"] & status_ok & plausible_ok
)
matched["UseInEnvironmentalCommunityModel"] = (
    matched["UseInCommunityAnalysis"].fillna(False) & matched["EnvironmentMatchOK"]
)
matched["UseInEnvironmentalBiomassModel"] = (
    matched["UseInBiomassAnalysis"].fillna(False) & matched["EnvironmentMatchOK"]
)
matched["Cephalopod_Presence"] = np.where(
    matched["SurveyStatus"].eq("SURVEYED_POSITIVE"),
    1,
    np.where(matched["SurveyStatus"].eq("SURVEYED_ZERO"), 0, np.nan),
)
matched["Cephalopod_Presence"] = matched["Cephalopod_Presence"].astype("Int64")


# ============================================================
# 5. 输出模型表和JSDM对齐矩阵
# ============================================================

model_base_cols = [
    "EventID",
    "EventID_Raw",
    "Year",
    "Month",
    "Station_Raw",
    "Station_Canonical",
    "LocationID",
    "Lon",
    "Lat",
    "EnvGridID_0125",
    "EnvGridLon_0125",
    "EnvGridLat_0125",
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
    "Chla_0125_TotalPixels",
    "Chla_0125_ValidFraction",
    "Chla_0125_MatchStatus",
    "Chla_4km_Nearest",
    "log1p_Chla_4km_Nearest",
    "Chla_4km_Nearest_Distance_km",
    "Chla_4km_Nearest_MatchStatus",
    "Depth_m",
    "EnvironmentMatchOK",
]

surveyed_table = matched.loc[
    matched["UseInCommunityAnalysis"].fillna(False), model_base_cols
].copy()

biomass_model_table = matched.loc[
    matched["UseInEnvironmentalBiomassModel"], model_base_cols
].copy()

presence_model_table = matched.loc[
    matched["UseInEnvironmentalCommunityModel"], model_base_cols
].copy()

jsdm_x_cols = [
    "EventID",
    "Year",
    "Month",
    "LocationID",
    "Lon",
    "Lat",
    "EnvGridID_0125",
    "EnvGridLon_0125",
    "EnvGridLat_0125",
    "SST",
    "SSS",
    "Chla_0125",
    "log1p_Chla_0125",
    "Depth_m",
]
jsdm_x = matched.loc[
    matched["UseInEnvironmentalCommunityModel"], jsdm_x_cols
].copy()

jsdm_event_order = jsdm_x["EventID"].tolist()
presence_indexed = presence_matrix.set_index("EventID")
biomass_indexed = biomass_matrix.set_index("EventID")

jsdm_y_presence = presence_indexed.loc[jsdm_event_order, species_columns].reset_index()
jsdm_y_biomass = biomass_indexed.loc[jsdm_event_order, species_columns].reset_index()

if jsdm_y_presence["EventID"].tolist() != jsdm_x["EventID"].tolist():
    raise AssertionError("JSDM X 与 Y_presence 行顺序不一致。")
if jsdm_y_biomass["EventID"].tolist() != jsdm_x["EventID"].tolist():
    raise AssertionError("JSDM X 与 Y_biomass 行顺序不一致。")


# ============================================================
# 6. 质控汇总
# ============================================================

environment_summary = summarize_environment(
    matched, environment_variables + comparison_variables
)
time_coverage = pd.concat(
    [sst_coverage, sss_coverage, chla_coverage], ignore_index=True
).sort_values(["Dataset", "Year", "Month"])
variable_metadata = pd.DataFrame(
    [
        sst_metadata,
        sss_metadata,
        chla_0125_metadata,
        chla_4km_metadata,
        depth_metadata,
    ]
)

status_rows = []
for variable in environment_variables + comparison_variables:
    counts = matched[f"{variable}_MatchStatus"].value_counts(dropna=False)
    for status_value, count in counts.items():
        status_rows.append(
            {
                "Variable": variable,
                "MatchStatus": status_value,
                "N_Events": int(count),
            }
        )
match_status_counts = pd.DataFrame(status_rows)

qc_mask = ~matched["EnvironmentMatchOK"]
qc_columns = [
    "EventID",
    "Year",
    "Month",
    "LocationID",
    "Lon",
    "Lat",
    "EnvGridID_0125",
    "SurveyStatus",
    "EnvironmentComplete",
    "EnvironmentMatchOK",
]
for variable in environment_variables:
    qc_columns.extend(
        [
            variable,
            f"{variable}_Distance_km",
            f"{variable}_MatchStatus",
            f"{variable}_Plausible",
            f"{variable}_UsedValidCellSearch",
            f"{variable}_OutsideGridBounds",
        ]
    )
qc_columns.extend(
    [
        "Chla_0125_AssignedGridLon",
        "Chla_0125_AssignedGridLat",
        "Chla_0125_GridLon",
        "Chla_0125_GridLat",
        "Chla_0125_ValidPixels",
        "Chla_0125_TotalPixels",
        "Chla_0125_ValidFraction",
        "Chla_0125_UsedAdjacentCell",
        "Chla_4km_Nearest",
        "Chla_4km_Nearest_GridLon",
        "Chla_4km_Nearest_GridLat",
        "Chla_4km_Nearest_Distance_km",
        "Chla_4km_Nearest_MatchStatus",
        "Chla_4km_Nearest_Plausible",
        "Chla_4km_Nearest_UsedValidCellSearch",
    ]
)
# 保持字段顺序且去除重复字段。
qc_columns = list(dict.fromkeys(qc_columns))
environment_qc_events = matched.loc[qc_mask, qc_columns].copy()

chla_4km_qc_columns = [
    "EventID",
    "Year",
    "Month",
    "LocationID",
    "Lon",
    "Lat",
    "SurveyStatus",
    "Chla_4km_Nearest",
    "Chla_4km_Nearest_GridLon",
    "Chla_4km_Nearest_GridLat",
    "Chla_4km_Nearest_Distance_km",
    "Chla_4km_Nearest_MatchStatus",
    "Chla_4km_Nearest_UsedValidCellSearch",
    "Chla_0125",
    "Chla_0125_MatchStatus",
]
chla_4km_qc_events = matched.loc[
    ~matched["Chla_4km_Nearest_MatchStatus"].eq("OK"),
    chla_4km_qc_columns,
].copy()

year_month_summary = (
    matched.groupby(["Year", "Month"])
    .agg(
        N_Events=("EventID", "nunique"),
        N_Surveyed=("UseInCommunityAnalysis", "sum"),
        N_EnvironmentComplete=("EnvironmentComplete", "sum"),
        N_EnvironmentMatchOK=("EnvironmentMatchOK", "sum"),
        N_UsableEnvironmentalCommunity=("UseInEnvironmentalCommunityModel", "sum"),
        N_UsableEnvironmentalBiomass=("UseInEnvironmentalBiomassModel", "sum"),
        SST_Mean=("SST", "mean"),
        SSS_Mean=("SSS", "mean"),
        Chla_0125_Median=("Chla_0125", "median"),
        Chla_4km_Median=("Chla_4km_Nearest", "median"),
        N_Chla0125_DirectOK=("Chla_0125_MatchStatus", lambda s: s.eq("OK").sum()),
        N_Chla0125_AdjacentOK=(
            "Chla_0125_MatchStatus",
            lambda s: s.eq("OK_ADJACENT").sum(),
        ),
        Depth_Mean=("Depth_m", "mean"),
    )
    .reset_index()
    .sort_values(["Year", "Month"])
)

comparison_rows = []
for (year, month), group in matched.groupby(["Year", "Month"], sort=True):
    both = group[["Chla_0125", "Chla_4km_Nearest"]].dropna()
    difference = (both["Chla_0125"] - both["Chla_4km_Nearest"]).abs()
    comparison_rows.append(
        {
            "Year": int(year),
            "Month": int(month),
            "N_Events": len(group),
            "N_Surveyed": int(group["UseInCommunityAnalysis"].fillna(False).sum()),
            "N_Chla0125_NonMissing": int(group["Chla_0125"].notna().sum()),
            "N_Chla0125_DirectOK": int(group["Chla_0125_MatchStatus"].eq("OK").sum()),
            "N_Chla0125_AdjacentOK": int(
                group["Chla_0125_MatchStatus"].eq("OK_ADJACENT").sum()
            ),
            "N_Chla0125_Unusable": int(
                (~group["Chla_0125_MatchStatus"].isin(["OK", "OK_ADJACENT"])).sum()
            ),
            "N_Chla4km_OK": int(group["Chla_4km_Nearest_MatchStatus"].eq("OK").sum()),
            "N_Chla4km_WarnOrMissing": int(
                (~group["Chla_4km_Nearest_MatchStatus"].eq("OK")).sum()
            ),
            "N_BothNonMissing": len(both),
            "Correlation_0125_vs_4km": (
                both["Chla_0125"].corr(both["Chla_4km_Nearest"])
                if len(both) >= 2
                else np.nan
            ),
            "MedianAbsoluteDifference": difference.median(),
            "MeanAbsoluteDifference": difference.mean(),
        }
    )
chla_resolution_comparison = pd.DataFrame(comparison_rows)

expected_time_check = {
    dataset: set(zip(group["Year"].astype(int), group["Month"].astype(int)))
    for dataset, group in time_coverage.loc[
        time_coverage["ExpectedByStudy"]
    ].groupby("Dataset")
}

summary_rows = [
    ("Step1 event rows", len(events)),
    ("Unique LocationID", events["LocationID"].nunique()),
    (
        "Unique 0.125-degree environmental grid cells",
        matched["EnvGridID_0125"].nunique(),
    ),
    ("Target latitude resolution degrees", target_lat_resolution),
    ("Target longitude resolution degrees", target_lon_resolution),
    ("Minimum valid 4 km Chla pixels per target cell", CHLA_MIN_VALID_PIXELS),
    ("Surveyed event rows", int(events["UseInCommunityAnalysis"].sum())),
    ("Environment complete rows", int(matched["EnvironmentComplete"].sum())),
    ("Environment match OK rows", int(matched["EnvironmentMatchOK"].sum())),
    (
        "Surveyed rows usable for environmental community model",
        int(matched["UseInEnvironmentalCommunityModel"].sum()),
    ),
    (
        "Rows usable for environmental biomass model",
        int(matched["UseInEnvironmentalBiomassModel"].sum()),
    ),
    ("JSDM X rows", len(jsdm_x)),
    ("JSDM Y presence rows", len(jsdm_y_presence)),
    ("JSDM Y biomass rows", len(jsdm_y_biomass)),
    ("Chla 0.125 direct OK rows", int(matched["Chla_0125_MatchStatus"].eq("OK").sum())),
    (
        "Chla 0.125 adjacent OK rows",
        int(matched["Chla_0125_MatchStatus"].eq("OK_ADJACENT").sum()),
    ),
    (
        "Chla 0.125 unusable rows",
        int((~matched["Chla_0125_MatchStatus"].isin(["OK", "OK_ADJACENT"])).sum()),
    ),
    (
        "Original 4 km Chla warn or missing rows",
        int((~matched["Chla_4km_Nearest_MatchStatus"].eq("OK")).sum()),
    ),
    ("ENSO included", False),
]
for dataset in ["SST", "SSS", "Chla_0125"]:
    available = expected_time_check.get(dataset, set())
    summary_rows.append(
        (f"{dataset} expected year-month layers available", len(available & EXPECTED_TIME_KEYS))
    )
step2_summary = pd.DataFrame(summary_rows, columns=["Metric", "Value"])


# ============================================================
# 7. 保存结果
# ============================================================

outputs = {
    "00_step2_summary.csv": step2_summary,
    "01_event_environment_matched.csv": matched,
    "02_environment_model_table_surveyed.csv": surveyed_table,
    "03_biomass_model_table.csv": biomass_model_table,
    "04_presence_model_table.csv": presence_model_table,
    "05_jsdm_X_environment.csv": jsdm_x,
    "06_jsdm_Y_presence.csv": jsdm_y_presence,
    "07_jsdm_Y_biomass.csv": jsdm_y_biomass,
    "08_environment_match_qc_events.csv": environment_qc_events,
    "09_environment_summary.csv": environment_summary,
    "10_environment_time_coverage.csv": time_coverage,
    "11_environment_variable_metadata.csv": variable_metadata,
    "12_match_status_counts.csv": match_status_counts,
    "13_environment_by_year_month_summary.csv": year_month_summary,
    "14_chla_regridding_summary.csv": chla_regridding_summary,
    "15_chla_resolution_comparison.csv": chla_resolution_comparison,
    "16_chla_4km_match_qc_events.csv": chla_4km_qc_events,
}

for filename, table in outputs.items():
    write_csv(table, filename)

zip_path = Path(str(OUTPUT_DIR) + ".zip")
with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
    for filename in outputs:
        archive.write(OUTPUT_DIR / filename, arcname=filename)


# ============================================================
# 8. 屏幕摘要
# ============================================================

print("\n" + "=" * 78)
print("第二步完成：Chla已聚合到SST/SSS的0.125°格网并完成环境匹配")
print("ENSO：本步骤未读取、未匹配、未输出")
print("=" * 78)
print(step2_summary.to_string(index=False))
print("\n环境变量汇总：")
print(environment_summary.to_string(index=False))
print("\n匹配状态统计：")
print(match_status_counts.to_string(index=False))
print("\n按年、月汇总：")
print(year_month_summary.to_string(index=False))
print("\n输出目录：", OUTPUT_DIR)
print("压缩包：", zip_path)
print("\n进入建模前请优先检查：")
print("1. 00_step2_summary.csv")
print("2. 08_environment_match_qc_events.csv")
print("3. 09_environment_summary.csv")
print("4. 10_environment_time_coverage.csv")
print("5. 11_environment_variable_metadata.csv")
print("6. 12_match_status_counts.csv")
print("7. 14_chla_regridding_summary.csv")
print("8. 15_chla_resolution_comparison.csv")
print("9. 16_chla_4km_match_qc_events.csv")

# Reproducibility supplement: preserve source product identifiers and measured grid spacing.
from environment_metadata import audit_environment_sources
metadata_files = audit_environment_sources([TEMPERATURE_NC, SALINITY_NC, CHLA_NC, DEPTH_NC], OUTPUT_DIR)
with zipfile.ZipFile(Path(str(OUTPUT_DIR) + '.zip'), 'a', compression=zipfile.ZIP_DEFLATED) as archive:
    for metadata_path in metadata_files:
        archive.write(metadata_path, arcname=metadata_path.name)
