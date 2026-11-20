"""
Section 3.3 — Spatial shifts and persistent hotspots
====================================================

Standalone Kaggle script for the Zhejiang coastal cephalopod project.

Required inputs
---------------
1. cephalopod_step3_analysis_master.zip
2. cephalopod_region_mapping_outputs.zip (or 00_cephalopod_region_mapping.csv)

Optional input
--------------
cephalopod_step5_beta_diversity*.zip.  If present, the script extracts the
already-computed temporal NMDS/PERMANOVA/PERMDISP evidence into a compact table.

Analysis rules
--------------
* Resource spatial indicators and hotspots: all 2,100 surveyed events; the
  568 surveyed zero-catch events remain true zeros.
* Community composition / regional Bray-Curtis tests: positive-catch events,
  all 25 species, square-root biomass.  Events are averaged within LocationID
  before regional inference so that the spatial replicate is an independent
  station rather than repeated station-years.
* Main centre/extent series: all surveyed stations in each Year x Month.
* Sampling-design sensitivities:
  - a complete 2017--2024 station panel within each month;
  - shared stations within each Year for April--November comparisons.
* Persistent high-resource sites: recurrence in the within-period top decile.
* Confirmatory clustered hotspots: one-sided permutation Getis-Ord Gi* using
  self + six nearest surveyed stations and within-period BH-FDR correction.

The code has no internet dependency and does not require scikit-learn,
geopandas, libpysal or esda.
"""

from __future__ import annotations

import io
import json
import math
import os
import shutil
import sys
import time
import warnings
import zipfile
from collections import OrderedDict
from itertools import combinations, product
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import ConvexHull, distance_matrix
from scipy.stats import chi2, kendalltau, theilslopes


# ============================================================================
# 0. User parameters
# ============================================================================

STEP3_ENV = "CEPHALOPOD_STEP3_SOURCE"
REGION_ENV = "CEPHALOPOD_REGION_MAPPING_SOURCE"
STEP5_ENV = "CEPHALOPOD_STEP5_SOURCE"  # optional
OUTPUT_ENV = "CEPHALOPOD_SECTION33_OUTPUT_DIR"

OUTPUT_DIR = Path(
    os.environ.get(
        OUTPUT_ENV,
        "/kaggle/working/cephalopod_section3_3_spatial_shifts_hotspots",
    )
)

EXPECTED_SURVEYED = 2100
EXPECTED_POSITIVE = 1532
EXPECTED_SPECIES = 25
EXPECTED_YEARS = tuple(range(2014, 2025))
EXPECTED_MONTHS = (4, 11)

# Reproducible uncertainty and permutation settings.  These defaults are
# publication-oriented but still practical on a Kaggle CPU session.
RANDOM_SEED = int(os.environ.get("CEPHALOPOD_SPATIAL_RANDOM_SEED", "20260902"))
N_BOOTSTRAP = int(os.environ.get("CEPHALOPOD_SPATIAL_BOOTSTRAPS", "999"))
N_REGION_PERMUTATIONS = int(
    os.environ.get("CEPHALOPOD_REGION_PERMUTATIONS", "999")
)
N_GISTAR_PERMUTATIONS = int(
    os.environ.get("CEPHALOPOD_GISTAR_PERMUTATIONS", "999")
)
GISTAR_K = int(os.environ.get("CEPHALOPOD_GISTAR_K", "6"))
FDR_ALPHA = float(os.environ.get("CEPHALOPOD_FDR_ALPHA", "0.05"))

RECENT_PANEL_START = int(os.environ.get("CEPHALOPOD_PANEL_START", "2017"))
PERSISTENCE_MIN_SURVEYS = int(
    os.environ.get("CEPHALOPOD_HOTSPOT_MIN_SURVEYS", "5")
)
PRIMARY_HOTSPOT_QUANTILE = float(
    os.environ.get("CEPHALOPOD_HOTSPOT_QUANTILE", "0.90")
)
HOTSPOT_QUANTILES = tuple(
    sorted(
        {
            PRIMARY_HOTSPOT_QUANTILE,
            0.80,
            0.85,
            0.90,
        }
    )
)
REGION_MIN_POSITIVE_EVENTS_SENSITIVITY = int(
    os.environ.get("CEPHALOPOD_REGION_MIN_POSITIVE_EVENTS", "5")
)

EARTH_RADIUS_KM = 6371.0088
ELLIPSE_LEVEL = 0.95
ELLIPSE_CHI2 = float(chi2.ppf(ELLIPSE_LEVEL, df=2))

MONTH_LABELS = {4: "April", 11: "November"}
MONTH_COLORS = {4: "#2B8CBE", 11: "#E6550D"}
REGION_COLORS = {"South": "#2CA25F", "Central": "#756BB1", "North": "#DE2D26"}

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.max_columns", 120)


# ============================================================================
# 1. General helpers and source discovery
# ============================================================================

def progress(message: str) -> None:
    print(f"[Section3.3] {message}", flush=True)


def stable_seed(*parts: object) -> int:
    """Create a deterministic 32-bit seed without Python hash randomisation."""
    text = "|".join(str(part) for part in parts)
    value = RANDOM_SEED
    for byte in text.encode("utf-8"):
        value = (value * 1664525 + byte + 1013904223) & 0xFFFFFFFF
    return int(value)


def candidate_roots() -> list[Path]:
    roots = [Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()]
    return [root for root in roots if root.exists()]


def source_has_member(path: Path, member: str) -> bool:
    try:
        if path.is_dir():
            return (path / member).exists()
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
            return member in names or any(name.endswith("/" + member) for name in names)
        return path.name == member
    except Exception:
        return False


def discover_required_source(
    env_name: str,
    preferred_names: Sequence[str],
    required_member: str,
) -> Path:
    explicit = os.environ.get(env_name, "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{env_name} points to a missing path: {path}")
        if not source_has_member(path, required_member):
            raise FileNotFoundError(
                f"{path} does not contain required table {required_member}"
            )
        return path

    candidates: list[Path] = []
    for root in candidate_roots():
        for name in preferred_names:
            candidates.extend(root.rglob(name))
    valid = [p for p in candidates if source_has_member(p, required_member)]
    valid = sorted(set(p.resolve() for p in valid), key=lambda p: (len(str(p)), str(p)))
    if not valid:
        raise FileNotFoundError(
            f"Cannot find {required_member}. Upload the required ZIP to Kaggle or set "
            f"os.environ['{env_name}'] to its complete path."
        )
    return valid[0]


def discover_optional_step5() -> Path | None:
    explicit = os.environ.get(STEP5_ENV, "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{STEP5_ENV} points to a missing path: {path}")
        return path

    candidates: list[Path] = []
    for root in candidate_roots():
        candidates.extend(root.rglob("cephalopod_step5_beta_diversity*.zip"))
    valid = [
        p.resolve()
        for p in candidates
        if source_has_member(p, "10_permanova_omnibus.csv")
    ]
    if not valid:
        return None
    # Prefer a region-labelled filename, then the newest modified file.
    valid.sort(
        key=lambda p: (
            "region" in p.name.lower(),
            p.stat().st_mtime,
            len(str(p)),
        ),
        reverse=True,
    )
    return valid[0]


def matching_member(archive: zipfile.ZipFile, member: str) -> str:
    names = archive.namelist()
    if member in names:
        return member
    matches = [name for name in names if name.endswith("/" + member)]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"ZIP does not contain a unique {member}")


def read_csv_from_source(source: Path, member: str) -> pd.DataFrame:
    if source.is_dir():
        return pd.read_csv(source / member, encoding="utf-8-sig", low_memory=False)
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            name = matching_member(archive, member)
            with archive.open(name) as stream:
                return pd.read_csv(stream, encoding="utf-8-sig", low_memory=False)
    if source.name == member or source.suffix.lower() == ".csv":
        return pd.read_csv(source, encoding="utf-8-sig", low_memory=False)
    raise ValueError(f"Unsupported source: {source}")


def optional_csv(source: Path | None, member: str) -> pd.DataFrame | None:
    if source is None:
        return None
    try:
        return read_csv_from_source(source, member)
    except Exception:
        return None


def require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{label} is missing columns: {missing}")


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return output
    p = values[valid]
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    output[np.where(valid)[0]] = restored
    return output


def safe_percent_change(start: float, end: float) -> float:
    if not np.isfinite(start) or start == 0 or not np.isfinite(end):
        return np.nan
    return (end - start) / abs(start) * 100.0


def wilson_interval(successes: int, trials: int, z: float = 1.95996398454) -> tuple[float, float]:
    if trials <= 0:
        return np.nan, np.nan
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


# ============================================================================
# 2. Spatial geometry and survey-based spatial indicators
# ============================================================================

class LocalProjection:
    """Transparent local equirectangular projection in kilometres."""

    def __init__(self, reference_lon: float, reference_lat: float):
        self.lon0 = float(reference_lon)
        self.lat0 = float(reference_lat)
        self.cos_lat0 = math.cos(math.radians(self.lat0))

    def forward(
        self, lon: Sequence[float] | np.ndarray, lat: Sequence[float] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        lon_array = np.asarray(lon, dtype=float)
        lat_array = np.asarray(lat, dtype=float)
        x = EARTH_RADIUS_KM * self.cos_lat0 * np.radians(lon_array - self.lon0)
        y = EARTH_RADIUS_KM * np.radians(lat_array - self.lat0)
        return x, y

    def inverse(
        self, x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        x_array = np.asarray(x, dtype=float)
        y_array = np.asarray(y, dtype=float)
        lon = self.lon0 + np.degrees(x_array / (EARTH_RADIUS_KM * self.cos_lat0))
        lat = self.lat0 + np.degrees(y_array / EARTH_RADIUS_KM)
        return lon, lat


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    value = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(value)))


def initial_bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def cardinal_direction(bearing: float) -> str:
    if not np.isfinite(bearing):
        return "NA"
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return labels[int((bearing + 22.5) // 45.0) % 8]


def weighted_core_metrics(
    frame: pd.DataFrame,
    weight_column: str,
    projection: LocalProjection,
) -> dict[str, float]:
    lon = frame["Lon"].to_numpy(float)
    lat = frame["Lat"].to_numpy(float)
    weights = frame[weight_column].to_numpy(float)
    x, y = projection.forward(lon, lat)
    total_weight = float(weights.sum())
    positive = weights > 0

    result: dict[str, float] = {
        "N_Surveyed": int(len(frame)),
        "N_Positive": int(positive.sum()),
        "Positive_Fraction": float(positive.mean()) if len(frame) else np.nan,
        "Total_Weight": total_weight,
        "Mean_Weight_AllSurveyed": float(weights.mean()) if len(frame) else np.nan,
        "Effective_Weighted_N": np.nan,
        "COG_Lon": np.nan,
        "COG_Lat": np.nan,
        "RMS_Radius_km": np.nan,
        "Ellipse95_MajorSemiAxis_km": np.nan,
        "Ellipse95_MinorSemiAxis_km": np.nan,
        "Ellipse95_Area_km2": np.nan,
        "Ellipse_Orientation_deg_from_East_CCW": np.nan,
        "PositiveStation_ConvexHull_km2": np.nan,
    }
    if total_weight <= 0 or not positive.any():
        return result

    centre_x = float(np.average(x, weights=weights))
    centre_y = float(np.average(y, weights=weights))
    centre_lon, centre_lat = projection.inverse([centre_x], [centre_y])
    dx, dy = x - centre_x, y - centre_y
    rms = math.sqrt(float(np.average(dx * dx + dy * dy, weights=weights)))

    covariance = np.array(
        [
            [np.average(dx * dx, weights=weights), np.average(dx * dy, weights=weights)],
            [np.average(dx * dy, weights=weights), np.average(dy * dy, weights=weights)],
        ],
        dtype=float,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]
    major = math.sqrt(ELLIPSE_CHI2 * eigenvalues[0])
    minor = math.sqrt(ELLIPSE_CHI2 * eigenvalues[1])
    orientation = (math.degrees(math.atan2(eigenvectors[1, 0], eigenvectors[0, 0])) + 180.0) % 180.0

    hull_area = np.nan
    positive_points = np.column_stack([x[positive], y[positive]])
    if len(positive_points) >= 3 and np.linalg.matrix_rank(positive_points - positive_points.mean(axis=0)) >= 2:
        try:
            hull_area = float(ConvexHull(positive_points).volume)
        except Exception:
            hull_area = np.nan

    result.update(
        {
            "Effective_Weighted_N": float(total_weight**2 / np.square(weights).sum()),
            "COG_Lon": float(centre_lon[0]),
            "COG_Lat": float(centre_lat[0]),
            "RMS_Radius_km": rms,
            "Ellipse95_MajorSemiAxis_km": major,
            "Ellipse95_MinorSemiAxis_km": minor,
            "Ellipse95_Area_km2": float(math.pi * major * minor),
            "Ellipse_Orientation_deg_from_East_CCW": orientation,
            "PositiveStation_ConvexHull_km2": hull_area,
        }
    )
    return result


def spatial_metrics_with_bootstrap(
    frame: pd.DataFrame,
    weight_column: str,
    projection: LocalProjection,
    seed: int,
    n_bootstrap: int,
) -> dict[str, float]:
    result = weighted_core_metrics(frame, weight_column, projection)
    ci_variables = ["COG_Lon", "COG_Lat", "RMS_Radius_km", "Ellipse95_Area_km2"]
    for variable in ci_variables:
        result[f"{variable}_CI95_Low"] = np.nan
        result[f"{variable}_CI95_High"] = np.nan
    if n_bootstrap <= 0 or len(frame) < 3 or result["Total_Weight"] <= 0:
        return result

    # Array implementation avoids repeatedly constructing DataFrames and
    # convex hulls inside the bootstrap loop.
    rng = np.random.default_rng(seed)
    boot_values = {variable: [] for variable in ci_variables}
    n = len(frame)
    lon = frame["Lon"].to_numpy(float)
    lat = frame["Lat"].to_numpy(float)
    x, y = projection.forward(lon, lat)
    weights = frame[weight_column].to_numpy(float)
    for _ in range(n_bootstrap):
        index = rng.integers(0, n, size=n)
        xb, yb, wb = x[index], y[index], weights[index]
        total = float(wb.sum())
        if total <= 0:
            continue
        centre_x = float(np.average(xb, weights=wb))
        centre_y = float(np.average(yb, weights=wb))
        centre_lon, centre_lat = projection.inverse([centre_x], [centre_y])
        dx, dy = xb - centre_x, yb - centre_y
        rms = math.sqrt(float(np.average(dx * dx + dy * dy, weights=wb)))
        covariance = np.array(
            [
                [np.average(dx * dx, weights=wb), np.average(dx * dy, weights=wb)],
                [np.average(dx * dy, weights=wb), np.average(dy * dy, weights=wb)],
            ]
        )
        eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
        area = float(
            math.pi
            * math.sqrt(ELLIPSE_CHI2 * eigenvalues[0])
            * math.sqrt(ELLIPSE_CHI2 * eigenvalues[1])
        )
        boot_values["COG_Lon"].append(float(centre_lon[0]))
        boot_values["COG_Lat"].append(float(centre_lat[0]))
        boot_values["RMS_Radius_km"].append(rms)
        boot_values["Ellipse95_Area_km2"].append(area)

    for variable, values in boot_values.items():
        if values:
            low, high = np.quantile(values, [0.025, 0.975])
            result[f"{variable}_CI95_Low"] = float(low)
            result[f"{variable}_CI95_High"] = float(high)
    result["Bootstrap_Replicates_Requested"] = int(n_bootstrap)
    result["Bootstrap_Replicates_Valid"] = int(len(boot_values["COG_Lon"]))
    return result


def build_panel_frames(events: pd.DataFrame) -> list[tuple[str, pd.DataFrame, str]]:
    panels: list[tuple[str, pd.DataFrame, str]] = [
        (
            "ALL_SURVEYED_2014_2024",
            events.copy(),
            "All surveyed stations available in each Year-Month; main full-period series",
        )
    ]

    recent_years = list(range(RECENT_PANEL_START, max(EXPECTED_YEARS) + 1))
    panels.append((f"ALL_SURVEYED_{RECENT_PANEL_START}_2024",
                   events.loc[events.Year.isin(recent_years)].copy(),
                   "Same time window as complete panel; all available surveyed locations"))
    recent_parts: list[pd.DataFrame] = []
    recent_notes: list[str] = []
    for month in EXPECTED_MONTHS:
        subset = events.loc[
            events["Year"].isin(recent_years) & events["Month"].eq(month)
        ]
        coverage = subset.groupby("LocationID")["Year"].nunique()
        complete_ids = coverage.index[coverage.eq(len(recent_years))]
        selected = subset.loc[subset["LocationID"].isin(complete_ids)].copy()
        recent_parts.append(selected)
        recent_notes.append(f"{MONTH_LABELS[month]}={len(complete_ids)} locations")
    recent_panel = pd.concat(recent_parts, ignore_index=True)
    panels.append(
        (
            f"COMPLETE_{RECENT_PANEL_START}_2024_WITHIN_MONTH",
            recent_panel,
            "; ".join(recent_notes) + "; each listed station surveyed every year within its month",
        )
    )

    paired_parts: list[pd.DataFrame] = []
    paired_counts: list[str] = []
    for year in EXPECTED_YEARS:
        year_frame = events.loc[events["Year"].eq(year)]
        april_ids = set(year_frame.loc[year_frame["Month"].eq(4), "LocationID"])
        november_ids = set(year_frame.loc[year_frame["Month"].eq(11), "LocationID"])
        common = april_ids & november_ids
        selected = year_frame.loc[year_frame["LocationID"].isin(common)].copy()
        paired_parts.append(selected)
        paired_counts.append(f"{year}={len(common)}")
    panels.append(
        (
            "WITHIN_YEAR_APRIL_NOVEMBER_SHARED_LOCATIONS",
            pd.concat(paired_parts, ignore_index=True),
            "Same LocationIDs used for April and November within each year; " + " | ".join(paired_counts),
        )
    )
    return panels


def calculate_all_spatial_metrics(
    events: pd.DataFrame,
    projection: LocalProjection,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weight_columns = OrderedDict(
        [
            ("Biomass", "Total_Biomass_kg_km2"),
            ("Density", "Total_Density_1000ind_km2"),
        ]
    )
    records: list[dict[str, object]] = []
    panel_metadata: list[dict[str, object]] = []
    for panel_name, panel, note in build_panel_frames(events):
        panel_metadata.append(
            {
                "Panel": panel_name,
                "N_Rows": len(panel),
                "N_Locations": panel["LocationID"].nunique(),
                "Year_Min": int(panel["Year"].min()),
                "Year_Max": int(panel["Year"].max()),
                "Definition": note,
            }
        )
        for (year, month), group in panel.groupby(["Year", "Month"], sort=True):
            for metric_name, weight_column in weight_columns.items():
                estimate = spatial_metrics_with_bootstrap(
                    group,
                    weight_column,
                    projection,
                    stable_seed("bootstrap", panel_name, year, month, metric_name),
                    N_BOOTSTRAP,
                )
                record: dict[str, object] = {
                    "Panel": panel_name,
                    "Year": int(year),
                    "Month": int(month),
                    "Season": MONTH_LABELS[int(month)],
                    "Weight_Basis": metric_name,
                    "Weight_Column": weight_column,
                }
                record.update(estimate)
                records.append(record)
    return pd.DataFrame(records), pd.DataFrame(panel_metadata)


def build_centre_shifts(metrics: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (panel, basis, month), group in metrics.groupby(
        ["Panel", "Weight_Basis", "Month"], sort=False
    ):
        group = group.sort_values("Year")
        rows = list(group.itertuples(index=False))
        for previous, current in zip(rows[:-1], rows[1:]):
            distance = haversine_km(
                previous.COG_Lon,
                previous.COG_Lat,
                current.COG_Lon,
                current.COG_Lat,
            )
            bearing = initial_bearing_deg(
                previous.COG_Lon,
                previous.COG_Lat,
                current.COG_Lon,
                current.COG_Lat,
            )
            records.append(
                {
                    "Panel": panel,
                    "Weight_Basis": basis,
                    "Month": int(month),
                    "Season": MONTH_LABELS[int(month)],
                    "From_Year": int(previous.Year),
                    "To_Year": int(current.Year),
                    "Year_Gap": int(current.Year - previous.Year),
                    "From_COG_Lon": previous.COG_Lon,
                    "From_COG_Lat": previous.COG_Lat,
                    "To_COG_Lon": current.COG_Lon,
                    "To_COG_Lat": current.COG_Lat,
                    "Shift_km": distance,
                    "Bearing_deg": bearing,
                    "Direction": cardinal_direction(bearing),
                    "Annualized_Shift_km": distance / max(1, current.Year - previous.Year),
                }
            )
    return pd.DataFrame(records)


def build_net_centre_shifts(metrics: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (panel, basis, month), group in metrics.groupby(
        ["Panel", "Weight_Basis", "Month"], sort=False
    ):
        group = group.sort_values("Year")
        if len(group) < 2:
            continue
        first, last = group.iloc[0], group.iloc[-1]
        distance = haversine_km(
            first.COG_Lon, first.COG_Lat, last.COG_Lon, last.COG_Lat
        )
        bearing = initial_bearing_deg(
            first.COG_Lon, first.COG_Lat, last.COG_Lon, last.COG_Lat
        )
        records.append(
            {
                "Panel": panel,
                "Weight_Basis": basis,
                "Month": int(month),
                "Season": MONTH_LABELS[int(month)],
                "Start_Year": int(first.Year),
                "End_Year": int(last.Year),
                "Start_COG_Lon": first.COG_Lon,
                "Start_COG_Lat": first.COG_Lat,
                "End_COG_Lon": last.COG_Lon,
                "End_COG_Lat": last.COG_Lat,
                "Net_Shift_km": distance,
                "Bearing_deg": bearing,
                "Direction": cardinal_direction(bearing),
            }
        )
    return pd.DataFrame(records)


def build_paired_seasonal_shifts(metrics: pd.DataFrame) -> pd.DataFrame:
    panel = metrics.loc[
        metrics["Panel"].eq("WITHIN_YEAR_APRIL_NOVEMBER_SHARED_LOCATIONS")
    ]
    records: list[dict[str, object]] = []
    for (basis, year), group in panel.groupby(["Weight_Basis", "Year"]):
        indexed = group.set_index("Month")
        if not {4, 11}.issubset(indexed.index):
            continue
        april, november = indexed.loc[4], indexed.loc[11]
        distance = haversine_km(
            april.COG_Lon, april.COG_Lat, november.COG_Lon, november.COG_Lat
        )
        bearing = initial_bearing_deg(
            april.COG_Lon, april.COG_Lat, november.COG_Lon, november.COG_Lat
        )
        records.append(
            {
                "Weight_Basis": basis,
                "Year": int(year),
                "N_Shared_Locations": int(april.N_Surveyed),
                "April_COG_Lon": april.COG_Lon,
                "April_COG_Lat": april.COG_Lat,
                "November_COG_Lon": november.COG_Lon,
                "November_COG_Lat": november.COG_Lat,
                "April_to_November_Shift_km": distance,
                "Bearing_deg": bearing,
                "Direction": cardinal_direction(bearing),
                "Delta_RMS_Radius_km_Nov_minus_Apr": november.RMS_Radius_km - april.RMS_Radius_km,
                "Delta_Ellipse95_Area_km2_Nov_minus_Apr": november.Ellipse95_Area_km2 - april.Ellipse95_Area_km2,
                "Delta_Positive_Fraction_Nov_minus_Apr": november.Positive_Fraction - april.Positive_Fraction,
            }
        )
    return pd.DataFrame(records)


def trend_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    variables = [
        "COG_Lon",
        "COG_Lat",
        "RMS_Radius_km",
        "Ellipse95_Area_km2",
        "Positive_Fraction",
        "PositiveStation_ConvexHull_km2",
    ]
    relevant = metrics.loc[
        metrics["Panel"].ne("WITHIN_YEAR_APRIL_NOVEMBER_SHARED_LOCATIONS")
    ]
    records: list[dict[str, object]] = []
    for (panel, basis, month), group in relevant.groupby(
        ["Panel", "Weight_Basis", "Month"], sort=False
    ):
        group = group.sort_values("Year")
        years = group["Year"].to_numpy(float)
        for variable in variables:
            values = group[variable].to_numpy(float)
            valid = np.isfinite(years) & np.isfinite(values)
            if valid.sum() >= 4:
                slope, intercept, low_slope, high_slope = theilslopes(
                    values[valid], years[valid], alpha=0.95
                )
                tau, p_value = kendalltau(years[valid], values[valid])
                first_value = float(values[valid][0])
                last_value = float(values[valid][-1])
                first_year = int(years[valid][0])
                last_year = int(years[valid][-1])
            else:
                slope = intercept = low_slope = high_slope = tau = p_value = np.nan
                first_value = last_value = np.nan
                first_year = last_year = pd.NA
            records.append(
                {
                    "Panel": panel,
                    "Weight_Basis": basis,
                    "Month": int(month),
                    "Season": MONTH_LABELS[int(month)],
                    "Metric": variable,
                    "N_Years": int(valid.sum()),
                    "N_Surveyed_Min_PerYear": int(group.N_Surveyed.min()),
                    "N_Surveyed_Max_PerYear": int(group.N_Surveyed.max()),
                    "N_Event_Total": int(group.N_Surveyed.sum()),
                    "Start_Year": first_year,
                    "End_Year": last_year,
                    "Start_Value": first_value,
                    "End_Value": last_value,
                    "Absolute_Change": last_value - first_value if np.isfinite(first_value) else np.nan,
                    "Percent_Change": safe_percent_change(first_value, last_value),
                    "TheilSen_Slope_perYear": slope,
                    "TheilSen_Slope_CI95_Low": low_slope,
                    "TheilSen_Slope_CI95_High": high_slope,
                    "Kendall_Tau": tau,
                    "Kendall_P": p_value,
                    "Trend_Direction": (
                        "Increase" if np.isfinite(slope) and slope > 0 else
                        "Decrease" if np.isfinite(slope) and slope < 0 else "No_change"
                    ),
                    "Kendall_Significant_0.05": bool(np.isfinite(p_value) and p_value < 0.05),
                }
            )
    result = pd.DataFrame(records)
    result["Kendall_P_FDR_within_panel_basis"] = np.nan
    for _, index in result.groupby(["Panel", "Weight_Basis"]).groups.items():
        result.loc[index, "Kendall_P_FDR_within_panel_basis"] = bh_adjust(
            result.loc[index, "Kendall_P"]
        )
    # One prespecified BH family: 6 spatial metrics x 2 months = 12 tests,
    # separately for each panel and weighting basis, identically across panels.
    result['Q_Family'] = '12 tests: 6 metrics x 2 months within Panel/Weight_Basis'
    result['Q_value'] = result['Kendall_P_FDR_within_panel_basis']
    result['Inference_Status'] = 'exploratory annual trends; serial independence not established'
    result['Residual_Interpretation'] = 'panel differences may reflect window, coverage, or both'
    return result


def exact_sign_flip_p(differences: Sequence[float]) -> float:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return np.nan
    observed = abs(values.mean())
    if n <= 20:
        all_codes = np.arange(2**n, dtype=np.uint64)[:, None]
        bits = ((all_codes >> np.arange(n, dtype=np.uint64)) & 1).astype(float)
        signs = bits * 2.0 - 1.0
        statistics = np.abs((signs * values).mean(axis=1))
        return float((statistics >= observed - 1e-15).mean())
    rng = np.random.default_rng(stable_seed("sign_flip", n))
    signs = rng.choice([-1.0, 1.0], size=(99999, n))
    statistics = np.abs((signs * values).mean(axis=1))
    return float((1 + np.sum(statistics >= observed)) / (len(statistics) + 1))


def paired_season_metric_tests(seasonal_shifts: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Delta_RMS_Radius_km_Nov_minus_Apr",
        "Delta_Ellipse95_Area_km2_Nov_minus_Apr",
        "Delta_Positive_Fraction_Nov_minus_Apr",
    ]
    records: list[dict[str, object]] = []
    for basis, group in seasonal_shifts.groupby("Weight_Basis"):
        for column in columns:
            values = group[column].dropna().to_numpy(float)
            records.append(
                {
                    "Weight_Basis": basis,
                    "Metric": column.replace("Delta_", "").replace("_Nov_minus_Apr", ""),
                    "Contrast": "November_minus_April_on_shared_stations",
                    "N_Paired_Years": len(values),
                    "Mean_Difference": float(values.mean()) if len(values) else np.nan,
                    "Median_Difference": float(np.median(values)) if len(values) else np.nan,
                    "Exact_SignFlip_P": exact_sign_flip_p(values),
                    "Years_November_Greater": int((values > 0).sum()),
                    "Years_April_Greater": int((values < 0).sum()),
                    "Years_Tied": int((values == 0).sum()),
                }
            )
    result = pd.DataFrame(records)
    result["Exact_SignFlip_P_FDR"] = bh_adjust(result["Exact_SignFlip_P"])
    return result


# ============================================================================
# 3. High-resource recurrence and permutation Getis-Ord Gi* hotspots
# ============================================================================

def persistence_class(rate: float, n_surveys: int) -> str:
    if n_surveys < PERSISTENCE_MIN_SURVEYS:
        return "INSUFFICIENT_COVERAGE"
    if rate >= 0.50:
        return "PERSISTENT_GE50PCT"
    if rate >= 0.25:
        return "RECURRENT_25_TO_49PCT"
    if rate > 0:
        return "OCCASIONAL_LT25PCT"
    return "NEVER"


def flag_relative_high_resource_events(
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    measure_columns = OrderedDict(
        [
            ("Biomass", "Total_Biomass_kg_km2"),
            ("Density", "Total_Density_1000ind_km2"),
        ]
    )
    flags: list[pd.DataFrame] = []
    thresholds: list[dict[str, object]] = []
    base_columns = [
        "EventID", "Year", "Month", "Season", "LocationID", "Lon", "Lat",
        "Region", "SurveyStatus", "Cephalopod_Presence",
    ]
    for (year, month), group in events.groupby(["Year", "Month"], sort=True):
        for measure, column in measure_columns.items():
            values = group[column].to_numpy(float)
            percent_rank = pd.Series(values).rank(method="average", pct=True).to_numpy(float)
            for quantile in HOTSPOT_QUANTILES:
                threshold = float(np.quantile(values, quantile))
                is_high = (values > 0) & (values >= threshold)
                part = group[base_columns].copy()
                part["Measure"] = measure
                part["Measure_Column"] = column
                part["Quantile"] = float(quantile)
                part["WithinPeriod_Threshold"] = threshold
                part["Observed_Value"] = values
                part["WithinPeriod_PercentRank"] = percent_rank
                part["HighResource_Flag"] = is_high
                flags.append(part)
                thresholds.append(
                    {
                        "Year": int(year),
                        "Month": int(month),
                        "Season": MONTH_LABELS[int(month)],
                        "Measure": measure,
                        "Quantile": float(quantile),
                        "N_Surveyed": len(group),
                        "N_Positive": int((values > 0).sum()),
                        "Threshold": threshold,
                        "N_HighResource": int(is_high.sum()),
                        "HighResource_Fraction": float(is_high.mean()),
                        "TieNote": "N may exceed nominal tail size when values tie at threshold",
                    }
                )
    event_flags = pd.concat(flags, ignore_index=True)
    threshold_table = pd.DataFrame(thresholds)

    group_columns = ["Measure", "Quantile", "LocationID", "Month"]
    records: list[dict[str, object]] = []
    for keys, group in event_flags.groupby(group_columns, sort=True):
        measure, quantile, location, month = keys
        n_surveys = len(group)
        n_high = int(group["HighResource_Flag"].sum())
        rate = n_high / n_surveys
        low, high = wilson_interval(n_high, n_surveys)
        row0 = group.iloc[0]
        records.append(
            {
                "Measure": measure,
                "Quantile": float(quantile),
                "LocationID": location,
                "Month": int(month),
                "Season": MONTH_LABELS[int(month)],
                "Region": row0["Region"],
                "Lon": row0["Lon"],
                "Lat": row0["Lat"],
                "N_Surveyed_Periods": n_surveys,
                "N_HighResource_Periods": n_high,
                "Persistence_Frequency": rate,
                "Persistence_Wilson95_Low": low,
                "Persistence_Wilson95_High": high,
                "Coverage_Adequate": n_surveys >= PERSISTENCE_MIN_SURVEYS,
                "Persistence_Class": persistence_class(rate, n_surveys),
                "Surveyed_Years": " | ".join(map(str, sorted(group["Year"].unique()))),
                "HighResource_Years": " | ".join(
                    map(str, sorted(group.loc[group["HighResource_Flag"], "Year"].unique()))
                ),
            }
        )
    persistence = pd.DataFrame(records)
    return event_flags, threshold_table, persistence


def permutation_gistar_period(
    group: pd.DataFrame,
    value_column: str,
    measure: str,
    n_permutations: int,
    k: int,
    seed: int,
) -> pd.DataFrame:
    frame = group.reset_index(drop=True).copy()
    n = len(frame)
    k_used = min(max(1, k), max(1, n - 1))
    coords = frame[["X_km", "Y_km"]].to_numpy(float)
    distances = distance_matrix(coords, coords)
    neighbors = np.argsort(distances, axis=1)[:, : k_used + 1]
    values_raw = frame[value_column].to_numpy(float)
    values = np.log1p(values_raw)
    observed = np.array([values[index].sum() for index in neighbors], dtype=float)

    rng = np.random.default_rng(seed)
    permuted_values = np.empty((n_permutations, n), dtype=float)
    for permutation in range(n_permutations):
        permuted_values[permutation] = rng.permutation(values)

    null_mean = np.empty(n, dtype=float)
    null_sd = np.empty(n, dtype=float)
    p_high = np.empty(n, dtype=float)
    for i, index in enumerate(neighbors):
        null = permuted_values[:, index].sum(axis=1)
        null_mean[i] = float(null.mean())
        null_sd[i] = float(null.std(ddof=1))
        p_high[i] = float((1 + np.sum(null >= observed[i] - 1e-15)) / (n_permutations + 1))

    z_score = np.divide(
        observed - null_mean,
        null_sd,
        out=np.full(n, np.nan),
        where=null_sd > 0,
    )
    q_high = bh_adjust(p_high)
    neighbor_max_distance = np.array(
        [distances[i, index].max() for i, index in enumerate(neighbors)], dtype=float
    )
    output = frame[
        [
            "EventID", "Year", "Month", "Season", "LocationID", "Lon", "Lat",
            "Region", "SurveyStatus", "Cephalopod_Presence",
        ]
    ].copy()
    output["Measure"] = measure
    output["Value_Column"] = value_column
    output["Observed_Value"] = values_raw
    output["Transformed_Value"] = values
    output["K_Nearest_ExcludingSelf"] = k_used
    output["Neighborhood_Size_IncludingSelf"] = k_used + 1
    output["Neighborhood_MaxDistance_km"] = neighbor_max_distance
    output["LocalSum_Observed"] = observed
    output["LocalSum_NullMean"] = null_mean
    output["LocalSum_NullSD"] = null_sd
    output["Permutation_Z"] = z_score
    output["P_OneSided_High"] = p_high
    output["P_FDR_WithinPeriod"] = q_high
    output["GiStar_Hotspot_FDR"] = q_high <= FDR_ALPHA
    output["Permutations"] = n_permutations
    output["Interpretation"] = (
        "High-value spatial cluster on log1p resource; one-sided local permutation test"
    )
    return output


def calculate_gistar(
    events: pd.DataFrame,
    projection: LocalProjection,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = events.copy()
    frame["X_km"], frame["Y_km"] = projection.forward(frame["Lon"], frame["Lat"])
    measures = OrderedDict(
        [
            ("Biomass", "Total_Biomass_kg_km2"),
            ("Density", "Total_Density_1000ind_km2"),
        ]
    )
    outputs: list[pd.DataFrame] = []
    for (year, month), group in frame.groupby(["Year", "Month"], sort=True):
        for measure, column in measures.items():
            outputs.append(
                permutation_gistar_period(
                    group,
                    column,
                    measure,
                    N_GISTAR_PERMUTATIONS,
                    GISTAR_K,
                    stable_seed("gistar", year, month, measure),
                )
            )
    event_results = pd.concat(outputs, ignore_index=True)

    records: list[dict[str, object]] = []
    for (measure, location, month), group in event_results.groupby(
        ["Measure", "LocationID", "Month"], sort=True
    ):
        n_surveys = len(group)
        n_hot = int(group["GiStar_Hotspot_FDR"].sum())
        rate = n_hot / n_surveys
        low, high = wilson_interval(n_hot, n_surveys)
        first = group.iloc[0]
        records.append(
            {
                "Measure": measure,
                "LocationID": location,
                "Month": int(month),
                "Season": MONTH_LABELS[int(month)],
                "Region": first["Region"],
                "Lon": first["Lon"],
                "Lat": first["Lat"],
                "N_Surveyed_Periods": n_surveys,
                "N_GiStar_Hotspot_Periods": n_hot,
                "GiStar_Hotspot_Frequency": rate,
                "Frequency_Wilson95_Low": low,
                "Frequency_Wilson95_High": high,
                "Coverage_Adequate": n_surveys >= PERSISTENCE_MIN_SURVEYS,
                "Persistence_Class": persistence_class(rate, n_surveys),
                "GiStar_Hotspot_Years": " | ".join(
                    map(
                        str,
                        sorted(
                            group.loc[group["GiStar_Hotspot_FDR"], "Year"].unique()
                        ),
                    )
                ),
            }
        )
    return event_results, pd.DataFrame(records)


# ============================================================================
# 4. Bray-Curtis regional community heterogeneity
# ============================================================================

def bray_curtis_matrix(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    numerator = np.abs(values[:, None, :] - values[None, :, :]).sum(axis=2)
    denominator = (values[:, None, :] + values[None, :, :]).sum(axis=2)
    distance = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0,
    )
    np.fill_diagonal(distance, 0.0)
    return distance


def gower_matrix(distance: np.ndarray) -> np.ndarray:
    n = distance.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    return -0.5 * centering @ np.square(distance) @ centering


def permanova_statistic_from_gower(
    gower: np.ndarray, labels: Sequence[object]
) -> tuple[float, float, float, float]:
    labels_array = np.asarray(labels)
    levels = pd.unique(labels_array)
    n = len(labels_array)
    n_groups = len(levels)
    ss_total = float(np.trace(gower))
    ss_between = 0.0
    for level in levels:
        index = np.flatnonzero(labels_array == level)
        ss_between += float(gower[np.ix_(index, index)].sum() / len(index))
    ss_within = ss_total - ss_between
    pseudo_f = (ss_between / (n_groups - 1)) / (ss_within / (n - n_groups))
    r2 = ss_between / ss_total
    return pseudo_f, r2, ss_between, ss_within


def run_permanova(
    distance: np.ndarray,
    labels: Sequence[object],
    permutations: int,
    seed: int,
) -> dict[str, object]:
    labels_array = np.asarray(labels)
    gower = gower_matrix(distance)
    observed_f, r2, ss_between, ss_within = permanova_statistic_from_gower(
        gower, labels_array
    )
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        permuted = rng.permutation(labels_array)
        permuted_f, _, _, _ = permanova_statistic_from_gower(gower, permuted)
        exceed += permuted_f >= observed_f - 1e-15
    levels, counts = np.unique(labels_array, return_counts=True)
    return {
        "N": len(labels_array),
        "N_Groups": len(levels),
        "Group_Levels": " | ".join(map(str, levels)),
        "Group_Sizes": " | ".join(f"{level}={count}" for level, count in zip(levels, counts)),
        "PERMANOVA_PseudoF": observed_f,
        "PERMANOVA_R2": r2,
        "PERMANOVA_SS_Between": ss_between,
        "PERMANOVA_SS_Within": ss_within,
        "PERMANOVA_P": (exceed + 1) / (permutations + 1),
        "PERMANOVA_Permutations": permutations,
    }


def lingoes_pcoa(distance: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    initial = gower_matrix(distance)
    eigenvalues = np.linalg.eigvalsh(initial)
    constant = max(0.0, -float(eigenvalues.min()))
    if constant > 0:
        corrected_squared = np.square(distance) + 2.0 * constant
        np.fill_diagonal(corrected_squared, 0.0)
        corrected = np.sqrt(np.clip(corrected_squared, 0.0, None))
    else:
        corrected = distance.copy()
    corrected_gower = gower_matrix(corrected)
    values, vectors = np.linalg.eigh(corrected_gower)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    positive = values > max(1e-12, values.max() * 1e-12)
    coordinates = vectors[:, positive] * np.sqrt(values[positive])
    return coordinates, constant, values


def dispersion_distances(
    coordinates: np.ndarray, labels: Sequence[object]
) -> np.ndarray:
    labels_array = np.asarray(labels)
    distances = np.empty(len(labels_array), dtype=float)
    for level in pd.unique(labels_array):
        index = np.flatnonzero(labels_array == level)
        centroid = coordinates[index].mean(axis=0)
        distances[index] = np.linalg.norm(coordinates[index] - centroid, axis=1)
    return distances


def one_way_f(values: Sequence[float], labels: Sequence[object]) -> float:
    values_array = np.asarray(values, dtype=float)
    labels_array = np.asarray(labels)
    levels = pd.unique(labels_array)
    grand = values_array.mean()
    ss_between = sum(
        np.sum(labels_array == level)
        * (values_array[labels_array == level].mean() - grand) ** 2
        for level in levels
    )
    ss_within = sum(
        np.square(
            values_array[labels_array == level]
            - values_array[labels_array == level].mean()
        ).sum()
        for level in levels
    )
    return float((ss_between / (len(levels) - 1)) / (ss_within / (len(values_array) - len(levels))))


def run_permdisp(
    distance: np.ndarray,
    labels: Sequence[object],
    permutations: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    labels_array = np.asarray(labels)
    coordinates, constant, eigenvalues = lingoes_pcoa(distance)
    distances = dispersion_distances(coordinates, labels_array)
    observed_f = one_way_f(distances, labels_array)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        permuted = rng.permutation(labels_array)
        permuted_distances = dispersion_distances(coordinates, permuted)
        permuted_f = one_way_f(permuted_distances, permuted)
        exceed += permuted_f >= observed_f - 1e-15
    summary = {
        "PERMDISP_F": observed_f,
        "PERMDISP_P": (exceed + 1) / (permutations + 1),
        "PERMDISP_Permutations": permutations,
        "Lingoes_Constant": constant,
        "PCoA_Positive_Axes": int((eigenvalues > 1e-12).sum()),
    }
    group_records = []
    for level in pd.unique(labels_array):
        values = distances[labels_array == level]
        group_records.append(
            {
                "Group": level,
                "N": len(values),
                "Mean_DistanceToCentroid": float(values.mean()),
                "SD_DistanceToCentroid": float(values.std(ddof=1)),
                "Median_DistanceToCentroid": float(np.median(values)),
            }
        )
    return summary, pd.DataFrame(group_records)


def joint_interpretation(permanova_p: float, permdisp_p: float) -> str:
    perma = np.isfinite(permanova_p) and permanova_p < FDR_ALPHA
    disp = np.isfinite(permdisp_p) and permdisp_p < FDR_ALPHA
    if perma and not disp:
        return "LOCATION_DIFFERENCE_SUPPORTED"
    if perma and disp:
        return "LOCATION_AND_DISPERSION_CONFOUNDED"
    if not perma and disp:
        return "DISPERSION_DIFFERENCE_ONLY"
    return "NO_SIGNIFICANT_DIFFERENCE"


def simper_table(
    matrix: np.ndarray,
    labels: Sequence[object],
    species_metadata: pd.DataFrame,
    group1: str,
    group2: str,
    scope: str,
) -> pd.DataFrame:
    labels_array = np.asarray(labels).astype(str)
    first = matrix[labels_array == str(group1)]
    second = matrix[labels_array == str(group2)]
    contributions = []
    for row1 in first:
        denominator = row1[None, :] + second
        pair_total = denominator.sum(axis=1)
        pair_contribution = np.divide(
            np.abs(row1[None, :] - second),
            pair_total[:, None],
            out=np.zeros_like(denominator),
            where=pair_total[:, None] > 0,
        )
        contributions.append(pair_contribution)
    stacked = np.vstack(contributions)
    mean_contribution = stacked.mean(axis=0)
    total = mean_contribution.sum()
    percent = mean_contribution / total * 100.0 if total > 0 else np.zeros_like(mean_contribution)
    output = species_metadata[["Species_Code", "Species_CN", "Species_Latin"]].copy()
    output["Scope"] = scope
    output["Contrast"] = f"{group1}_vs_{group2}"
    output["Group1"] = group1
    output["Group2"] = group2
    output["N_Group1"] = len(first)
    output["N_Group2"] = len(second)
    output["Average_Contribution"] = mean_contribution
    output["Percent_Contribution"] = percent
    output["Mean_Group1_SqrtBiomass"] = first.mean(axis=0)
    output["Mean_Group2_SqrtBiomass"] = second.mean(axis=0)
    output["Higher_Mean_Group"] = np.where(
        output["Mean_Group1_SqrtBiomass"] >= output["Mean_Group2_SqrtBiomass"],
        group1,
        group2,
    )
    output = output.sort_values("Average_Contribution", ascending=False, kind="stable")
    output["Rank"] = np.arange(1, len(output) + 1)
    output["Cumulative_Percent"] = output["Percent_Contribution"].cumsum()
    output["Included_Through_80Percent"] = (
        output["Cumulative_Percent"].shift(fill_value=0) < 80.0
    )
    output["Mean_BrayCurtis_BetweenGroups"] = total
    return output


def build_station_profiles(
    events: pd.DataFrame,
    biomass: pd.DataFrame,
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    species_codes = [column for column in biomass.columns if column != "EventID"]
    positive_meta = events.loc[
        events["Total_Biomass_kg_km2"].gt(0),
        ["EventID", "LocationID", "Year", "Month"],
    ]
    positive = positive_meta.merge(biomass, on="EventID", how="left", validate="one_to_one")
    positive[species_codes] = np.sqrt(positive[species_codes].to_numpy(float))

    scopes: dict[str, pd.DataFrame] = {
        "ALL_POSITIVE_EVENTS_STATION_MEAN": positive,
        "APRIL_POSITIVE_EVENTS_STATION_MEAN": positive.loc[positive["Month"].eq(4)],
        "NOVEMBER_POSITIVE_EVENTS_STATION_MEAN": positive.loc[positive["Month"].eq(11)],
    }
    metadata_parts: list[pd.DataFrame] = []
    matrices: dict[str, np.ndarray] = {}
    profile_rows: list[pd.DataFrame] = []
    for scope, subset in scopes.items():
        counts = subset.groupby("LocationID").size().rename("N_PositiveEvents")
        means = subset.groupby("LocationID", as_index=False)[species_codes].mean()
        means = means.merge(counts, on="LocationID", validate="one_to_one")
        means = means.merge(
            mapping[["LocationID", "Region", "Region_Order", "Lon", "Lat"]],
            on="LocationID",
            how="left",
            validate="one_to_one",
        )
        means.insert(0, "Scope", scope)
        means["Profile_Sum"] = means[species_codes].sum(axis=1)
        profile_rows.append(means.copy())
        matrices[scope] = means[species_codes].to_numpy(float)
        metadata_parts.append(
            means[
                [
                    "Scope", "LocationID", "Region", "Region_Order", "Lon", "Lat",
                    "N_PositiveEvents", "Profile_Sum",
                ]
            ].copy()
        )

    # A prespecified temporal-coverage sensitivity for the all-season profile.
    full = profile_rows[0]
    sensitive = full.loc[
        full["N_PositiveEvents"].ge(REGION_MIN_POSITIVE_EVENTS_SENSITIVITY)
    ].copy()
    sensitive["Scope"] = (
        f"ALL_STATION_MEAN_MIN{REGION_MIN_POSITIVE_EVENTS_SENSITIVITY}_POSITIVE_EVENTS"
    )
    scope_name = str(sensitive["Scope"].iloc[0])
    profile_rows.append(sensitive)
    matrices[scope_name] = sensitive[species_codes].to_numpy(float)
    metadata_parts.append(
        sensitive[
            [
                "Scope", "LocationID", "Region", "Region_Order", "Lon", "Lat",
                "N_PositiveEvents", "Profile_Sum",
            ]
        ].copy()
    )
    return pd.concat(profile_rows, ignore_index=True), matrices, pd.concat(metadata_parts, ignore_index=True)


def run_region_community_analysis(
    station_profiles: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    species_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    omnibus_records: list[dict[str, object]] = []
    dispersion_groups: list[pd.DataFrame] = []
    pairwise_records: list[dict[str, object]] = []
    simper_outputs: list[pd.DataFrame] = []

    for scope, matrix in matrices.items():
        metadata = station_profiles.loc[station_profiles["Scope"].eq(scope)].reset_index(drop=True)
        labels = metadata["Region"].astype(str).to_numpy()
        distance = bray_curtis_matrix(matrix)
        permanova = run_permanova(
            distance,
            labels,
            N_REGION_PERMUTATIONS,
            stable_seed("region_permanova", scope),
        )
        permdisp, group_summary = run_permdisp(
            distance,
            labels,
            N_REGION_PERMUTATIONS,
            stable_seed("region_permdisp", scope),
        )
        record: dict[str, object] = {"Scope": scope, "Status": "OK"}
        record.update(permanova)
        record.update(permdisp)
        record["Joint_Interpretation"] = joint_interpretation(
            float(record["PERMANOVA_P"]), float(record["PERMDISP_P"])
        )
        omnibus_records.append(record)
        group_summary.insert(0, "Scope", scope)
        dispersion_groups.append(group_summary)

        regions = (
            metadata[["Region", "Region_Order"]]
            .drop_duplicates()
            .sort_values("Region_Order")["Region"]
            .astype(str)
            .tolist()
        )
        local_pair_rows: list[dict[str, object]] = []
        for group1, group2 in combinations(regions, 2):
            keep = np.isin(labels, [group1, group2])
            submatrix = matrix[keep]
            sublabels = labels[keep]
            subdistance = bray_curtis_matrix(submatrix)
            perma_pair = run_permanova(
                subdistance,
                sublabels,
                N_REGION_PERMUTATIONS,
                stable_seed("pair_permanova", scope, group1, group2),
            )
            disp_pair, _ = run_permdisp(
                subdistance,
                sublabels,
                N_REGION_PERMUTATIONS,
                stable_seed("pair_permdisp", scope, group1, group2),
            )
            row: dict[str, object] = {
                "Scope": scope,
                "Contrast": f"{group1}_vs_{group2}",
                "Group1": group1,
                "Group2": group2,
            }
            row.update(perma_pair)
            row.update(disp_pair)
            local_pair_rows.append(row)
            simper_outputs.append(
                simper_table(
                    submatrix,
                    sublabels,
                    species_metadata,
                    group1,
                    group2,
                    scope,
                )
            )
        local = pd.DataFrame(local_pair_rows)
        local["PERMANOVA_P_FDR_within_scope"] = bh_adjust(local["PERMANOVA_P"])
        local["PERMDISP_P_FDR_within_scope"] = bh_adjust(local["PERMDISP_P"])
        local["PERMANOVA_FDR_Significant"] = local["PERMANOVA_P_FDR_within_scope"] < FDR_ALPHA
        local["PERMDISP_FDR_Significant"] = local["PERMDISP_P_FDR_within_scope"] < FDR_ALPHA
        local["Joint_Interpretation_FDR"] = [
            joint_interpretation(perma, disp)
            for perma, disp in zip(
                local["PERMANOVA_P_FDR_within_scope"],
                local["PERMDISP_P_FDR_within_scope"],
            )
        ]
        pairwise_records.extend(local.to_dict(orient="records"))

    omnibus = pd.DataFrame(omnibus_records)
    omnibus["PERMANOVA_P_FDR_across_scopes"] = bh_adjust(omnibus["PERMANOVA_P"])
    omnibus["PERMDISP_P_FDR_across_scopes"] = bh_adjust(omnibus["PERMDISP_P"])
    pairwise = pd.DataFrame(pairwise_records)
    dispersion = pd.concat(dispersion_groups, ignore_index=True)
    simper = pd.concat(simper_outputs, ignore_index=True)
    simper = simper.merge(
        pairwise[
            [
                "Scope", "Contrast", "PERMANOVA_P_FDR_within_scope",
                "PERMDISP_P_FDR_within_scope", "Joint_Interpretation_FDR",
            ]
        ],
        on=["Scope", "Contrast"],
        how="left",
        validate="many_to_one",
    )
    top80 = simper.loc[
        simper["Included_Through_80Percent"]
        & simper["PERMANOVA_P_FDR_within_scope"].lt(FDR_ALPHA)
    ].copy()
    return omnibus, pairwise, dispersion, simper, top80


# ============================================================================
# 5. Existing Step 5 evidence, summaries and figures
# ============================================================================

def extract_existing_step5_evidence(source: Path | None) -> pd.DataFrame:
    columns = [
        "Source", "Evidence_Type", "Test", "N", "Effect_Size_R2", "Statistic",
        "P_Value", "PERMDISP_P", "Result", "Reusable_For_Section3_3", "Caution",
    ]
    if source is None:
        return pd.DataFrame(
            [[
                "NOT_FOUND", "Availability", "Step5 optional source", np.nan,
                np.nan, np.nan, np.nan, np.nan, "NOT_AVAILABLE",
                "Run or attach Step5 for temporal beta-diversity evidence",
                "Spatial indicators in this script are unaffected",
            ]],
            columns=columns,
        )

    summary = optional_csv(source, "00_step5_summary.csv")
    permanova = optional_csv(source, "10_permanova_omnibus.csv")
    permdisp = optional_csv(source, "11_permdisp_omnibus.csv")
    records: list[dict[str, object]] = []
    if summary is not None and {"Metric", "Value"}.issubset(summary.columns):
        lookup = dict(zip(summary["Metric"].astype(str), summary["Value"]))
        records.append(
            {
                "Source": str(source),
                "Evidence_Type": "NMDS",
                "Test": "NMDS dimensional adequacy",
                "N": pd.to_numeric(lookup.get("Positive events analyzed"), errors="coerce"),
                "Effect_Size_R2": np.nan,
                "Statistic": pd.to_numeric(lookup.get("NMDS 2D Stress-1"), errors="coerce"),
                "P_Value": np.nan,
                "PERMDISP_P": np.nan,
                "Result": f"Recommended dimensions: {lookup.get('NMDS recommended dimensions', 'NA')}",
                "Reusable_For_Section3_3": "Ordination display of community heterogeneity",
                "Caution": str(lookup.get("NMDS recommendation status", "Check stress table")),
            }
        )
        records.append(
            {
                "Source": str(source),
                "Evidence_Type": "RegionAvailability",
                "Test": "Formal region analysis in supplied Step5",
                "N": np.nan,
                "Effect_Size_R2": np.nan,
                "Statistic": np.nan,
                "P_Value": np.nan,
                "PERMDISP_P": np.nan,
                "Result": str(lookup.get("Region analysis status", "UNKNOWN")),
                "Reusable_For_Section3_3": "Only reusable if status is READY/COMPLETED",
                "Caution": str(lookup.get("Region message", "")),
            }
        )

    if permanova is not None and permdisp is not None and "Test" in permanova.columns:
        dispersion_lookup = permdisp.set_index("Test") if "Test" in permdisp.columns else None
        for row in permanova.itertuples(index=False):
            test = str(row.Test)
            disp_p = np.nan
            if dispersion_lookup is not None and test in dispersion_lookup.index:
                disp_p = pd.to_numeric(
                    dispersion_lookup.loc[test, "P_Value"], errors="coerce"
                )
                if isinstance(disp_p, pd.Series):
                    disp_p = disp_p.iloc[0]
            records.append(
                {
                    "Source": str(source),
                    "Evidence_Type": "PERMANOVA_PERMDISP",
                    "Test": test,
                    "N": getattr(row, "N", np.nan),
                    "Effect_Size_R2": getattr(row, "R2", np.nan),
                    "Statistic": getattr(row, "PseudoF", np.nan),
                    "P_Value": getattr(row, "P_Value", np.nan),
                    "PERMDISP_P": disp_p,
                    "Result": joint_interpretation(
                        float(getattr(row, "P_Value", np.nan)), float(disp_p)
                    ),
                    "Reusable_For_Section3_3": (
                        "April-November composition difference" if "Month_" in test
                        else "Temporal community reorganization"
                    ),
                    "Caution": "Interpret PERMANOVA jointly with PERMDISP",
                }
            )
    return pd.DataFrame(records, columns=columns)


def region_resource_summary(events: pd.DataFrame) -> pd.DataFrame:
    return (
        events.groupby(["Year", "Month", "Season", "Region"], as_index=False)
        .agg(
            N_Surveyed=("EventID", "size"),
            N_Positive=("Cephalopod_Presence", "sum"),
            Positive_Fraction=("Cephalopod_Presence", "mean"),
            Mean_Biomass_AllSurveyed=("Total_Biomass_kg_km2", "mean"),
            Median_Biomass_AllSurveyed=("Total_Biomass_kg_km2", "median"),
            Total_Biomass_Index=("Total_Biomass_kg_km2", "sum"),
            Mean_Density_AllSurveyed=("Total_Density_1000ind_km2", "mean"),
            Total_Density_Index=("Total_Density_1000ind_km2", "sum"),
        )
        .sort_values(["Year", "Month", "Region"])
    )


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "pdf"):
        path = figure_dir / f"{stem}.{extension}"
        for attempt in range(2):
            fig.savefig(path, facecolor="white")
            if extension != "png":
                break
            try:
                with Image.open(path) as image:
                    image.verify()
                break
            except Exception:
                if attempt == 1:
                    raise
                path.unlink(missing_ok=True)
        paths.append(path)
    plt.close(fig)
    return paths


def add_region_background(ax: plt.Axes, mapping: pd.DataFrame) -> None:
    for region, group in mapping.groupby("Region"):
        ax.scatter(
            group["Lon"], group["Lat"], s=13,
            facecolor=REGION_COLORS.get(str(region), "#BDBDBD"),
            edgecolor="none", alpha=0.22, zorder=1,
        )
    for boundary in (28.80, 30.00):
        ax.axhline(boundary, color="#777777", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlim(mapping["Lon"].min() - 0.15, mapping["Lon"].max() + 0.15)
    ax.set_ylim(mapping["Lat"].min() - 0.15, mapping["Lat"].max() + 0.15)
    ax.set_xlabel("Longitude (degrees E)")
    ax.set_ylabel("Latitude (degrees N)")
    ax.set_aspect(1.0 / math.cos(math.radians(mapping["Lat"].mean())))
    ax.grid(color="#EEEEEE", linewidth=0.5, zorder=0)


def plot_centre_trajectories(
    metrics: pd.DataFrame, mapping: pd.DataFrame, output_dir: Path
) -> list[Path]:
    subset = metrics.loc[
        metrics["Panel"].eq("ALL_SURVEYED_2014_2024")
        & metrics["Weight_Basis"].eq("Biomass")
    ].copy()
    fig, ax = plt.subplots(figsize=(7.6, 8.2))
    add_region_background(ax, mapping)
    norm = Normalize(min(EXPECTED_YEARS), max(EXPECTED_YEARS))
    cmap = matplotlib.colormaps.get_cmap("viridis")
    for month in EXPECTED_MONTHS:
        group = subset.loc[subset["Month"].eq(month)].sort_values("Year")
        ax.plot(
            group["COG_Lon"], group["COG_Lat"],
            color=MONTH_COLORS[month], linewidth=1.8, alpha=0.85,
            label=f"{MONTH_LABELS[month]} trajectory", zorder=3,
        )
        ax.errorbar(
            group["COG_Lon"], group["COG_Lat"],
            xerr=np.vstack(
                [
                    group["COG_Lon"] - group["COG_Lon_CI95_Low"],
                    group["COG_Lon_CI95_High"] - group["COG_Lon"],
                ]
            ),
            yerr=np.vstack(
                [
                    group["COG_Lat"] - group["COG_Lat_CI95_Low"],
                    group["COG_Lat_CI95_High"] - group["COG_Lat"],
                ]
            ),
            fmt="none", ecolor=MONTH_COLORS[month], elinewidth=0.65,
            alpha=0.28, capsize=0, zorder=2,
        )
        ax.scatter(
            group["COG_Lon"], group["COG_Lat"],
            c=group["Year"], cmap=cmap, norm=norm,
            s=52, marker="o" if month == 4 else "s",
            edgecolor=MONTH_COLORS[month], linewidth=1.0, zorder=4,
        )
        for row in group.itertuples(index=False):
            if row.Year in (2014, 2024):
                ax.annotate(
                    str(row.Year), (row.COG_Lon, row.COG_Lat),
                    xytext=(4, 5 if month == 4 else -11),
                    textcoords="offset points", fontsize=8,
                )
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Year")
    ax.set_title("Biomass-weighted centre-of-gravity trajectories\nall surveyed events; bars are bootstrap marginal 95% CIs")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return save_figure(fig, output_dir, "Fig01_biomass_COG_trajectories")


def plot_extent_metrics(metrics: pd.DataFrame, output_dir: Path) -> list[Path]:
    subset = metrics.loc[
        metrics["Panel"].eq("ALL_SURVEYED_2014_2024")
        & metrics["Weight_Basis"].eq("Biomass")
    ].copy()
    specifications = [
        ("RMS_Radius_km", "Weighted RMS radius (km)"),
        ("Ellipse95_Area_km2", "95% ellipse area (km²)"),
        ("Positive_Fraction", "Positive-station fraction"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 9.0), sharex=True)
    for ax, (column, label) in zip(axes, specifications):
        for month in EXPECTED_MONTHS:
            group = subset.loc[subset["Month"].eq(month)].sort_values("Year")
            ax.plot(
                group["Year"], group[column], marker="o", linewidth=1.7,
                color=MONTH_COLORS[month], label=MONTH_LABELS[month],
            )
            low_column, high_column = f"{column}_CI95_Low", f"{column}_CI95_High"
            if low_column in group and high_column in group:
                ax.fill_between(
                    group["Year"], group[low_column], group[high_column],
                    color=MONTH_COLORS[month], alpha=0.14, linewidth=0,
                )
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    axes[0].set_title("Distributional extent and occupancy through time")
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("Year")
    axes[-1].set_xticks(EXPECTED_YEARS)
    fig.tight_layout()
    return save_figure(fig, output_dir, "Fig02_distribution_extent_timeseries")


def plot_paired_seasonal_shifts(
    seasonal: pd.DataFrame, mapping: pd.DataFrame, output_dir: Path
) -> list[Path]:
    group = seasonal.loc[seasonal["Weight_Basis"].eq("Biomass")].sort_values("Year")
    fig, ax = plt.subplots(figsize=(7.6, 8.2))
    add_region_background(ax, mapping)
    norm = Normalize(min(EXPECTED_YEARS), max(EXPECTED_YEARS))
    cmap = matplotlib.colormaps.get_cmap("plasma")
    for row in group.itertuples(index=False):
        color = cmap(norm(row.Year))
        ax.annotate(
            "",
            xy=(row.November_COG_Lon, row.November_COG_Lat),
            xytext=(row.April_COG_Lon, row.April_COG_Lat),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.6, alpha=0.85),
            zorder=3,
        )
        ax.scatter(row.April_COG_Lon, row.April_COG_Lat, marker="o", s=25, color=color, zorder=4)
        ax.scatter(row.November_COG_Lon, row.November_COG_Lat, marker="s", s=25, color=color, zorder=4)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Year")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555", label="April COG"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="#555555", label="November COG"),
        ],
        frameon=False,
    )
    ax.set_title("April-to-November biomass COG shifts\ncomputed on shared stations within each year")
    fig.tight_layout()
    return save_figure(fig, output_dir, "Fig03_paired_seasonal_COG_shifts")


def plot_persistence_map(
    persistence: pd.DataFrame,
    mapping: pd.DataFrame,
    output_dir: Path,
    measure: str,
    value_column: str,
    filename: str,
    title: str,
) -> list[Path]:
    subset = persistence.loc[
        persistence["Measure"].eq(measure) & persistence["Coverage_Adequate"]
    ].copy()
    if "Quantile" in subset.columns:
        subset = subset.loc[np.isclose(subset["Quantile"], PRIMARY_HOTSPOT_QUANTILE)]
    fig, axes = plt.subplots(
        1, 2, figsize=(12.0, 6.5), sharex=True, sharey=True,
        layout="constrained",
    )
    scatter = None
    for ax, month in zip(axes, EXPECTED_MONTHS):
        add_region_background(ax, mapping)
        group = subset.loc[subset["Month"].eq(month)]
        scatter = ax.scatter(
            group["Lon"], group["Lat"], c=group[value_column],
            cmap="magma", vmin=0, vmax=1,
            s=22 + 5 * group["N_Surveyed_Periods"],
            edgecolor=np.where(
                group["Persistence_Class"].eq("PERSISTENT_GE50PCT"),
                "#00FFFF", "#222222",
            ),
            linewidth=np.where(
                group["Persistence_Class"].eq("PERSISTENT_GE50PCT"), 1.7, 0.35
            ),
            zorder=4,
        )
        ax.set_title(MONTH_LABELS[month])
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes, fraction=0.025, pad=0.03)
        cbar.set_label("Hotspot persistence frequency")
    fig.suptitle(title + "\ncyan outline: frequency >= 0.50; size: number of surveyed years")
    return save_figure(fig, output_dir, filename)


def plot_region_effects(
    omnibus: pd.DataFrame, output_dir: Path
) -> list[Path]:
    frame = omnibus.copy()
    labels = (
        frame["Scope"]
        .str.replace("_POSITIVE_EVENTS_STATION_MEAN", "", regex=False)
        .str.replace("ALL_STATION_MEAN_", "Sensitivity: ", regex=False)
        .str.replace("_", " ", regex=False)
    )
    colors = np.where(frame["PERMANOVA_P_FDR_across_scopes"].lt(FDR_ALPHA), "#4C78A8", "#BDBDBD")
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    x = np.arange(len(frame))
    bars = ax.bar(x, frame["PERMANOVA_R2"], color=colors, edgecolor="#222222", linewidth=0.7)
    for bar, p_disp in zip(bars, frame["PERMDISP_P_FDR_across_scopes"]):
        if p_disp < FDR_ALPHA:
            bar.set_hatch("//")
    for i, row in frame.reset_index(drop=True).iterrows():
        ax.text(
            i, row["PERMANOVA_R2"] + 0.004,
            f"R²={row['PERMANOVA_R2']:.3f}\np={row['PERMANOVA_P']:.3g}",
            ha="center", va="bottom", fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Regional PERMANOVA R²")
    ax.set_title("Regional community heterogeneity at independent station level\nhatching indicates significant PERMDISP after across-scope FDR")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    upper = max(0.05, float(frame["PERMANOVA_R2"].max()) * 1.35)
    ax.set_ylim(0, upper)
    fig.tight_layout()
    return save_figure(fig, output_dir, "Fig06_region_PERMANOVA_PERMDISP")


def plot_region_simper(top80: pd.DataFrame, output_dir: Path) -> list[Path]:
    scope = "ALL_POSITIVE_EVENTS_STATION_MEAN"
    frame = top80.loc[top80["Scope"].eq(scope)].copy()
    contrasts = frame["Contrast"].drop_duplicates().tolist()
    if not contrasts:
        return []
    fig, axes = plt.subplots(
        1, len(contrasts), figsize=(5.3 * len(contrasts), 7.2), squeeze=False
    )
    for ax, contrast in zip(axes.ravel(), contrasts):
        group = frame.loc[frame["Contrast"].eq(contrast)].sort_values(
            "Percent_Contribution", ascending=True
        )
        colors = [REGION_COLORS.get(str(value), "#777777") for value in group["Higher_Mean_Group"]]
        y = np.arange(len(group))
        ax.barh(y, group["Percent_Contribution"], color=colors, edgecolor="#222222", linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(group["Species_Latin"], fontstyle="italic", fontsize=8)
        ax.set_xlabel("SIMPER contribution (%)")
        ax.set_title(contrast.replace("_", " "))
        ax.grid(axis="x", color="#E5E5E5", linewidth=0.5)
    fig.suptitle("Species contributing through cumulative 80% of regional Bray-Curtis dissimilarity", y=1.01)
    used_regions = list(dict.fromkeys(frame["Higher_Mean_Group"].astype(str)))
    fig.legend(
        handles=[
            Patch(
                facecolor=REGION_COLORS.get(region, "#777777"),
                edgecolor="#222222",
                label=f"Higher mean in {region}",
            )
            for region in used_regions
        ],
        loc="lower center",
        ncol=max(1, len(used_regions)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
    )
    fig.tight_layout()
    return save_figure(fig, output_dir, "Fig07_region_SIMPER_top80")


def build_qc_table(
    events: pd.DataFrame,
    biomass: pd.DataFrame,
    density: pd.DataFrame,
    mapping: pd.DataFrame,
    spatial_metrics: pd.DataFrame,
    top_flags: pd.DataFrame,
    gistar: pd.DataFrame,
) -> pd.DataFrame:
    species_columns = [column for column in biomass if column != "EventID"]
    biomass_sum = biomass[species_columns].sum(axis=1).to_numpy(float)
    density_columns = [column for column in density if column != "EventID"]
    density_sum = density[density_columns].sum(axis=1).to_numpy(float)
    zero_index = events["Total_Biomass_kg_km2"].eq(0).to_numpy()
    checks = [
        ("Surveyed event count", EXPECTED_SURVEYED, len(events), len(events) == EXPECTED_SURVEYED, "FATAL"),
        ("Unique EventID", len(events), events["EventID"].nunique(), events["EventID"].is_unique, "FATAL"),
        ("Positive event count", EXPECTED_POSITIVE, int(events["Cephalopod_Presence"].sum()), int(events["Cephalopod_Presence"].sum()) == EXPECTED_POSITIVE, "FATAL"),
        ("Species count", EXPECTED_SPECIES, len(species_columns), len(species_columns) == EXPECTED_SPECIES, "FATAL"),
        ("Biomass matrix EventID alignment", True, events["EventID"].equals(biomass["EventID"]), events["EventID"].equals(biomass["EventID"]), "FATAL"),
        ("Density matrix EventID alignment", True, events["EventID"].equals(density["EventID"]), events["EventID"].equals(density["EventID"]), "FATAL"),
        ("Species biomass sums match event total", True, bool(np.allclose(biomass_sum, events["Total_Biomass_kg_km2"], rtol=1e-8, atol=1e-8)), bool(np.allclose(biomass_sum, events["Total_Biomass_kg_km2"], rtol=1e-8, atol=1e-8)), "FATAL"),
        ("Species density sums match event total", True, bool(np.allclose(density_sum, events["Total_Density_1000ind_km2"], rtol=1e-8, atol=1e-8)), bool(np.allclose(density_sum, events["Total_Density_1000ind_km2"], rtol=1e-8, atol=1e-8)), "FATAL"),
        ("Zero-catch biomass rows all zero", True, bool(np.allclose(biomass_sum[zero_index], 0)), bool(np.allclose(biomass_sum[zero_index], 0)), "FATAL"),
        ("Formal LocationID mapping count", 124, mapping["LocationID"].nunique(), mapping["LocationID"].nunique() == 124, "FATAL"),
        ("Every event receives Region", 0, int(events["Region"].isna().sum()), not events["Region"].isna().any(), "FATAL"),
        ("Exactly 22 Year-Month periods", 22, events[["Year", "Month"]].drop_duplicates().shape[0], events[["Year", "Month"]].drop_duplicates().shape[0] == 22, "FATAL"),
        ("All-surveyed biomass spatial metrics complete", 22, len(spatial_metrics.loc[spatial_metrics["Panel"].eq("ALL_SURVEYED_2014_2024") & spatial_metrics["Weight_Basis"].eq("Biomass")]), len(spatial_metrics.loc[spatial_metrics["Panel"].eq("ALL_SURVEYED_2014_2024") & spatial_metrics["Weight_Basis"].eq("Biomass")]) == 22, "FATAL"),
        ("Relative hotspot flags cover all events at primary biomass threshold", len(events), len(top_flags.loc[top_flags["Measure"].eq("Biomass") & np.isclose(top_flags["Quantile"], PRIMARY_HOTSPOT_QUANTILE)]), len(top_flags.loc[top_flags["Measure"].eq("Biomass") & np.isclose(top_flags["Quantile"], PRIMARY_HOTSPOT_QUANTILE)]) == len(events), "FATAL"),
        ("GiStar covers all events for biomass", len(events), len(gistar.loc[gistar["Measure"].eq("Biomass")]), len(gistar.loc[gistar["Measure"].eq("Biomass")]) == len(events), "FATAL"),
        (
            "Recent complete panel has >=20 stations per month",
            ">=20",
            int(
                spatial_metrics.loc[
                    spatial_metrics["Panel"].str.startswith("COMPLETE_")
                ]
                .groupby("Month")["N_Surveyed"]
                .min()
                .min()
            ),
            bool(
                (
                    spatial_metrics.loc[
                        spatial_metrics["Panel"].str.startswith("COMPLETE_")
                    ]
                    .groupby("Month")["N_Surveyed"]
                    .min()
                    >= 20
                ).all()
            ),
            "WARNING",
        ),
    ]
    return pd.DataFrame(checks, columns=["Check", "Expected", "Observed", "Pass", "Severity"])


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    minimum = 1.0 / (N_REGION_PERMUTATIONS + 1)
    return f"<{minimum * 1.01:.3g}" if value <= minimum + 1e-15 else f"{value:.3g}"


def make_readme(
    step3_source: Path,
    region_source: Path,
    step5_source: Path | None,
    metrics: pd.DataFrame,
    net_shifts: pd.DataFrame,
    seasonal_tests: pd.DataFrame,
    hotspot_primary: pd.DataFrame,
    gistar_persistence: pd.DataFrame,
    region_omnibus: pd.DataFrame,
    region_pairwise: pd.DataFrame,
    qc: pd.DataFrame,
) -> str:
    main_net = net_shifts.loc[
        net_shifts["Panel"].eq("ALL_SURVEYED_2014_2024")
        & net_shifts["Weight_Basis"].eq("Biomass")
    ]
    net_lines = [
        f"- {row.Season}: {row.Start_Year}-{row.End_Year} net shift "
        f"{row.Net_Shift_km:.1f} km toward {row.Direction} "
        f"(bearing {row.Bearing_deg:.1f} degrees)."
        for row in main_net.itertuples(index=False)
    ]
    overall_region = region_omnibus.loc[
        region_omnibus["Scope"].eq("ALL_POSITIVE_EVENTS_STATION_MEAN")
    ].iloc[0]
    pair_lines = [
        f"- {row.Contrast}: R2={row.PERMANOVA_R2:.4f}, "
        f"PERMANOVA q={format_p(row.PERMANOVA_P_FDR_within_scope)}, "
        f"PERMDISP q={format_p(row.PERMDISP_P_FDR_within_scope)}, "
        f"{row.Joint_Interpretation_FDR}."
        for row in region_pairwise.loc[
            region_pairwise["Scope"].eq("ALL_POSITIVE_EVENTS_STATION_MEAN")
        ].itertuples(index=False)
    ]
    persistent = hotspot_primary.loc[
        hotspot_primary["Persistence_Class"].eq("PERSISTENT_GE50PCT")
    ]
    hotspot_lines = []
    for month in EXPECTED_MONTHS:
        names = persistent.loc[persistent["Month"].eq(month), "LocationID"].astype(str).tolist()
        hotspot_lines.append(
            f"- {MONTH_LABELS[month]} top-decile persistent locations "
            f"(>=50%, >= {PERSISTENCE_MIN_SURVEYS} surveyed years): "
            + (", ".join(names) if names else "none")
            + "."
        )
    gi_count = int(
        gistar_persistence.loc[
            gistar_persistence["Measure"].eq("Biomass")
            & gistar_persistence["Persistence_Class"].eq("PERSISTENT_GE50PCT")
        ].shape[0]
    )
    seasonal_lines = [
        f"- {row.Weight_Basis} {row.Metric}: mean November-April difference "
        f"{row.Mean_Difference:.4g}; exact paired sign-flip p="
        f"{format_p(row.Exact_SignFlip_P)}, FDR q={format_p(row.Exact_SignFlip_P_FDR)}."
        for row in seasonal_tests.itertuples(index=False)
    ]
    return f"""Section 3.3 — Spatial shifts and persistent hotspots
=======================================================

INPUTS
------
- Step 3: {step3_source}
- Formal region mapping: {region_source}
- Optional Step 5 evidence: {step5_source if step5_source else 'not supplied'}

ESTIMANDS AND ANALYSIS UNITS
----------------------------
1. Centre, extent, occupancy and hotspots use all {len(metrics.loc[metrics['Panel'].eq('ALL_SURVEYED_2014_2024') & metrics['Weight_Basis'].eq('Biomass')])} Year-Month summaries derived from all 2,100 surveyed events. Surveyed zero catches are retained as zero.
2. Community spatial heterogeneity uses all 25 species and square-root biomass from positive events, averaged first within LocationID. The regional test therefore uses independent station profiles, not 1,532 repeated events as independent spatial replicates.
3. The main 2014-2024 spatial series uses all stations sampled in each period. A complete {RECENT_PANEL_START}-2024 within-month panel and within-year shared-station April-November analysis diagnose sampling-footprint sensitivity.
4. Centre of gravity is resource weighted. RMS radius and the 95% weighted standard ellipse describe spread. Convex-hull area is descriptive and should not be the sole expansion metric.

REGIONAL COMMUNITY HETEROGENEITY
--------------------------------
- Overall station-level PERMANOVA: pseudo-F={overall_region.PERMANOVA_PseudoF:.4f}, R2={overall_region.PERMANOVA_R2:.4f}, p={format_p(overall_region.PERMANOVA_P)}.
- Overall PERMDISP: F={overall_region.PERMDISP_F:.4f}, p={format_p(overall_region.PERMDISP_P)}.
- Joint interpretation: {overall_region.Joint_Interpretation}.
{chr(10).join(pair_lines)}

CENTRE SHIFTS
-------------
{chr(10).join(net_lines)}

PAIRED APRIL-NOVEMBER EXTENT TESTS
----------------------------------
{chr(10).join(seasonal_lines)}

PERSISTENT HIGH-RESOURCE LOCATIONS
----------------------------------
The primary recurrence definition is biomass >= the within-Year-Month {PRIMARY_HOTSPOT_QUANTILE:.0%} quantile, positive biomass, at least {PERSISTENCE_MIN_SURVEYS} sampled years, and recurrence >=50%.
{chr(10).join(hotspot_lines)}
- Persistent permutation Gi* biomass clusters under the same recurrence rule: {gi_count} LocationID-Month combinations.
- A top-decile location is a recurrent high-resource station, whereas Gi* is the formal local high-value clustering result. Do not call a top-decile station a statistically significant spatial cluster unless Gi* also supports it.

INTERPRETATION RULES
--------------------
- If PERMANOVA and PERMDISP are both significant, report that the regional difference may contain both centroid/composition and within-region heterogeneity components.
- Report continuous changes in RMS radius/ellipse area. “Expansion” means an increase and “contraction” a decrease; significance comes from the trend or paired test, not from the direction label alone.
- Compare the full series with the complete recent panel before attributing a shift to ecology, because early November sampling coverage was smaller.
- Persistent hotspots are conditional on survey coverage; sites with fewer than {PERSISTENCE_MIN_SURVEYS} observations are marked INSUFFICIENT_COVERAGE rather than treated as non-hotspots.

METHOD REFERENCES
-----------------
- Woillez, Rivoirard & Petitgas (2009), survey-based spatial indicators, DOI: 10.1051/alr/2009017.
- Anderson (2006), PERMDISP, DOI: 10.1111/j.1541-0420.2005.00440.x.
- Getis & Ord (1992), local G statistics, DOI: 10.1111/j.1538-4632.1992.tb00261.x.

QUALITY CONTROL
---------------
- Fatal checks passed: {bool(qc.loc[qc['Severity'].eq('FATAL'), 'Pass'].all())}.
- Warning checks failed: {int((qc['Severity'].eq('WARNING') & ~qc['Pass']).sum())}.
- See 25_qc_checks.csv for every check.
"""


# ============================================================================
# 6. Main workflow
# ============================================================================

def main() -> None:
    started = time.time()
    setup_plot_style()
    progress("Locating Step 3 and formal region mapping inputs...")
    step3_source = discover_required_source(
        STEP3_ENV,
        ["cephalopod_step3_analysis_master.zip"],
        "02_analysis_master_v1_surveyed_all25.csv",
    )
    region_source = discover_required_source(
        REGION_ENV,
        ["cephalopod_region_mapping_outputs.zip", "00_cephalopod_region_mapping.csv"],
        "00_cephalopod_region_mapping.csv",
    )
    step5_source = discover_optional_step5()
    progress(f"Step 3: {step3_source}")
    progress(f"Region mapping: {region_source}")
    progress(f"Optional Step 5: {step5_source if step5_source else 'not found'}")

    events = read_csv_from_source(
        step3_source, "02_analysis_master_v1_surveyed_all25.csv"
    )
    biomass = read_csv_from_source(
        step3_source, "08_Y_biomass_all25_surveyed.csv"
    )
    density = read_csv_from_source(
        step3_source, "09_Y_density_all25_surveyed.csv"
    )
    species_metadata = read_csv_from_source(
        step3_source, "06_taxonomy_analysis_dictionary.csv"
    )
    mapping = read_csv_from_source(
        region_source, "00_cephalopod_region_mapping.csv"
    )

    require_columns(
        events,
        [
            "EventID", "Year", "Month", "LocationID", "Lon", "Lat", "SurveyStatus",
            "Total_Biomass_kg_km2", "Total_Density_1000ind_km2",
        ],
        "surveyed master",
    )
    require_columns(
        mapping,
        ["LocationID", "Region", "Region_Order", "Lon", "Lat"],
        "formal region mapping",
    )
    require_columns(
        species_metadata,
        ["Species_Code", "Species_CN", "Species_Latin"],
        "species metadata",
    )

    # Normalize and validate core tables before any analysis.
    events = events.copy()
    events["Year"] = pd.to_numeric(events["Year"], errors="raise").astype(int)
    events["Month"] = pd.to_numeric(events["Month"], errors="raise").astype(int)
    events["Lon"] = pd.to_numeric(events["Lon"], errors="raise")
    events["Lat"] = pd.to_numeric(events["Lat"], errors="raise")
    events["Total_Biomass_kg_km2"] = pd.to_numeric(
        events["Total_Biomass_kg_km2"], errors="raise"
    )
    events["Total_Density_1000ind_km2"] = pd.to_numeric(
        events["Total_Density_1000ind_km2"], errors="raise"
    )
    events["Season"] = events["Month"].map(MONTH_LABELS)
    events["Cephalopod_Presence"] = events["Total_Biomass_kg_km2"].gt(0)
    mapping = mapping.drop_duplicates("LocationID").copy()
    mapping = mapping.sort_values(["Region_Order", "LocationID"]).reset_index(drop=True)

    if not events["EventID"].is_unique:
        raise AssertionError("Surveyed master contains duplicate EventID values")
    if set(events["Year"].unique()) != set(EXPECTED_YEARS):
        raise AssertionError(f"Unexpected years: {sorted(events['Year'].unique())}")
    if set(events["Month"].unique()) != set(EXPECTED_MONTHS):
        raise AssertionError(f"Unexpected months: {sorted(events['Month'].unique())}")
    if (events[["Total_Biomass_kg_km2", "Total_Density_1000ind_km2"]].to_numpy() < 0).any():
        raise AssertionError("Negative biomass or density detected")

    mapping_for_merge = mapping.rename(columns={"Lon": "Mapping_Lon", "Lat": "Mapping_Lat"})
    events = events.merge(
        mapping_for_merge[
            [
                "LocationID", "Region", "Region_Order", "Region_Definition",
                "Mapping_Lon", "Mapping_Lat",
            ]
        ],
        on="LocationID",
        how="left",
        validate="many_to_one",
    )
    if events["Region"].isna().any():
        unknown = sorted(events.loc[events["Region"].isna(), "LocationID"].unique())
        raise AssertionError(f"Formal region mapping misses LocationID values: {unknown}")
    coordinate_difference = np.maximum(
        np.abs(events["Lon"] - events["Mapping_Lon"]),
        np.abs(events["Lat"] - events["Mapping_Lat"]),
    )
    if coordinate_difference.max() > 1e-5:
        raise AssertionError(
            f"Event versus formal mapping coordinate mismatch; maximum={coordinate_difference.max()} degrees"
        )

    # Exact EventID order is mandatory because spatial weights and species data
    # are joined by row after this point.
    if not events["EventID"].equals(biomass["EventID"]):
        raise AssertionError("Biomass matrix EventID order does not match master")
    if not events["EventID"].equals(density["EventID"]):
        raise AssertionError("Density matrix EventID order does not match master")
    species_codes = [column for column in biomass.columns if column != "EventID"]
    if species_codes != [column for column in density.columns if column != "EventID"]:
        raise AssertionError("Biomass and density species columns differ")
    if len(species_codes) != EXPECTED_SPECIES:
        raise AssertionError(f"Expected 25 species but found {len(species_codes)}")
    species_metadata = (
        species_metadata.set_index("Species_Code").reindex(species_codes).reset_index()
    )
    if species_metadata["Species_Latin"].isna().any():
        raise AssertionError("Species dictionary does not cover all matrix columns")

    projection = LocalProjection(
        reference_lon=float(mapping["Lon"].mean()),
        reference_lat=float(mapping["Lat"].mean()),
    )

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    progress("Calculating COG, weighted extent, occupancy and sampling-panel sensitivities...")
    spatial_metrics, panel_metadata = calculate_all_spatial_metrics(events, projection)
    consecutive_shifts = build_centre_shifts(spatial_metrics)
    net_shifts = build_net_centre_shifts(spatial_metrics)
    seasonal_shifts = build_paired_seasonal_shifts(spatial_metrics)
    spatial_trends = trend_summary(spatial_metrics)
    seasonal_tests = paired_season_metric_tests(seasonal_shifts)

    progress("Flagging recurrent high-resource stations...")
    top_flags, hotspot_thresholds, hotspot_persistence = (
        flag_relative_high_resource_events(events)
    )
    hotspot_primary = hotspot_persistence.loc[
        hotspot_persistence["Measure"].eq("Biomass")
        & np.isclose(hotspot_persistence["Quantile"], PRIMARY_HOTSPOT_QUANTILE)
    ].copy()

    progress(
        f"Running permutation Gi* ({N_GISTAR_PERMUTATIONS} permutations per period and measure)..."
    )
    gistar_events, gistar_persistence = calculate_gistar(events, projection)

    progress("Running independent-station regional PERMANOVA, PERMDISP and SIMPER...")
    station_profiles, profile_matrices, station_profile_metadata = build_station_profiles(
        events, biomass, mapping
    )
    (
        region_omnibus,
        region_pairwise,
        region_dispersion,
        region_simper,
        region_simper_top80,
    ) = run_region_community_analysis(
        station_profiles,
        profile_matrices,
        species_metadata,
    )
    regional_resources = region_resource_summary(events)
    existing_step5 = extract_existing_step5_evidence(step5_source)

    qc = build_qc_table(
        events, biomass, density, mapping, spatial_metrics, top_flags, gistar_events
    )
    fatal_failures = qc.loc[qc["Severity"].eq("FATAL") & ~qc["Pass"]]
    if len(fatal_failures):
        raise AssertionError(
            "Fatal QC failures:\n" + fatal_failures.to_string(index=False)
        )

    progress("Rendering publication and diagnostic figures...")
    figure_paths: list[Path] = []
    figure_paths.extend(plot_centre_trajectories(spatial_metrics, mapping, OUTPUT_DIR))
    figure_paths.extend(plot_extent_metrics(spatial_metrics, OUTPUT_DIR))
    figure_paths.extend(plot_paired_seasonal_shifts(seasonal_shifts, mapping, OUTPUT_DIR))
    figure_paths.extend(
        plot_persistence_map(
            hotspot_primary,
            mapping,
            OUTPUT_DIR,
            "Biomass",
            "Persistence_Frequency",
            "Fig04_topdecile_biomass_persistence",
            "Persistent within-period top-decile biomass locations",
        )
    )
    figure_paths.extend(
        plot_persistence_map(
            gistar_persistence,
            mapping,
            OUTPUT_DIR,
            "Biomass",
            "GiStar_Hotspot_Frequency",
            "Fig05_GiStar_biomass_hotspot_persistence",
            "Persistent permutation Gi* high-biomass clusters",
        )
    )
    figure_paths.extend(plot_region_effects(region_omnibus, OUTPUT_DIR))
    figure_paths.extend(plot_region_simper(region_simper_top80, OUTPUT_DIR))

    elapsed_minutes = (time.time() - started) / 60.0
    main_net = net_shifts.loc[
        net_shifts["Panel"].eq("ALL_SURVEYED_2014_2024")
        & net_shifts["Weight_Basis"].eq("Biomass")
    ]
    overall_region = region_omnibus.loc[
        region_omnibus["Scope"].eq("ALL_POSITIVE_EVENTS_STATION_MEAN")
    ].iloc[0]
    summary = pd.DataFrame(
        [
            ("Section", "3.3 Spatial shifts and persistent hotspots"),
            ("Surveyed events used for resource spatial metrics", len(events)),
            ("Positive events", int(events["Cephalopod_Presence"].sum())),
            ("Surveyed zero-catch events retained", int((~events["Cephalopod_Presence"]).sum())),
            ("Species in regional community analysis", len(species_codes)),
            ("Formal LocationIDs", mapping["LocationID"].nunique()),
            ("Formal regions", mapping["Region"].nunique()),
            ("Year-Month periods", events[["Year", "Month"]].drop_duplicates().shape[0]),
            ("Biomass COG April net shift km", float(main_net.loc[main_net["Month"].eq(4), "Net_Shift_km"].iloc[0])),
            ("Biomass COG November net shift km", float(main_net.loc[main_net["Month"].eq(11), "Net_Shift_km"].iloc[0])),
            ("Top-decile persistent biomass LocationID-Month combinations", int(hotspot_primary["Persistence_Class"].eq("PERSISTENT_GE50PCT").sum())),
            ("Persistent GiStar biomass LocationID-Month combinations", int(gistar_persistence.loc[gistar_persistence["Measure"].eq("Biomass"), "Persistence_Class"].eq("PERSISTENT_GE50PCT").sum())),
            ("Overall regional PERMANOVA R2", float(overall_region["PERMANOVA_R2"])),
            ("Overall regional PERMANOVA p", float(overall_region["PERMANOVA_P"])),
            ("Overall regional PERMDISP p", float(overall_region["PERMDISP_P"])),
            ("Overall regional joint interpretation", overall_region["Joint_Interpretation"]),
            ("Bootstrap replicates per spatial summary", N_BOOTSTRAP),
            ("Regional permutations per test", N_REGION_PERMUTATIONS),
            ("GiStar permutations per period and measure", N_GISTAR_PERMUTATIONS),
            ("Random seed", RANDOM_SEED),
            ("Elapsed minutes", elapsed_minutes),
            ("Fatal QC passed", True),
        ],
        columns=["Metric", "Value"],
    )

    inclusion_rules = pd.DataFrame(
        [
            ("Spatial resource indicators", "2,100 surveyed events", "Keep surveyed zeros as 0", "Biomass main; density sensitivity"),
            ("Main time series", "All surveyed stations per Year-Month", "2014-2024", "Sampling footprint can vary; compare sensitivities"),
            ("Recent complete panel", f"Stations surveyed every year within month, {RECENT_PANEL_START}-2024", "Month-specific fixed station set", "Sampling-design sensitivity"),
            ("Seasonal COG/extent", "Stations surveyed in both April and November within each year", "Same sites within each annual comparison", "Paired spatial comparison"),
            ("Regional community heterogeneity", "Positive events, 25 species, station-level mean sqrt biomass", "One independent LocationID per row", "Conditional-on-positive composition"),
            ("High-resource persistence", f"Within-period >= {PRIMARY_HOTSPOT_QUANTILE:.0%} biomass quantile and >0", f">= {PERSISTENCE_MIN_SURVEYS} surveyed periods", "Persistent if recurrence >=50%"),
            ("Formal hotspot", "Permutation Getis-Ord Gi* on log1p resource", f"Self + {GISTAR_K} nearest sites; one-sided; within-period FDR", "Spatial clustering inference"),
        ],
        columns=["Analysis", "Sample", "Rule", "Interpretation"],
    )

    method_metadata = pd.DataFrame(
        [
            ("Projection", f"Local equirectangular km coordinates centered at {projection.lon0:.6f}E, {projection.lat0:.6f}N", "Used only for short-range distances/covariance within study domain"),
            ("COG", "Resource-weighted mean x and y", "Location indicator"),
            ("RMS radius", "sqrt(weighted mean squared distance to COG)", "Primary continuous spreading indicator"),
            ("Standard ellipse", f"Weighted covariance ellipse at {ELLIPSE_LEVEL:.0%}", "Area and axes describe extent/orientation"),
            ("Convex hull", "Hull of positive surveyed stations", "Descriptive; sensitive to sampling edge and outliers"),
            ("COG uncertainty", f"Station bootstrap, {N_BOOTSTRAP} replicates", "Marginal percentile 95% CIs"),
            ("Temporal trends", "Theil-Sen slope plus Kendall tau", "FDR within panel and weight basis"),
            ("Seasonal extent test", "Exact paired sign-flip test across 11 annual shared-station contrasts", "FDR across reported metrics"),
            ("Regional composition", "Bray-Curtis on station mean sqrt biomass", "PERMANOVA interpreted jointly with PERMDISP"),
            ("PERMDISP", "Lingoes-corrected PCoA distances to group centroid", "Anderson 2006"),
            ("SIMPER", "All cross-region independent-station pairs", "Descriptive contribution, not species-level hypothesis test"),
            ("Relative high-resource site", f"Within-Year-Month top {(1-PRIMARY_HOTSPOT_QUANTILE):.0%}", "Recurrence measure; not a formal spatial cluster test"),
            ("GiStar", f"Local randomization on log1p values, k={GISTAR_K}, {N_GISTAR_PERMUTATIONS} permutations", "One-sided high cluster; BH-FDR within period"),
        ],
        columns=["Method", "Specification", "Note"],
    )

    output_tables: OrderedDict[str, pd.DataFrame] = OrderedDict(
        [
            ("00_section33_summary.csv", summary),
            ("01_analysis_inclusion_rules.csv", inclusion_rules),
            ("02_station_region_mapping_used.csv", mapping),
            ("03_sampling_panel_metadata.csv", panel_metadata),
            ("04_spatial_metrics_year_month.csv", spatial_metrics),
            ("05_COG_consecutive_year_shifts.csv", consecutive_shifts),
            ("06_COG_net_period_shifts.csv", net_shifts),
            ("07_paired_April_November_COG_shifts.csv", seasonal_shifts),
            ("08_spatial_metric_temporal_trends.csv", spatial_trends),
            ("08b_unified_trend_sensitivity.csv", spatial_trends),
            ("09_paired_seasonal_extent_tests.csv", seasonal_tests),
            ("10_high_resource_period_thresholds.csv", hotspot_thresholds),
            ("11_high_resource_event_flags_all_thresholds.csv", top_flags),
            ("12_high_resource_persistence_all_thresholds.csv", hotspot_persistence),
            ("13_high_resource_persistence_primary_biomass.csv", hotspot_primary),
            ("14_GiStar_event_results.csv", gistar_events),
            ("15_GiStar_hotspot_persistence.csv", gistar_persistence),
            ("16_region_resource_summary_year_month.csv", regional_resources),
            ("17_region_station_community_profiles.csv", station_profiles),
            ("18_region_PERMANOVA_PERMDISP_omnibus.csv", region_omnibus),
            ("19_region_pairwise_PERMANOVA_PERMDISP.csv", region_pairwise),
            ("20_region_PERMDISP_group_distances.csv", region_dispersion),
            ("21_region_SIMPER_all_species.csv", region_simper),
            ("22_region_SIMPER_top80.csv", region_simper_top80),
            ("23_existing_step5_evidence_reusable.csv", existing_step5),
            ("24_method_metadata.csv", method_metadata),
            ("25_qc_checks.csv", qc),
        ]
    )

    output_paths: list[Path] = []
    for filename, table in output_tables.items():
        path = OUTPUT_DIR / filename
        table.to_csv(path, index=False, encoding="utf-8-sig")
        output_paths.append(path)

    readme = make_readme(
        step3_source,
        region_source,
        step5_source,
        spatial_metrics,
        net_shifts,
        seasonal_tests,
        hotspot_primary,
        gistar_persistence,
        region_omnibus,
        region_pairwise,
        qc,
    )
    readme_path = OUTPUT_DIR / "README_section3_3.txt"
    readme_path.write_text("REVISED INTERPRETATION: annual trend tests are exploratory. Compare full-period, same-window all-available, and fixed-location panels in 08b_unified_trend_sensitivity.csv before making a trend claim. Coverage alone is not an established explanation for differences.\n\n" + readme, encoding="utf-8")
    output_paths.append(readme_path)
    output_paths.extend(figure_paths)

    manifest_rows = []
    for filename, table in output_tables.items():
        manifest_rows.append(
            {
                "File": filename,
                "Rows": len(table),
                "Columns": len(table.columns),
                "Purpose": "Section 3.3 tabular output",
            }
        )
    for path in figure_paths:
        manifest_rows.append(
            {
                "File": str(path.relative_to(OUTPUT_DIR)),
                "Rows": pd.NA,
                "Columns": pd.NA,
                "Purpose": "Section 3.3 publication/QC figure",
            }
        )
    manifest_rows.append(
        {
            "File": "README_section3_3.txt",
            "Rows": pd.NA,
            "Columns": pd.NA,
            "Purpose": "Methods, exact headline results and interpretation rules",
        }
    )
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = OUTPUT_DIR / "26_output_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    output_paths.append(manifest_path)

    zip_path = OUTPUT_DIR.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in output_paths:
            archive.write(path, arcname=str(path.relative_to(OUTPUT_DIR)))

    progress("Complete.")
    print("=" * 78)
    print("SECTION 3.3 ANALYSIS COMPLETE")
    print("=" * 78)
    print(f"Surveyed events: {len(events)} (positive={int(events['Cephalopod_Presence'].sum())}, zero={int((~events['Cephalopod_Presence']).sum())})")
    print(f"Formal mapped locations: {mapping['LocationID'].nunique()} in {mapping['Region'].nunique()} regions")
    for row in main_net.itertuples(index=False):
        print(f"Biomass COG {row.Season} net shift: {row.Net_Shift_km:.1f} km {row.Direction}")
    print(
        "Regional community PERMANOVA: "
        f"R2={overall_region.PERMANOVA_R2:.4f}, p={format_p(overall_region.PERMANOVA_P)}; "
        f"PERMDISP p={format_p(overall_region.PERMDISP_P)}"
    )
    print(f"Fatal QC passed: {bool(qc.loc[qc['Severity'].eq('FATAL'), 'Pass'].all())}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"ZIP: {zip_path}")


if __name__ == "__main__":
    main()
