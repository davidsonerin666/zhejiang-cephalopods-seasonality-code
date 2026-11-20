# -*- coding: utf-8 -*-
"""
浙江近岸头足类第五步（正式区域版）：β多样性与群落重组

这是可作为单个.py文件直接运行的整理版。正式区域映射会在任何耗时分析前
自动定位、读取、核对并写到统一路径，不再依赖“先失败一次再生成模板”的
循环流程。运行前只需把第三步Master ZIP和正式区域映射CSV/ZIP加入Kaggle。

如自动发现多个不同版本，请在运行前设置：

    CEPHALOPOD_STEP3_ZIP=/完整路径/cephalopod_step3_analysis_master.zip
    CEPHALOPOD_REGION_MAPPING_SOURCE=/完整路径/cephalopod_region_mapping_outputs.zip
"""

from __future__ import annotations

import importlib.util
import hashlib
import io
import inspect
import math
import os
import time
import warnings
import zipfile
from collections import OrderedDict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import scipy
import sklearn
from packaging.version import Version

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch

from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ConstantInputWarning, f_oneway, spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.manifold import smacof


# ============================================================
# A. 正式区域映射自动定位与严格验收（耗时分析前执行）
# ============================================================

RUN_STARTED_UTC = datetime.now(timezone.utc).isoformat()

FORMAL_MAPPING_FILENAMES = {
    "cephalopod_region_mapping.csv",
    "00_cephalopod_region_mapping.csv",
}
FORMAL_REGION_ORDER = {"South": 1, "Central": 2, "North": 3}
FORMAL_ALL_LOCATION_COUNTS = {"South": 36, "Central": 41, "North": 47}


def _mapping_hash(frame: pd.DataFrame) -> str:
    columns = [
        "LocationID", "Region", "Region_Order", "Region_Definition"
    ]
    payload = (
        frame[columns]
        .sort_values("LocationID", kind="stable")
        .reset_index(drop=True)
        .to_csv(index=False, lineterminator="\n")
        .encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def _validate_formal_mapping(
    frame: pd.DataFrame,
    source_label: str,
) -> pd.DataFrame:
    required = [
        "LocationID", "Region", "Region_Order", "Region_Definition"
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{source_label}缺少字段：{missing}")

    result = frame.copy()
    result["LocationID"] = result["LocationID"].astype("string").str.strip()
    result["Region"] = result["Region"].astype("string").str.strip()
    result["Region_Definition"] = (
        result["Region_Definition"].astype("string").str.strip()
    )
    result["Region_Order"] = pd.to_numeric(
        result["Region_Order"], errors="coerce"
    ).astype("Int64")

    blank_location = result["LocationID"].isna() | result["LocationID"].eq("")
    blank_region = result["Region"].isna() | result["Region"].eq("")
    if blank_location.any() or blank_region.any():
        raise ValueError(
            f"{source_label}含空白LocationID={int(blank_location.sum())}，"
            f"空白Region={int(blank_region.sum())}"
        )
    if result["LocationID"].duplicated().any():
        duplicates = result.loc[
            result["LocationID"].duplicated(keep=False), "LocationID"
        ].astype(str).unique()
        raise ValueError(
            f"{source_label}含重复LocationID：" + ", ".join(duplicates[:20])
        )

    expected_locations = int(
        os.environ.get("CEPHALOPOD_EXPECTED_ALL_LOCATIONS", "124")
    )
    if len(result) != expected_locations:
        raise ValueError(
            f"{source_label}应有{expected_locations}个LocationID，"
            f"实际为{len(result)}个"
        )

    observed_regions = set(result["Region"].astype(str))
    if observed_regions != set(FORMAL_REGION_ORDER):
        raise ValueError(
            f"{source_label}区域应为{sorted(FORMAL_REGION_ORDER)}，"
            f"实际为{sorted(observed_regions)}"
        )
    order_lookup = (
        result[["Region", "Region_Order"]]
        .drop_duplicates()
        .set_index("Region")["Region_Order"]
        .astype(int)
        .to_dict()
    )
    if order_lookup != FORMAL_REGION_ORDER:
        raise ValueError(
            f"{source_label}的Region_Order不正确：{order_lookup}"
        )
    counts = result["Region"].value_counts().to_dict()
    if counts != FORMAL_ALL_LOCATION_COUNTS:
        raise ValueError(
            f"{source_label}的三区标准站位数应为"
            f"{FORMAL_ALL_LOCATION_COUNTS}，实际为{counts}"
        )
    for region, group in result.groupby("Region", sort=False):
        if group["Region_Definition"].nunique(dropna=True) != 1:
            raise ValueError(f"{source_label}中{region}对应多个分区定义")

    preferred = [
        "LocationID", "Lon", "Lat", "Station_Canonical", "Region",
        "Region_Order", "Region_Definition",
    ]
    first = [column for column in preferred if column in result.columns]
    rest = [column for column in result.columns if column not in first]
    return (
        result[first + rest]
        .sort_values(["Region_Order", "LocationID"], kind="stable")
        .reset_index(drop=True)
    )


def _read_mapping_candidate(path: Path) -> tuple[pd.DataFrame, str]:
    if path.suffix.lower() != ".zip":
        return pd.read_csv(path, encoding="utf-8-sig"), str(path)

    with zipfile.ZipFile(path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise zipfile.BadZipFile(f"CRC检查失败：{bad_member}")
        matches: list[str] = []
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"ZIP含不安全路径：{info.filename}")
            if member.name in FORMAL_MAPPING_FILENAMES:
                matches.append(info.filename)
        preferred = [
            name for name in matches
            if PurePosixPath(name).name == "00_cephalopod_region_mapping.csv"
        ]
        selected = preferred or matches
        if len(selected) != 1:
            raise FileNotFoundError(
                f"{path}中正式映射CSV数量应为1，实际为{len(selected)}："
                f"{selected}"
            )
        member_name = selected[0]
        frame = pd.read_csv(
            io.BytesIO(archive.read(member_name)), encoding="utf-8-sig"
        )
    return frame, f"{path}::{member_name}"


def _discover_formal_mapping_candidates(target: Path) -> list[Path]:
    explicit = os.environ.get(
        "CEPHALOPOD_REGION_MAPPING_SOURCE", ""
    ).strip()
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(
                "CEPHALOPOD_REGION_MAPPING_SOURCE不存在：" + str(path)
            )
        return [path.resolve()]

    candidates: list[Path] = []
    if target.exists():
        candidates.append(target)
    for root in [Path("/kaggle/working"), Path("/kaggle/input")]:
        if not root.exists():
            continue
        for filename in sorted(FORMAL_MAPPING_FILENAMES):
            candidates.extend(root.rglob(filename))
        candidates.extend(
            root.rglob("cephalopod_region_mapping_outputs*.zip")
        )

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if not unique:
        raise FileNotFoundError(
            "未找到正式区域映射。请将"
            "cephalopod_region_mapping_outputs.zip加入Kaggle，或设置：\n"
            "os.environ['CEPHALOPOD_REGION_MAPPING_SOURCE']='ZIP完整路径'"
        )
    return unique


def prepare_formal_region_mapping() -> tuple[Path, str, pd.DataFrame]:
    target = Path(
        os.environ.get(
            "CEPHALOPOD_REGION_MAPPING",
            "/kaggle/working/cephalopod_region_mapping.csv",
        )
    )
    valid: list[tuple[Path, pd.DataFrame, str]] = []
    rejected: list[str] = []
    for path in _discover_formal_mapping_candidates(target):
        try:
            frame, label = _read_mapping_candidate(path)
            frame = _validate_formal_mapping(frame, label)
            valid.append((path, frame, label))
        except Exception as exc:
            rejected.append(f"{path}: {type(exc).__name__}: {exc}")
    if not valid:
        raise RuntimeError(
            "找到的区域映射均未通过验收：\n" + "\n".join(rejected)
        )
    fingerprints = {_mapping_hash(frame) for _, frame, _ in valid}
    if len(fingerprints) != 1:
        raise RuntimeError(
            "发现多个内容不同的正式区域映射，请显式设置"
            "CEPHALOPOD_REGION_MAPPING_SOURCE：\n"
            + "\n".join(str(path) for path, _, _ in valid)
        )

    _, mapping, source_label = valid[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    mapping.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(target)
    written = _validate_formal_mapping(
        pd.read_csv(target, encoding="utf-8-sig"), str(target)
    )
    if _mapping_hash(written) != _mapping_hash(mapping):
        raise AssertionError("正式区域映射写出后内容发生变化。")
    return target, source_label, mapping


FORMAL_MAPPING_PATH, FORMAL_MAPPING_SOURCE, FORMAL_MAPPING_TABLE = (
    prepare_formal_region_mapping()
)
os.environ["CEPHALOPOD_REGION_MAPPING"] = str(FORMAL_MAPPING_PATH)
os.environ["CEPHALOPOD_REQUIRE_REGION_MAPPING"] = "1"
os.environ.setdefault("CEPHALOPOD_MIN_REGION_LOCATIONS", "5")
os.environ.setdefault("CEPHALOPOD_MAX_SIMPER_CONTRASTS", "0")
os.environ.setdefault(
    "CEPHALOPOD_STEP5_OUTPUT_DIR",
    "/kaggle/working/cephalopod_step5_beta_diversity_region",
)

print("=" * 78, flush=True)
print("正式区域映射已定位并通过验收", flush=True)
print(f"来源：{FORMAL_MAPPING_SOURCE}", flush=True)
print(f"统一读取路径：{FORMAL_MAPPING_PATH}", flush=True)
print("三区全部标准站位：South=43, Central=28, North=53", flush=True)
print("=" * 78, flush=True)


# ============================================================
# 0. 参数
# ============================================================

STEP3_DIR = Path(
    os.environ.get(
        "CEPHALOPOD_STEP3_DIR",
        "/kaggle/working/cephalopod_step3_analysis_master",
    )
)
STEP3_DIR_EXPLICIT = "CEPHALOPOD_STEP3_DIR" in os.environ
STEP3_ZIP = Path(
    os.environ.get(
        "CEPHALOPOD_STEP3_ZIP",
        "/kaggle/working/cephalopod_step3_analysis_master.zip",
    )
)
STEP3_ZIP_EXPLICIT = "CEPHALOPOD_STEP3_ZIP" in os.environ
OUTPUT_DIR = Path(
    os.environ.get(
        "CEPHALOPOD_STEP5_OUTPUT_DIR",
        "/kaggle/working/cephalopod_step5_beta_diversity_region",
    )
)
FIGURE_DIR = OUTPUT_DIR / "figures"
REGION_MAPPING_PATH = Path(
    os.environ.get(
        "CEPHALOPOD_REGION_MAPPING",
        "/kaggle/working/cephalopod_region_mapping.csv",
    )
)

SURVEYED_MASTER_NAME = "02_analysis_master_v1_surveyed_all25.csv"
LONG_MASTER_NAME = "04_analysis_master_v1_long_surveyed_all25.csv"
SPECIES_FILTER_NAME = "05_species_filter_summary.csv"

EXPECTED_SURVEYED = int(os.environ.get("CEPHALOPOD_EXPECTED_SURVEYED_EVENTS", "2100"))
EXPECTED_POSITIVE = int(os.environ.get("CEPHALOPOD_EXPECTED_POSITIVE_EVENTS", "1532"))
EXPECTED_ZERO = int(os.environ.get("CEPHALOPOD_EXPECTED_ZERO_EVENTS", "568"))
EXPECTED_SPECIES = int(os.environ.get("CEPHALOPOD_EXPECTED_SPECIES_COUNT", "25"))

N_PERMUTATIONS = int(os.environ.get("CEPHALOPOD_PERMUTATIONS", "999"))
N_PAIRWISE_PERMUTATIONS = int(
    os.environ.get("CEPHALOPOD_PAIRWISE_PERMUTATIONS", "999")
)
RANDOM_SEED = int(os.environ.get("CEPHALOPOD_RANDOM_SEED", "20260827"))
NMDS_RANDOM_STARTS = int(os.environ.get("CEPHALOPOD_NMDS_RANDOM_STARTS", "8"))
NMDS_MAX_ITER = int(os.environ.get("CEPHALOPOD_NMDS_MAX_ITER", "1000"))
NMDS_EPS = float(os.environ.get("CEPHALOPOD_NMDS_EPS", "1e-6"))
NMDS_VALIDATION_ITER = int(
    os.environ.get("CEPHALOPOD_NMDS_VALIDATION_ITER", "100")
)
NMDS_MAX_VALIDATION_ROUNDS = int(
    os.environ.get("CEPHALOPOD_NMDS_MAX_VALIDATION_ROUNDS", "3")
)
NMDS_STRESS_STABILITY_TOL = float(
    os.environ.get("CEPHALOPOD_NMDS_STRESS_STABILITY_TOL", "5e-5")
)
NMDS_CONFIGURATION_STABILITY_TOL = float(
    os.environ.get("CEPHALOPOD_NMDS_CONFIGURATION_STABILITY_TOL", "5e-3")
)
NMDS_OBJECTIVE_WORSENING_TOL = float(
    os.environ.get("CEPHALOPOD_NMDS_OBJECTIVE_WORSENING_TOL", "1e-8")
)
MIN_SKLEARN_VERSION = Version("1.7")
MIN_SHARED_LOCATIONS = int(
    os.environ.get("CEPHALOPOD_MIN_SHARED_LOCATIONS", "8")
)
MIN_REGION_LOCATIONS = int(
    os.environ.get("CEPHALOPOD_MIN_REGION_LOCATIONS", "5")
)
REQUIRE_REGION_MAPPING = str(
    os.environ.get("CEPHALOPOD_REQUIRE_REGION_MAPPING", "1")
).strip().lower() in {"1", "true", "yes", "y"}
FDR_ALPHA = float(os.environ.get("CEPHALOPOD_FDR_ALPHA", "0.05"))
SIMPER_CUMULATIVE_PERCENT = float(
    os.environ.get("CEPHALOPOD_SIMPER_CUMULATIVE_PERCENT", "80")
)
MAX_SIMPER_CONTRASTS = int(
    os.environ.get("CEPHALOPOD_MAX_SIMPER_CONTRASTS", "0")
)
FIGURE_DPI = int(os.environ.get("CEPHALOPOD_FIGURE_DPI", "600"))
SAVE_FULL_DISTANCE_CSV = str(
    os.environ.get("CEPHALOPOD_SAVE_FULL_DISTANCE_CSV", "0")
).strip().lower() in {"1", "true", "yes", "y"}

STATUS_POSITIVE = "SURVEYED_POSITIVE"
STATUS_ZERO = "SURVEYED_ZERO"
MONTH_LABELS = {4: "April", 11: "November"}
MONTH_COLORS = {4: "#0072B2", 11: "#D55E00"}

warnings.resetwarnings()
warnings.filterwarnings("default")

if Version(sklearn.__version__) < MIN_SKLEARN_VERSION:
    raise RuntimeError(
        "NMDS要求scikit-learn>=1.7，因为1.7修复了SMACOF收敛判据。"
        f"当前版本为{sklearn.__version__}。请在Kaggle启用Internet后先运行："
        "!pip install -q --upgrade 'scikit-learn>=1.7,<1.9'，然后Restart Session。"
    )
if NMDS_MAX_ITER < 100:
    raise ValueError("CEPHALOPOD_NMDS_MAX_ITER建议至少为100。")
if NMDS_VALIDATION_ITER < 10:
    raise ValueError("CEPHALOPOD_NMDS_VALIDATION_ITER必须至少为10。")
if NMDS_MAX_VALIDATION_ROUNDS < 1:
    raise ValueError("CEPHALOPOD_NMDS_MAX_VALIDATION_ROUNDS必须至少为1。")
if NMDS_EPS <= 0:
    raise ValueError("CEPHALOPOD_NMDS_EPS必须为正数。")
if NMDS_STRESS_STABILITY_TOL <= 0 or NMDS_CONFIGURATION_STABILITY_TOL <= 0:
    raise ValueError("NMDS稳定性阈值必须为正数。")
if NMDS_RANDOM_STARTS < 0:
    raise ValueError("CEPHALOPOD_NMDS_RANDOM_STARTS不能为负数。")
if MAX_SIMPER_CONTRASTS < 0:
    raise ValueError("CEPHALOPOD_MAX_SIMPER_CONTRASTS不能为负数；0表示不设上限。")
if MIN_REGION_LOCATIONS < 2:
    raise ValueError("CEPHALOPOD_MIN_REGION_LOCATIONS必须至少为2。")


# ============================================================
# 1. 文件和通用函数
# ============================================================

def progress(message: str) -> None:
    elapsed_minutes = (time.time() - RUN_START_MONOTONIC) / 60.0
    print(f"[Step5 {elapsed_minutes:6.1f} min] {message}", flush=True)


RUN_START_MONOTONIC = time.time()


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"ZIP中存在不安全路径：{member.filename}")
        archive.extractall(destination)


def locate_required_parent(root: Path, required: list[str]) -> Path | None:
    if all((root / name).exists() for name in required):
        return root
    candidates = []
    for first_name in [required[0]]:
        for file_path in root.rglob(first_name):
            parent = file_path.parent
            if all((parent / name).exists() for name in required):
                candidates.append(parent)
    candidates = sorted(set(candidates))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise FileNotFoundError(
            "解压后发现多个可能的第三步目录，请设置CEPHALOPOD_STEP3_DIR：\n"
            + "\n".join(str(p) for p in candidates)
        )
    return None


def locate_step3_dir() -> Path:
    required = [SURVEYED_MASTER_NAME, LONG_MASTER_NAME, SPECIES_FILTER_NAME]
    direct = locate_required_parent(STEP3_DIR, required) if STEP3_DIR.exists() else None
    if direct is not None and (STEP3_DIR_EXPLICIT or not STEP3_ZIP_EXPLICIT):
        return direct

    zip_candidates: list[Path] = []
    if STEP3_ZIP_EXPLICIT:
        if not STEP3_ZIP.exists():
            raise FileNotFoundError(f"CEPHALOPOD_STEP3_ZIP不存在：{STEP3_ZIP}")
        zip_candidates.append(STEP3_ZIP)
    else:
        if STEP3_ZIP.exists():
            zip_candidates.append(STEP3_ZIP)
        for base in [Path("/kaggle/input"), Path("/kaggle/working")]:
            if base.exists():
                for candidate in base.rglob("cephalopod_step3_analysis_master*.zip"):
                    if candidate not in zip_candidates:
                        zip_candidates.append(candidate)

    if not zip_candidates:
        raise FileNotFoundError(
            "未找到第三步Master目录或cephalopod_step3_analysis_master.zip。\n"
            "请先运行第三步，或设置CEPHALOPOD_STEP3_DIR/CEPHALOPOD_STEP3_ZIP。"
        )
    if len(zip_candidates) > 1:
        raise FileNotFoundError(
            "发现多个同名第三步ZIP，无法安全自动选择：\n"
            + "\n".join(str(p) for p in zip_candidates)
            + "\n请设置CEPHALOPOD_STEP3_ZIP。"
        )

    selected_zip = zip_candidates[0]
    digest = hashlib.sha256()
    with selected_zip.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    # ZIP内容不同则使用不同目录，避免Kaggle交互式重跑混入旧版本文件。
    extract_dir = Path("/kaggle/working") / (
        "cephalopod_step3_analysis_master_input_" + digest.hexdigest()[:12]
    )
    if locate_required_parent(extract_dir, required) is None:
        safe_extract_zip(selected_zip, extract_dir)
    resolved = locate_required_parent(extract_dir, required)
    if resolved is None:
        raise FileNotFoundError("第三步ZIP中缺少所需Master文件。")
    return resolved


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def write_csv(table: pd.DataFrame, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def require_columns(df: pd.DataFrame, columns: list[str], table_name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"{table_name}缺少字段：{missing}")


def bh_fdr(p_values: pd.Series) -> pd.Series:
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(float)
    result = np.full(len(values), np.nan, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return pd.Series(result, index=p_values.index)
    idx = np.flatnonzero(valid)
    p = values[valid]
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    inverse = np.empty(m, dtype=int)
    inverse[order] = np.arange(m)
    result[idx] = adjusted[inverse]
    return pd.Series(result, index=p_values.index)


def combine_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series("ALL", index=df.index, dtype="string")
    return df[columns].astype("string").agg("::".join, axis=1)


def make_block_indices(block_values: pd.Series | None, n: int) -> list[np.ndarray]:
    if block_values is None:
        return [np.arange(n, dtype=int)]
    codes = pd.Categorical(block_values.astype(str)).codes
    return [np.flatnonzero(codes == code) for code in np.unique(codes)]


def permute_within_blocks(
    values: np.ndarray,
    block_indices: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    permuted = values.copy()
    for indices in block_indices:
        if len(indices) > 1:
            permuted[indices] = rng.permutation(permuted[indices])
    return permuted


def count_exchangeable_blocks(labels: np.ndarray, blocks: list[np.ndarray]) -> int:
    return int(
        sum(len(idx) > 1 and np.unique(labels[idx]).size > 1 for idx in blocks)
    )


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    paths = []
    for ext in ["png", "pdf", "svg"]:
        final_path = FIGURE_DIR / f"{stem}.{ext}"
        temporary_path = FIGURE_DIR / f".{stem}.{ext}.tmp"
        kwargs: dict[str, object] = {
            "bbox_inches": "tight",
            "facecolor": "white",
        }
        if ext == "png":
            kwargs["dpi"] = FIGURE_DPI
        fig.savefig(temporary_path, format=ext, **kwargs)
        temporary_path.replace(final_path)
        paths.append(final_path)
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


# ============================================================
# 2. NMDS
# ============================================================

def pcoa_initial_configuration(distance_matrix: np.ndarray, dimensions: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    d2 = distance_matrix**2
    row_mean = d2.mean(axis=1, keepdims=True)
    gower = -0.5 * (d2 - row_mean - row_mean.T + d2.mean())
    eigenvalues, eigenvectors = eigh(
        gower,
        subset_by_index=[n - dimensions, n - 1],
        check_finite=False,
    )
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    return eigenvectors * np.sqrt(eigenvalues)


def stress_and_shepard(
    distance_matrix: np.ndarray,
    coordinates: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    original = squareform(distance_matrix, checks=False)
    fitted = pdist(coordinates, metric="euclidean")
    valid = np.isfinite(original) & np.isfinite(fitted) & (original > 0)
    x = original[valid]
    y = fitted[valid]
    if len(x) == 0:
        return np.nan, np.nan, x, y, y
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    disparities = iso.fit_transform(x, y)
    denom_disp = np.sum(disparities**2)
    if denom_disp > 0:
        disparities = disparities * math.sqrt(np.sum(y**2) / denom_disp)
    denominator = np.sum(y**2)
    stress1 = (
        math.sqrt(np.sum((y - disparities) ** 2) / denominator)
        if denominator > 0
        else np.nan
    )
    shepard_rho = float(spearmanr(x, y).statistic)
    return stress1, shepard_rho, x, y, disparities


def call_smacof(
    distance_matrix: np.ndarray,
    dimensions: int,
    init: np.ndarray | None,
    n_init: int,
    seed: int,
    max_iter: int,
    eps: float,
) -> tuple[np.ndarray, float, int, list[str]]:
    signature = inspect.signature(smacof).parameters
    kwargs: dict[str, object] = {
        "metric": False,
        "n_components": dimensions,
        "init": init,
        "n_init": n_init,
        "max_iter": max_iter,
        "eps": eps,
        "random_state": seed,
        "return_n_iter": True,
    }
    if "n_jobs" in signature:
        kwargs["n_jobs"] = -1
    if "normalized_stress" in signature:
        kwargs["normalized_stress"] = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = smacof(distance_matrix, **kwargs)
    coordinates, sklearn_stress, iterations = result
    return (
        np.asarray(coordinates, dtype=float),
        float(sklearn_stress),
        int(iterations),
        [f"{item.category.__name__}: {item.message}" for item in caught],
    )


def run_smacof_with_convergence_check(
    distance_matrix: np.ndarray,
    dimensions: int,
    init: np.ndarray | None,
    initial_n_init: int,
    seed: int,
) -> dict[str, object]:
    """运行非度量SMACOF，并用独立续跑验证解是否真正稳定。"""
    start_time = time.time()
    primary, primary_objective, primary_iterations, warnings_primary = call_smacof(
        distance_matrix=distance_matrix,
        dimensions=dimensions,
        init=init,
        n_init=initial_n_init,
        seed=seed,
        max_iter=NMDS_MAX_ITER,
        eps=NMDS_EPS,
    )
    primary_oriented = orient_coordinates(primary)
    primary_stress1, primary_rho, _, _, _ = stress_and_shepard(
        distance_matrix, primary_oriented
    )

    # 连续续跑最多若干轮；以前一轮为基准判断最后一轮是否稳定。第一轮若仍
    # 明显改善，不会立即误判为失败，而会继续验证最终构型。
    current = primary
    previous_oriented = primary_oriented
    previous_stress1 = primary_stress1
    previous_objective = primary_objective
    coordinates = primary_oriented
    stress1 = primary_stress1
    rho = primary_rho
    validated_objective = primary_objective
    validation_iterations = 0
    validation_rounds = 0
    validation_stopped_before_max = False
    configuration_relative_change = np.inf
    last_stress_change = np.inf
    last_objective_change = np.inf
    warnings_validation: list[str] = []

    for validation_round in range(1, NMDS_MAX_VALIDATION_ROUNDS + 1):
        validated, validated_objective, completed, round_warnings = call_smacof(
            distance_matrix=distance_matrix,
            dimensions=dimensions,
            init=current,
            n_init=1,
            seed=seed + 100_000 + validation_round,
            max_iter=NMDS_VALIDATION_ITER,
            eps=NMDS_EPS,
        )
        validation_rounds = validation_round
        validation_iterations += completed
        warnings_validation.extend(round_warnings)
        coordinates = orient_coordinates(validated)
        stress1, rho, _, _, _ = stress_and_shepard(distance_matrix, coordinates)

        previous_fit_distances = pdist(previous_oriented)
        current_fit_distances = pdist(coordinates)
        configuration_scale = max(
            float(np.sqrt(np.mean(np.square(previous_fit_distances)))),
            np.finfo(float).eps,
        )
        configuration_relative_change = float(
            np.sqrt(np.mean(np.square(current_fit_distances - previous_fit_distances)))
            / configuration_scale
        )
        last_stress_change = float(stress1 - previous_stress1)
        last_objective_change = float(validated_objective - previous_objective)
        validation_stopped_before_max = completed < NMDS_VALIDATION_ITER

        stress_stable_now = abs(last_stress_change) <= NMDS_STRESS_STABILITY_TOL
        configuration_stable_now = (
            configuration_relative_change <= NMDS_CONFIGURATION_STABILITY_TOL
        )
        objective_not_worse_now = (
            last_objective_change <= NMDS_OBJECTIVE_WORSENING_TOL
        )
        if (
            validation_stopped_before_max
            and stress_stable_now
            and configuration_stable_now
            and objective_not_worse_now
        ):
            break

        current = validated
        previous_oriented = coordinates
        previous_stress1 = stress1
        previous_objective = validated_objective

    final_minus_primary_stress = float(stress1 - primary_stress1)
    final_minus_primary_objective = float(validated_objective - primary_objective)
    primary_stopped_before_max = primary_iterations < NMDS_MAX_ITER
    finite_solution = bool(
        np.isfinite(coordinates).all()
        and np.isfinite(stress1)
        and np.isfinite(rho)
        and np.isfinite(validated_objective)
    )
    stress_stable = abs(last_stress_change) <= NMDS_STRESS_STABILITY_TOL
    configuration_stable = (
        configuration_relative_change <= NMDS_CONFIGURATION_STABILITY_TOL
    )
    objective_not_worse = last_objective_change <= NMDS_OBJECTIVE_WORSENING_TOL
    convergence_ok = bool(
        primary_stopped_before_max
        and validation_stopped_before_max
        and finite_solution
        and stress_stable
        and configuration_stable
        and objective_not_worse
    )

    if convergence_ok:
        convergence_status = "CONVERGED_AND_VALIDATED"
    elif not primary_stopped_before_max:
        convergence_status = "PRIMARY_MAX_ITER_REACHED"
    elif not validation_stopped_before_max:
        convergence_status = "VALIDATION_MAX_ITER_REACHED"
    elif not finite_solution:
        convergence_status = "NONFINITE_SOLUTION"
    elif not stress_stable:
        convergence_status = "STRESS_NOT_STABLE_ON_VALIDATION"
    elif not configuration_stable:
        convergence_status = "CONFIGURATION_NOT_STABLE_ON_VALIDATION"
    else:
        convergence_status = "OBJECTIVE_WORSENED_ON_VALIDATION"

    return {
        "Coordinates": coordinates,
        "Stress1": stress1,
        "PrimaryStress1": primary_stress1,
        "FinalMinusPrimaryStress1": final_minus_primary_stress,
        "LastValidationStressChange": last_stress_change,
        "Shepard_Spearman": rho,
        "SklearnNormalizedStress": validated_objective,
        "PrimarySklearnNormalizedStress": primary_objective,
        "FinalMinusPrimaryObjective": final_minus_primary_objective,
        "LastValidationObjectiveChange": last_objective_change,
        "ConfigurationRelativeChange": configuration_relative_change,
        "PrimaryIterations": primary_iterations,
        "ValidationIterations": validation_iterations,
        "ValidationRounds": validation_rounds,
        "Iterations": primary_iterations + validation_iterations,
        "PrimaryStoppedBeforeMax": primary_stopped_before_max,
        "ValidationStoppedBeforeMax": validation_stopped_before_max,
        "StressStable": stress_stable,
        "ConfigurationStable": configuration_stable,
        "ObjectiveNotWorse": objective_not_worse,
        "ConvergenceOK": convergence_ok,
        "ConvergenceStatus": convergence_status,
        "WarningMessages": " | ".join(warnings_primary + warnings_validation),
        "RuntimeSeconds": time.time() - start_time,
    }


def orient_coordinates(coordinates: np.ndarray) -> np.ndarray:
    result = coordinates.copy()
    result -= result.mean(axis=0, keepdims=True)
    for axis in range(result.shape[1]):
        idx = int(np.argmax(np.abs(result[:, axis])))
        if result[idx, axis] < 0:
            result[:, axis] *= -1
    return result


def run_nmds_all_dimensions(
    distance_matrix: np.ndarray,
) -> tuple[dict[int, np.ndarray], pd.DataFrame, dict[str, np.ndarray]]:
    solutions: dict[int, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    shepard_payload: dict[str, np.ndarray] = {}

    for dimensions in [2, 3, 4]:
        progress(f"NMDS {dimensions}D：运行PCoA初始解")
        initial = pcoa_initial_configuration(distance_matrix, dimensions)
        pcoa_candidate = run_smacof_with_convergence_check(
            distance_matrix,
            dimensions,
            initial,
            initial_n_init=1,
            seed=RANDOM_SEED + dimensions,
        )
        pcoa_candidate["StartType"] = "PCoA_initialization"
        candidates = [pcoa_candidate]

        if NMDS_RANDOM_STARTS > 0:
            progress(
                f"NMDS {dimensions}D：运行{NMDS_RANDOM_STARTS}个随机初始解"
            )
            random_candidate = run_smacof_with_convergence_check(
                distance_matrix,
                dimensions,
                init=None,
                initial_n_init=NMDS_RANDOM_STARTS,
                seed=RANDOM_SEED + dimensions * 1_000,
            )
            random_candidate[
                "StartType"
            ] = f"Best_of_{NMDS_RANDOM_STARTS}_random_starts"
            candidates.append(random_candidate)

        converged_candidates = [
            item for item in candidates if bool(item["ConvergenceOK"])
        ]
        if not converged_candidates:
            diagnostic = pd.DataFrame(
                [
                    {
                        key: value
                        for key, value in candidate.items()
                        if key != "Coordinates"
                    }
                    for candidate in candidates
                ]
            )
            raise RuntimeError(
                f"{dimensions}D NMDS没有任何通过收敛检查的解。请提高"
                "CEPHALOPOD_NMDS_MAX_ITER或检查运行环境。\n"
                + diagnostic.to_string(index=False)
            )

        best = min(converged_candidates, key=lambda item: item["Stress1"])
        progress(
            f"NMDS {dimensions}D完成：Stress-1={float(best['Stress1']):.4f}，"
            f"状态={best['ConvergenceStatus']}"
        )
        solutions[dimensions] = np.asarray(best["Coordinates"])
        for candidate in candidates:
            rows.append(
                {
                    "Dimensions": dimensions,
                    "StartType": candidate["StartType"],
                    "Stress1": candidate["Stress1"],
                    "PrimaryStress1": candidate["PrimaryStress1"],
                    "FinalMinusPrimaryStress1": candidate[
                        "FinalMinusPrimaryStress1"
                    ],
                    "LastValidationStressChange": candidate[
                        "LastValidationStressChange"
                    ],
                    "Shepard_Spearman": candidate["Shepard_Spearman"],
                    "SklearnNormalizedStress": candidate[
                        "SklearnNormalizedStress"
                    ],
                    "PrimarySklearnNormalizedStress": candidate[
                        "PrimarySklearnNormalizedStress"
                    ],
                    "FinalMinusPrimaryObjective": candidate[
                        "FinalMinusPrimaryObjective"
                    ],
                    "LastValidationObjectiveChange": candidate[
                        "LastValidationObjectiveChange"
                    ],
                    "ConfigurationRelativeChange": candidate[
                        "ConfigurationRelativeChange"
                    ],
                    "PrimaryIterations": candidate["PrimaryIterations"],
                    "ValidationIterations": candidate["ValidationIterations"],
                    "ValidationRounds": candidate["ValidationRounds"],
                    "Iterations": candidate["Iterations"],
                    "PrimaryStoppedBeforeMax": candidate[
                        "PrimaryStoppedBeforeMax"
                    ],
                    "ValidationStoppedBeforeMax": candidate[
                        "ValidationStoppedBeforeMax"
                    ],
                    "StressStable": candidate["StressStable"],
                    "ConfigurationStable": candidate["ConfigurationStable"],
                    "ObjectiveNotWorse": candidate["ObjectiveNotWorse"],
                    "ConvergenceOK": candidate["ConvergenceOK"],
                    "ConvergenceStatus": candidate["ConvergenceStatus"],
                    "WarningMessages": candidate["WarningMessages"],
                    "RuntimeSeconds": candidate["RuntimeSeconds"],
                    "Selected": candidate is best,
                }
            )

        if dimensions == 2:
            _, _, x, fitted, disparities = stress_and_shepard(
                distance_matrix, solutions[2]
            )
            shepard_payload = {
                "OriginalDissimilarity": x,
                "OrdinationDistance": fitted,
                "Disparity": disparities,
            }

    stress_table = pd.DataFrame(rows)
    selected = stress_table.loc[stress_table["Selected"]].copy()
    acceptable = selected.loc[
        selected["ConvergenceOK"].astype(bool) & selected["Stress1"].le(0.20),
        "Dimensions",
    ]
    recommended: int | None = int(acceptable.min()) if len(acceptable) else None
    stress_table["AcceptableStress"] = stress_table["Stress1"].le(0.20)
    stress_table["RecommendedDimensions"] = (
        recommended if recommended is not None else pd.NA
    )
    stress_table["RecommendationStatus"] = (
        f"LOWEST_DIMENSION_AT_OR_BELOW_0.20_{recommended}D_USE_WITH_CAUTION"
        if recommended is not None
        else "NO_ACCEPTABLE_2D_TO_4D_SOLUTION"
    )
    stress_table["TwoDInterpretation"] = np.where(
        stress_table["Dimensions"].eq(2) & stress_table["Selected"],
        np.where(
            stress_table["Stress1"].le(0.10),
            "Good",
            np.where(
                stress_table["Stress1"].le(0.20),
                "Usable_with_caution",
                "Poor_interpret_cautiously",
            ),
        ),
        "Not_applicable",
    )
    return solutions, stress_table, shepard_payload


# ============================================================
# 3. PERMANOVA与PERMDISP
# ============================================================

def permanova_components(
    squared_condensed: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, float, int, int]:
    n = len(labels)
    groups = np.unique(labels)
    group_sizes = np.bincount(labels, minlength=int(groups.max()) + 1)
    total_ss = float(squared_condensed.sum() / n)
    same = labels[pair_i] == labels[pair_j]
    same_i = pair_i[same]
    within_ss = float(
        np.sum(
            squared_condensed[same]
            / group_sizes[labels[same_i]]
        )
    )
    between_ss = total_ss - within_ss
    df_between = len(groups) - 1
    df_within = n - len(groups)
    pseudo_f = (
        (between_ss / df_between) / (within_ss / df_within)
        if df_between > 0 and df_within > 0 and within_ss > 0
        else np.nan
    )
    r2 = between_ss / total_ss if total_ss > 0 else np.nan
    return pseudo_f, r2, within_ss, df_between, df_within


def permanova_test(
    distance_matrix: np.ndarray,
    labels: pd.Series,
    blocks: pd.Series | None,
    permutations: int,
    seed: int,
    test_name: str,
    restriction: str,
) -> dict[str, object]:
    categorical = pd.Categorical(labels.astype(str))
    codes = categorical.codes.astype(np.int16)
    n = len(codes)
    groups = list(categorical.categories)
    if len(groups) < 2 or n <= len(groups):
        return {
            "Test": test_name,
            "Status": "SKIPPED",
            "SkipReason": "Fewer than two groups or insufficient residual degrees of freedom",
        }

    condensed = squareform(distance_matrix, checks=False)
    squared = condensed**2
    pair_i, pair_j = np.triu_indices(n, 1)
    observed_f, r2, within_ss, df_between, df_within = permanova_components(
        squared, pair_i, pair_j, codes
    )
    block_indices = make_block_indices(blocks, n)
    exchangeable = count_exchangeable_blocks(codes, block_indices)
    if exchangeable == 0:
        return {
            "Test": test_name,
            "Status": "SKIPPED",
            "SkipReason": "No exchangeable block contains more than one group",
            "N": n,
            "N_Groups": len(groups),
        }

    rng = np.random.default_rng(seed)
    exceedances = 0
    finite_permutations = 0
    for _ in range(permutations):
        permuted = permute_within_blocks(codes, block_indices, rng)
        permuted_f, _, _, _, _ = permanova_components(
            squared, pair_i, pair_j, permuted
        )
        if np.isfinite(permuted_f):
            finite_permutations += 1
            if permuted_f >= observed_f - 1e-12:
                exceedances += 1
    p_value = (exceedances + 1) / (finite_permutations + 1)
    return {
        "Test": test_name,
        "Status": "OK",
        "SkipReason": "",
        "N": n,
        "N_Groups": len(groups),
        "GroupLevels": " | ".join(groups),
        "GroupSizes": " | ".join(
            f"{group}={int(np.sum(codes == idx))}" for idx, group in enumerate(groups)
        ),
        "DF_Between": df_between,
        "DF_Within": df_within,
        "PseudoF": observed_f,
        "R2": r2,
        "WithinSS": within_ss,
        "P_Value": p_value,
        "PermutationsRequested": permutations,
        "PermutationsCompleted": finite_permutations,
        "Restriction": restriction,
        "N_ExchangeableBlocks": exchangeable,
    }


def lingoes_corrected_squared_distances(
    distance_matrix: np.ndarray,
) -> tuple[np.ndarray, float]:
    n = distance_matrix.shape[0]
    d2 = distance_matrix**2
    row_mean = d2.mean(axis=1, keepdims=True)
    gower = -0.5 * (d2 - row_mean - row_mean.T + d2.mean())
    min_eigen = float(
        eigh(
            gower,
            subset_by_index=[0, 0],
            eigvals_only=True,
            check_finite=False,
        )[0]
    )
    constant = max(0.0, -min_eigen)
    corrected = d2.copy()
    if constant > 1e-12:
        corrected += 2.0 * constant
        np.fill_diagonal(corrected, 0.0)
    return corrected, constant


def distances_to_group_centroids(
    corrected_squared: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    distances = np.full(len(labels), np.nan, dtype=float)
    for group in np.unique(labels):
        idx = np.flatnonzero(labels == group)
        sub = corrected_squared[np.ix_(idx, idx)]
        squared_distance = sub.mean(axis=1) - 0.5 * sub.mean()
        distances[idx] = np.sqrt(np.maximum(squared_distance, 0.0))
    return distances


def one_way_f(values: np.ndarray, labels: np.ndarray) -> float:
    arrays = [values[labels == group] for group in np.unique(labels)]
    if len(arrays) < 2 or any(len(array) < 2 for array in arrays):
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(f_oneway(*arrays).statistic)


def permdisp_test(
    distance_matrix: np.ndarray,
    labels: pd.Series,
    blocks: pd.Series | None,
    permutations: int,
    seed: int,
    test_name: str,
    restriction: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    categorical = pd.Categorical(labels.astype(str))
    codes = categorical.codes.astype(np.int16)
    groups = list(categorical.categories)
    n = len(codes)
    if len(groups) < 2:
        return (
            {
                "Test": test_name,
                "Status": "SKIPPED",
                "SkipReason": "Fewer than two groups",
            },
            pd.DataFrame(),
        )

    corrected, lingoes_constant = lingoes_corrected_squared_distances(distance_matrix)
    centroid_distances = distances_to_group_centroids(corrected, codes)
    observed_f = one_way_f(centroid_distances, codes)
    block_indices = make_block_indices(blocks, n)
    exchangeable = count_exchangeable_blocks(codes, block_indices)
    if exchangeable == 0 or not np.isfinite(observed_f):
        return (
            {
                "Test": test_name,
                "Status": "SKIPPED",
                "SkipReason": "No exchangeable blocks or undefined F statistic",
                "N": n,
                "N_Groups": len(groups),
            },
            pd.DataFrame(),
        )

    rng = np.random.default_rng(seed)
    exceedances = 0
    finite_permutations = 0
    for _ in range(permutations):
        permuted_values = permute_within_blocks(
            centroid_distances, block_indices, rng
        )
        permuted_f = one_way_f(permuted_values, codes)
        if np.isfinite(permuted_f):
            finite_permutations += 1
            if permuted_f >= observed_f - 1e-12:
                exceedances += 1
    p_value = (exceedances + 1) / (finite_permutations + 1)

    group_rows = []
    for idx, group in enumerate(groups):
        values = centroid_distances[codes == idx]
        group_rows.append(
            {
                "Test": test_name,
                "Group": group,
                "N": len(values),
                "Mean_DistanceToCentroid": float(np.mean(values)),
                "Median_DistanceToCentroid": float(np.median(values)),
                "SD_DistanceToCentroid": float(np.std(values, ddof=1))
                if len(values) > 1
                else 0.0,
            }
        )
    return (
        {
            "Test": test_name,
            "Status": "OK",
            "SkipReason": "",
            "N": n,
            "N_Groups": len(groups),
            "GroupLevels": " | ".join(groups),
            "F": observed_f,
            "P_Value": p_value,
            "LingoesConstant": lingoes_constant,
            "PermutationsRequested": permutations,
            "PermutationsCompleted": finite_permutations,
            "Restriction": restriction,
            "N_ExchangeableBlocks": exchangeable,
        },
        pd.DataFrame(group_rows),
    )


def run_permanova_permdisp(
    distance_matrix: np.ndarray,
    metadata: pd.DataFrame,
    group_col: str,
    block_cols: list[str],
    permutations: int,
    seed: int,
    test_name: str,
    restriction: str,
) -> tuple[dict[str, object], dict[str, object], pd.DataFrame]:
    labels = metadata[group_col].astype(str).reset_index(drop=True)
    blocks = (
        combine_columns(metadata.reset_index(drop=True), block_cols)
        if block_cols
        else None
    )
    permanova = permanova_test(
        distance_matrix,
        labels,
        blocks,
        permutations,
        seed,
        test_name,
        restriction,
    )
    permdisp, dispersion_groups = permdisp_test(
        distance_matrix,
        labels,
        blocks,
        permutations,
        seed + 1_000_000,
        test_name,
        restriction,
    )
    return permanova, permdisp, dispersion_groups


# ============================================================
# 4. SIMPER
# ============================================================

def simper_two_groups(
    transformed: np.ndarray,
    raw: np.ndarray,
    labels: pd.Series,
    group1: str,
    group2: str,
    species_metadata: pd.DataFrame,
    contrast_name: str,
    source_test: str,
    permdisp_significant: bool | None,
    pair_ids: pd.Series | None = None,
    pair_id_name: str | None = None,
    block_size: int = 128,
) -> pd.DataFrame:
    """Two-group SIMPER.

    pair_ids=None gives classical all-cross-pair SIMPER. Supplying pair_ids
    enforces one-to-one matched SIMPER and uses only the two observations in
    each matched block.
    """
    if len(transformed) != len(raw) or len(transformed) != len(labels):
        raise ValueError("SIMPER矩阵与分组标签行数不一致。")
    label_values = labels.astype(str).to_numpy()
    group1 = str(group1)
    group2 = str(group2)

    if pair_ids is not None:
        if len(pair_ids) != len(labels):
            raise ValueError("SIMPER配对ID与分组标签行数不一致。")
        pairing = pd.DataFrame(
            {
                "MatrixIndex": np.arange(len(labels), dtype=int),
                "Group": label_values,
                "PairID": pair_ids.astype("string").str.strip(),
            }
        )
        pairing = pairing.loc[pairing["Group"].isin([group1, group2])].copy()
        if pairing["PairID"].isna().any() or pairing["PairID"].eq("").any():
            raise AssertionError(f"{contrast_name}的配对ID含缺失或空字符串。")
        if pairing.duplicated(["PairID", "Group"]).any():
            bad = pairing.loc[
                pairing.duplicated(["PairID", "Group"], keep=False),
                ["PairID", "Group"],
            ].drop_duplicates()
            raise AssertionError(
                f"{contrast_name}存在同一配对ID-组的重复事件：\n"
                + bad.head(20).to_string(index=False)
            )
        first = pairing.loc[pairing["Group"].eq(group1)].set_index("PairID")
        second = pairing.loc[pairing["Group"].eq(group2)].set_index("PairID")
        common_pairs = first.index.intersection(second.index, sort=False)
        if len(common_pairs) == 0:
            raise AssertionError(f"{contrast_name}没有完整配对。")
        if len(common_pairs) != len(first) or len(common_pairs) != len(second):
            raise AssertionError(
                f"{contrast_name}并非完整一对一设计："
                f"{group1}={len(first)}, {group2}={len(second)}, 完整配对={len(common_pairs)}。"
            )
        idx1 = first.loc[common_pairs, "MatrixIndex"].to_numpy(int)
        idx2 = second.loc[common_pairs, "MatrixIndex"].to_numpy(int)
        simper_design = "Paired_OneToOne"
        n_pairs: object = len(common_pairs)
    else:
        idx1 = np.flatnonzero(label_values == group1)
        idx2 = np.flatnonzero(label_values == group2)
        if len(idx1) == 0 or len(idx2) == 0:
            return pd.DataFrame()
        simper_design = "Classical_AllCrossPairs"
        n_pairs = pd.NA

    x1 = transformed[idx1]
    x2 = transformed[idx2]
    raw1 = raw[idx1]
    raw2 = raw[idx2]
    totals1 = x1.sum(axis=1)
    totals2 = x2.sum(axis=1)
    n_species = transformed.shape[1]
    sums = np.zeros(n_species, dtype=float)
    sums_sq = np.zeros(n_species, dtype=float)
    count = 0

    if pair_ids is not None:
        denominator = totals1 + totals2
        valid = denominator > 0
        count = int(valid.sum())
        paired_contributions = np.divide(
            np.abs(x1 - x2),
            denominator[:, None],
            out=np.zeros_like(x1),
            where=valid[:, None],
        )
        sums = paired_contributions[valid].sum(axis=0)
        sums_sq = np.square(paired_contributions[valid]).sum(axis=0)
    else:
        for start in range(0, len(x1), block_size):
            stop = min(start + block_size, len(x1))
            a = x1[start:stop]
            denominator = totals1[start:stop, None] + totals2[None, :]
            valid = denominator > 0
            pair_count = int(valid.sum())
            count += pair_count
            for species_idx in range(n_species):
                numerator = np.abs(
                    a[:, species_idx, None] - x2[None, :, species_idx]
                )
                contribution = np.divide(
                    numerator,
                    denominator,
                    out=np.zeros_like(numerator),
                    where=valid,
                )
                sums[species_idx] += contribution[valid].sum()
                sums_sq[species_idx] += np.square(contribution[valid]).sum()

    means = sums / count if count else np.full(n_species, np.nan)
    if count > 1:
        variances = np.maximum((sums_sq - count * means**2) / (count - 1), 0.0)
        sds = np.sqrt(variances)
    else:
        sds = np.full(n_species, np.nan)
    total_mean = np.nansum(means)
    percent = 100 * means / total_mean if total_mean > 0 else np.full(n_species, np.nan)

    result = species_metadata[
        ["Species_Code", "Species_CN", "Species_Latin", "Species_Order"]
    ].copy()
    result.insert(0, "SourceTest", source_test)
    result.insert(1, "Contrast", contrast_name)
    result.insert(2, "SIMPER_Design", simper_design)
    result.insert(3, "PairID_Column", pair_id_name if pair_ids is not None else pd.NA)
    result["Group1"] = str(group1)
    result["Group2"] = str(group2)
    result["N_Group1"] = len(idx1)
    result["N_Group2"] = len(idx2)
    result["N_MatchedPairs"] = n_pairs
    result["N_BetweenGroupPairs"] = count
    result["Mean_RawBiomass_Group1_kg_km2"] = raw1.mean(axis=0)
    result["Mean_RawBiomass_Group2_kg_km2"] = raw2.mean(axis=0)
    result["Mean_SqrtBiomass_Group1"] = x1.mean(axis=0)
    result["Mean_SqrtBiomass_Group2"] = x2.mean(axis=0)
    result["Average_BrayCurtis_Contribution"] = means
    result["SD_Contribution"] = sds
    result["Average_to_SD_Ratio"] = np.divide(
        means,
        sds,
        out=np.full_like(means, np.nan),
        where=sds > 0,
    )
    result["Percent_Contribution"] = percent
    result["Mean_BrayCurtis_BetweenGroups"] = total_mean
    result["PERMDISP_Significant"] = permdisp_significant
    result = result.sort_values(
        "Average_BrayCurtis_Contribution", ascending=False, kind="stable"
    ).reset_index(drop=True)
    result["SIMPER_Rank"] = np.arange(1, len(result) + 1)
    result["Cumulative_Percent"] = result["Percent_Contribution"].cumsum()
    previous = result["Cumulative_Percent"].shift(fill_value=0)
    result["Selected_Through_CumulativeThreshold"] = previous.lt(
        SIMPER_CUMULATIVE_PERCENT
    )
    return result


# ============================================================
# 5. 读取并验收数据
# ============================================================

progress("定位并读取第三步Master")
INPUT_DIR = locate_step3_dir()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RUN_STATUS_PATH = OUTPUT_DIR / "RUN_STATUS.txt"
RUN_STATUS_PATH.write_text(
    "Status: RUNNING\n"
    f"RunStartedUTC: {RUN_STARTED_UTC}\n"
    f"RegionMappingSource: {FORMAL_MAPPING_SOURCE}\n",
    encoding="utf-8",
)

surveyed = read_csv(INPUT_DIR / SURVEYED_MASTER_NAME)
long_df = read_csv(INPUT_DIR / LONG_MASTER_NAME)
species_filter = read_csv(INPUT_DIR / SPECIES_FILTER_NAME)

require_columns(
    surveyed,
    [
        "EventID",
        "Year",
        "Month",
        "Period",
        "LocationID",
        "Lon",
        "Lat",
        "SurveyStatus",
        "Total_Biomass_kg_km2",
        "Species_Richness",
    ],
    "已调查事件Master",
)
require_columns(
    long_df,
    [
        "EventID",
        "Year",
        "Month",
        "LocationID",
        "SurveyStatus",
        "Species_Code",
        "Species_CN",
        "Species_Latin",
        "Species_Order",
        "Presence",
        "Biomass_kg_km2",
    ],
    "事件×物种长表",
)
require_columns(
    species_filter,
    ["Species_Code", "Species_CN", "Species_Latin"],
    "物种字典",
)

for col in ["Year", "Month", "Total_Biomass_kg_km2", "Species_Richness"]:
    surveyed[col] = pd.to_numeric(surveyed[col], errors="raise")
for col in ["Year", "Month", "Species_Order", "Presence", "Biomass_kg_km2"]:
    long_df[col] = pd.to_numeric(long_df[col], errors="raise")

surveyed["Year"] = surveyed["Year"].astype(int)
surveyed["Month"] = surveyed["Month"].astype(int)
long_df["Year"] = long_df["Year"].astype(int)
long_df["Month"] = long_df["Month"].astype(int)

if not surveyed["EventID"].is_unique:
    raise AssertionError("已调查事件Master的EventID不唯一。")
if not surveyed["SurveyStatus"].isin([STATUS_POSITIVE, STATUS_ZERO]).all():
    raise AssertionError("已调查事件Master中出现未知或未调查状态。")

species_metadata = (
    species_filter[
        [c for c in ["Species_Code", "Species_CN", "Species_Latin"] if c in species_filter]
    ]
    .drop_duplicates("Species_Code")
    .copy()
)
species_order_lookup = (
    long_df[["Species_Code", "Species_Order"]]
    .drop_duplicates("Species_Code")
    .set_index("Species_Code")["Species_Order"]
)
species_metadata["Species_Order"] = species_metadata["Species_Code"].map(
    species_order_lookup
)
species_metadata = species_metadata.sort_values("Species_Order", kind="stable").reset_index(
    drop=True
)
species_codes = species_metadata["Species_Code"].astype(str).tolist()

surveyed_count = len(surveyed)
positive_events = surveyed.loc[
    surveyed["SurveyStatus"].eq(STATUS_POSITIVE)
].copy()
zero_events = surveyed.loc[surveyed["SurveyStatus"].eq(STATUS_ZERO)].copy()

observed_counts = {
    "Surveyed": surveyed_count,
    "Positive": len(positive_events),
    "Zero": len(zero_events),
    "Species": len(species_codes),
    "LongRows": len(long_df),
}
expected_counts = {
    "Surveyed": EXPECTED_SURVEYED,
    "Positive": EXPECTED_POSITIVE,
    "Zero": EXPECTED_ZERO,
    "Species": EXPECTED_SPECIES,
    "LongRows": EXPECTED_SURVEYED * EXPECTED_SPECIES,
}
mismatches = [
    f"{key}: expected {expected_counts[key]}, observed {observed_counts[key]}"
    for key in expected_counts
    if expected_counts[key] != observed_counts[key]
]
if mismatches:
    raise AssertionError("冻结计数检查未通过：\n" + "\n".join(mismatches))

positive_events = positive_events.sort_values(
    ["Year", "Month", "LocationID", "EventID"], kind="stable"
).reset_index(drop=True)
positive_events["YearMonth"] = (
    positive_events["Year"].astype(str)
    + "-"
    + positive_events["Month"].astype(str).str.zfill(2)
)
positive_events["MonthLabel"] = positive_events["Month"].map(MONTH_LABELS)
positive_events["StationYear"] = (
    positive_events["LocationID"].astype(str)
    + "::"
    + positive_events["Year"].astype(str)
)

positive_long = long_df.loc[
    long_df["EventID"].isin(positive_events["EventID"])
].copy()
duplicate_pairs = int(
    positive_long[["EventID", "Species_Code"]].duplicated().sum()
)
if duplicate_pairs:
    raise AssertionError("正捕获长表存在重复EventID-Species_Code。")

biomass_matrix_df = (
    positive_long.pivot(
        index="EventID", columns="Species_Code", values="Biomass_kg_km2"
    )
    .reindex(index=positive_events["EventID"], columns=species_codes)
)
if biomass_matrix_df.isna().any().any():
    raise AssertionError("正捕获生物量矩阵含NA。")
if (biomass_matrix_df < 0).any().any():
    raise AssertionError("正捕获生物量矩阵含负值。")
if biomass_matrix_df.sum(axis=1).le(0).any():
    raise AssertionError("正捕获事件中出现全零群落。")

event_totals = positive_events.set_index("EventID")["Total_Biomass_kg_km2"]
max_biomass_reconciliation = float(
    (biomass_matrix_df.sum(axis=1) - event_totals).abs().max()
)
if max_biomass_reconciliation > 1e-6:
    raise AssertionError("25种生物量加总与事件总生物量不一致。")

matrix_raw = biomass_matrix_df.to_numpy(dtype=float)
matrix_sqrt = np.sqrt(matrix_raw)
event_metadata = positive_events[
    [
        "EventID",
        "Year",
        "Month",
        "MonthLabel",
        "Period",
        "YearMonth",
        "LocationID",
        "StationYear",
        "Lon",
        "Lat",
        "Species_Richness",
        "Total_Biomass_kg_km2",
    ]
].reset_index(drop=True)
event_metadata["MatrixRow"] = np.arange(len(event_metadata), dtype=int)
progress(
    f"数据验收通过：正捕获事件={len(event_metadata)}，"
    f"物种={len(species_codes)}"
)


# ============================================================
# 6. 区域映射
# ============================================================

location_template = (
    surveyed.groupby("LocationID", as_index=False)
    .agg(
        Lon=("Lon", "first"),
        Lat=("Lat", "first"),
        Station_Canonical=("Station_Canonical", "first")
        if "Station_Canonical" in surveyed.columns
        else ("LocationID", "first"),
    )
    .sort_values("LocationID", kind="stable")
)
location_template["Region"] = ""
location_template["Region_Order"] = ""
location_template["Region_Definition"] = ""

region_status = "NOT_PROVIDED"
region_message = (
    "No Region field or complete LocationID-Region mapping; temporal analyses completed, "
    "regional analyses skipped"
)
region_mapping_used: pd.DataFrame | None = None
region_mapping_source = "NONE"
region_mapping_qc_rows: list[dict[str, object]] = []
region_mapping_output_columns = [
    "LocationID",
    "Region",
    "Region_Order",
    "Region_Definition",
    "Lon",
    "Lat",
    "Station_Canonical",
]
region_mapping_used_output = pd.DataFrame(columns=region_mapping_output_columns)
region_group_sizes = pd.DataFrame(
    columns=["Region", "N_PositiveLocations", "N_PositiveEvents"]
)

if REGION_MAPPING_PATH.exists():
    candidate = read_csv(REGION_MAPPING_PATH)
    require_columns(candidate, ["LocationID", "Region"], "区域映射表")
    for optional_column in ["Region_Order", "Region_Definition"]:
        if optional_column not in candidate.columns:
            candidate[optional_column] = pd.NA
    candidate = candidate[
        ["LocationID", "Region", "Region_Order", "Region_Definition"]
    ].copy()
    region_mapping_source = str(REGION_MAPPING_PATH)
    region_mapping_used = candidate
elif "Region" in surveyed.columns and surveyed["Region"].notna().any():
    keep_columns = ["LocationID", "Region"]
    for optional_column in ["Region_Order", "Region_Definition"]:
        if optional_column in surveyed.columns:
            keep_columns.append(optional_column)
    candidate = surveyed[keep_columns].copy()
    for optional_column in ["Region_Order", "Region_Definition"]:
        if optional_column not in candidate.columns:
            candidate[optional_column] = pd.NA
    region_mapping_source = "Embedded Region field in surveyed master"
    region_mapping_used = candidate

if region_mapping_used is not None:
    region_mapping_used["LocationID"] = (
        region_mapping_used["LocationID"].astype("string").str.strip()
    )
    region_mapping_used["Region"] = (
        region_mapping_used["Region"].astype("string").str.strip()
    )
    region_mapping_used["Region_Definition"] = (
        region_mapping_used["Region_Definition"].astype("string").str.strip()
    )
    region_mapping_used["Region_Order"] = pd.to_numeric(
        region_mapping_used["Region_Order"], errors="coerce"
    ).astype("Int64")
    region_mapping_used = region_mapping_used.loc[
        region_mapping_used["LocationID"].notna()
        & region_mapping_used["LocationID"].ne("")
        & region_mapping_used["Region"].notna()
        & region_mapping_used["Region"].ne("")
    ].copy()
    region_mapping_used = region_mapping_used.drop_duplicates()

    duplicated_locations = sorted(
        region_mapping_used.loc[
            region_mapping_used["LocationID"].duplicated(keep=False), "LocationID"
        ].astype(str).unique()
    )
    if duplicated_locations:
        raise AssertionError(
            "区域映射表中同一LocationID对应多个区域："
            + ", ".join(duplicated_locations[:20])
        )

    # 一个区域只能有一个顺序和一套定义，避免同名区域的含义不一致。
    for metadata_column in ["Region_Order", "Region_Definition"]:
        conflicts = (
            region_mapping_used.dropna(subset=[metadata_column])
            .groupby("Region")[metadata_column]
            .nunique(dropna=True)
        )
        bad_regions = conflicts.loc[conflicts.gt(1)].index.astype(str).tolist()
        if bad_regions:
            raise AssertionError(
                f"同一区域存在多个{metadata_column}：" + ", ".join(bad_regions)
            )

    known_locations = set(location_template["LocationID"].astype(str))
    mapped_locations = set(region_mapping_used["LocationID"].astype(str))
    unknown_locations = sorted(mapped_locations - known_locations)
    if unknown_locations:
        raise AssertionError(
            "区域映射表含未知LocationID：" + ", ".join(unknown_locations[:20])
        )

    positive_locations = set(event_metadata["LocationID"].astype(str))
    missing_region_locations = sorted(positive_locations - mapped_locations)
    positive_mapping = region_mapping_used.loc[
        region_mapping_used["LocationID"].astype(str).isin(positive_locations)
    ].copy()
    n_regions = positive_mapping["Region"].nunique(dropna=True)
    location_counts = positive_mapping["Region"].value_counts().sort_index()
    min_region_count = int(location_counts.min()) if len(location_counts) else 0

    if not missing_region_locations and n_regions >= 2 and min_region_count >= MIN_REGION_LOCATIONS:
        region_status = "READY"
        region_message = (
            f"Formal mapping validated; {n_regions} regions; "
            f"minimum positive locations per region={min_region_count}"
        )
    else:
        region_status = "INCOMPLETE"
        region_message = (
            f"Missing positive-event locations={len(missing_region_locations)}; "
            f"defined regions={n_regions}; minimum locations in a region={min_region_count}; "
            f"required minimum={MIN_REGION_LOCATIONS}; regional analyses skipped"
        )

    event_metadata = event_metadata.merge(
        region_mapping_used,
        on="LocationID",
        how="left",
        validate="many_to_one",
    )
    region_mapping_used_output = (
        region_mapping_used.merge(
            location_template[
                ["LocationID", "Lon", "Lat", "Station_Canonical"]
            ],
            on="LocationID",
            how="left",
            validate="one_to_one",
        )
        .reindex(columns=region_mapping_output_columns)
        .sort_values(["Region_Order", "Region", "LocationID"], kind="stable")
        .reset_index(drop=True)
    )
    if len(positive_mapping):
        event_counts = (
            event_metadata.dropna(subset=["Region"])
            .groupby("Region")
            .size()
            .rename("N_PositiveEvents")
        )
        region_group_sizes = (
            location_counts.rename("N_PositiveLocations")
            .to_frame()
            .join(event_counts, how="left")
            .fillna({"N_PositiveEvents": 0})
            .reset_index()
        )
        region_group_sizes["N_PositiveEvents"] = region_group_sizes[
            "N_PositiveEvents"
        ].astype(int)
else:
    missing_region_locations = sorted(event_metadata["LocationID"].astype(str).unique())
    n_regions = 0
    min_region_count = 0

mapping_supplied = region_mapping_used is not None


def region_qc_row(check: str, expected: object, observed: object, passed: bool) -> dict[str, object]:
    if not mapping_supplied:
        status = "FAIL" if REQUIRE_REGION_MAPPING else "SKIPPED"
        pass_value: object = False if REQUIRE_REGION_MAPPING else pd.NA
    else:
        status = "PASS" if passed else "FAIL"
        pass_value = bool(passed)
    return {
        "Check": check,
        "Expected": expected,
        "Observed": observed,
        "Status": status,
        "Pass": pass_value,
    }


region_mapping_qc_rows.extend(
    [
        region_qc_row(
            "Formal region mapping supplied",
            True,
            mapping_supplied,
            mapping_supplied,
        ),
        region_qc_row(
            "Positive LocationID coverage",
            event_metadata["LocationID"].nunique(),
            event_metadata["LocationID"].nunique() - len(missing_region_locations),
            len(missing_region_locations) == 0,
        ),
        region_qc_row(
            "At least two formal regions",
            ">=2",
            n_regions,
            n_regions >= 2,
        ),
        region_qc_row(
            "Minimum positive locations per region",
            f">={MIN_REGION_LOCATIONS}",
            min_region_count,
            min_region_count >= MIN_REGION_LOCATIONS,
        ),
    ]
)
region_mapping_qc = pd.DataFrame(region_mapping_qc_rows)
region_mapping_qc.insert(0, "MappingSource", region_mapping_source)
region_mapping_qc.insert(1, "RegionStatus", region_status)

if REQUIRE_REGION_MAPPING and region_status != "READY":
    raise AssertionError(
        "已设置CEPHALOPOD_REQUIRE_REGION_MAPPING=1，但正式区域映射未通过。\n"
        + region_message
    )


# ============================================================
# 7. Bray–Curtis与NMDS
# ============================================================

progress("计算1,532个正捕获事件的Bray-Curtis距离")
distance_condensed = pdist(matrix_sqrt, metric="braycurtis")
distance_matrix = squareform(distance_condensed)
if not np.isfinite(distance_matrix).all():
    raise AssertionError("Bray–Curtis距离含非有限值。")
if not np.allclose(distance_matrix, distance_matrix.T, atol=1e-12):
    raise AssertionError("Bray–Curtis距离矩阵不对称。")
if not np.allclose(np.diag(distance_matrix), 0.0, atol=1e-12):
    raise AssertionError("Bray–Curtis距离矩阵对角线不为0。")

progress("开始2D、3D和4D NMDS；这是本步骤最耗时的阶段")
nmds_solutions, nmds_stress, shepard_payload = run_nmds_all_dimensions(
    distance_matrix
)
progress("全部NMDS维数完成")
nmds_coordinate_tables: dict[int, pd.DataFrame] = {}
for dimensions, coordinates in nmds_solutions.items():
    table = event_metadata.copy()
    for axis in range(dimensions):
        table[f"NMDS{axis + 1}"] = coordinates[:, axis]
    nmds_coordinate_tables[dimensions] = table


# ============================================================
# 8. 主PERMANOVA/PERMDISP
# ============================================================

progress("运行4项总体PERMANOVA与PERMDISP")
omnibus_permanova_rows: list[dict[str, object]] = []
omnibus_permdisp_rows: list[dict[str, object]] = []
dispersion_group_tables: list[pd.DataFrame] = []


def append_omnibus_test(
    subset: pd.DataFrame,
    group_col: str,
    block_cols: list[str],
    test_name: str,
    restriction: str,
    seed_offset: int,
) -> None:
    indices = subset["MatrixRow"].to_numpy(int)
    sub_distance = distance_matrix[np.ix_(indices, indices)]
    permanova, permdisp, group_table = run_permanova_permdisp(
        sub_distance,
        subset.reset_index(drop=True),
        group_col,
        block_cols,
        N_PERMUTATIONS,
        RANDOM_SEED + seed_offset,
        test_name,
        restriction,
    )
    omnibus_permanova_rows.append(permanova)
    omnibus_permdisp_rows.append(permdisp)
    if len(group_table):
        dispersion_group_tables.append(group_table)


# 22个YearMonth，站位内受限置换
yearmonth_subset = event_metadata.loc[
    event_metadata.groupby("LocationID")["YearMonth"].transform("nunique").ge(2)
].copy()
append_omnibus_test(
    yearmonth_subset,
    "YearMonth",
    ["LocationID"],
    "YearMonth_22group_one_factor",
    "Permutations within LocationID",
    10,
)

# 严格配对月份比较：同一站位同一年两个季节均正捕获
station_year_month_counts = event_metadata.groupby("StationYear")["Month"].nunique()
paired_station_years = station_year_month_counts.loc[
    station_year_month_counts.eq(2)
].index
month_paired_subset = event_metadata.loc[
    event_metadata["StationYear"].isin(paired_station_years)
].copy()
append_omnibus_test(
    month_paired_subset,
    "MonthLabel",
    ["StationYear"],
    "Month_April_vs_November_paired",
    "Permutations within matched LocationID-Year",
    20,
)

# 分月份年际比较
for month, label, seed_offset in [(4, "April", 30), (11, "November", 40)]:
    subset = event_metadata.loc[event_metadata["Month"].eq(month)].copy()
    repeat_locations = subset.groupby("LocationID")["Year"].nunique()
    subset = subset.loc[
        subset["LocationID"].isin(repeat_locations.loc[repeat_locations.ge(2)].index)
    ]
    append_omnibus_test(
        subset,
        "Year",
        ["LocationID"],
        f"Year_within_{label}",
        "Permutations within LocationID",
        seed_offset,
    )

omnibus_permanova = pd.DataFrame(omnibus_permanova_rows)
omnibus_permdisp = pd.DataFrame(omnibus_permdisp_rows)
dispersion_groups = (
    pd.concat(dispersion_group_tables, ignore_index=True)
    if dispersion_group_tables
    else pd.DataFrame()
)


# ============================================================
# 9. 成对时间比较
# ============================================================

progress("运行受限置换的时间成对比较")
pairwise_rows: list[dict[str, object]] = []
pairwise_contexts: dict[str, dict[str, object]] = {}
pair_test_counter = 0


def run_pairwise_subset(
    subset: pd.DataFrame,
    group_col: str,
    group1: str,
    group2: str,
    block_cols: list[str],
    family: str,
    contrast: str,
    restriction: str,
    context: dict[str, object],
) -> None:
    global pair_test_counter
    pair_test_counter += 1
    if pair_test_counter == 1 or pair_test_counter % 10 == 0:
        progress(f"时间成对检验：已开始第{pair_test_counter}项")
    indices = subset["MatrixRow"].to_numpy(int)
    sub_distance = distance_matrix[np.ix_(indices, indices)]
    test_name = f"{family}::{contrast}"
    permanova, permdisp, _ = run_permanova_permdisp(
        sub_distance,
        subset.reset_index(drop=True),
        group_col,
        block_cols,
        N_PAIRWISE_PERMUTATIONS,
        RANDOM_SEED + 10_000 + pair_test_counter * 3,
        test_name,
        restriction,
    )
    row = {
        "Family": family,
        "Contrast": contrast,
        "Group1": str(group1),
        "Group2": str(group2),
        "N_Events": len(subset),
        "N_Locations": subset["LocationID"].nunique(),
        "Restriction": restriction,
        "PERMANOVA_Status": permanova.get("Status"),
        "PERMANOVA_PseudoF": permanova.get("PseudoF", np.nan),
        "PERMANOVA_R2": permanova.get("R2", np.nan),
        "PERMANOVA_P": permanova.get("P_Value", np.nan),
        "PERMANOVA_Permutations": permanova.get("PermutationsCompleted", 0),
        "PERMDISP_Status": permdisp.get("Status"),
        "PERMDISP_F": permdisp.get("F", np.nan),
        "PERMDISP_P": permdisp.get("P_Value", np.nan),
        "PERMDISP_LingoesConstant": permdisp.get("LingoesConstant", np.nan),
    }
    row.update(context)
    pairwise_rows.append(row)
    pairwise_contexts[f"{family}::{contrast}"] = {
        "Subset": subset.copy(),
        "GroupColumn": group_col,
        "Group1": str(group1),
        "Group2": str(group2),
        "PairIDColumn": block_cols[0] if len(block_cols) == 1 else None,
    }


# 每年4月—11月配对比较
for year in sorted(event_metadata["Year"].unique()):
    subset = event_metadata.loc[event_metadata["Year"].eq(year)].copy()
    shared = subset.groupby("LocationID")["Month"].nunique()
    shared_locations = shared.loc[shared.eq(2)].index
    subset = subset.loc[subset["LocationID"].isin(shared_locations)].copy()
    if len(shared_locations) < MIN_SHARED_LOCATIONS:
        pairwise_rows.append(
            {
                "Family": "Month_within_Year",
                "Contrast": f"{year}:April_vs_November",
                "Group1": "April",
                "Group2": "November",
                "N_Events": len(subset),
                "N_Locations": len(shared_locations),
                "Restriction": "Permutations within LocationID",
                "PERMANOVA_Status": "SKIPPED",
                "SkipReason": f"Shared positive locations < {MIN_SHARED_LOCATIONS}",
                "Year": year,
            }
        )
        continue
    run_pairwise_subset(
        subset,
        "MonthLabel",
        "April",
        "November",
        ["LocationID"],
        "Month_within_Year",
        f"{year}:April_vs_November",
        "Permutations within shared LocationID",
        {"Year": year, "Month": pd.NA},
    )

# 每个月内所有年份成对比较，共同正捕获站位设计
years = sorted(event_metadata["Year"].unique())
for month in [4, 11]:
    month_label = MONTH_LABELS[month]
    month_data = event_metadata.loc[event_metadata["Month"].eq(month)].copy()
    for year1, year2 in combinations(years, 2):
        subset = month_data.loc[month_data["Year"].isin([year1, year2])].copy()
        shared = subset.groupby("LocationID")["Year"].nunique()
        shared_locations = shared.loc[shared.eq(2)].index
        subset = subset.loc[subset["LocationID"].isin(shared_locations)].copy()
        contrast = f"{month_label}:{year1}_vs_{year2}"
        if len(shared_locations) < MIN_SHARED_LOCATIONS:
            pairwise_rows.append(
                {
                    "Family": f"Year_within_{month_label}",
                    "Contrast": contrast,
                    "Group1": str(year1),
                    "Group2": str(year2),
                    "N_Events": len(subset),
                    "N_Locations": len(shared_locations),
                    "Restriction": "Permutations within LocationID",
                    "PERMANOVA_Status": "SKIPPED",
                    "SkipReason": f"Shared positive locations < {MIN_SHARED_LOCATIONS}",
                    "Year": pd.NA,
                    "Month": month,
                }
            )
            continue
        subset["YearLabel"] = subset["Year"].astype(str)
        run_pairwise_subset(
            subset,
            "YearLabel",
            str(year1),
            str(year2),
            ["LocationID"],
            f"Year_within_{month_label}",
            contrast,
            "Permutations within shared LocationID",
            {"Year": pd.NA, "Month": month},
        )

pairwise_results = pd.DataFrame(pairwise_rows)
if len(pairwise_results):
    pairwise_results["PERMANOVA_P_FDR"] = np.nan
    pairwise_results["PERMDISP_P_FDR"] = np.nan
    for family, idx in pairwise_results.groupby("Family").groups.items():
        pairwise_results.loc[idx, "PERMANOVA_P_FDR"] = bh_fdr(
            pairwise_results.loc[idx, "PERMANOVA_P"]
        )
        pairwise_results.loc[idx, "PERMDISP_P_FDR"] = bh_fdr(
            pairwise_results.loc[idx, "PERMDISP_P"]
        )
    pairwise_results["PERMANOVA_FDR_Significant"] = pairwise_results[
        "PERMANOVA_P_FDR"
    ].lt(FDR_ALPHA)
    pairwise_results["PERMDISP_FDR_Significant"] = pairwise_results[
        "PERMDISP_P_FDR"
    ].lt(FDR_ALPHA)
progress(f"时间成对比较完成：共{len(pairwise_results)}项")


# ============================================================
# 10. 区域主分析（仅在完整映射时运行）
# ============================================================

progress("运行120个LocationID长期均值的区域PERMANOVA/PERMDISP")
region_analysis_rows: list[dict[str, object]] = []
REGION_PAIRWISE_COLUMNS = [
    "Family",
    "Contrast",
    "Group1",
    "Group2",
    "N_Events",
    "N_Locations",
    "PERMANOVA_Status",
    "SkipReason",
    "PERMANOVA_PseudoF",
    "PERMANOVA_R2",
    "PERMANOVA_P",
    "PERMDISP_Status",
    "PERMDISP_F",
    "PERMDISP_P",
    "PERMANOVA_P_FDR",
    "PERMDISP_P_FDR",
    "PERMANOVA_FDR_Significant",
    "PERMDISP_FDR_Significant",
]
region_pairwise = pd.DataFrame(columns=REGION_PAIRWISE_COLUMNS)
region_station_coverage = pd.DataFrame(
    columns=[
        "LocationID",
        "Region",
        "Region_Order",
        "N_PositiveEvents",
        "N_PositiveYears",
        "N_PositiveMonths",
        "FirstPositiveYear",
        "LastPositiveYear",
    ]
)
region_simper_tables: list[pd.DataFrame] = []

if region_status == "READY":
    region_labels = event_metadata[
        ["LocationID", "Region", "Region_Order", "Region_Definition"]
    ].drop_duplicates("LocationID")
    station_raw_df = pd.DataFrame(matrix_raw, columns=species_codes)
    station_raw_df.insert(0, "LocationID", event_metadata["LocationID"].to_numpy())
    station_raw = station_raw_df.groupby("LocationID", sort=True)[species_codes].mean()
    station_sqrt_df = pd.DataFrame(matrix_sqrt, columns=species_codes)
    station_sqrt_df.insert(0, "LocationID", event_metadata["LocationID"].to_numpy())
    # 先对每个正捕获事件进行sqrt变换，再在LocationID内求均值，明确估计对象为
    # “正捕获条件下的长期平均群落组成”，而不是包含零捕获的总体资源状态。
    station_sqrt = station_sqrt_df.groupby("LocationID", sort=True)[species_codes].mean()
    station_meta = (
        event_metadata.groupby("LocationID", as_index=True)
        .agg(
            Lon=("Lon", "first"),
            Lat=("Lat", "first"),
            N_PositiveEvents=("EventID", "size"),
            N_PositiveYears=("Year", "nunique"),
            N_PositiveMonths=("Month", "nunique"),
            FirstPositiveYear=("Year", "min"),
            LastPositiveYear=("Year", "max"),
        )
        .join(region_labels.set_index("LocationID"), how="left")
        .loc[station_sqrt.index]
        .reset_index()
    )
    station_raw = station_raw.loc[station_sqrt.index]
    station_matrix_raw = station_raw.to_numpy(float)
    station_matrix_sqrt = station_sqrt.to_numpy(float)
    region_station_coverage = station_meta[
        [
            "LocationID",
            "Region",
            "Region_Order",
            "N_PositiveEvents",
            "N_PositiveYears",
            "N_PositiveMonths",
            "FirstPositiveYear",
            "LastPositiveYear",
        ]
    ].copy()
    station_distance = squareform(pdist(station_matrix_sqrt, metric="braycurtis"))
    region_perm, region_disp, region_disp_groups = run_permanova_permdisp(
        station_distance,
        station_meta,
        "Region",
        [],
        N_PERMUTATIONS,
        RANDOM_SEED + 500_000,
        "Region_station_level",
        "Unrestricted permutations of independent LocationID rows",
    )
    region_analysis_rows.extend(
        [
            {"Analysis": "PERMANOVA", **region_perm},
            {"Analysis": "PERMDISP", **region_disp},
        ]
    )
    if len(region_disp_groups):
        region_disp_groups.insert(0, "AnalysisScope", "Region_station_level")
        dispersion_groups = pd.concat(
            [dispersion_groups, region_disp_groups], ignore_index=True
        )

    region_pair_rows = []
    region_names = (
        station_meta[["Region", "Region_Order"]]
        .drop_duplicates("Region")
        .sort_values(["Region_Order", "Region"], kind="stable", na_position="last")
        ["Region"]
        .astype(str)
        .tolist()
    )
    for idx, (region1, region2) in enumerate(combinations(region_names, 2)):
        mask = station_meta["Region"].astype(str).isin([region1, region2])
        sub_meta = station_meta.loc[mask].reset_index(drop=True)
        counts = sub_meta["Region"].astype(str).value_counts()
        if counts.min() < MIN_REGION_LOCATIONS:
            region_pair_rows.append(
                {
                    "Family": "Region_station_level",
                    "Contrast": f"{region1}_vs_{region2}",
                    "Group1": region1,
                    "Group2": region2,
                    "N_Events": len(sub_meta),
                    "N_Locations": len(sub_meta),
                    "PERMANOVA_Status": "SKIPPED",
                    "SkipReason": f"At least one region has < {MIN_REGION_LOCATIONS} locations",
                }
            )
            continue
        original_indices = np.flatnonzero(mask.to_numpy())
        sub_distance = station_distance[np.ix_(original_indices, original_indices)]
        perm, disp, _ = run_permanova_permdisp(
            sub_distance,
            sub_meta,
            "Region",
            [],
            N_PAIRWISE_PERMUTATIONS,
            RANDOM_SEED + 510_000 + idx * 5,
            f"Region::{region1}_vs_{region2}",
            "Unrestricted LocationID-level permutations",
        )
        region_pair_rows.append(
            {
                "Family": "Region_station_level",
                "Contrast": f"{region1}_vs_{region2}",
                "Group1": region1,
                "Group2": region2,
                "N_Events": len(sub_meta),
                "N_Locations": len(sub_meta),
                "PERMANOVA_Status": perm.get("Status"),
                "PERMANOVA_PseudoF": perm.get("PseudoF", np.nan),
                "PERMANOVA_R2": perm.get("R2", np.nan),
                "PERMANOVA_P": perm.get("P_Value", np.nan),
                "PERMDISP_Status": disp.get("Status"),
                "PERMDISP_F": disp.get("F", np.nan),
                "PERMDISP_P": disp.get("P_Value", np.nan),
            }
        )
    region_pairwise = pd.DataFrame(region_pair_rows).reindex(
        columns=REGION_PAIRWISE_COLUMNS
    )
    if len(region_pairwise):
        region_pairwise["PERMANOVA_P_FDR"] = bh_fdr(region_pairwise["PERMANOVA_P"])
        region_pairwise["PERMDISP_P_FDR"] = bh_fdr(region_pairwise["PERMDISP_P"])
        region_pairwise["PERMANOVA_FDR_Significant"] = region_pairwise[
            "PERMANOVA_P_FDR"
        ].lt(FDR_ALPHA)
        region_pairwise["PERMDISP_FDR_Significant"] = region_pairwise[
            "PERMDISP_P_FDR"
        ].lt(FDR_ALPHA)

        for _, row in region_pairwise.loc[
            region_pairwise["PERMANOVA_FDR_Significant"].fillna(False)
        ].iterrows():
            mask = station_meta["Region"].astype(str).isin([row["Group1"], row["Group2"]])
            region_simper = simper_two_groups(
                station_matrix_sqrt[mask.to_numpy()],
                station_matrix_raw[mask.to_numpy()],
                station_meta.loc[mask, "Region"].reset_index(drop=True),
                str(row["Group1"]),
                str(row["Group2"]),
                species_metadata,
                str(row["Contrast"]),
                "Region_station_level",
                bool(row["PERMDISP_FDR_Significant"]),
            )
            if len(region_simper):
                region_simper_tables.append(region_simper)
else:
    region_analysis_rows.append(
        {
            "Analysis": "Region",
            "Status": "SKIPPED",
            "SkipReason": region_message,
        }
    )

region_analysis_summary = pd.DataFrame(region_analysis_rows)
progress(f"区域分析完成：状态={region_status}")


# ============================================================
# 11. SIMPER：季节主比较与显著成对比较
# ============================================================

progress("运行配对季节SIMPER及全部符合条件的显著对比")
simper_tables: list[pd.DataFrame] = []

# 严格配对4月—11月始终计算
paired_indices = month_paired_subset["MatrixRow"].to_numpy(int)
month_omnibus_disp_row = omnibus_permdisp.loc[
    omnibus_permdisp["Test"].eq("Month_April_vs_November_paired")
]
month_disp_significant = (
    bool(month_omnibus_disp_row["P_Value"].iat[0] < FDR_ALPHA)
    if len(month_omnibus_disp_row)
    and pd.notna(month_omnibus_disp_row["P_Value"].iat[0])
    else None
)
month_simper = simper_two_groups(
    matrix_sqrt[paired_indices],
    matrix_raw[paired_indices],
    month_paired_subset["MonthLabel"].reset_index(drop=True),
    "April",
    "November",
    species_metadata,
    "Paired_April_vs_November",
    "Month_April_vs_November_paired",
    month_disp_significant,
    pair_ids=month_paired_subset["StationYear"].reset_index(drop=True),
    pair_id_name="StationYear",
)
if len(month_simper):
    simper_tables.append(month_simper)

# 对全部FDR显著时间成对比较追加SIMPER。若用户主动设置非零上限，
# 按比较家族轮转抽取，避免某一家族占满全部名额。
selected_pairs = pd.DataFrame()
if len(pairwise_results):
    eligible_pairs = pairwise_results.loc[
        pairwise_results["PERMANOVA_FDR_Significant"].fillna(False)
    ].copy()
    eligible_pairs["FamilyRank"] = (
        eligible_pairs.sort_values(
            ["PERMANOVA_P_FDR", "PERMANOVA_R2"],
            ascending=[True, False],
            kind="stable",
        )
        .groupby("Family")
        .cumcount()
    )
    eligible_pairs = eligible_pairs.sort_values(
        ["FamilyRank", "Family", "PERMANOVA_P_FDR", "PERMANOVA_R2"],
        ascending=[True, True, True, False],
        kind="stable",
    )
    selected_pairs = (
        eligible_pairs
        if MAX_SIMPER_CONTRASTS == 0
        else eligible_pairs.head(MAX_SIMPER_CONTRASTS)
    )
    for simper_index, (_, row) in enumerate(
        selected_pairs.iterrows(), start=1
    ):
        if simper_index == 1 or simper_index % 10 == 0:
            progress(f"时间SIMPER：{simper_index}/{len(selected_pairs)}项")
        key = f"{row['Family']}::{row['Contrast']}"
        context = pairwise_contexts.get(key)
        if context is None:
            continue
        subset = context["Subset"]
        indices = subset["MatrixRow"].to_numpy(int)
        pair_id_column = context["PairIDColumn"]
        if pair_id_column is None:
            raise AssertionError(
                f"时间成对SIMPER缺少唯一配对字段：{row['Family']}::{row['Contrast']}"
            )
        simper = simper_two_groups(
            matrix_sqrt[indices],
            matrix_raw[indices],
            subset[context["GroupColumn"]].reset_index(drop=True),
            context["Group1"],
            context["Group2"],
            species_metadata,
            str(row["Contrast"]),
            str(row["Family"]),
            bool(row["PERMDISP_FDR_Significant"]),
            pair_ids=subset[pair_id_column].reset_index(drop=True),
            pair_id_name=str(pair_id_column),
        )
        if len(simper):
            simper_tables.append(simper)

if region_simper_tables:
    simper_tables.extend(region_simper_tables)

simper_results = (
    pd.concat(simper_tables, ignore_index=True)
    if simper_tables
    else pd.DataFrame()
)

simper_output_keys = (
    set(
        zip(
            simper_results["SourceTest"].astype(str),
            simper_results["Contrast"].astype(str),
        )
    )
    if len(simper_results)
    else set()
)
simper_coverage_rows: list[dict[str, object]] = [
    {
        "SourceTest": "Month_April_vs_November_paired",
        "Contrast": "Paired_April_vs_November",
        "SIMPER_Design": "Paired_OneToOne",
        "PairID_Column": "StationYear",
        "EligibleForSIMPER": True,
        "IncludedInSIMPER": (
            "Month_April_vs_November_paired",
            "Paired_April_vs_November",
        )
        in simper_output_keys,
        "SelectionReason": "Primary paired seasonal contrast; always included",
    }
]

if len(pairwise_results):
    selected_time_keys = set(
        zip(selected_pairs["Family"].astype(str), selected_pairs["Contrast"].astype(str))
    )
    for _, row in pairwise_results.iterrows():
        key = (str(row["Family"]), str(row["Contrast"]))
        coverage_context = pairwise_contexts.get(f"{key[0]}::{key[1]}")
        eligible = bool(row["PERMANOVA_FDR_Significant"])
        selected = key in selected_time_keys
        simper_coverage_rows.append(
            {
                "SourceTest": key[0],
                "Contrast": key[1],
                "SIMPER_Design": "Paired_OneToOne",
                "PairID_Column": (
                    coverage_context.get("PairIDColumn")
                    if coverage_context is not None
                    else pd.NA
                ),
                "EligibleForSIMPER": eligible,
                "IncludedInSIMPER": key in simper_output_keys,
                "SelectionReason": (
                    "FDR-significant; included"
                    if eligible and selected
                    else "FDR-significant but excluded by user-specified cap"
                    if eligible
                    else "PERMANOVA not FDR-significant"
                ),
            }
        )

if len(region_pairwise):
    for _, row in region_pairwise.iterrows():
        key = ("Region_station_level", str(row["Contrast"]))
        eligible = bool(row.get("PERMANOVA_FDR_Significant", False))
        simper_coverage_rows.append(
            {
                "SourceTest": key[0],
                "Contrast": key[1],
                "SIMPER_Design": "Classical_AllCrossPairs",
                "PairID_Column": pd.NA,
                "EligibleForSIMPER": eligible,
                "IncludedInSIMPER": key in simper_output_keys,
                "SelectionReason": (
                    "Regional PERMANOVA FDR-significant; included"
                    if eligible
                    else "Regional PERMANOVA not FDR-significant or skipped"
                ),
            }
        )

simper_contrast_coverage = pd.DataFrame(simper_coverage_rows)
simper_eligible = simper_contrast_coverage["EligibleForSIMPER"].astype(bool)
simper_full_coverage = bool(
    simper_contrast_coverage.loc[simper_eligible, "IncludedInSIMPER"]
    .astype(bool)
    .all()
)
progress(
    f"SIMPER完成：符合条件={int(simper_eligible.sum())}项，"
    f"实际输出={len(simper_output_keys)}项"
)


# ============================================================
# 12. 质控、摘要和方法
# ============================================================

selected_nmds = nmds_stress.loc[nmds_stress["Selected"]].copy()
two_d_stress = float(
    selected_nmds.loc[selected_nmds["Dimensions"].eq(2), "Stress1"].iat[0]
)
recommended_value = nmds_stress["RecommendedDimensions"].iat[0]
recommended_dimensions: int | None = (
    None if pd.isna(recommended_value) else int(recommended_value)
)
recommended_dimensions_label = (
    str(recommended_dimensions)
    if recommended_dimensions is not None
    else "None: all selected 2D-4D solutions have Stress-1 > 0.20"
)
nmds_recommendation_status = str(nmds_stress["RecommendationStatus"].iat[0])
nmds_convergence_ok = bool(selected_nmds["ConvergenceOK"].astype(bool).all())
two_d_interpretable = bool(two_d_stress <= 0.20 and nmds_convergence_ok)
nmds_acceptable_found = recommended_dimensions is not None
paired_station_year_count = month_paired_subset["StationYear"].nunique()

qc_rows = [
    ("Surveyed events", EXPECTED_SURVEYED, surveyed_count, surveyed_count == EXPECTED_SURVEYED),
    ("Positive events used for beta diversity", EXPECTED_POSITIVE, len(event_metadata), len(event_metadata) == EXPECTED_POSITIVE),
    ("Zero events excluded from composition", EXPECTED_ZERO, len(zero_events), len(zero_events) == EXPECTED_ZERO),
    ("Species retained", EXPECTED_SPECIES, len(species_codes), len(species_codes) == EXPECTED_SPECIES),
    ("Duplicate positive EventID-Species", 0, duplicate_pairs, duplicate_pairs == 0),
    ("Matrix NA count", 0, int(np.isnan(matrix_raw).sum()), not np.isnan(matrix_raw).any()),
    ("Matrix negative count", 0, int((matrix_raw < 0).sum()), not (matrix_raw < 0).any()),
    ("All-zero positive event rows", 0, int((matrix_raw.sum(axis=1) <= 0).sum()), not (matrix_raw.sum(axis=1) <= 0).any()),
    ("Max biomass reconciliation difference", 0.0, max_biomass_reconciliation, max_biomass_reconciliation <= 1e-6),
    ("Bray-Curtis finite", True, bool(np.isfinite(distance_matrix).all()), bool(np.isfinite(distance_matrix).all())),
    ("Bray-Curtis symmetry max error", 0.0, float(np.max(np.abs(distance_matrix-distance_matrix.T))), np.allclose(distance_matrix,distance_matrix.T,atol=1e-12)),
    ("Bray-Curtis diagonal max", 0.0, float(np.max(np.abs(np.diag(distance_matrix)))), np.allclose(np.diag(distance_matrix),0.0,atol=1e-12)),
    ("Paired positive station-years", 530, paired_station_year_count, paired_station_year_count == 530),
    ("NMDS 2D finite", True, bool(np.isfinite(nmds_solutions[2]).all()), bool(np.isfinite(nmds_solutions[2]).all())),
    ("Omnibus PERMANOVA completed", 4, int(omnibus_permanova["Status"].eq("OK").sum()), int(omnibus_permanova["Status"].eq("OK").sum()) == 4),
    ("Omnibus PERMDISP completed", 4, int(omnibus_permdisp["Status"].eq("OK").sum()), int(omnibus_permdisp["Status"].eq("OK").sum()) == 4),
]
qc_checks = pd.DataFrame(qc_rows, columns=["Check", "Expected", "Observed", "Pass"])
qc_checks["Severity"] = "ERROR"
qc_checks = pd.concat(
    [
        qc_checks,
        pd.DataFrame(
            [
                {
                    "Check": "All selected NMDS solutions converged",
                    "Expected": True,
                    "Observed": nmds_convergence_ok,
                    "Pass": nmds_convergence_ok,
                    "Severity": "ERROR",
                },
                {
                    "Check": "Selected NMDS validation stability",
                    "Expected": (
                        f"|delta Stress-1|<={NMDS_STRESS_STABILITY_TOL:g}; "
                        f"configuration change<={NMDS_CONFIGURATION_STABILITY_TOL:g}"
                    ),
                    "Observed": " | ".join(
                        f"{int(row.Dimensions)}D: dStress="
                        f"{row.LastValidationStressChange:.3g}, dConfig="
                        f"{row.ConfigurationRelativeChange:.3g}"
                        for row in selected_nmds.itertuples()
                    ),
                    "Pass": bool(
                        selected_nmds["StressStable"].astype(bool).all()
                        and selected_nmds["ConfigurationStable"].astype(bool).all()
                        and selected_nmds["ObjectiveNotWorse"].astype(bool).all()
                    ),
                    "Severity": "ERROR",
                },
                {
                    "Check": "2D NMDS interpretable",
                    "Expected": "Stress-1 <= 0.20",
                    "Observed": two_d_stress,
                    "Pass": two_d_stress <= 0.20,
                    "Severity": "WARNING",
                },
                {
                    "Check": "Acceptable NMDS dimension found",
                    "Expected": "At least one selected solution with Stress-1 <= 0.20",
                    "Observed": nmds_recommendation_status,
                    "Pass": nmds_acceptable_found,
                    "Severity": "WARNING",
                },
                {
                    "Check": "SIMPER eligible contrast coverage",
                    "Expected": int(simper_eligible.sum()),
                    "Observed": int(
                        simper_contrast_coverage.loc[
                            simper_eligible, "IncludedInSIMPER"
                        ].astype(bool).sum()
                    ),
                    "Pass": simper_full_coverage,
                    "Severity": "ERROR" if MAX_SIMPER_CONTRASTS == 0 else "WARNING",
                },
                {
                    "Check": "Formal region analysis completed",
                    "Expected": "READY",
                    "Observed": region_status,
                    "Pass": region_status == "READY",
                    "Severity": "WARNING",
                },
            ]
        ),
    ],
    ignore_index=True,
)
fatal_qc_failures = qc_checks.loc[
    qc_checks["Severity"].eq("ERROR") & ~qc_checks["Pass"].astype(bool)
]
if len(fatal_qc_failures):
    raise AssertionError(
        "第五步关键质控未通过：\n" + fatal_qc_failures.to_string(index=False)
    )

method_rows = [
    ("Run started UTC", RUN_STARTED_UTC),
    ("Step3 input directory", str(INPUT_DIR)),
    ("Step5 output directory", str(OUTPUT_DIR)),
    ("Analysis unit", "One positive-catch EventID = Year + Month + LocationID"),
    ("Surveyed events retained in QC", surveyed_count),
    ("Positive events in composition analysis", len(event_metadata)),
    ("Zero-catch handling", "568 zero-catch events excluded from compositional distances; retained in QC"),
    ("Species retained", len(species_codes)),
    ("Primary community response", "sqrt(Biomass_kg_km2) for all 25 species"),
    ("Dissimilarity", "Bray-Curtis"),
    ("NMDS dimensions evaluated", "2, 3, 4"),
    ("NMDS initialization", f"Each dimension: PCoA start and best of {NMDS_RANDOM_STARTS} random starts"),
    ("NMDS convergence tolerance", NMDS_EPS),
    ("NMDS maximum primary iterations", NMDS_MAX_ITER),
    ("NMDS independent validation iterations", NMDS_VALIDATION_ITER),
    ("NMDS maximum validation rounds", NMDS_MAX_VALIDATION_ROUNDS),
    ("NMDS Stress-1 stability tolerance", NMDS_STRESS_STABILITY_TOL),
    ("NMDS configuration stability tolerance", NMDS_CONFIGURATION_STABILITY_TOL),
    ("NMDS recommended dimensions", recommended_dimensions_label),
    ("NMDS recommendation status", nmds_recommendation_status),
    ("NMDS 2D Stress-1", two_d_stress),
    ("Python package versions", f"numpy={np.__version__}; scipy={scipy.__version__}; scikit-learn={sklearn.__version__}; pandas={pd.__version__}"),
    ("Omnibus permutations", N_PERMUTATIONS),
    ("Pairwise permutations", N_PAIRWISE_PERMUTATIONS),
    ("Month primary comparison", "Only station-year blocks positive in both April and November"),
    ("Month permutation restriction", "Within LocationID-Year"),
    ("Year permutation restriction", "Within LocationID, separately by month"),
    ("YearMonth 22-group test", "One-factor YearMonth group test; not a factorial Year x Month interaction"),
    ("YearMonth permutation restriction", "Within LocationID"),
    ("PERMDISP Euclidean correction", "Lingoes correction before distance-to-centroid calculation"),
    ("Multiple testing", "Benjamini-Hochberg FDR within each pairwise family"),
    ("SIMPER basis", "sqrt biomass Bray-Curtis contribution"),
    ("Temporal SIMPER design", "One-to-one matched pairs using StationYear or LocationID"),
    ("Regional SIMPER design", "Classical all-cross-pair SIMPER among independent LocationID means"),
    ("SIMPER contrast selection", "All FDR-significant pairwise contrasts" if MAX_SIMPER_CONTRASTS == 0 else f"Family-balanced cap of {MAX_SIMPER_CONTRASTS} contrasts"),
    ("SIMPER interpretation", "Descriptive contribution decomposition; not a species-level significance test"),
    ("Region analysis status", region_status),
    ("Region mapping source", region_mapping_source),
    ("Region mapping requirement", REQUIRE_REGION_MAPPING),
    ("Region analysis estimand", "Conditional positive-catch assemblage composition"),
    ("Region analysis design", "Mean of event-level sqrt biomass per LocationID; one independent station row"),
    ("Random seed", RANDOM_SEED),
]
method_metadata = pd.DataFrame(method_rows, columns=["Item", "Definition"])

summary_rows = [
    ("Step", "Step5 beta diversity and assemblage reorganization"),
    ("Surveyed events in QC", surveyed_count),
    ("Positive events analyzed", len(event_metadata)),
    ("Zero events excluded from compositional distance", len(zero_events)),
    ("Positive-event locations", event_metadata["LocationID"].nunique()),
    ("Species retained", len(species_codes)),
    ("Paired positive LocationID-Year blocks", paired_station_year_count),
    ("Paired month-comparison events", len(month_paired_subset)),
    ("Mean Bray-Curtis distance", float(distance_condensed.mean())),
    ("Median Bray-Curtis distance", float(np.median(distance_condensed))),
    ("NMDS 2D Stress-1", two_d_stress),
    ("NMDS recommended dimensions", recommended_dimensions_label),
    ("NMDS recommendation status", nmds_recommendation_status),
    ("NMDS convergence passed", nmds_convergence_ok),
    ("Omnibus PERMANOVA tests completed", int(omnibus_permanova["Status"].eq("OK").sum())),
    ("Pairwise PERMANOVA tests completed", int(pairwise_results["PERMANOVA_Status"].eq("OK").sum())),
    ("Pairwise PERMANOVA FDR significant", int(pairwise_results.get("PERMANOVA_FDR_Significant", pd.Series(dtype=bool)).fillna(False).sum())),
    ("SIMPER eligible contrasts", int(simper_eligible.sum())),
    ("SIMPER contrasts output", len(simper_output_keys)),
    ("SIMPER full eligible coverage", simper_full_coverage),
    ("Region analysis status", region_status),
    ("Region message", region_message),
]
step5_summary = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

sample_inclusion = pd.DataFrame(
    [
        {
            "SurveyStatus": STATUS_POSITIVE,
            "N_Events": len(positive_events),
            "UseInBrayNMDS_PERMANOVA_SIMPER": True,
            "Reason": "Positive community composition is defined",
        },
        {
            "SurveyStatus": STATUS_ZERO,
            "N_Events": len(zero_events),
            "UseInBrayNMDS_PERMANOVA_SIMPER": False,
            "Reason": "All-zero sample has no assemblage composition; retained for occurrence/resource analyses",
        },
    ]
)


# ============================================================
# 13. 图件
# ============================================================

progress("生成论文图件")
configure_plot_style()
figure_paths: list[Path] = []


def add_data_ellipse(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    level_chi2: float = 5.991,
) -> None:
    if len(x) < 3:
        return
    covariance = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = math.degrees(math.atan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * np.sqrt(np.maximum(eigenvalues, 0) * level_chi2)
    ellipse = Ellipse(
        (np.mean(x), np.mean(y)),
        width,
        height,
        angle=angle,
        facecolor="none",
        edgecolor=color,
        linewidth=1.5,
    )
    ax.add_patch(ellipse)


# Fig01：2D NMDS季节
coords2 = nmds_coordinate_tables[2]
fig, ax = plt.subplots(figsize=(8.2, 6.4))
for month in [4, 11]:
    subset = coords2.loc[coords2["Month"].eq(month)]
    color = MONTH_COLORS[month]
    ax.scatter(
        subset["NMDS1"],
        subset["NMDS2"],
        s=12,
        alpha=0.30,
        color=color,
        edgecolor="none",
        label=f"{MONTH_LABELS[month]} (n={len(subset)})",
    )
    add_data_ellipse(
        ax,
        subset["NMDS1"].to_numpy(float),
        subset["NMDS2"].to_numpy(float),
        color,
    )
    ax.scatter(
        subset["NMDS1"].mean(),
        subset["NMDS2"].mean(),
        marker="X",
        s=80,
        color=color,
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )
ax.set_xlabel("NMDS1")
ax.set_ylabel("NMDS2")
ax.set_title(
    f"Cephalopod assemblage composition by survey month\n"
    f"Bray–Curtis on sqrt biomass; 2D Stress-1 = {two_d_stress:.3f}"
)
ax.grid(color="#E3E3E3", linewidth=0.5)
ax.legend(frameon=False)
if two_d_interpretable:
    figure_paths.extend(save_figure(fig, "Fig01_NMDS_month"))
else:
    plt.close(fig)

# Fig02：年份质心轨迹
centroids = (
    coords2.groupby(["Year", "Month"], as_index=False)
    .agg(N=("EventID", "size"), NMDS1=("NMDS1", "mean"), NMDS2=("NMDS2", "mean"))
    .sort_values(["Month", "Year"])
)
fig, ax = plt.subplots(figsize=(8.5, 6.6))
for month in [4, 11]:
    subset = centroids.loc[centroids["Month"].eq(month)].sort_values("Year")
    color = MONTH_COLORS[month]
    ax.plot(
        subset["NMDS1"],
        subset["NMDS2"],
        color=color,
        marker="o",
        linewidth=1.4,
        markersize=4,
        label=MONTH_LABELS[month],
    )
    for _, row in subset.iterrows():
        ax.annotate(
            str(int(row["Year"])),
            (row["NMDS1"], row["NMDS2"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
            color=color,
        )
ax.set_xlabel("NMDS1 centroid")
ax.set_ylabel("NMDS2 centroid")
ax.set_title("Interannual trajectories of assemblage centroids")
ax.grid(color="#E3E3E3", linewidth=0.5)
ax.legend(frameon=False)
if two_d_interpretable:
    figure_paths.extend(save_figure(fig, "Fig02_NMDS_year_centroid_trajectory"))
else:
    plt.close(fig)

# Fig03：NMDS维数—Stress
stress_plot = selected_nmds.sort_values("Dimensions")
fig, ax = plt.subplots(figsize=(6.5, 4.6))
ax.plot(
    stress_plot["Dimensions"],
    stress_plot["Stress1"],
    marker="o",
    color="#2C7FB8",
    linewidth=1.5,
)
ax.axhline(0.20, color="#D95F0E", linestyle="--", linewidth=1, label="Stress = 0.20")
ax.axhline(0.10, color="#31A354", linestyle=":", linewidth=1, label="Stress = 0.10")
for _, row in stress_plot.iterrows():
    ax.annotate(
        f"{row['Stress1']:.3f}",
        (row["Dimensions"], row["Stress1"]),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        fontsize=8,
    )
ax.set_xticks([2, 3, 4])
ax.set_xlabel("NMDS dimensions")
ax.set_ylabel("Kruskal Stress-1")
ax.set_title("NMDS dimensionality assessment")
ax.grid(axis="y", color="#E3E3E3", linewidth=0.5)
ax.legend(frameon=False)
figure_paths.extend(save_figure(fig, "Fig03_NMDS_stress_dimensions"))

# Fig04：主PERMANOVA效应量，阴影表示PERMDISP显著
perm_plot = omnibus_permanova.loc[omnibus_permanova["Status"].eq("OK")].copy()
perm_plot = perm_plot.merge(
    omnibus_permdisp[["Test", "P_Value"]].rename(columns={"P_Value": "PERMDISP_P"}),
    on="Test",
    how="left",
)
test_display_labels = {
    "YearMonth_22group_one_factor": "Year-month groups\n(22-level one-factor test)",
    "Month_April_vs_November_paired": "April vs November\n(paired stations)",
    "Year_within_April": "Year within April",
    "Year_within_November": "Year within November",
}
perm_plot["DisplayLabel"] = perm_plot["Test"].map(test_display_labels).fillna(perm_plot["Test"])
fig, ax = plt.subplots(figsize=(8.3, 4.8))
x = np.arange(len(perm_plot))
bars = ax.bar(x, perm_plot["R2"], color="#4C78A8", edgecolor="#1F1F1F", linewidth=0.5)
for bar, (_, row) in zip(bars, perm_plot.iterrows()):
    if pd.notna(row["PERMDISP_P"]) and row["PERMDISP_P"] < FDR_ALPHA:
        bar.set_hatch("///")
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"P={row['P_Value']:.3g}",
        ha="center",
        va="bottom",
        fontsize=7,
    )
ax.set_xticks(x)
ax.set_xticklabels(perm_plot["DisplayLabel"], rotation=18, ha="right")
ax.set_ylabel("PERMANOVA R²")
ax.set_title("Temporal differences in cephalopod assemblage composition")
ax.grid(axis="y", color="#E3E3E3", linewidth=0.5)
ax.legend(
    handles=[Patch(facecolor="#4C78A8", edgecolor="#1F1F1F", hatch="///", label="PERMDISP P < 0.05")],
    frameon=False,
    loc="upper right",
)
figure_paths.extend(save_figure(fig, "Fig04_PERMANOVA_effect_sizes"))

# Fig05：4月—11月SIMPER
month_simper_plot = month_simper.head(min(15, len(month_simper))).sort_values(
    "Percent_Contribution", ascending=True
)
fig, ax = plt.subplots(figsize=(8.0, 6.2))
ax.barh(
    np.arange(len(month_simper_plot)),
    month_simper_plot["Percent_Contribution"],
    color="#7FCDBB",
    edgecolor="none",
)
ax.set_yticks(np.arange(len(month_simper_plot)))
ax.set_yticklabels(month_simper_plot["Species_Latin"])
for tick in ax.get_yticklabels():
    tick.set_fontstyle("italic")
ax.set_xlabel("SIMPER contribution to April–November dissimilarity (%)")
ax.set_title("Species contributions to seasonal assemblage dissimilarity")
ax.grid(axis="x", color="#E3E3E3", linewidth=0.5)
figure_paths.extend(save_figure(fig, "Fig05_SIMPER_April_November"))

# Fig06：Shepard图，抽样防止矢量文件过大
rng_plot = np.random.default_rng(RANDOM_SEED)
n_shepard = len(shepard_payload["OriginalDissimilarity"])
sample_size = min(80_000, n_shepard)
sample_idx = rng_plot.choice(n_shepard, size=sample_size, replace=False)
fig, ax = plt.subplots(figsize=(6.5, 5.2))
ax.scatter(
    shepard_payload["OriginalDissimilarity"][sample_idx],
    shepard_payload["OrdinationDistance"][sample_idx],
    s=5,
    alpha=0.10,
    color="#4C78A8",
    edgecolor="none",
    rasterized=True,
    label="Pairwise distances (sampled)",
)
# 用全体距离的分箱均值绘制单调拟合趋势；避免对抽样点连线产生锯齿假象。
shepard_curve = pd.DataFrame(
    {
        "Observed": shepard_payload["OriginalDissimilarity"],
        "Disparity": shepard_payload["Disparity"],
    }
)
shepard_curve["Bin"] = pd.cut(
    shepard_curve["Observed"],
    bins=np.linspace(0.0, 1.0, 201),
    include_lowest=True,
)
shepard_curve = (
    shepard_curve.groupby("Bin", observed=True, as_index=False)
    .agg(Observed=("Observed", "mean"), Disparity=("Disparity", "mean"))
    .dropna(subset=["Observed", "Disparity"])
    .sort_values("Observed")
)
ax.plot(
    shepard_curve["Observed"],
    shepard_curve["Disparity"],
    color="#D95F0E",
    linewidth=1.6,
    label="Monotonic fit (binned)",
)
ax.set_xlabel("Observed Bray–Curtis dissimilarity")
ax.set_ylabel("NMDS ordination distance")
ax.set_title("Shepard diagram for the selected 2D NMDS solution")
ax.grid(color="#E3E3E3", linewidth=0.5)
ax.legend(frameon=False, loc="upper left")
if two_d_interpretable:
    figure_paths.extend(save_figure(fig, "Fig06_NMDS_Shepard"))
else:
    plt.close(fig)

# Matplotlib在含rasterized图层的PDF/SVG保存过程中可能留下隐藏的临时PNG。
for temporary_file in FIGURE_DIR.glob(".*.tmp"):
    temporary_file.unlink(missing_ok=True)


# ============================================================
# 14. 输出
# ============================================================

progress("写出CSV、距离文件、图件清单和ZIP")
output_tables: OrderedDict[str, pd.DataFrame] = OrderedDict(
    [
        ("00_step5_summary.csv", step5_summary),
        ("01_sample_inclusion_rules.csv", sample_inclusion),
        ("02_species_metadata_all25.csv", species_metadata),
        ("03_positive_event_metadata.csv", event_metadata),
        (
            "04_species_matrix_biomass_raw_positive.csv",
            biomass_matrix_df.reset_index(),
        ),
        (
            "05_species_matrix_sqrt_biomass_positive.csv",
            pd.DataFrame(matrix_sqrt, columns=species_codes).assign(
                EventID=event_metadata["EventID"].to_numpy()
            )[["EventID"] + species_codes],
        ),
        ("06_nmds_stress_by_dimension.csv", nmds_stress),
        ("07_nmds_coordinates_2D.csv", nmds_coordinate_tables[2]),
        ("08_nmds_coordinates_3D.csv", nmds_coordinate_tables[3]),
        ("09_nmds_coordinates_4D.csv", nmds_coordinate_tables[4]),
        ("10_permanova_omnibus.csv", omnibus_permanova),
        ("11_permdisp_omnibus.csv", omnibus_permdisp),
        ("12_permdisp_group_distances.csv", dispersion_groups),
        ("13_pairwise_permanova_permdisp.csv", pairwise_results),
        ("14_simper_results.csv", simper_results),
        ("14b_simper_contrast_coverage.csv", simper_contrast_coverage),
        ("15_region_mapping_template.csv", location_template),
        ("15a_region_mapping_used.csv", region_mapping_used_output),
        ("15b_region_mapping_qc.csv", region_mapping_qc),
        ("15c_region_group_sizes.csv", region_group_sizes),
        ("15d_region_station_temporal_coverage.csv", region_station_coverage),
        ("16_region_analysis_summary.csv", region_analysis_summary),
        ("17_region_pairwise_results.csv", region_pairwise),
        ("18_method_metadata.csv", method_metadata),
        ("19_qc_checks.csv", qc_checks),
    ]
)

output_paths: list[Path] = []
for filename, table in output_tables.items():
    output_paths.append(write_csv(table, filename))

distance_event_ids_path = write_csv(
    event_metadata[["MatrixRow", "EventID"]],
    "20_bray_curtis_event_order.csv",
)
output_paths.append(distance_event_ids_path)

distance_npz_path = OUTPUT_DIR / "21_bray_curtis_condensed_sqrt_biomass.npz"
np.savez_compressed(
    distance_npz_path,
    condensed_distance=distance_condensed.astype(np.float32),
    event_id=event_metadata["EventID"].astype(str).to_numpy(dtype=str),
    transform=np.array(["sqrt biomass"], dtype="U32"),
    metric=np.array(["Bray-Curtis"], dtype="U32"),
)
output_paths.append(distance_npz_path)

if SAVE_FULL_DISTANCE_CSV:
    full_distance_path = OUTPUT_DIR / "22_bray_curtis_full_distance_matrix.csv.gz"
    pd.DataFrame(
        distance_matrix,
        index=event_metadata["EventID"],
        columns=event_metadata["EventID"],
    ).to_csv(full_distance_path, compression="gzip", encoding="utf-8-sig")
    output_paths.append(full_distance_path)

manifest_rows = []
for filename, table in output_tables.items():
    manifest_rows.append(
        {
            "File": filename,
            "Rows": len(table),
            "Columns": len(table.columns),
            "Purpose": "Tabular analysis output",
        }
    )
manifest_rows.extend(
    [
        {
            "File": distance_event_ids_path.name,
            "Rows": len(event_metadata),
            "Columns": 2,
            "Purpose": "Event order for condensed Bray-Curtis distance",
        },
        {
            "File": distance_npz_path.name,
            "Rows": pd.NA,
            "Columns": pd.NA,
            "Purpose": "Compressed condensed Bray-Curtis distance",
        },
    ]
)
if SAVE_FULL_DISTANCE_CSV:
    manifest_rows.append(
        {
            "File": full_distance_path.name,
            "Rows": len(event_metadata),
            "Columns": len(event_metadata),
            "Purpose": "Optional full Bray-Curtis distance matrix",
        }
    )
for path in figure_paths:
    manifest_rows.append(
        {
            "File": str(path.relative_to(OUTPUT_DIR)),
            "Rows": pd.NA,
            "Columns": pd.NA,
            "Purpose": "Publication figure",
        }
    )
manifest_rows.extend(
    [
        {
            "File": "README_step5.txt",
            "Rows": pd.NA,
            "Columns": pd.NA,
            "Purpose": "Methods, design, and interpretation notes",
        },
        {
            "File": "22_output_manifest.csv",
            "Rows": pd.NA,
            "Columns": 4,
            "Purpose": "Output manifest",
        },
    ]
)
output_manifest = pd.DataFrame(manifest_rows)
manifest_path = write_csv(output_manifest, "22_output_manifest.csv")
output_paths.append(manifest_path)

readme = f"""浙江近岸头足类第五步：β多样性与群落重组
============================================================

分析样本
--------
- 已调查事件质控总数：{surveyed_count}
- 正捕获事件：{len(event_metadata)}，进入群落组成分析
- 零捕获事件：{len(zero_events)}，不进入Bray-Curtis，但没有从调查数据库删除
- 物种：{len(species_codes)}

主分析
------
- 群落矩阵：sqrt(Biomass_kg_km2)
- 距离：Bray-Curtis
- NMDS：2D、3D、4D
- 2D Stress-1：{two_d_stress:.6f}
- 最低达到Stress-1阈值的维数：{recommended_dimensions_label}
- 阈值状态：{nmds_recommendation_status}
- 每个维数比较PCoA起点与{NMDS_RANDOM_STARTS}个随机起点中的最佳解；
- 主优化使用eps={NMDS_EPS:g}、max_iter={NMDS_MAX_ITER}；随后每轮最多续跑
  {NMDS_VALIDATION_ITER}次、最多{NMDS_MAX_VALIDATION_ROUNDS}轮，验证Stress-1和构型稳定性；
- 所选解主优化+验证迭代数：{" | ".join(f"{int(row.Dimensions)}D={int(row.PrimaryIterations)}+{int(row.ValidationIterations)}" for row in selected_nmds.itertuples())}
- 运行环境：numpy={np.__version__}；scipy={scipy.__version__}；
  scikit-learn={sklearn.__version__}；pandas={pd.__version__}

置换设计
--------
- YearMonth：LocationID内受限置换；
- 月份主比较：仅使用同一LocationID-Year两个季节均正捕获的配对事件，
  在LocationID-Year内置换；
- 年份比较：分别在4月和11月内，在LocationID内置换；
- 成对比较：仅使用两个组都存在正捕获的共同LocationID；
- FDR：在各成对比较家族内进行Benjamini-Hochberg校正；
- PERMDISP：使用Lingoes校正后的到组质心距离。

SIMPER
------
- 时间SIMPER使用真正一对一配对：季节主比较按StationYear，时间成对比较按LocationID；
- 区域SIMPER仅对独立站位长期均值使用经典全部跨组样本对；
- 14_simper_results.csv的SIMPER_Design、PairID_Column和N_MatchedPairs记录设计；
- 默认包含全部FDR显著时间成对比较；
- 当前符合条件：{int(simper_eligible.sum())}项，实际输出：{len(simper_output_keys)}项；
- 14b_simper_contrast_coverage.csv逐项记录是否纳入，完整覆盖={simper_full_coverage}；
- 只有主动设置CEPHALOPOD_MAX_SIMPER_CONTRASTS为正数时才限制数量，
  并按比较家族轮转选择；
- SIMPER仅用于说明哪些物种贡献了Bray-Curtis差异，不能替代物种级显著性检验。

区域分析
--------
- 状态：{region_status}
- 说明：{region_message}
- 正式区域映射在任何NMDS或置换计算开始前自动定位并严格验收；
- 本次映射来源：{FORMAL_MAPPING_SOURCE}
- 正式映射必须覆盖124个唯一LocationID，三区全部标准站位数必须为
  South=43、Central=28、North=53；任一检查失败时程序立即停止；
- 15b_region_mapping_qc.csv记录正式映射覆盖和分组数量检查；
- 15a_region_mapping_used.csv保存实际采用的区域、区域顺序和区域定义；
- 15c_region_group_sizes.csv记录各区域正捕获站位数和事件数；
- 15d_region_station_temporal_coverage.csv记录各站位的正捕获时间覆盖；
- 每个正捕获LocationID必须且只能属于一个区域，每区至少
  {MIN_REGION_LOCATIONS}个正捕获站位；
- 不应根据NMDS结果事后划区。

关键解释规则
------------
- PERMANOVA显著且PERMDISP不显著：较支持群落质心/组成差异；
- PERMANOVA和PERMDISP均显著：差异可能同时包含组中心移动和组内异质性变化；
- 2D Stress > 0.20时，不输出二维NMDS分组图、质心轨迹图和Shepard图；
- 若2—4D均无Stress <= 0.20的收敛解，推荐维数明确显示为None；
- 未经成对检验和FDR校正，不对具体年份作显著差异结论。
"""
readme_path = OUTPUT_DIR / "README_step5.txt"
readme_path.write_text(readme, encoding="utf-8")
output_paths.append(readme_path)

output_paths.extend(figure_paths)

zip_path = Path(str(OUTPUT_DIR) + ".zip")
temporary_zip_path = Path(str(zip_path) + ".tmp")
with zipfile.ZipFile(
    temporary_zip_path,
    "w",
    compression=zipfile.ZIP_DEFLATED,
) as archive:
    for path in output_paths:
        archive.write(path, arcname=str(path.relative_to(OUTPUT_DIR)))
temporary_zip_path.replace(zip_path)
run_completed_utc = datetime.now(timezone.utc).isoformat()
RUN_STATUS_PATH.write_text(
    "Status: SUCCESS\n"
    f"RunStartedUTC: {RUN_STARTED_UTC}\n"
    f"RunCompletedUTC: {run_completed_utc}\n"
    f"OutputZIP: {zip_path}\n",
    encoding="utf-8",
)
os.environ["CEPHALOPOD_STEP5_REGION_SOURCE"] = str(zip_path)


# ============================================================
# 15. Kaggle控制台摘要
# ============================================================

print("=" * 78)
print("第五步完成：浙江近岸头足类β多样性与群落重组")
print("=" * 78)
print(f"已调查事件（质控）：{surveyed_count:,}")
print(f"正捕获事件（群落组成）：{len(event_metadata):,}")
print(f"零捕获事件（不进入Bray-Curtis）：{len(zero_events):,}")
print(f"全部物种：{len(species_codes)}")
print(f"配对月份站位年：{paired_station_year_count:,}（{len(month_paired_subset):,}个事件）")
print(f"2D NMDS Stress-1：{two_d_stress:.4f}")
print(f"推荐NMDS维数：{recommended_dimensions_label}")
print(
    "NMDS所选解总迭代："
    + " | ".join(
        f"{int(row.Dimensions)}D={int(row.Iterations)}"
        for row in selected_nmds.itertuples()
    )
)
print(f"主PERMANOVA完成：{int(omnibus_permanova['Status'].eq('OK').sum())}项")
print(f"成对PERMANOVA完成：{int(pairwise_results['PERMANOVA_Status'].eq('OK').sum())}项")
print(
    "成对PERMANOVA经FDR显著："
    f"{int(pairwise_results['PERMANOVA_FDR_Significant'].fillna(False).sum())}项"
)
print(f"SIMPER符合条件/输出：{int(simper_eligible.sum())}/{len(simper_output_keys)}项")
print(f"区域分析状态：{region_status}")
print(f"关键质控通过：{not len(fatal_qc_failures)}")
print(
    "质控警告："
    f"{int((qc_checks['Severity'].eq('WARNING') & ~qc_checks['Pass'].astype(bool)).sum())}项"
)
print(f"输出目录：{OUTPUT_DIR}")
print(f"ZIP文件：{zip_path}")
print(f"请下载：{zip_path.name}")


# ============================================================================
# 第二阶段：区域PERMANOVA、PERMDISP及区域SIMPER详细整理
# 主分析成功后自动执行；不会重新运行NMDS或置换。
# ============================================================================

os.environ.setdefault(
    "CEPHALOPOD_EXPECTED_REGION_PERMUTATIONS",
    os.environ.get("CEPHALOPOD_PERMUTATIONS", "999"),
)

import math
import os
import zipfile
from collections import OrderedDict
from pathlib import Path, PurePosixPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


# ============================================================
# 0. 参数
# ============================================================

SOURCE_EXPLICIT = os.environ.get(
    "CEPHALOPOD_STEP5_REGION_SOURCE", ""
).strip()
OUTPUT_DIR = Path(
    os.environ.get(
        "CEPHALOPOD_STEP5B_OUTPUT_DIR",
        "/kaggle/working/cephalopod_step5b_region_detailed_analysis",
    )
)
FIGURE_DIR = OUTPUT_DIR / "figures"

ALPHA = float(os.environ.get("CEPHALOPOD_REGION_ALPHA", "0.05"))
SIMPER_CUMULATIVE_PERCENT = float(
    os.environ.get("CEPHALOPOD_SIMPER_CUMULATIVE_PERCENT", "80")
)
LOW_COVERAGE_EVENT_THRESHOLD = int(
    os.environ.get("CEPHALOPOD_REGION_LOW_COVERAGE_EVENTS", "3")
)
EXPECTED_POSITIVE_EVENTS = int(
    os.environ.get("CEPHALOPOD_EXPECTED_POSITIVE_EVENTS", "1532")
)
EXPECTED_POSITIVE_LOCATIONS = int(
    os.environ.get("CEPHALOPOD_EXPECTED_POSITIVE_LOCATIONS", "120")
)
EXPECTED_STANDARD_LOCATIONS = int(
    os.environ.get("CEPHALOPOD_EXPECTED_ALL_LOCATIONS", "124")
)
EXPECTED_SPECIES = int(
    os.environ.get("CEPHALOPOD_EXPECTED_SPECIES_COUNT", "25")
)
EXPECTED_PERMUTATIONS = int(
    os.environ.get("CEPHALOPOD_EXPECTED_REGION_PERMUTATIONS", "999")
)
FIGURE_DPI = int(os.environ.get("CEPHALOPOD_FIGURE_DPI", "600"))

REGION_COLORS = {
    "South": "#D55E00",
    "Central": "#009E73",
    "North": "#0072B2",
}

REQUIRED_FILES = [
    "02_species_metadata_all25.csv",
    "03_positive_event_metadata.csv",
    "04_species_matrix_biomass_raw_positive.csv",
    "05_species_matrix_sqrt_biomass_positive.csv",
    "12_permdisp_group_distances.csv",
    "14_simper_results.csv",
    "14b_simper_contrast_coverage.csv",
    "15a_region_mapping_used.csv",
    "15b_region_mapping_qc.csv",
    "15c_region_group_sizes.csv",
    "15d_region_station_temporal_coverage.csv",
    "16_region_analysis_summary.csv",
    "17_region_pairwise_results.csv",
    "18_method_metadata.csv",
    "19_qc_checks.csv",
]


# ============================================================
# 1. 通用函数和输入定位
# ============================================================

def progress(message: str) -> None:
    print(f"[Step5b] {message}", flush=True)


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"{name}缺少字段：{missing}")


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin(["true", "1", "yes", "y", "pass", "ok"])


def zip_member_map(zip_path: Path) -> dict[str, str]:
    matches: dict[str, list[str]] = {name: [] for name in REQUIRED_FILES}
    with zipfile.ZipFile(zip_path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise zipfile.BadZipFile(f"ZIP成员CRC检查失败：{bad_member}")
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"ZIP含不安全路径：{info.filename}")
            if member.name in matches:
                matches[member.name].append(info.filename)

    missing = [name for name, found in matches.items() if not found]
    duplicated = {name: found for name, found in matches.items() if len(found) > 1}
    if missing:
        raise FileNotFoundError(f"区域版第五步ZIP缺少文件：{missing}")
    if duplicated:
        raise RuntimeError(f"区域版第五步ZIP存在多个同名结果：{duplicated}")
    return {name: found[0] for name, found in matches.items()}


def directory_file_map(directory: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for name in REQUIRED_FILES:
        direct = directory / name
        if direct.exists():
            result[name] = direct
            continue
        found = list(directory.rglob(name))
        if len(found) != 1:
            raise FileNotFoundError(
                f"{directory}中{name}数量应为1，实际为{len(found)}"
            )
        result[name] = found[0]
    return result


def is_valid_source(path: Path) -> bool:
    try:
        if path.is_dir():
            directory_file_map(path)
            return True
        if path.is_file() and path.suffix.lower() == ".zip":
            zip_member_map(path)
            return True
    except Exception:
        return False
    return False


def locate_source() -> Path:
    if SOURCE_EXPLICIT:
        explicit = Path(SOURCE_EXPLICIT)
        if not is_valid_source(explicit):
            raise FileNotFoundError(
                "CEPHALOPOD_STEP5_REGION_SOURCE不是有效的区域版第五步结果："
                f"{explicit}"
            )
        return explicit

    preferred = [
        Path("/kaggle/working/cephalopod_step5_beta_diversity_region"),
        Path("/kaggle/working/cephalopod_step5_beta_diversity_region.zip"),
    ]
    for candidate in preferred:
        if is_valid_source(candidate):
            return candidate

    candidates: list[Path] = []
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for candidate in input_root.rglob(
            "cephalopod_step5_beta_diversity_region*.zip"
        ):
            if is_valid_source(candidate):
                candidates.append(candidate.resolve())
    candidates = sorted(set(candidates))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            "没有找到cephalopod_step5_beta_diversity_region结果。请上传ZIP，"
            "或设置CEPHALOPOD_STEP5_REGION_SOURCE。"
        )
    raise RuntimeError(
        "发现多个区域版第五步ZIP，请显式设置CEPHALOPOD_STEP5_REGION_SOURCE：\n"
        + "\n".join(str(path) for path in candidates)
    )


def read_source_csv(source: Path, filename: str) -> pd.DataFrame:
    if source.is_dir():
        paths = directory_file_map(source)
        return pd.read_csv(paths[filename], encoding="utf-8-sig")
    members = zip_member_map(source)
    with zipfile.ZipFile(source, "r") as archive:
        with archive.open(members[filename], "r") as stream:
            return pd.read_csv(stream, encoding="utf-8-sig")


def joint_interpretation(
    permanova_significant: bool,
    permdisp_significant: bool,
) -> tuple[str, str]:
    if permanova_significant and permdisp_significant:
        return (
            "PERMANOVA_AND_PERMDISP_SIGNIFICANT",
            "群落差异显著，但同时存在组内离散度差异；不能解释为纯粹的群落质心移动。",
        )
    if permanova_significant and not permdisp_significant:
        return (
            "CENTROID_DIFFERENCE_SUPPORTED",
            "PERMANOVA显著且PERMDISP不显著，较支持群落质心或组成差异。",
        )
    if not permanova_significant and permdisp_significant:
        return (
            "DISPERSION_DIFFERENCE_ONLY",
            "未发现显著质心差异，但组内离散度存在显著差异。",
        )
    return (
        "NO_SIGNIFICANT_DIFFERENCE",
        "PERMANOVA和PERMDISP均未达到显著水平。",
    )


def p_text(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    value = float(value)
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in ["png", "pdf", "svg"]:
        path = FIGURE_DIR / f"{stem}.{suffix}"
        fig.savefig(
            path,
            dpi=FIGURE_DPI if suffix == "png" else None,
            bbox_inches="tight",
        )
        paths.append(path)
    plt.close(fig)
    return paths


# ============================================================
# 2. 读取并严格验收区域结果
# ============================================================

progress("定位区域版第五步结果……")
SOURCE = locate_source()
progress(f"读取：{SOURCE}")

tables = {
    name: read_source_csv(SOURCE, name)
    for name in REQUIRED_FILES
}

species_metadata = tables["02_species_metadata_all25.csv"]
event_metadata = tables["03_positive_event_metadata.csv"]
matrix_raw_df = tables["04_species_matrix_biomass_raw_positive.csv"]
matrix_sqrt_df = tables["05_species_matrix_sqrt_biomass_positive.csv"]
dispersion_all = tables["12_permdisp_group_distances.csv"]
simper_all = tables["14_simper_results.csv"]
simper_coverage = tables["14b_simper_contrast_coverage.csv"]
region_mapping = tables["15a_region_mapping_used.csv"]
region_mapping_qc = tables["15b_region_mapping_qc.csv"]
region_group_sizes = tables["15c_region_group_sizes.csv"]
station_coverage = tables["15d_region_station_temporal_coverage.csv"]
region_summary = tables["16_region_analysis_summary.csv"]
region_pairwise = tables["17_region_pairwise_results.csv"]
method_metadata = tables["18_method_metadata.csv"]
step5_qc = tables["19_qc_checks.csv"]

require_columns(
    species_metadata,
    ["Species_Code", "Species_CN", "Species_Latin", "Species_Order"],
    "物种字典",
)
require_columns(
    event_metadata,
    ["EventID", "Year", "Month", "LocationID", "Region", "Lon", "Lat"],
    "正捕获事件元数据",
)
require_columns(
    region_mapping,
    ["LocationID", "Region", "Region_Order", "Region_Definition"],
    "正式区域映射",
)
require_columns(
    region_summary,
    [
        "Analysis", "Test", "Status", "N", "N_Groups", "P_Value",
        "PermutationsRequested", "PermutationsCompleted",
    ],
    "区域总体检验",
)
require_columns(
    region_pairwise,
    [
        "Contrast", "Group1", "Group2", "N_Locations",
        "PERMANOVA_Status", "PERMANOVA_R2", "PERMANOVA_P_FDR",
        "PERMDISP_Status", "PERMDISP_P_FDR",
        "PERMANOVA_FDR_Significant", "PERMDISP_FDR_Significant",
    ],
    "区域成对比较",
)
require_columns(
    simper_coverage,
    [
        "SourceTest", "Contrast", "EligibleForSIMPER",
        "IncludedInSIMPER", "SelectionReason",
    ],
    "SIMPER对比覆盖表",
)

species_metadata = species_metadata.sort_values(
    "Species_Order", kind="stable"
).reset_index(drop=True)
species_codes = species_metadata["Species_Code"].astype(str).tolist()

for frame_name, frame in [
    ("原始生物量矩阵", matrix_raw_df),
    ("平方根生物量矩阵", matrix_sqrt_df),
]:
    require_columns(frame, ["EventID"] + species_codes, frame_name)

event_metadata["EventID"] = event_metadata["EventID"].astype(str)
event_metadata["LocationID"] = event_metadata["LocationID"].astype(str)
event_metadata["Region"] = event_metadata["Region"].astype(str)
matrix_raw_df["EventID"] = matrix_raw_df["EventID"].astype(str)
matrix_sqrt_df["EventID"] = matrix_sqrt_df["EventID"].astype(str)

event_alignment = (
    event_metadata["EventID"].equals(matrix_raw_df["EventID"])
    and event_metadata["EventID"].equals(matrix_sqrt_df["EventID"])
)
matrix_raw = matrix_raw_df[species_codes].to_numpy(float)
matrix_sqrt = matrix_sqrt_df[species_codes].to_numpy(float)

if not event_metadata["EventID"].is_unique:
    raise AssertionError("正捕获事件EventID不唯一。")
if not np.isfinite(matrix_raw).all() or not np.isfinite(matrix_sqrt).all():
    raise AssertionError("物种矩阵含非有限值。")
if (matrix_raw < 0).any() or (matrix_sqrt < 0).any():
    raise AssertionError("物种矩阵含负值。")
if not np.allclose(np.sqrt(matrix_raw), matrix_sqrt, atol=1e-10, rtol=1e-10):
    raise AssertionError("平方根矩阵与原始生物量矩阵不一致。")

method_dict = dict(
    zip(method_metadata["Item"].astype(str), method_metadata["Definition"])
)
mapping_qc_pass = as_bool(region_mapping_qc["Pass"]).all()
step5_error_qc = step5_qc.loc[step5_qc["Severity"].astype(str).eq("ERROR")]
step5_fatal_pass = as_bool(step5_error_qc["Pass"]).all()

region_order_table = (
    region_mapping[["Region", "Region_Order", "Region_Definition"]]
    .drop_duplicates()
    .sort_values(["Region_Order", "Region"], kind="stable")
)
if region_order_table["Region"].duplicated().any():
    raise AssertionError("同一区域对应多个Region_Order或Region_Definition。")
region_order = dict(
    zip(region_order_table["Region"].astype(str), region_order_table["Region_Order"])
)
regions = region_order_table["Region"].astype(str).tolist()

permanova_row = region_summary.loc[
    region_summary["Analysis"].astype(str).eq("PERMANOVA")
].copy()
permdisp_row = region_summary.loc[
    region_summary["Analysis"].astype(str).eq("PERMDISP")
].copy()
if len(permanova_row) != 1 or len(permdisp_row) != 1:
    raise AssertionError("区域总体PERMANOVA和PERMDISP必须各有且仅有1行。")
permanova_row = permanova_row.iloc[0]
permdisp_row = permdisp_row.iloc[0]

permanova_significant = bool(float(permanova_row["P_Value"]) < ALPHA)
permdisp_significant = bool(float(permdisp_row["P_Value"]) < ALPHA)
omnibus_code, omnibus_cn = joint_interpretation(
    permanova_significant, permdisp_significant
)

region_pairwise = region_pairwise.copy()
region_pairwise["PERMANOVA_FDR_Significant"] = as_bool(
    region_pairwise["PERMANOVA_FDR_Significant"]
)
region_pairwise["PERMDISP_FDR_Significant"] = as_bool(
    region_pairwise["PERMDISP_FDR_Significant"]
)
pairwise_flag_consistent = bool(
    region_pairwise["PERMANOVA_FDR_Significant"].equals(
        region_pairwise["PERMANOVA_P_FDR"].astype(float).lt(ALPHA)
    )
    and region_pairwise["PERMDISP_FDR_Significant"].equals(
        region_pairwise["PERMDISP_P_FDR"].astype(float).lt(ALPHA)
    )
)

joint_codes: list[str] = []
joint_cn: list[str] = []
for row in region_pairwise.itertuples(index=False):
    code, message = joint_interpretation(
        bool(row.PERMANOVA_FDR_Significant),
        bool(row.PERMDISP_FDR_Significant),
    )
    joint_codes.append(code)
    joint_cn.append(message)
region_pairwise["JointInterpretationCode"] = joint_codes
region_pairwise["JointInterpretation_CN"] = joint_cn
region_pairwise["CentroidOnlyInferenceAllowed"] = (
    region_pairwise["PERMANOVA_FDR_Significant"]
    & ~region_pairwise["PERMDISP_FDR_Significant"]
)

region_simper = simper_all.loc[
    simper_all["SourceTest"].astype(str).eq("Region_station_level")
].copy()
expected_simper_contrasts = set(
    region_pairwise.loc[
        region_pairwise["PERMANOVA_FDR_Significant"], "Contrast"
    ].astype(str)
)
actual_simper_contrasts = set(region_simper["Contrast"].astype(str))
regional_coverage = simper_coverage.loc[
    simper_coverage["SourceTest"].astype(str).eq("Region_station_level")
].copy()
regional_coverage["EligibleForSIMPER"] = as_bool(
    regional_coverage["EligibleForSIMPER"]
)
regional_coverage["IncludedInSIMPER"] = as_bool(
    regional_coverage["IncludedInSIMPER"]
)
coverage_eligible_contrasts = set(
    regional_coverage.loc[
        regional_coverage["EligibleForSIMPER"], "Contrast"
    ].astype(str)
)
coverage_included_contrasts = set(
    regional_coverage.loc[
        regional_coverage["IncludedInSIMPER"], "Contrast"
    ].astype(str)
)

progress("重建120个站位的长期平均群落组成……")
station_raw_df = pd.DataFrame(matrix_raw, columns=species_codes)
station_raw_df.insert(0, "LocationID", event_metadata["LocationID"].to_numpy())
station_sqrt_df = pd.DataFrame(matrix_sqrt, columns=species_codes)
station_sqrt_df.insert(0, "LocationID", event_metadata["LocationID"].to_numpy())
station_raw = station_raw_df.groupby("LocationID", sort=True)[species_codes].mean()
station_sqrt = station_sqrt_df.groupby("LocationID", sort=True)[species_codes].mean()

station_meta = (
    event_metadata.groupby("LocationID", as_index=True)
    .agg(
        Region=("Region", "first"),
        Lon=("Lon", "first"),
        Lat=("Lat", "first"),
        N_PositiveEvents=("EventID", "size"),
        N_PositiveYears=("Year", "nunique"),
        N_PositiveMonths=("Month", "nunique"),
        FirstPositiveYear=("Year", "min"),
        LastPositiveYear=("Year", "max"),
    )
    .loc[station_raw.index]
)
station_meta["Region_Order"] = station_meta["Region"].map(region_order)

station_mean_raw_output = station_meta.reset_index().merge(
    station_raw.reset_index(), on="LocationID", validate="one_to_one"
)
station_mean_sqrt_output = station_meta.reset_index().merge(
    station_sqrt.reset_index(), on="LocationID", validate="one_to_one"
)

profile_rows: list[dict[str, object]] = []
for region in regions:
    location_ids = station_meta.index[station_meta["Region"].eq(region)]
    for species_row in species_metadata.itertuples(index=False):
        code = str(species_row.Species_Code)
        raw_values = station_raw.loc[location_ids, code].to_numpy(float)
        sqrt_values = station_sqrt.loc[location_ids, code].to_numpy(float)
        profile_rows.append(
            {
                "Region": region,
                "Region_Order": region_order.get(region),
                "N_Locations": len(location_ids),
                "Species_Code": code,
                "Species_CN": species_row.Species_CN,
                "Species_Latin": species_row.Species_Latin,
                "Species_Order": species_row.Species_Order,
                "Mean_LocationLongTermRawBiomass_kg_km2": raw_values.mean(),
                "Median_LocationLongTermRawBiomass_kg_km2": np.median(raw_values),
                "SD_LocationLongTermRawBiomass_kg_km2": raw_values.std(ddof=1)
                if len(raw_values) > 1 else np.nan,
                "Mean_LocationLongTermSqrtBiomass": sqrt_values.mean(),
                "N_Locations_WithSpecies": int((raw_values > 0).sum()),
                "Location_Prevalence": float((raw_values > 0).mean()),
            }
        )
region_species_profiles = pd.DataFrame(profile_rows)
region_species_profiles["Rank_by_MeanSqrtBiomass"] = (
    region_species_profiles.groupby("Region")[
        "Mean_LocationLongTermSqrtBiomass"
    ]
    .rank(method="first", ascending=False)
    .astype(int)
)
region_species_profiles = region_species_profiles.sort_values(
    ["Region_Order", "Rank_by_MeanSqrtBiomass"], kind="stable"
).reset_index(drop=True)


# ============================================================
# 3. 深化PERMDISP、时间覆盖和SIMPER解释
# ============================================================

progress("联合解释区域PERMANOVA与PERMDISP……")
omnibus_interpretation = pd.DataFrame(
    [
        {
            "Test": "Region_station_level",
            "N_Locations": int(permanova_row["N"]),
            "N_Regions": int(permanova_row["N_Groups"]),
            "PERMANOVA_PseudoF": float(permanova_row["PseudoF"]),
            "PERMANOVA_R2": float(permanova_row["R2"]),
            "PERMANOVA_P": float(permanova_row["P_Value"]),
            "PERMANOVA_Permutations": int(permanova_row["PermutationsCompleted"]),
            "PERMANOVA_Significant": permanova_significant,
            "PERMDISP_F": float(permdisp_row["F"]),
            "PERMDISP_P": float(permdisp_row["P_Value"]),
            "PERMDISP_Permutations": int(permdisp_row["PermutationsCompleted"]),
            "PERMDISP_Significant": permdisp_significant,
            "JointInterpretationCode": omnibus_code,
            "JointInterpretation_CN": omnibus_cn,
        }
    ]
)

region_dispersion = dispersion_all.loc[
    dispersion_all["Test"].astype(str).eq("Region_station_level")
].copy()
if len(region_dispersion) != len(regions):
    raise AssertionError(
        f"区域PERMDISP组摘要应有{len(regions)}行，实际为{len(region_dispersion)}。"
    )
region_dispersion["Region_Order"] = region_dispersion["Group"].map(region_order)
region_dispersion["LingoesConstant"] = float(permdisp_row["LingoesConstant"])
region_dispersion = region_dispersion.sort_values(
    ["Region_Order", "Group"], kind="stable"
).reset_index(drop=True)

coverage = station_coverage.copy()
coverage["Region_Order"] = coverage["Region"].map(region_order)
coverage_summary_rows: list[dict[str, object]] = []
for region in regions:
    group = coverage.loc[coverage["Region"].astype(str).eq(region)]
    coverage_summary_rows.append(
        {
            "Region": region,
            "Region_Order": region_order.get(region),
            "N_Locations": len(group),
            "Mean_PositiveEvents": group["N_PositiveEvents"].mean(),
            "Median_PositiveEvents": group["N_PositiveEvents"].median(),
            "Q1_PositiveEvents": group["N_PositiveEvents"].quantile(0.25),
            "Q3_PositiveEvents": group["N_PositiveEvents"].quantile(0.75),
            "Min_PositiveEvents": group["N_PositiveEvents"].min(),
            "Max_PositiveEvents": group["N_PositiveEvents"].max(),
            "Mean_PositiveYears": group["N_PositiveYears"].mean(),
            "Median_PositiveYears": group["N_PositiveYears"].median(),
            "N_Locations_BelowDiagnosticThreshold": int(
                group["N_PositiveEvents"].lt(LOW_COVERAGE_EVENT_THRESHOLD).sum()
            ),
            "DiagnosticThreshold_PositiveEvents": LOW_COVERAGE_EVENT_THRESHOLD,
        }
    )
coverage_summary = pd.DataFrame(coverage_summary_rows)
low_coverage_locations = coverage.loc[
    coverage["N_PositiveEvents"].lt(LOW_COVERAGE_EVENT_THRESHOLD)
].sort_values(["Region_Order", "N_PositiveEvents", "LocationID"], kind="stable")
low_coverage_locations = low_coverage_locations.assign(
    DiagnosticFlag=(
        f"N_PositiveEvents < {LOW_COVERAGE_EVENT_THRESHOLD}; descriptive flag only, not excluded"
    )
)

progress("整理区域SIMPER贡献及方向……")
if len(region_simper):
    region_simper = region_simper.sort_values(
        ["Contrast", "SIMPER_Rank"], kind="stable"
    ).reset_index(drop=True)
    require_columns(
        region_simper,
        [
            "Contrast", "SIMPER_Design", "Species_Code", "Species_CN",
            "Species_Latin", "Group1", "Group2", "N_Group1", "N_Group2",
            "N_BetweenGroupPairs", "Mean_RawBiomass_Group1_kg_km2",
            "Mean_RawBiomass_Group2_kg_km2", "Percent_Contribution",
            "Mean_BrayCurtis_BetweenGroups", "SIMPER_Rank",
            "Cumulative_Percent",
        ],
        "区域SIMPER",
    )
    region_simper["Mean_Raw_Difference_Group2_minus_Group1_kg_km2"] = (
        region_simper["Mean_RawBiomass_Group2_kg_km2"]
        - region_simper["Mean_RawBiomass_Group1_kg_km2"]
    )
    tolerance = 1e-12
    region_simper["HigherMeanRegion"] = np.where(
        region_simper[
            "Mean_Raw_Difference_Group2_minus_Group1_kg_km2"
        ].gt(tolerance),
        region_simper["Group2"],
        np.where(
            region_simper[
                "Mean_Raw_Difference_Group2_minus_Group1_kg_km2"
            ].lt(-tolerance),
            region_simper["Group1"],
            "Equal",
        ),
    )
    region_simper["Selected_Top80"] = (
        region_simper.groupby("Contrast")["Cumulative_Percent"]
        .shift(fill_value=0)
        .lt(SIMPER_CUMULATIVE_PERCENT)
    )
    pair_info = region_pairwise[
        [
            "Contrast", "PERMANOVA_R2", "PERMANOVA_P_FDR",
            "PERMDISP_P_FDR", "PERMDISP_FDR_Significant",
            "JointInterpretationCode",
        ]
    ]
    region_simper = region_simper.merge(
        pair_info, on="Contrast", how="left", validate="many_to_one"
    )
    region_simper["SIMPER_InferenceNote"] = np.where(
        region_simper["PERMDISP_FDR_Significant"],
        "Descriptive only; PERMDISP is also significant for this contrast.",
        "Descriptive contribution decomposition; not a species-level significance test.",
    )
    region_simper_top80 = region_simper.loc[
        region_simper["Selected_Top80"]
    ].copy()

    simper_summary_rows: list[dict[str, object]] = []
    for contrast, group in region_simper.groupby("Contrast", sort=False):
        group = group.sort_values("SIMPER_Rank", kind="stable")
        top = group.loc[group["Selected_Top80"]]
        first = group.iloc[0]
        simper_summary_rows.append(
            {
                "Contrast": contrast,
                "Group1": first["Group1"],
                "Group2": first["Group2"],
                "N_Group1": int(first["N_Group1"]),
                "N_Group2": int(first["N_Group2"]),
                "N_BetweenGroupPairs": int(first["N_BetweenGroupPairs"]),
                "Mean_BrayCurtis_BetweenGroups": float(
                    first["Mean_BrayCurtis_BetweenGroups"]
                ),
                "PERMANOVA_R2": float(first["PERMANOVA_R2"]),
                "PERMANOVA_P_FDR": float(first["PERMANOVA_P_FDR"]),
                "PERMDISP_P_FDR": float(first["PERMDISP_P_FDR"]),
                "PERMDISP_FDR_Significant": bool(
                    first["PERMDISP_FDR_Significant"]
                ),
                "JointInterpretationCode": first["JointInterpretationCode"],
                "N_Species_Through80": len(top),
                "Cumulative_Percent_Through80": float(
                    top["Cumulative_Percent"].max()
                ) if len(top) else np.nan,
                "Top80_Species_CN": " | ".join(top["Species_CN"].astype(str)),
                "Top80_Species_Latin": " | ".join(
                    top["Species_Latin"].astype(str)
                ),
            }
        )
    simper_contrast_summary = pd.DataFrame(simper_summary_rows)

    recurrence = (
        region_simper.groupby(
            ["Species_Code", "Species_CN", "Species_Latin"], as_index=False
        )
        .agg(
            N_Regional_SIMPER_Contrasts=("Contrast", "nunique"),
            N_Contrasts_Top3=("SIMPER_Rank", lambda s: int((s <= 3).sum())),
            N_Contrasts_Through80=("Selected_Top80", "sum"),
            Mean_Percent_Contribution=("Percent_Contribution", "mean"),
            Max_Percent_Contribution=("Percent_Contribution", "max"),
            Contrasts=("Contrast", lambda s: " | ".join(dict.fromkeys(s.astype(str)))),
        )
        .sort_values(
            ["N_Contrasts_Through80", "Mean_Percent_Contribution"],
            ascending=[False, False],
            kind="stable",
        )
        .reset_index(drop=True)
    )
else:
    region_simper_top80 = region_simper.copy()
    simper_contrast_summary = pd.DataFrame(
        columns=[
            "Contrast", "Group1", "Group2", "N_Group1", "N_Group2",
            "N_BetweenGroupPairs", "Mean_BrayCurtis_BetweenGroups",
            "PERMANOVA_R2", "PERMANOVA_P_FDR", "PERMDISP_P_FDR",
            "PERMDISP_FDR_Significant", "JointInterpretationCode",
            "N_Species_Through80", "Cumulative_Percent_Through80",
            "Top80_Species_CN", "Top80_Species_Latin",
        ]
    )
    recurrence = pd.DataFrame(
        columns=[
            "Species_Code", "Species_CN", "Species_Latin",
            "N_Regional_SIMPER_Contrasts", "N_Contrasts_Top3",
            "N_Contrasts_Through80", "Mean_Percent_Contribution",
            "Max_Percent_Contribution", "Contrasts",
        ]
    )


# ============================================================
# 4. 质量检查
# ============================================================

def qc_row(
    check: str,
    expected: object,
    observed: object,
    passed: bool,
    severity: str = "ERROR",
) -> dict[str, object]:
    return {
        "Check": check,
        "Expected": expected,
        "Observed": observed,
        "Pass": bool(passed),
        "Severity": severity,
    }


mapping_locations = region_mapping["LocationID"].astype(str).nunique()
positive_locations = event_metadata["LocationID"].nunique()
expected_pairwise_count = math.comb(len(regions), 2)

simper_rows_per_contrast_ok = True
simper_percent_ok = True
simper_design_ok = True
simper_pairs_ok = True
simper_profile_max_error = 0.0
if len(region_simper):
    counts = region_simper.groupby("Contrast").size()
    simper_rows_per_contrast_ok = bool(counts.eq(EXPECTED_SPECIES).all())
    percent_sums = region_simper.groupby("Contrast")["Percent_Contribution"].sum()
    simper_percent_ok = bool(np.allclose(percent_sums, 100.0, atol=1e-8))
    simper_design_ok = bool(
        region_simper["SIMPER_Design"].astype(str).eq(
            "Classical_AllCrossPairs"
        ).all()
    )
    first_rows = region_simper.sort_values("SIMPER_Rank").groupby(
        "Contrast", as_index=False
    ).first()
    simper_pairs_ok = bool(
        (
            first_rows["N_BetweenGroupPairs"]
            == first_rows["N_Group1"] * first_rows["N_Group2"]
        ).all()
    )

    # 复核SIMPER中的区域原始均值是否等于120个站位长期均值的区域平均。
    profile_lookup = region_species_profiles.set_index(
        ["Region", "Species_Code"]
    )["Mean_LocationLongTermRawBiomass_kg_km2"]
    errors: list[float] = []
    for row in region_simper.itertuples(index=False):
        errors.append(
            abs(
                float(row.Mean_RawBiomass_Group1_kg_km2)
                - float(profile_lookup.loc[(str(row.Group1), str(row.Species_Code))])
            )
        )
        errors.append(
            abs(
                float(row.Mean_RawBiomass_Group2_kg_km2)
                - float(profile_lookup.loc[(str(row.Group2), str(row.Species_Code))])
            )
        )
    simper_profile_max_error = max(errors) if errors else 0.0

qc_checks = pd.DataFrame(
    [
        qc_row("Step5 fatal QC", True, step5_fatal_pass, step5_fatal_pass),
        qc_row("Region mapping QC", True, mapping_qc_pass, mapping_qc_pass),
        qc_row(
            "Region analysis status",
            "READY",
            method_dict.get("Region analysis status"),
            str(method_dict.get("Region analysis status")) == "READY",
        ),
        qc_row(
            "Positive EventID count",
            EXPECTED_POSITIVE_EVENTS,
            event_metadata["EventID"].nunique(),
            event_metadata["EventID"].nunique() == EXPECTED_POSITIVE_EVENTS,
        ),
        qc_row(
            "Positive LocationID count",
            EXPECTED_POSITIVE_LOCATIONS,
            positive_locations,
            positive_locations == EXPECTED_POSITIVE_LOCATIONS,
        ),
        qc_row(
            "Formal mapping LocationID count",
            EXPECTED_STANDARD_LOCATIONS,
            mapping_locations,
            mapping_locations == EXPECTED_STANDARD_LOCATIONS,
        ),
        qc_row(
            "Species count",
            EXPECTED_SPECIES,
            len(species_codes),
            len(species_codes) == EXPECTED_SPECIES,
        ),
        qc_row("Event matrix alignment", True, event_alignment, event_alignment),
        qc_row(
            "Region omnibus tests completed",
            2,
            int(region_summary["Status"].astype(str).eq("OK").sum()),
            int(region_summary["Status"].astype(str).eq("OK").sum()) == 2,
        ),
        qc_row(
            "Region permutations completed",
            EXPECTED_PERMUTATIONS,
            int(region_summary["PermutationsCompleted"].min()),
            bool(
                region_summary["PermutationsCompleted"].eq(
                    EXPECTED_PERMUTATIONS
                ).all()
            ),
        ),
        qc_row(
            "Region pairwise comparisons",
            expected_pairwise_count,
            len(region_pairwise),
            len(region_pairwise) == expected_pairwise_count,
        ),
        qc_row(
            "Pairwise FDR flags match adjusted p-values",
            True,
            pairwise_flag_consistent,
            pairwise_flag_consistent,
        ),
        qc_row(
            "SIMPER coverage eligibility matches significant contrasts",
            sorted(expected_simper_contrasts),
            sorted(coverage_eligible_contrasts),
            expected_simper_contrasts == coverage_eligible_contrasts,
        ),
        qc_row(
            "SIMPER coverage inclusion matches output",
            sorted(actual_simper_contrasts),
            sorted(coverage_included_contrasts),
            actual_simper_contrasts == coverage_included_contrasts,
        ),
        qc_row(
            "Significant regional contrasts covered by SIMPER",
            sorted(expected_simper_contrasts),
            sorted(actual_simper_contrasts),
            expected_simper_contrasts == actual_simper_contrasts,
        ),
        qc_row(
            "SIMPER species rows per contrast",
            EXPECTED_SPECIES,
            "all exact" if simper_rows_per_contrast_ok else "mismatch",
            simper_rows_per_contrast_ok,
        ),
        qc_row(
            "SIMPER contribution sums",
            100.0,
            "all exact" if simper_percent_ok else "mismatch",
            simper_percent_ok,
        ),
        qc_row(
            "Regional SIMPER design",
            "Classical_AllCrossPairs",
            "all exact" if simper_design_ok else "mismatch",
            simper_design_ok,
        ),
        qc_row(
            "Regional SIMPER pair count",
            "N_Group1 * N_Group2",
            "all exact" if simper_pairs_ok else "mismatch",
            simper_pairs_ok,
        ),
        qc_row(
            "SIMPER regional raw-mean reconciliation",
            "<=1e-10",
            simper_profile_max_error,
            simper_profile_max_error <= 1e-10,
        ),
        qc_row(
            "Centroid-only regional inference",
            "Use caution when PERMDISP is significant",
            omnibus_code,
            not (permanova_significant and permdisp_significant),
            severity="WARNING",
        ),
    ]
)

fatal_failures = qc_checks.loc[
    qc_checks["Severity"].eq("ERROR") & ~qc_checks["Pass"]
]
if len(fatal_failures):
    raise AssertionError(
        "区域深度分析质量检查失败：\n"
        + fatal_failures[["Check", "Expected", "Observed"]].to_string(index=False)
    )


# ============================================================
# 5. 图件
# ============================================================

progress("生成区域统计和SIMPER图件……")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
figure_paths: list[Path] = []

# Fig01：总体与成对PERMANOVA效应量，并用阴影标记PERMDISP显著。
effect_rows = [
    {
        "Label": "Overall regions",
        "R2": float(permanova_row["R2"]),
        "P": float(permanova_row["P_Value"]),
        "P_Label": "p",
        "PERMANOVA_Significant": permanova_significant,
        "PERMDISP_Significant": permdisp_significant,
    }
]
for row in region_pairwise.itertuples(index=False):
    effect_rows.append(
        {
            "Label": f"{row.Group1} vs {row.Group2}",
            "R2": float(row.PERMANOVA_R2),
            "P": float(row.PERMANOVA_P_FDR),
            "P_Label": "q",
            "PERMANOVA_Significant": bool(row.PERMANOVA_FDR_Significant),
            "PERMDISP_Significant": bool(row.PERMDISP_FDR_Significant),
        }
    )
effect_df = pd.DataFrame(effect_rows)
fig, ax = plt.subplots(figsize=(9.0, 5.2))
x = np.arange(len(effect_df))
bars = ax.bar(
    x,
    effect_df["R2"],
    color=np.where(effect_df["PERMANOVA_Significant"], "#4C78A8", "#B8B8B8"),
    edgecolor="#222222",
    linewidth=0.8,
)
for bar, disp_sig in zip(bars, effect_df["PERMDISP_Significant"]):
    if disp_sig:
        bar.set_hatch("//")
for idx, row in effect_df.iterrows():
    ax.text(
        idx,
        row["R2"] + max(effect_df["R2"].max() * 0.025, 0.001),
        f"R²={row['R2']:.3f}\n{row['P_Label']}={p_text(row['P'])}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
ax.set_xticks(x)
ax.set_xticklabels(effect_df["Label"], rotation=18, ha="right")
ax.set_ylabel("PERMANOVA R²")
ax.set_title("Regional assemblage differences and dispersion diagnostics")
ax.set_ylim(0, effect_df["R2"].max() * 1.32)
ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
ax.legend(
    handles=[
        Patch(facecolor="#4C78A8", edgecolor="#222222", label="PERMANOVA significant"),
        Patch(facecolor="#B8B8B8", edgecolor="#222222", label="PERMANOVA not significant"),
        Patch(facecolor="white", edgecolor="#222222", hatch="//", label="PERMDISP significant"),
    ],
    frameon=False,
    fontsize=8,
    loc="upper right",
)
fig.tight_layout()
figure_paths.extend(save_figure(fig, "Fig01_region_PERMANOVA_PERMDISP"))

# Fig02：各区域到组质心距离。
fig, ax = plt.subplots(figsize=(7.2, 5.2))
groups = region_dispersion["Group"].astype(str).tolist()
means = region_dispersion["Mean_DistanceToCentroid"].to_numpy(float)
sds = region_dispersion["SD_DistanceToCentroid"].to_numpy(float)
x = np.arange(len(groups))
ax.bar(
    x,
    means,
    yerr=sds,
    capsize=5,
    color=[REGION_COLORS.get(group, "#777777") for group in groups],
    edgecolor="#222222",
    linewidth=0.8,
)
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel("Mean distance to group centroid (Lingoes-corrected)")
ax.set_title(
    "Regional multivariate dispersion\n"
    f"PERMDISP F={float(permdisp_row['F']):.3f}, p={p_text(permdisp_row['P_Value'])}"
)
ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
fig.tight_layout()
figure_paths.extend(save_figure(fig, "Fig02_region_PERMDISP_distances"))

# Fig03：显著区域对比的SIMPER top80。
if len(region_simper_top80):
    contrasts = simper_contrast_summary["Contrast"].astype(str).tolist()
    n_panels = len(contrasts)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(9.2, max(4.2, 4.2 * n_panels)),
        squeeze=False,
    )
    for axis, contrast in zip(axes.ravel(), contrasts):
        group = region_simper_top80.loc[
            region_simper_top80["Contrast"].astype(str).eq(contrast)
        ].sort_values("Percent_Contribution", ascending=True, kind="stable")
        colors = [
            REGION_COLORS.get(str(value), "#777777")
            for value in group["HigherMeanRegion"]
        ]
        axis.barh(
            np.arange(len(group)),
            group["Percent_Contribution"],
            color=colors,
            edgecolor="#222222",
            linewidth=0.5,
        )
        axis.set_yticks(np.arange(len(group)))
        axis.set_yticklabels(group["Species_Latin"], fontstyle="italic")
        axis.set_xlabel("SIMPER contribution (%)")
        axis.set_title(
            contrast.replace("_", " ")
            + " — species through cumulative 80%"
        )
        axis.grid(axis="x", color="#E5E5E5", linewidth=0.6)
        for idx, value in enumerate(group["Percent_Contribution"]):
            axis.text(value + 0.15, idx, f"{value:.1f}%", va="center", fontsize=8)
        legend_regions = list(
            dict.fromkeys(group["HigherMeanRegion"].astype(str).tolist())
        )
        axis.legend(
            handles=[
                Patch(
                    facecolor=REGION_COLORS.get(region, "#777777"),
                    edgecolor="#222222",
                    label=f"Higher mean in {region}",
                )
                for region in legend_regions
            ],
            frameon=False,
            fontsize=8,
            loc="lower right",
        )
    fig.tight_layout()
    figure_paths.extend(save_figure(fig, "Fig03_region_SIMPER_top80"))

# Fig04：SIMPER top80物种的三区长期平均生物量热图。
if len(region_simper_top80):
    selected_codes = list(
        dict.fromkeys(region_simper_top80["Species_Code"].astype(str).tolist())
    )
else:
    overall_species = (
        region_species_profiles.groupby(
            ["Species_Code", "Species_Latin"], as_index=False
        )["Mean_LocationLongTermSqrtBiomass"]
        .mean()
        .nlargest(min(10, EXPECTED_SPECIES), "Mean_LocationLongTermSqrtBiomass")
    )
    selected_codes = overall_species["Species_Code"].astype(str).tolist()

heat = region_species_profiles.loc[
    region_species_profiles["Species_Code"].astype(str).isin(selected_codes)
].copy()
species_order = (
    heat.groupby(["Species_Code", "Species_Latin"], as_index=False)[
        "Mean_LocationLongTermSqrtBiomass"
    ]
    .mean()
    .sort_values("Mean_LocationLongTermSqrtBiomass", ascending=False)
)
selected_codes = species_order["Species_Code"].astype(str).tolist()
latin_lookup = dict(
    zip(species_metadata["Species_Code"].astype(str), species_metadata["Species_Latin"])
)
heat_matrix = (
    heat.pivot(
        index="Species_Code",
        columns="Region",
        values="Mean_LocationLongTermRawBiomass_kg_km2",
    )
    .reindex(index=selected_codes, columns=regions)
)
display_matrix = np.log1p(heat_matrix.to_numpy(float))
fig_height = max(4.8, 0.45 * len(selected_codes) + 1.8)
fig, ax = plt.subplots(figsize=(7.4, fig_height))
image = ax.imshow(display_matrix, cmap="YlGnBu", aspect="auto")
ax.set_xticks(np.arange(len(regions)))
ax.set_xticklabels(regions)
ax.set_yticks(np.arange(len(selected_codes)))
ax.set_yticklabels(
    [latin_lookup.get(code, code) for code in selected_codes],
    fontstyle="italic",
)
ax.set_title("Regional mean biomass profiles of SIMPER-selected species")
for i in range(display_matrix.shape[0]):
    for j in range(display_matrix.shape[1]):
        value = heat_matrix.iloc[i, j]
        text_color = "white" if display_matrix[i, j] > np.nanmedian(display_matrix) else "black"
        ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7, color=text_color)
cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("log1p(mean location-level raw biomass)")
fig.tight_layout()
figure_paths.extend(save_figure(fig, "Fig04_region_species_profiles"))

# Fig05：各区域站位正捕获时间覆盖。
fig, ax = plt.subplots(figsize=(7.5, 5.3))
coverage_data = [
    coverage.loc[coverage["Region"].astype(str).eq(region), "N_PositiveEvents"].to_numpy(float)
    for region in regions
]
try:
    box = ax.boxplot(
        coverage_data,
        tick_labels=regions,
        patch_artist=True,
        showfliers=False,
    )
except TypeError:  # Matplotlib < 3.9
    box = ax.boxplot(
        coverage_data,
        labels=regions,
        patch_artist=True,
        showfliers=False,
    )
for patch, region in zip(box["boxes"], regions):
    patch.set_facecolor(REGION_COLORS.get(region, "#777777"))
    patch.set_alpha(0.55)
rng = np.random.default_rng(20260901)
for idx, (region, values) in enumerate(zip(regions, coverage_data), start=1):
    jitter = rng.normal(0, 0.045, size=len(values))
    ax.scatter(
        np.full(len(values), idx) + jitter,
        values,
        s=18,
        color=REGION_COLORS.get(region, "#777777"),
        edgecolor="#222222",
        linewidth=0.3,
        alpha=0.75,
        zorder=3,
    )
ax.axhline(
    LOW_COVERAGE_EVENT_THRESHOLD,
    color="#CC3311",
    linestyle="--",
    linewidth=1.1,
    label=f"Diagnostic threshold = {LOW_COVERAGE_EVENT_THRESHOLD} positive events",
)
ax.set_ylabel("Positive-catch events per LocationID")
ax.set_title("Temporal coverage of station-level regional assemblage means")
ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
figure_paths.extend(save_figure(fig, "Fig05_region_station_temporal_coverage"))


# ============================================================
# 6. 输出表格、解释说明和ZIP
# ============================================================

summary_metrics = OrderedDict(
    [
        ("Source", str(SOURCE)),
        ("Analysis unit", "One independent LocationID long-term positive-catch mean"),
        ("Positive locations", positive_locations),
        ("Regions", len(regions)),
        ("Region levels", " | ".join(regions)),
        ("Species", len(species_codes)),
        ("PERMANOVA R2", float(permanova_row["R2"])),
        ("PERMANOVA p", float(permanova_row["P_Value"])),
        ("PERMDISP p", float(permdisp_row["P_Value"])),
        ("Joint interpretation", omnibus_code),
        (
            "Pairwise PERMANOVA FDR significant",
            int(region_pairwise["PERMANOVA_FDR_Significant"].sum()),
        ),
        (
            "Pairwise PERMDISP FDR significant",
            int(region_pairwise["PERMDISP_FDR_Significant"].sum()),
        ),
        ("Regional SIMPER contrasts", len(actual_simper_contrasts)),
        ("Regional SIMPER full coverage", expected_simper_contrasts == actual_simper_contrasts),
        (
            "Low temporal coverage locations",
            len(low_coverage_locations),
        ),
        ("Fatal QC passed", len(fatal_failures) == 0),
    ]
)
region_detailed_summary = pd.DataFrame(
    {"Metric": list(summary_metrics.keys()), "Value": list(summary_metrics.values())}
)

output_tables: OrderedDict[str, pd.DataFrame] = OrderedDict(
    [
        ("00_region_detailed_summary.csv", region_detailed_summary),
        ("01_region_omnibus_joint_interpretation.csv", omnibus_interpretation),
        ("02_region_pairwise_joint_interpretation.csv", region_pairwise),
        ("03_region_permdisp_group_summary.csv", region_dispersion),
        ("04_region_temporal_coverage_summary.csv", coverage_summary),
        ("05_region_low_temporal_coverage_locations.csv", low_coverage_locations),
        ("06_station_mean_raw_composition.csv", station_mean_raw_output),
        ("07_station_mean_sqrt_composition.csv", station_mean_sqrt_output),
        ("08_region_species_mean_profiles.csv", region_species_profiles),
        ("09_region_simper_contrast_summary.csv", simper_contrast_summary),
        ("10_region_simper_all_species.csv", region_simper),
        ("11_region_simper_top80_species.csv", region_simper_top80),
        ("12_region_simper_species_recurrence.csv", recurrence),
        ("13_qc_checks.csv", qc_checks),
    ]
)

output_paths: list[Path] = []
for filename, table in output_tables.items():
    path = OUTPUT_DIR / filename
    table.to_csv(path, index=False, encoding="utf-8-sig")
    output_paths.append(path)

pairwise_lines = []
for row in region_pairwise.itertuples(index=False):
    pairwise_lines.append(
        f"- {row.Contrast}: R2={float(row.PERMANOVA_R2):.4f}, "
        f"PERMANOVA q={p_text(row.PERMANOVA_P_FDR)}, "
        f"PERMDISP q={p_text(row.PERMDISP_P_FDR)}, "
        f"{row.JointInterpretationCode}"
    )

simper_lines = []
for row in simper_contrast_summary.itertuples(index=False):
    simper_lines.append(
        f"- {row.Contrast}: mean Bray-Curtis={float(row.Mean_BrayCurtis_BetweenGroups):.4f}; "
        f"top80 species={row.Top80_Species_CN}"
    )
if not simper_lines:
    simper_lines.append("- No FDR-significant regional PERMANOVA contrast required SIMPER.")

readme = f"""浙江近岸头足类：区域PERMANOVA、PERMDISP与SIMPER深度分析
================================================================

输入与估计对象
------------
- 输入：{SOURCE}
- 本脚本没有重新运行NMDS或任何置换检验，只整理和复核已完成的正式结果。
- 区域分析单位：每个LocationID一行，共{positive_locations}个独立站位。
- 每个站位的组成：先对每个正捕获事件的生物量开平方，再在LocationID内求长期均值。
- 结论条件：正捕获条件下的长期平均群落组成；不代表包含零捕获的总体资源状态。
- 区域：{" | ".join(regions)}。

总体检验
--------
- PERMANOVA: pseudo-F={float(permanova_row['PseudoF']):.4f},
  R2={float(permanova_row['R2']):.4f}, p={p_text(permanova_row['P_Value'])},
  permutations={int(permanova_row['PermutationsCompleted'])}。
- PERMDISP: F={float(permdisp_row['F']):.4f},
  p={p_text(permdisp_row['P_Value'])},
  permutations={int(permdisp_row['PermutationsCompleted'])}。
- 联合解释：{omnibus_cn}

区域成对比较
------------
{chr(10).join(pairwise_lines)}

区域SIMPER
---------
{chr(10).join(simper_lines)}
- SIMPER使用独立LocationID长期均值之间的全部跨组站位对。
- SIMPER是描述性贡献分解，不是物种级显著性检验。
- 当相应PERMDISP显著时，SIMPER贡献可能同时反映群落中心和异质性变化。

时间覆盖诊断
------------
- N_PositiveEvents < {LOW_COVERAGE_EVENT_THRESHOLD}仅作诊断标记，不自动删除站位。
- 区域检验使用站位作为独立重复，不能用1,532个事件作为空间独立样本量。

图件
----
- Fig01：总体和成对PERMANOVA R2；斜线表示相应PERMDISP显著。
- Fig02：Lingoes校正后各区域到组质心距离。
- Fig03：显著区域对比的SIMPER累计80%物种及方向。
- Fig04：SIMPER所选物种在三区的长期平均原始生物量。
- Fig05：三区站位长期均值的正捕获时间覆盖。

论文解释底线
------------
- PERMANOVA显著且PERMDISP不显著：较支持群落质心/组成差异。
- PERMANOVA和PERMDISP均显著：不能写成纯粹质心差异，应报告离散度混杂。
- 不把SIMPER物种称为“显著差异物种”；只能称为主要贡献物种。
"""
readme_path = OUTPUT_DIR / "README_region_detailed_analysis.txt"
readme_path.write_text(readme, encoding="utf-8")
output_paths.append(readme_path)
output_paths.extend(figure_paths)

manifest_rows: list[dict[str, object]] = []
for filename, table in output_tables.items():
    manifest_rows.append(
        {
            "File": filename,
            "Rows": len(table),
            "Columns": len(table.columns),
            "Purpose": "Detailed regional tabular result",
        }
    )
for path in figure_paths:
    manifest_rows.append(
        {
            "File": str(path.relative_to(OUTPUT_DIR)),
            "Rows": pd.NA,
            "Columns": pd.NA,
            "Purpose": "Regional publication/QC figure",
        }
    )
manifest_rows.extend(
    [
        {
            "File": readme_path.name,
            "Rows": pd.NA,
            "Columns": pd.NA,
            "Purpose": "Methods and interpretation notes",
        },
        {
            "File": "14_output_manifest.csv",
            "Rows": pd.NA,
            "Columns": 4,
            "Purpose": "Output manifest",
        },
    ]
)
manifest = pd.DataFrame(manifest_rows)
manifest_path = OUTPUT_DIR / "14_output_manifest.csv"
manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
output_paths.append(manifest_path)

zip_path = Path(str(OUTPUT_DIR) + ".zip")
temporary_zip_path = Path(str(zip_path) + ".tmp")
with zipfile.ZipFile(
    temporary_zip_path,
    "w",
    compression=zipfile.ZIP_DEFLATED,
) as archive:
    for path in output_paths:
        archive.write(path, arcname=str(path.relative_to(OUTPUT_DIR)))
temporary_zip_path.replace(zip_path)

progress("区域深度分析完成。")
print("=" * 78)
print("区域总体PERMANOVA/PERMDISP")
print("=" * 78)
print(
    f"PERMANOVA: F={float(permanova_row['PseudoF']):.4f}, "
    f"R2={float(permanova_row['R2']):.4f}, p={p_text(permanova_row['P_Value'])}"
)
print(
    f"PERMDISP: F={float(permdisp_row['F']):.4f}, "
    f"p={p_text(permdisp_row['P_Value'])}"
)
print(f"联合解释：{omnibus_cn}")
print(f"区域成对比较：{len(region_pairwise)}项")
print(
    "PERMANOVA经FDR显著："
    f"{int(region_pairwise['PERMANOVA_FDR_Significant'].sum())}项"
)
print(f"区域SIMPER：{len(actual_simper_contrasts)}项，覆盖完整={expected_simper_contrasts == actual_simper_contrasts}")
print(f"致命QC通过：{len(fatal_failures) == 0}")
print(f"输出目录：{OUTPUT_DIR}")
print(f"ZIP文件：{zip_path}")
print(f"请下载：{zip_path.name}")
