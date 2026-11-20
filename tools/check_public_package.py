"""Review the Chapter 2 code + four public NetCDF environmental subsets.
This is a STATIC audit, not a scientific reproduction or Git history audit.
No separate ENVIRONMENT_SHA256.json file is required: reference hashes are
recorded here from the 2026-11-09 data-only archive.
"""
from pathlib import Path
import hashlib
import json
import re
import sys
import ast

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENVS = {
    "data/environment/Chla_4km_April_November_2014_2024.nc": (661366, "39f80686a0aa80fe1b76a7d8bebd7796a6b3a7aaa2c872382c04b1552a8d9e0d"),
    "data/environment/Depth_0.005°_2024.nc": (1862988, "5345e6c3afe94695b6387c49d126c475739524ee88d54bc5edfdb0937e801df8"),
    "data/environment/Salinity_0.125deg_April_November_2014_2024.nc": (449693, "bd7ef9577693dfa669a1c6a92d784d98bfd7fb4e552a50b1b52561f068e0729d"),
    "data/environment/Temperature_0.125deg_April_November_2014_2024.nc": (508398, "31155812f3aff89365d0451958f99a74cff57d1458c37da9331f101be6fbb939"),
}
errors = []
expected = set((ROOT / "PUBLIC_FILES.txt").read_text(encoding="utf-8").splitlines())
actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()
          and not ({".git", "__pycache__", ".ipynb_checkpoints"} & set(p.relative_to(ROOT).parts))}
for p in sorted(actual - expected): errors.append(f"Unexpected extra file: {p}")
for p in sorted(expected - actual): errors.append(f"Missing file: {p}")
if expected.intersection(EXPECTED_ENVS) != set(EXPECTED_ENVS):
    errors.append("Manifest must include exactly the four reviewed NetCDF files")
for name in sorted(expected & actual):
    p = ROOT / name
    if p.is_symlink():
        errors.append(f"Symbolic link not allowed: {name}")
        continue
    if name in EXPECTED_ENVS:
        size, digest = EXPECTED_ENVS[name]
        b = p.read_bytes()
        if len(b) != size: errors.append(f"NC size mismatch: {name}")
        if hashlib.sha256(b).hexdigest() != digest: errors.append(f"NC SHA256 mismatch: {name}")
        if not (b.startswith(b"CDF") or b.startswith(b"\x89HDF\r\n\x1a\n")):
            errors.append(f"NC magic mismatch: {name}")
        continue
    if name.lower().endswith((".csv", ".xlsx", ".xls", ".zip", ".rar", ".h5", ".pkl", ".parquet")):
        errors.append(f"Unapproved data-like file: {name}")
        continue
    try: content = p.read_text(encoding="utf-8")
    except UnicodeError:
        errors.append(f"Unreviewed binary file: {name}")
        continue
    if name.endswith(".py"):
        try: ast.parse(content, filename=name)
        except SyntaxError as e: errors.append(f"Invalid Python file: {name}, {e}")
    if name.endswith(".ipynb"):
        try:
            nb = json.loads(content)
            for i, cell in enumerate(nb.get("cells", [])):
                if cell.get("outputs") or cell.get("attachments") or cell.get("execution_count") is not None:
                    errors.append(f"Notebook output or attachment: {name}, cell {i}")
                if cell.get("metadata"):
                    errors.append(f"Notebook cell metadata not empty: {name}, cell {i}")
        except (ValueError, TypeError) as e:
            errors.append(f"Invalid notebook: {name}, {e}")
    if name != "tools/check_public_package.py":
        patterns = [r"/kaggle/input/datasets/[^\s\"']+", r"/workspace/scratch/[^\s\"']+",
                    r"gh[pousr]_[A-Za-z0-9]{20,}", r"github_pat_[A-Za-z0-9_]{20,}",
                    r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----", r"AKIA[0-9A-Z]{16}"]
        if any(re.search(pattern, content) for pattern in patterns):
            errors.append(f"Potential private path or credential: {name}")
if errors:
    print("FAIL: " + str(len(errors)) + " issues")
    for e in errors: print(" - " + e)
    sys.exit(1)
print(f"PASS: {len(actual)} reviewed files; four NetCDF SHA256 digests match source subsets; notebook outputs absent.")
print("WARNING: This does not verify remote GitHub history, file licenses, scientific results, or unreviewed future commits.")
