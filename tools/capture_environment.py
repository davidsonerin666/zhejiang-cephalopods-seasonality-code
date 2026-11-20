"""Record installed analysis-package versions without inspecting research data."""
from pathlib import Path
from importlib import metadata
import argparse, json, platform, sys
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "patsy",
            "matplotlib", "seaborn", "xarray", "netCDF4", "openpyxl", "packaging",
            "Pillow", "threadpoolctl"]
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Private run record path; defaults to stdout")
    args = parser.parse_args()
    versions = {}
    for package in PACKAGES:
        try: versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError: versions[package] = "NOT_INSTALLED"
    record = {"python_version": platform.python_version(), "operating_system": platform.system(),
              "packages": versions, "scope": "Installed versions only; not evidence of a successful analysis run"}
    text = json.dumps(record, indent=2) + "\n"
    if args.out:
        root = Path(__file__).resolve().parents[1]
        target = args.out.expanduser().resolve()
        if target == root or root in target.parents:
            raise SystemExit("Save execution records outside the public source repository.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print("Environment record saved to the requested private path.")
    else:
        print(text, end="")
if __name__ == "__main__":
    main()
