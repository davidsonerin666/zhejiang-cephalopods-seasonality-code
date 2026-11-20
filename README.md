# Zhejiang coastal cephalopods — Chapter 2 statistical analysis code

Research code and four publicly sourced gridded environmental NetCDF subsets for surveys in April and November, 2014–2024. This package is a **pre-release candidate for academic review**.
## Repository contents

- `scripts/`: 13 Python statistical and modelling scripts (including data cleaning, alpha/beta diversity, environmental matching, GEE, conditional dbRDA). The three standalone publication-figure scripts are omitted.
- `notebooks/Run_Chapter2_Private.ipynb`: unexecuted analysis entry point.
- `tools/`: environment information capture and static public-file audit.
- `data/environment/`: four environmental grids: sea temperature, salinity, chlorophyll-a and GEBCO_2024 bathymetry, with their original product IDs, acknowledgments and DOIs documented in the directory README.
- `run_pipeline.py`, `requirements.txt`, `config.example.json`, `INPUT_SCHEMA.md`, `MANUSCRIPT_CODE_MAP.md`, `PUBLIC_FILES.txt`: workflow and reproducibility documentation.

## Restricted data (NOT included)

**This package does not contain private fisheries survey CSV/Excel, confidential station data, event-level environment matches, catch records, species matrices, residuals or model predictions.** The source RAR archive and full analysis/output ZIP files must not be uploaded. Reproducing manuscript statistics requires separate lawful access to the private survey dataset.

## Running analyses

A local `config.local.json` (excluded by `.gitignore`) can be created from `config.example.json`; configure the authorized private fisheries input outside this repository. From repository root:

```bash
python run_pipeline.py --config config.local.json --stages all
```

## Publication and rights

The environmental data are attributed to Copernicus Marine Service and GEBCO; see `data/environment/README.md`. Confirm applicable redistribution terms and author/institutional permission to publish the scripts. A software LICENSE is not supplied; rights owners should choose an appropriate license separately.

Run `python tools/check_public_package.py` for **static** file safety checks. This does not audit GitHub commit history. If a GitHub account is restricted, seek explicit guidance from GitHub Support and do not use another account to circumvent enforcement.
