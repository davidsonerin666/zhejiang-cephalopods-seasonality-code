# Pre-release update (2026-11-09): supervisor-requested scope

- Removed three pure figure scripts and the stand-alone Figure 9 drawing instructions.
- Kept statistical conditional-dbRDA code, even when it prepares numerical ordination coordinates used by a chart.
- Removed `map` and `figures` from the `run_pipeline.py` stage list; `--stages all` now executes statistical-analysis stages only.
- Added four unchanged public-source NetCDF environmental grid subsets; NO private survey CSV from the original archive.
- Updated `.gitignore` to allow-list only the four gridded files and updated their source attribution/byte digests.
- Replaced project README and revised the statistical mapping to identify remaining exact-reproduction limitations.

This review is a static safety/source audit, not a new full fisheries-data run. All statistical scripts are otherwise left intact (some include embedded plotting routines).
