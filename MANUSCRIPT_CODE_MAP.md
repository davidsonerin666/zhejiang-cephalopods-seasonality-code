# Chapter 2 statistical code mapping (no stand-alone figure sources)

This is an inventory of analysis methods and code; it does not certify exact reproduction of the current manuscript figures.

| Analysis | Main retained script(s) | Output or purpose |
|---|---|---|
| Data quality, event status and taxonomy | `step1_clean.py`, `taxonomy_revision.py` | 0 versus unsurveyed NA, station mapping, taxon names |
| Environmental matching and metadata | `step2_environment.py`, `environment_metadata.py` | SST, SSS, Chla and bathymetry, matching QC |
| Event master | `step3_master.py` | Presence/biomass matrices and analysis event tables |
| Biomass, occurrence and resource statistics | `step4_resources.py` | Seasonal and annual response summaries |
| IRI, dominance and species groups | `step4b_composition.py` | IRI contribution and taxon composition |
| Alpha diversity | `step4c_alpha.py` | Richness, Shannon, Simpson, Pielou and paired effects |
| Beta diversity and spatial structuring | `step5_beta.py`, `section33_spatial.py` | NMDS/PERMANOVA/PERMDISP/SIMPER, centroid, hotspots, coverage sensitivity |
| Environmental association / model validation | `step6_environment_models.py` | GEE, cross-validation, Wald tests, environmental dbRDA |
| Independent conditional-dbRDA check | `recheck_conditional_dbrda.py` | Recomputed conditional tests from private event data |
| Statistical reconstruction of conditional ordination coordinates | `rebuild_partial_ordination.py` | SVD, conditioned coordinates and diagnostics; not a figure renderer |

**Limitations:** The recovered `step6_environment_models.py` is not yet established as the identical original version that produced every final figure/table. A separate private reconstruction of 4-month RMSE 59.04, 11-month RMSE 350.48 and Chla q = 0.079 exists, but is not silently substituted into this source package. The original Supplementary Figure S4 Moran permutation code has not been fully recovered. The mixed analysis scripts also contain embedded plotting functions, which remain intact so that the statistical stages are not inadvertently changed.

The repository omits the pure plotting scripts (`figure1_map.py`, `publication_figures.py`, `draw_figure9.py`) at the supervisor's request. Those scripts are not necessary for statistical run stage selection. Keep private fisheries and any generated plot data/coordinate outputs outside the public repository.
