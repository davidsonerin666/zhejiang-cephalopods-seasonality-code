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

