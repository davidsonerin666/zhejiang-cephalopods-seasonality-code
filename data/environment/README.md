# Public environmental raster subsets only (NOT survey observations)

The four NetCDF files are source-derived gridded environmental subsets for Zhejiang coastal waters, with 22 April/November layers for 2014–2024 (except the static bathymetry). They were extracted **without changing numerical values** from the previously archived `数据-261016.rar`. The original RAR also contains a PRIVATE fisheries CSV and **must not be shared**. No private survey records, observation dates linked to stations, station coordinates, catch values, model predictions or event-level matched environment tables are included here.

| File | Source dataset | Description |
| --- | --- | --- |
| `Temperature_0.125deg_April_November_2014_2024.nc` | Copernicus Marine ARMOR3D physical product (DOI 10.48670/moi-00052) | Multidepth sea temperature; 22 month layers, 16 depths, 0.125° grid |
| `Salinity_0.125deg_April_November_2014_2024.nc` | Copernicus Marine ARMOR3D physical product (DOI 10.48670/moi-00052) | Multidepth salinity; same grid/time |
| `Chla_4km_April_November_2014_2024.nc` | Copernicus Marine global ocean colour product (DOI 10.48670/moi-00283) | Chlorophyll-a, 4 km subset; 22 month layers |
| `Depth_0.005°_2024.nc` | **GEBCO_2024**, DOI 10.5285/1c44ce99-0a0d-5f4f-e063-7086abc0ea0f | Elevation (negative at sea), 15-arc-second native grid subset; filenames use the approximately 0.005° label |

Product identifiers embedded in the physical files: `MULTIOBS_GLO_PHY_TSUV_3D_MYNRT_015_012`; chlorophyll: `OCEANCOLOUR_GLO_BGC_L4_MY_009_108`. Retain source metadata and use the matching dataset versions, not a newer GEBCO release, to maintain numerical consistency.

**Attribution for redistributed Copernicus Marine information**: E.U. Copernicus Marine Service Information; https://doi.org/10.48670/moi-00052 and https://doi.org/10.48670/moi-00283. See https://help.marine.copernicus.eu/en/articles/4444611-citing-copernicus-marine-products-and-services and the applicable Copernicus Marine Licence Agreement. **GEBCO** terms: https://www.gebco.net/data-products/gridded-bathymetry/terms-of-use; credit GEBCO Bathymetric Compilation Group (2024), `GEBCO_2024 Grid`, DOI above. Public permission to redistribute these products does not extend to private field survey data.

`ENVIRONMENT_SHA256.json` in the repository root gives an integrity digest for each file. The scripts require a separate, authorized fisheries input for statistical reproduction. The `enso.csv` in the original archive is not required by the present Chapter 2 pipeline and is not included.
