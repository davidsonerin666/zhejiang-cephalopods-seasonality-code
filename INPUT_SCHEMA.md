# Input schema without observations

No real or synthetic research records are included. These are literal column names recognized in the recovered private source CSV. The example configuration uses a generic filename, `fishery_long.csv`.

| Literal column | Type/unit | Meaning |
| --- | --- | --- |
| 年 | Integer | Survey year |
| 月 | Integer | Survey month, April or November |
| 站位 | String | Original station identifier; private |
| 经度 | Numeric degrees | Longitude; private |
| 纬度 | Numeric degrees | Latitude; private |
| 渔获名称 | String | Original Chinese taxon label |
| 拉丁名 | String | Original scientific-name label |
| 目 | String | Order label |
| 科 | String | Family label |
| 拖速3节时的1h重量(g) | Numeric g | One-hour catch weight as stored in the source |
| 拖速3节时的1h尾数(ind) | Numeric individuals | One-hour number as stored in the source |
| 生物量(kg/km2) | Numeric kg km-2 | Provided biomass density; the cleaning script does not fix density-conversion discrepancies |
| 尾数密度(千尾/km2) | Numeric 10^3 individuals km-2 | Provided numerical density; retain these units |
| 调查状态 | String | Survey status used to distinguish unsurveyed missing values from surveyed zero catches |

## Missing values and zeros

Unsurveyed records remain missing; surveyed zero-catch events contain zero catch/density. The original status labels and conflicts are interpreted by `scripts/step1_clean.py`. Retain UTF-8 encoding and do not convert missing values to zero globally. Species and station mapping are generated from private inputs.

## Environment files

Four public gridded environmental NetCDF files are included in `data/environment/`: temperature, salinity, chlorophyll-a and bathymetry. Original product metadata and attributions are documented in `data/environment/README.md`.

Grid variable names, depth selection and matching are handled by `scripts/step2_environment.py`. Source product metadata can be inspected with `scripts/environment_metadata.py`. File names alone do not establish product-native resolution.

Private fisheries survey records and event-level matched environmental data are not included.

## Derived inputs

The workflow generates private event/master tables, species matrices, model inputs, predictions, residuals and logs. These should remain outside the public repository.
