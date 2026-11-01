# Data

Download the source datasets from their providers. Versions, identifiers, filters
and destination paths are configured in `configs/sources.yaml` and
`configs/cities/`. Follow the preparation commands in the running guide.
[Preparation](REPRODUCE.md#prepare-the-data).

| Dataset used | Access | Citation |
| --- | --- | --- |
| CrowdFlood v1, record 19921295 | Public ZIP | Chen et al. (2026), [10.5281/zenodo.19921295](https://zenodo.org/records/19921295) |
| geoBoundaries CHN ADM2, commit `9469f09592ced973a3448cf66b6100b741b64c0d` | Public download | Runfola et al. (2020), [10.1371/journal.pone.0231866](https://doi.org/10.1371/journal.pone.0231866) |
| Copernicus DEM GLO-30 DGED 2024_1 | CDSE registration | European Space Agency and Airbus (2022), [10.5270/ESA-c5d3d65](https://doi.org/10.5270/ESA-c5d3d65) |
| ESA WorldCover 2021 v200 | Public S3 downloads | Zanaga et al. (2022), [10.5281/zenodo.7254221](https://zenodo.org/records/7254221) |
| GPM IMERG Final V07B, half-hourly 0.1° | NASA Earthdata account and GES DISC authorization | Huffman et al. (2023), [10.5067/GPM/IMERG/3B-HH/07](https://doi.org/10.5067/GPM/IMERG/3B-HH/07) |
| Historical OpenStreetMap through ohsome v1 | HeiGIT geometry queries | OpenStreetMap contributors and Raifer et al. (2019), [10.1186/s40965-019-0061-3](https://doi.org/10.1186/s40965-019-0061-3) |
| WorldPop R2024B v1, China unconstrained 100 m, 2021/2023 | Public raster downloads | WorldPop (2025), [release statement](https://data.worldpop.org/repo/prj/Global_2015_2030/R2024B/doc/Global2_Release_Statement_R2024B_v1.pdf) |

Complete citations are supplied in BibTeX. [data_references.bib](data_references.bib).

## Download details

- **CrowdFlood**. Keep `CrowdFlood dataset.zip` intact. Preparation extracts
  `Zhengzhou_City_2021_7_19.csv` and `Beijing_2023_7_30.csv`, applies `pt_type_1=1`
  and merges records within 200 m cells. Expected totals are 791 and 296 recorded
  cells. [Pinned ZIP](https://zenodo.org/api/records/19921295/files/CrowdFlood%20dataset.zip/content).
- **Boundaries**. Select `Zhengzhoushi` (`17275852B14866353147502`) and
  `Beijingshi` (`17275852B40522342172572`).
  [Pinned GeoJSON](https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/9469f09592ced973a3448cf66b6100b741b64c0d/releaseData/gbOpen/CHN/ADM2/geoBoundaries-CHN-ADM2.geojson).
- **Elevation**. The six DSM product IDs are fixed in the source configuration.
  The CLI requests CDSE credentials interactively.
  [Registration](https://documentation.dataspace.copernicus.eu/Registration.html).
- **Land cover**. Preparation downloads tiles N33E111 and N39E114 using the
  configured public S3 URLs. [Access and terms](https://esa-worldcover.org/en/data-access).
- **Rainfall**. Use collection `GPM_3IMERGHH`, version `07`, with filenames ending
  in `V07B.HDF5`. City settings include the 72-hour antecedent period and exact
  boundary filenames. [Earthdata registration](https://urs.earthdata.nasa.gov/),
  [dataset](https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGHH_07/summary).
- **Roads and waterways**. Use snapshots `2021-08-04T08:58:00Z` and
  `2023-08-11T13:42:00Z`. The client uses v1. HeiGIT's newer v2 service requires
  an account/API key. [v1 metadata](https://api.ohsome.org/v1/metadata),
  [current API documentation](https://docs.ohsome.org/ohsome-api/v2/).
- **Population**. Use the two R2024B/v1 count rasters selected in
  `configs/sources.yaml`. Record the release and acquisition date when citing
  them. [WorldPop release](https://data.worldpop.org/repo/prj/Global_2015_2030/R2024B/).

## Storage and terms

Original files go under `data/raw/<source>/`. Derived grids and attributes go to
`data/interim/<city>/`. These local directories are excluded from Git. Shared
experiment aggregates and model settings are in `results/`.

CrowdFlood and WorldCover use CC BY 4.0. Obtain CrowdFlood records directly from
Zenodo under its terms. Follow Copernicus GLO-30's licence, NASA's data policy,
geoBoundaries attribution requirements and OSM's ODbL attribution. Check the terms
for each WorldPop asset, including the ODbL exception for products derived from
OSM/building data.
[Copernicus terms](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM),
[NASA policy](https://gpm.nasa.gov/data/policy), [OSM terms](https://www.openstreetmap.org/copyright),
[WorldPop guidance](https://www.worldpop.org/faq/).
