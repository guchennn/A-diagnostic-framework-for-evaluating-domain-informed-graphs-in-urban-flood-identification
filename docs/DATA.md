# Data acquisition and attribution

Provider access checked on 31 October 2026.
The frozen input contract is `configs/sources.yaml`, with event windows in
`configs/cities/{zhengzhou,beijing}.yaml`. Obtain all seven datasets independently from
the providers below, using your own accounts and accepting their applicable terms.
Raw provider files, individual reports, identifiers, text, toponyms, coordinates,
location-linked ledgers and cell-level predictions are excluded from this release.
CrowdFlood non-redistribution is a project privacy choice. Its provider licence is CC BY 4.0.

## Acquisition workspace and commands

Use a private checkout with Python 3.12 and run commands from its root. All paths below
are relative to that root. Keep `data/` and generated acquisition receipts outside public
Git history. The CLI supports `--root <workspace>` for a workspace containing `configs/`.
It uses fixed destination paths rather than a general file-import argument.

```sh
uv sync --frozen
uv run floodgnn fetch --city all --source all
uv run floodgnn prepare --city all --through worldcover
uv run floodgnn prepare --city all --through osm
uv run floodgnn prepare --city all --through worldpop
uv run floodgnn prepare --city all --through imerg
uv run floodgnn prepare --city all --through dsm
uv run floodgnn prepare --city all --through terrain
uv run floodgnn prepare --city all --through features
uv run floodgnn prepare --city all --through design
```

Run sequentially, resolving any provider failure before proceeding. `fetch --source all`
fetches CrowdFlood, geoBoundaries and the WorldCover tile index. WorldCover preparation
selects and downloads its map rasters. The `osm`, `worldpop`, `imerg` and `dsm` preparation
targets each acquire their named source and rerun labels/WorldCover prerequisites.
After DSM preparation, use `terrain`, `features` or `design` to consume existing DSM and
final-grid inputs. Preserve `artifacts/protocol_lock.json`.

Acquisition writes `<asset>.source.json` receipts and
`artifacts/manifests/raw_source_manifest.parquet`, recording bytes, SHA-256, source
identity and retrieval time. Keep those private with the acquired files. If downloading
manually, preserve the exact provider bytes and destination filenames below, then run
the corresponding CLI stage to validate and register them. API responses require their
request/provenance metadata. Preserve complete receipts for reuse. First-download
SHA-256 values are locally computed where the provider supplies no frozen digest.
The reference hashes below come from recorded project acquisition reports, not a new
full download during this documentation check. ETags are never substituted for SHA-256.

## 1. CrowdFlood v1

Use Zenodo record **19921295**, DOI **10.5281/zenodo.19921295**.
The concept record **19921294** identifies the version family. Download the pinned v1
archive without login or an application. Cite Chen et al. (2026), key
`chen2026crowdflood` in [data_references.bib](data_references.bib).
[Provider record](https://zenodo.org/records/19921295).

- [Exact ZIP](https://zenodo.org/api/records/19921295/files/CrowdFlood%20dataset.zip/content)
- Destination `data/raw/crowdflood_v1/19921295/CrowdFlood dataset.zip`
- Size `16290257` bytes, provider MD5 `53b7389ef3495f172d73d1dc54180518`
- Recorded SHA-256 `68f6c1662b629878c50b9874d20780c5a6ddcf6835ead0661bf8106965ed9c52`

Keep the ZIP intact. Label preparation extracts `CrowdFlood dataset/csv/` members
`Zhengzhou_City_2021_7_19.csv` and `Beijing_2023_7_30.csv` into the archive directory's
`events/` subdirectory. Event identifiers are `Zhengzhou City_2021_7_19` and
`Beijing_2023_7_30`. STRICT-LOC is the frozen `pt_type_1=1` filter.

## 2. geoBoundaries CHN ADM2

Use `gbOpen/CHN/ADM2` at commit
`9469f09592ced973a3448cf66b6100b741b64c0d`, boundary ID `CHN-ADM2-17275852`,
source year 2017 and build date 12 December 2023. Public download requires no account.
The pinned metadata records PDDL v1.0 for the source boundary. geoBoundaries requests
CC BY 4.0 attribution. Cite Runfola et al. (2020), `runfola2020geoboundaries`, and retain
the commit identifier. [Access and attribution](https://www.geoboundaries.org/countryDownloads.html),
[pinned provider metadata](https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/9469f09592ced973a3448cf66b6100b741b64c0d/releaseData/gbOpen/CHN/ADM2/geoBoundaries-CHN-ADM2-metaData.json).

- [Exact GeoJSON](https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/9469f09592ced973a3448cf66b6100b741b64c0d/releaseData/gbOpen/CHN/ADM2/geoBoundaries-CHN-ADM2.geojson)
- Destination `data/raw/geoboundaries_chn_adm2/9469f095/geoBoundaries-CHN-ADM2.geojson`
- Size `7618692` bytes, SHA-256 `2b68d8a808742fc6d7acd769584db960d8fc2c25b9f1d20e3e98c72e9f1c4d34`
- Preserve 2,391 features and `OGC:CRS84`. Select `Zhengzhoushi` / `17275852B14866353147502`
  and `Beijingshi` / `17275852B40522342172572` exactly as configured.

## 3. Copernicus DEM GLO-30 DGED 2024_1

Use collection `CCM`, dataset `COP-DEM_GLO-30-DGED/2024_1`, product type
`SAR_DGE_30_A4AD`. This DSM includes surface objects. The input is GeoTIFF with horizontal
EPSG:4326 and vertical EPSG:3855. Cite `esa2022copdem`, DOI **10.5270/ESA-c5d3d65**,
and specify the 2024_1 delivery. Follow the GLO-30 licence and its Article 6 attribution
and modified-product notices. [Collection and access](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM),
[provider licence, GLO-30 section](https://dataspace.copernicus.eu/sites/default/files/media/files/2025-06/copernicus_contributing_mission_data_access_v2_cop_dem_licenses.pdf#page=22).

Register and verify your own CDSE account, and complete applicable CCM registration/user-category
steps. General-public registered users can access GLO-30. The CLI prompts interactively for
CDSE credentials and holds tokens in memory. Acquire with `prepare --through dsm`.
[Registration](https://documentation.dataspace.copernicus.eu/Registration.html),
[CCM eligibility and registration](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions).

The catalogue is `https://catalogue.dataspace.copernicus.eu/odata/v1/Products`.
The downloader uses `https://download.dataspace.copernicus.eu/odata/v1` and the frozen
UUID/product name in `configs/sources.yaml` to address each product's `DEM` GeoTIFF leaf.
Destination pattern is
`data/raw/cop_dem_glo30_dged_2024_1/2024_1/<tile>/Copernicus_DSM_10_<Nxx>_00_<Exxx>_00_DEM.tif`.
Each of the following six files is `51897972` bytes. The UUID and recorded SHA-256 identify
the exact tile. [OData collection guidance](https://documentation.dataspace.copernicus.eu/Data/Others/CCM.html).

| Tile | Product UUID | Recorded SHA-256 |
|---|---|---|
| N34E113 | `679359b8-91a0-40ba-a5de-40fc4c70e526` | `0eb30f2b035c7cc45233db12507e7c74f9ad7411bd6c14deeeb06d1a3a76f2cd` |
| N35E113 | `861f53dd-85a8-44ed-a7d1-27f7638c8d56` | `55165977c070777731ce7704c26dae5a321b3f8546fc002f000a8faab45783ff` |
| N39E115 | `21efee57-54a2-479e-8e04-145c0a2ad78b` | `11904f489d8b8ec0ad73f9d5a2eb0206ce2212976c312f4c997a42c0664c74d7` |
| N39E116 | `bfe10121-32c7-444a-a217-d9160893d5be` | `d8bfeaf81b9d5e884577444e875103aca28c1b5dbc1cc8c2ced0ffbd5bbdc6c5` |
| N40E115 | `c328b460-61e5-4360-a111-3b627a4c2114` | `30d287ca6edb0bebe161221144aad2bdc58caf36a9231bcdf5fc067e20603f1c` |
| N40E116 | `41829d3d-52eb-4871-a99d-840b483383dc` | `e888a0842aa94603b04d92a9ba44871bf9739aa5eb2669282a071504457f0a2f` |

## 4. ESA WorldCover 2021 v200

Use the **2021 v200 Map** product. Cite Zanaga et al. (2022), `zanaga2022worldcover`,
DOI **10.5281/zenodo.7254221**. Access through the public S3 HTTPS URLs is anonymous.
The provider's Terrascope route separately requires an account. WorldCover is CC BY 4.0.
For maps, use the provider credit “© ESA WorldCover project 2021 / Contains modified
Copernicus Sentinel data (2021) processed by ESA WorldCover consortium”.
[Provider access, licence and citation](https://esa-worldcover.org/en/data-access).

The CLI stores the [tile index](https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/esa_worldcover_grid.fgb)
at `data/raw/worldcover_2021_v200/v200/esa_worldcover_grid.fgb`.
It is `516760` bytes with recorded SHA-256
`f56a3935e6819c95f0ca89e7dc9583e9d0fa92644e549ff9c5b8b4858fc831ac`.
The two rasters go in `data/raw/worldcover_2021_v200/v200/map/`, retaining their filenames.

| Tile and exact download | Bytes | Recorded SHA-256 |
|---|---|---|
| [N33E111](https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N33E111_Map.tif) | 130476058 | `d70b317b612aaabc4eda86f1a4ef468f93fc7be8b6e9c738ab682f9856ff330c` |
| [N39E114](https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N39E114_Map.tif) | 108133854 | `1abcb25bdbc0030eb2656dfba9881a333e0535f88a6d5ecee719c9ac476ea9ea` |

## 5. NASA GPM IMERG Final Run V07B

Use `GPM_3IMERGHH`, collection version `07`, CMR concept `C2723754847-GES_DISC`,
file suffix `V07B.HDF5`. Cite Huffman et al. (2023), `huffman2023imerg`, DOI
**10.5067/GPM/IMERG/3B-HH/07**, and record your actual acquisition date. The selected
variable is `Grid/precipitation`, in mm h-1, at 30-minute intervals.
[GES DISC dataset](https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGHH_07/summary),
[NASA usage and citation policy](https://gpm.nasa.gov/data/policy).

Create your own [NASA Earthdata Login](https://urs.earthdata.nasa.gov/), and authorize
GES DISC access if prompted. The CLI uses `earthaccess.login(strategy="interactive",
persist=False)`, searches the frozen CMR collection and retrieves complete granules.
Files go to `data/raw/imerg_v07b/GPM_3IMERGHH.07/<YYYY>/<DDD>/<provider filename>`,
preserving the year/day-of-year path returned by GES DISC. Provider checksums and
locally computed SHA-256 are recorded in the granule receipts/manifests.

| City | UTC granule coverage, including 72-hour antecedent period | Count |
|---|---|---|
| Zhengzhou | 2021-07-16 11:00:00 through 2021-08-04 08:59:59.999 | 908 |
| Beijing | 2023-07-27 06:30:00 through 2023-08-11 13:59:59.999 | 735 |

The first and last exact filenames are frozen in each city YAML. For example, Zhengzhou
starts with `3B-HHR.MS.MRG.3IMERG.20210716-S110000-E112959.0660.V07B.HDF5`.
Retain the Final Run V07B identity when using a catalogue that also offers newer products.

## 6. Historical OpenStreetMap through HeiGIT ohsome

Use OSM snapshots at `2021-08-04T08:58:00Z` for Zhengzhou and
`2023-08-11T13:42:00Z` for Beijing. These are the configured event-end timestamps.
Credit “© OpenStreetMap contributors” and ODbL 1.0, using `openstreetmap_contributors`.
Cite the extraction infrastructure separately with `raifer2019oshdb` and `heigit_ohsome`.
[OSM attribution and licence](https://www.openstreetmap.org/copyright).

The frozen base is `https://api.ohsome.org/v1`. Although the source contract names
`/elements/geometry`, the acquisition code requests paired `/elements/bbox` and
`/elementsFullHistory/geometry` responses. It partitions buffered AOIs into 0.25-degree
boxes, keeps tags and metadata with `clipGeometry=false`, and deduplicates by
`@osmId` and `@version`. Preserve the exact roads/waterways filters in `configs/sources.yaml`.
Full-history requests cover the snapshot timestamp through one second later.

Destination pattern is
`data/raw/ohsome_v1/<apiVersion>/<city>/<YYYYMMDDTHHMMSSZ>/<roads|waterways>/part-<0000>.<bbox|full-history>.geojson`.
The actual API version, replication sequence, request SHA-256 and response SHA-256 are
captured during acquisition. API response byte identity can change with backend revisions.

At this check the unauthenticated v1 metadata endpoint returned API `1.10.4` and a
history endpoint date of 27 July 2026. The v1 documentation URL redirects to v2, whose
instructions require a free HeiGIT account/API key. The frozen client sends v1 requests
without an API key. Full extraction was not rerun, and v2 compatibility is unverified.
[Live v1 metadata](https://api.ohsome.org/v1/metadata),
[current API documentation and registration](https://docs.ohsome.org/ohsome-api/v2/).

## 7. WorldPop R2024B v1, China unconstrained population counts

Use **R2024B/v1**, 100 m (3 arc-second), unconstrained population counts for **2021**
(Zhengzhou) and **2023** (Beijing). Values are persons per grid cell. This is the beta
release documented in 2025. Cite `worldpop2025global2` and `worldpop_r2024b_chn`, and
acknowledge WorldPop. These pinned files are publicly downloadable without login.
[Release statement](https://data.worldpop.org/repo/prj/Global_2015_2030/R2024B/doc/Global2_Release_Statement_R2024B_v1.pdf),
[WorldPop citation guidance](https://www.worldpop.org/faq/).

WorldPop publishes general CC BY 4.0 terms and an ODbL exception for datasets derived
from OSM or Microsoft building/road inputs. The project source YAML leaves per-asset
licensing unresolved, while its acquisition report records ODbL for derived outputs.
Retain this distinction and check the applicable asset terms when downloading. An
asset-specific DOI for these two pinned files was not verified, so the bibliography
identifies their release and exact file URLs. [Provider licensing statement](https://hub.worldpop.org/geodata/summary?id=80032).

Both files go in `data/raw/worldpop_r2024b/R2024B/v1/100m/unconstrained/`, retaining the
provider filenames. Use `prepare --through worldpop` after the public-source fetch.
The inspected standalone `fetch --source worldpop_r2024b` branch expects four return
values from a function that returns three. That branch requires a code correction
before use. The preparation route consumes the three-value return directly.

| Exact download | Bytes | Recorded SHA-256 |
|---|---|---|
| [chn_pop_2021_UC_100m_R2024B_v1.tif](https://data.worldpop.org/GIS/Population/Global_2015_2030/R2024B/2021/CHN/v1/100m/unconstrained/chn_pop_2021_UC_100m_R2024B_v1.tif) | 2092461699 | `14cd2439fc2e00e6f1b95d4fb36e3b7a72a074d1d85aa05cae7bb0d13079b773` |
| [chn_pop_2023_UC_100m_R2024B_v1.tif](https://data.worldpop.org/GIS/Population/Global_2015_2030/R2024B/2023/CHN/v1/100m/unconstrained/chn_pop_2023_UC_100m_R2024B_v1.tif) | 2138386657 | `58dad7d4daa0d2b1ade8ab9af41d118879a6629150bdfee5830dbf7d25b0906c` |

## Verification limits and access blockers

Public asset header checks succeeded for CrowdFlood, the pinned boundary, WorldCover
index and both map tiles, and both WorldPop files, with sizes matching the frozen
records. The CDSE catalogue resolved the N34E113 UUID. Authenticated CDSE leaf downloads
and NASA granule downloads were not attempted. They require the reader's credentials.
The NASA CMR metadata request failed in this verification environment with a TLS EOF,
and the GES DISC page exposed only its script shell to the web reader. Availability of
the complete frozen V07B inventory therefore remains to be checked during acquisition.
The ohsome metadata check establishes service response only. Account requirements and
provider version changes may affect future extraction.

This guide was checked against `sources.py`, `audit.py`, `cli.py`, `features.py`,
`terrain.py`, `imerg.py`, `osm.py`, `worldpop.py`, the source/city YAML files, acquisition
reports and the v8 reproducibility records. The commands were inspected rather than
executed end to end. Reconstruct private prepared inputs and complete the repository's
protocol/environment checks before training.
