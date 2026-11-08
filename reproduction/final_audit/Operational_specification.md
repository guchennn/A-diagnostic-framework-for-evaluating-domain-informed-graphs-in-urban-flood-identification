# Data, terrain attributes and graph operations

This specification documents the implemented frozen workflow; it does not introduce a new preparation protocol. `feature_operations.csv` lists all 18 attributes, units, source versions, resampling operators, temporal rules, coverage requirements and missing-value conventions directly from `configs/feature_schema.yaml`.

## Spatial domains and alignment

Zhengzhou and Beijing use 200 m cells in EPSG:32649 and EPSG:32650. The eligible study grid is the accepted built-up extent plus its 2 km context, clipped to the selected administrative unit, excluding cells with water fraction above 0.5 or DSM coverage below 0.8. It contains 25,155 and 61,443 cells, including 791 and 296 STRICT-LOC positive cells. Feature joins are one-to-one by grid ID and do not change geometry or eligibility.

The terrain computation domain is the aligned bounding rectangle of the administrative AOI buffered by 10 km. It is larger than the eligible study grid and distinct from the full eligible-city prediction graph. Copernicus GLO-30 DGED 2024_1 is a digital surface model (horizontal EPSG:4326, vertical EPSG:3855); valid-area weighted averaging forms its aligned 200 m surface. Priority-Flood fills depressions using eight-neighbour connectivity. Valid cells bordering the domain or missing terrain seed the flood queue, with deterministic elevation/row/column ordering. These edge rules define the available terrain context; they do not claim an unbounded catchment reconstruction.

## Terrain attributes and routing

All six terrain attributes use the conditioned extended DSM and are then extracted for eligible cells before spatial splitting. They are not recalculated on held-out graphs.

| Attribute | Implemented operation |
| --- | --- |
| Elevation | Filled 200 m mean elevation, metres. |
| Slope | Horn 3 × 3 operator, degrees; all nine values must be valid. |
| Relative elevation | Centre minus valid-cell mean in a centred 5 × 5 window (1 km), metres. |
| Local relief | Maximum minus minimum in that window, metres; at least 13 valid cells. |
| Log flow accumulation | `ln(1 + 0.04 N)`, with N the number of upstream cells including the receiver. `0.04 N` is contributing area numerically expressed in km². |
| Topographic wetness | `ln[(a + 1 m) / tan(max(beta, 0.5 degrees))]`, with `a = 200 N m` and beta the Horn slope. The offset and floor are implementation choices. |

D8 selects the greatest strictly positive elevation drop divided by cardinal or diagonal distance. Equal-gradient ties retain the first direction in the fixed order E, SE, S, SW, W, NW, N, NE. Drainable flats use an integer FlatMask without an epsilon perturbation of elevation. Accumulation includes the receiver and rejects routing cycles. Conditioning and routing are performed on the extended surface before restricting endpoints to eligible cells.

G0 has no external edges. G1 includes both directions of eight-neighbour adjacency. G2 retains each eligible cell's terrain-derived downstream edge only when both endpoints are eligible, with unit weight, no self-loop or duplicate and outdegree at most one. An ineligible downstream receiver is not replaced by another eligible neighbour. Thus the full G2 graph is already an eligible-domain restriction, not the complete extended terrain-routing network.

Spatial evaluation uses five outer and three inner folds, 4 km blocks and a 1 km polygon-distance buffer. Each partition graph keeps only edges with both endpoints in that partition; crossing edges and edges touching buffer nodes are removed. The frozen outer-test graph was verified to equal that endpoint restriction exactly. Terrain-derived covariates remain unchanged by graph pruning. Imputation and standardization are subsequently fitted only on permitted sampled training cells.

## Other environmental attributes and temporal availability

IMERG Final V07B assigns the nearest native pixel to each grid centre and multiplies half-hour rates by 0.5 h. Exact local record windows are 19 July 2021 19:03–4 August 2021 16:58 for Zhengzhou and 30 July 2023 14:50–11 August 2023 21:42 for Beijing (UTC+8). Event totals select half-hour midpoints in the closed window. Antecedent totals use the preceding 144 half-hour midpoints, excluding the start. Maximum six-hour rainfall takes complete sequences of 12 intervals, advanced by one interval. Selected intervals are integrated without endpoint clipping. Incomplete sums are missing; maximum rainfall uses only complete finite windows. Event-window inputs can include rainfall after an individual report and define retrospective information.

WorldCover 2021 v200 categorical data use nearest-neighbour reprojection followed by aligned 20 × 20 aggregation. Fractions divide selected-class area by valid area and require at least 50% valid coverage. Vegetation combines classes 10, 20, 30, 90, 95 and 100; built-up, water and cropland are classes 50, 80 and 40. The waterbody distance is measured from cell centre to the nearest class-80 10 m pixel boundary, capped at 10 km.

OSM history snapshots correspond to the record-window end. Road and waterway lengths are clipped to cells and expressed in km/km²; distance to the nearest eligible waterway is in metres and capped at 10 km. An absence of intersecting eligible lines is a valid zero density. WorldPop R2024B uses the corresponding 2021 or 2023 count layer, sum-preserving reprojection, valid-area density in persons/km² and `ln(1 + density)`, requiring 50% valid coverage. The prepared feature audit found no missing terrain attributes; Beijing population density has two missing cells, handled by training-only imputation.

## Diagnostic accounting

Local comparisons use fixed FF-G2-1 against both FF-G0-1 (own) and FF-G1-1 (adjacency), 18 attributes and 100 revealed positives. Held-out D8 groups are fixed for all references: no incoming support, incoming support without a two-edge upstream path, or such a path (multi-hop). AP terms retain the full test ranking and sum across groups to the complete AP contrast. Background-relative rank uses half credit for ties. Five draws and three refits are averaged within each fold, then the five folds equally. The 10,000 paired fold-resampling intervals are descriptive, conditional on these fits and partitions.

Migration compares each cell's group in the full eligible-city graph with its group in its own outer-test graph, counting each cell once. Pooled transition counts are cell weighted, unlike the equal-fold performance estimates. Test-target edge retention divides retained test–test edges by full-city edges whose target belongs to that test set. Train-plus-test retention uses all full-city edges as its denominator and is reported separately. Group relabelling sensitivity changes only group membership, retaining exactly the same held-out predictions. Stable group labels do not imply unchanged upstream neighbours or establish the causal effect of graph pruning.

## Implementation entry points

`terrain.py` defines DSM conditioning, D8 routing and six terrain attributes; `graphs.py` defines eligible and partition restrictions. `features.py`, `imerg.py` and `feature_schema.yaml` define environmental alignment and temporal rules. `evaluation_design.py` and `split_config.yaml` define spatial isolation. `training.py` defines training-only preprocessing. The accompanying source snapshot and SHA-256 provenance identify the actual inspected files. The README separates aggregate verification, replay from frozen local products and post-DSM preparation.
