# Data Documentation

This repository contains data artifacts for **The Spatial Illusion: Beyond Correlation in GNN-Based EV Forecasting**.

The original paper used the controlled Washington benchmark and observed Caltech ACN sessions. The camera-ready revision adds UrbanEV geographic validation. These sources represent **different types of evidence** and must be described separately.

| Dataset | Geographic / physical context | Forecast target | Main location |
|---|---|---|---|
| Controlled Washington | Observed geographic units, coordinates, and EV-registration-derived scale | Synthetic hourly charging demand | `data/processed_v2/` |
| Caltech ACN | Observed sessions and electrical pod relationships | Hourly energy reconstructed from session/current-trace data | `data/real_acn/` |
| UrbanEV | Published regional charging observations and geographic distances | Published estimated regional charging energy | `revision_inputs/urbanev_*/data/` |

Paths above are relative to the repository root. UrbanEV snapshots are stored outside this `data/` directory to preserve the original inputs.

## Controlled Washington benchmark

```text
Geographic nodes:             581
Duration:                     365 days
Hourly observations per node: 8,760
Data seeds:                   42, 52, 62
```

Observed inputs provide geographic units, coordinates/centroids, EV-registration counts, and relative node demand scale. The hourly EV charging-demand trajectories are **synthetically generated**, not measured charging loads.

Relevant implementation files:

```text
experiments_v2/generator.py
experiments_v2/graphs.py
```

Controlled conditions:

```text
True coupling:
0.00, 0.25, 0.50, 0.75, 1.00

False-signal shared-shock strengths:
0.05, 0.10, 0.15, 0.20
with coupling fixed to zero
```

The geographic graph is a symmetric six-nearest-neighbor graph with 2,179 undirected edges. The benchmark controls lagged spatial dependence and shared contemporaneous shocks to test graph attribution under known generative assumptions.

See [experiments_v2/README.md](../experiments_v2/README.md) for the frozen model, generator, and graph-control protocol. The revision does not replace these original inputs or add a generator-sensitivity sweep.

## Caltech ACN observed data

The Caltech validation uses observed charging sessions from the Adaptive Charging Network. The hourly target is reconstructed from session/current-trace information; it is not the synthetic Washington target.

Prepared artifacts are stored under:

```text
data/real_acn/
```

Key prepared inputs:

```text
caltech_hourly_energy.csv
caltech_hourly_energy.npz
caltech_station_mapping.csv
caltech_pod_adjacency.npy
caltech_pod_edges.csv
caltech_pod_graph_metadata.json
caltech_selected_stations.csv
caltech_selected_sessions.csv
```

Original preparation and graph counts:

```text
Downloaded session metadata records: 31,424
Selected sessions:                  24,985
Usable measured-current sessions:   24,251
Selected EVSEs:                     37
Hourly matrix:                      (16080, 37)
Pod-connected EVSEs:                16
Isolated EVSEs in the full graph:    21
Undirected pod edges:               56
```

The primary neural experiment uses the **16 pod-connected EVSEs**, forming two eight-node cliques. The original RSPS analysis includes both the full 37-node set and the connected 16-node subset. The electrical pod graph is not a geographic graph.

Station ordering must remain consistent across energy columns, station mappings, adjacency matrices, edge lists, and model indices. Adding baselines or training seeds does not require rebuilding the prepared dataset.

See [real_acn/README.md](real_acn/README.md) for data details and [real_validation/README.md](../real_validation/README.md) for reconstruction steps and known timestamp-unit compatibility issues. In particular, verify timestamp units before rebuilding with a different pandas environment. The revision experiments use the preserved prepared matrix.

## UrbanEV geographic validation

The revision uses the published [UrbanEV dataset](https://github.com/IntelligentSystemsLab/UrbanEV). Cite [Li et al., Scientific Data (2025)](https://doi.org/10.1038/s41597-025-04874-4) when using it.

The completed revision experiments used source commit:

```text
44f2aa0c8d89f192bce00bafb0def74a21b39c68
```

The downloader creates a separate snapshot under:

```text
revision_inputs/urbanev_<commit-prefix>_<run-suffix>/
```

Each snapshot includes downloaded data, upstream documentation/license, a checksum manifest, and an inspection report. Preserve this snapshot for exact reproduction: a fresh download resolves the current upstream commit, which may differ.

### Files and target definition

| File | Purpose |
|---|---|
| `data/volume.csv` | Primary target: published estimated regional energy |
| `data/duration.csv` | Published charging-duration series; downloaded for inspection, not the primary target in the completed experiment |
| `data/inf.csv` | Station metadata, including zone identifiers and coordinates |
| `data/distance.csv` | Published distances used to construct the revision geographic graph |
| `data/adj.csv` | Published adjacency, retained as source data; not the graph used in the completed revision run |
| `download_manifest.json` | Source revision, file URLs, sizes, and checksums |
| `inspection_report.txt` | File shapes, timestamp coverage, and missing-value checks |

```text
Regions:                      275
Hourly observations:          4,344
Coverage:                     2022-09-01 through 2023-02-28
Geographic graph:             symmetric union of five nearest neighbors
Undirected graph edges:       843
Node degrees:                 5–10
```

All 275 published regions are retained. The new graph is built from published distances, not charging-target correlations. No extra load-based filtering or outcome-driven graph selection is performed by the revision runner.

### Limitations to preserve in reporting

- `volume.csv` contains **estimated energy**, not directly metered charging energy. Do not describe it as a new measured-energy dataset.
- Published preprocessing includes cleaning and bidirectional imputation before our chronological split. No missing values in the downloaded matrix does not prove that upstream preprocessing is free of future-information leakage.
- The distance matrix has column labels but no explicit row identifiers. The runner assumes row order matches the published column order and checks alignment, symmetry, and zero diagonal. These checks do not independently establish the geographic mapping.
- Timestamps are retained as published, without converting them to UTC.
- Only 28 daily forecast windows are available in the designated test period. Interpret the geographic results within this scope.

UrbanEV is published under CC0; retain the upstream license, provenance, and appropriate citation when sharing its artifacts.

## Chronological splitting

No random temporal split is used for these forecasting experiments. Training-period statistics are used for normalization in the repository's evaluation code, subject to the upstream UrbanEV preprocessing limitation noted above.

| Dataset | Train | Validation | Test |
|---|---|---|---|
| Washington | 250 days | 50 days | 65 days |
| Caltech, UTC | 2018-05-01 to 2019-08-12: 469 days | 2019-08-13 to 2019-11-20: 100 days | 2019-11-21 to 2020-02-29: 101 days |
| UrbanEV, published timestamps | 2022-09-01 to 2023-01-04: 126 days | 2023-01-05 to 2023-01-31: 27 days | 2023-02-01 to 2023-02-28: 28 days |

All three forecasting setups use 168-hour histories, 24-hour forecast horizons, and a 24-hour stride.

| Dataset | Training windows | Validation windows | Test windows |
|---|---:|---:|---:|
| Washington | 243 | 50 | 65 |
| Caltech | 462 | 100 | 101 |
| UrbanEV | 119 | 27 | 28 |

The first seven training days provide historical context. Validation/test windows can use preceding observed history, including earlier observations within their own period, while keeping forecast targets inside the designated split. Do not randomly shuffle the temporal split when reproducing the paper.

RSPS uses standardized daily totals and weekly residual differences rather than the neural hourly prediction target. The Caltech and UrbanEV diagnostic fits use training target days only and score validation target days only. Their residual-MSE scores should not be presented as neural test-MAE improvements.

## Revision inputs, outputs, and figures

Additional Caltech baselines and seed runs reuse the prepared ACN data. UrbanEV acquisition and evaluation use separate root-level scripts:

```text
download_urbanev.py
run_urbanev.py
run_urbanev_rsps.py
```

Preserve the distinction between data and model outputs:

```text
data/processed_v2/    Original controlled benchmark inputs
data/real_acn/        Original prepared ACN inputs and local raw records
revision_inputs/     Versioned UrbanEV inputs and provenance
results_v2/          Original Washington results
results_real/        Original Caltech results
revision_runs/       Camera-ready experiment outputs
paper_exports/       Generated revision figures and paper handoff packages
```

Generate figures from saved result files; do not regenerate the underlying datasets just to revise the paper. See the [script documentation](../scripts/README.md) for commands and the [root README](../README.md) for experimental findings.

## Version control and backups

Normally exclude these from Git:

```text
Raw ACN time-series JSON downloads
Private credentials and API tokens
Virtual environments and Python caches
Temporary logs and machine-specific caches
Large regenerated intermediate arrays
Large prediction arrays and model checkpoints
Redownloadable UrbanEV data copies, when preserved in a separate backup
```

Keep small provenance files, source revisions, checksum manifests, preprocessing documentation, final result CSVs, and frozen graph/permutation artifacts trackable when appropriate. Do not apply blanket ignore rules to every CSV, JSON, or NPZ file; some are essential reproducibility artifacts.

Ignored files still need a local or external backup. Git only preserves files that are tracked and committed, and `.gitignore` does not remove already tracked data or old history. Original ACN raw-session filenames containing colons can also prevent a direct Windows checkout; preserve the Ubuntu copy or a suitable archive.

Third-party raw data should only be redistributed when its license or terms permit it. Never commit private credentials. Keep the complete experiment data separately from the smaller paper-export ZIP, which is not a full backup.
