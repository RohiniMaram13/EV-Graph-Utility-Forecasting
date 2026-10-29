# Real-Data Validation Pipeline

This directory contains the **original Caltech ACN preparation and validation pipeline**. Camera-ready additions use separate Python scripts at the repository root and write to `revision_runs/`.

Run commands from the repository root with the configured environment active.

## Choose the appropriate workflow

- **Prepared inputs and completed results already exist:** keep them. Do not repeat the raw-data pipeline merely to generate figures or refine the manuscript.
- **Reproduce the original experiment:** preserve existing inputs/results first, then use the original pipeline documented below.
- **Reproduce camera-ready additions:** use the root-level scripts listed at the end of this document. They preserve the original experiment outputs.

See the [root README](../README.md) for the complete findings and [results documentation](../results_real/README.md) for the distinction between original three-seed and added five-seed Caltech results.

## Original pipeline

```text
ACN session metadata
-> inspect coverage
-> freeze pre-COVID interval
-> select active stations
-> download selected current traces
-> reconstruct hourly energy
-> map stationID to ACN-Sim spaceID
-> construct CC/AV pod graph
-> prepare chronological split and RSPS
-> run matched real-graph neural validation
```

Counts and numerical values below describe the frozen original data and saved outputs. A fresh API response or a different software environment is not guaranteed to reproduce them automatically.

## Step 1 — Verify API access

```bash
python real_validation/test_acn.py
```

A successful response reports HTTP status 200. Keep API tokens local and outside version control.

## Step 2 — Inspect dataset

```bash
python real_validation/inspect_acn.py
```

Original metadata snapshot:

```text
Total sessions:   31,424
Unique stations:  55
```

## Step 3 — Inspect timeline

```bash
python real_validation/inspect_timeline.py
```

Frozen paper interval: **2018-05-01 through 2020-02-29**, comprising 670 complete days.

## Step 4 — Select real dataset

```bash
python real_validation/select_real_dataset.py
```

The original selection rule requires at least **300 active days**.

```text
Selected stations: 37
Selected sessions: 24,985
Selected-session metadata energy: 223,827.48 kWh
```

This metadata-energy total is not the same quantity as the energy retained after current-trace processing.

Outputs:

```text
data/real_acn/caltech_selected_stations.csv
data/real_acn/caltech_selected_sessions.csv
```

The station selection is part of the frozen original protocol. Do not change it based on forecasting results.

## Step 5 — Download selected time series

```bash
python real_validation/download_selected_timeseries.py
```

The original download contains **24,985 session JSON records**. Preserve a separate backup of these raw records if excluded from Git.

Some original raw-session filenames contain colons and cannot be checked out directly on Windows. Retain the Ubuntu copy or a suitable archive; `.gitignore` does not repair filenames already present in Git history.

## Step 6 — Build hourly matrix

```bash
python real_validation/build_hourly_matrix.py
```

### Timestamp compatibility before rebuilding

The existing reconstruction code assumes that timestamp integer values from `.asi8` are in nanoseconds when converting intervals to seconds. A local check with pandas 3.0.5 exposed microsecond-resolution values for which that assumption gives incorrect elapsed times. Verify timestamp units or correct the conversion before using this script in a different environment. This documentation update does not patch that code.

For the camera-ready additions, use the preserved prepared matrix rather than rebuilding raw data unnecessarily.

### Original reconstruction report

```text
Hourly matrix shape:            (16080, 37)
Usable time-series sessions:     24,251
Missing chargingCurrent:            694
Bad time series:                      0
Zero-current + positive kWh:         40
Input energy:               218,229.383 kWh
Reported represented energy:218,216.734 kWh
Reported energy retention:      99.9942%
Charging hours:                 12,486
All-zero hours:                  3,594
Charging-hour fraction:          77.65%
```

The original reconstruction report's represented-energy value is retained above. A later float64 summation of the saved float32 NPZ matrix yielded approximately **218,216.699714 kWh**, a difference of about **0.034286 kWh**. Do not silently substitute one quantity for the other. The exact cause of this small discrepancy has not been independently established; identify the source used when reporting an energy total.

Outputs:

```text
data/real_acn/caltech_hourly_energy.csv
data/real_acn/caltech_hourly_energy.npz
```

Check matrix shape, nonnegative finite values, continuous hourly timestamps, station ordering, and represented energy before accepting any independently rebuilt matrix.

## Step 7 — Build station mapping

```bash
python real_validation/build_station_mapping.py
```

Original mapping checks:

```text
Selected stations:            37
Exactly one spaceID:          37
Multiple spaceIDs:             0
Mapped into ACN-Sim:          37 / 37
```

Output:

```text
data/real_acn/caltech_station_mapping.csv
```

Keep the hourly matrix, mapping, graph, edge list, and model node indices in the same station order. Do not independently sort one artifact without updating every dependent artifact.

## Step 8 — Build physical graph

```bash
python real_validation/build_real_graph.py
```

Original graph checks:

```text
Nodes:                 37
Pod-connected nodes:   16
Isolated nodes:        21
Undirected edges:      56
Degree 0 nodes:        21
Degree 7 nodes:        16
```

The connected subgraph consists of two eight-node cliques. The primary neural experiment uses only these **16 connected nodes**. This pod graph is not a geographic graph and does not represent all possible electrical relationships.

Outputs:

```text
data/real_acn/caltech_pod_adjacency.npy
data/real_acn/caltech_pod_edges.csv
data/real_acn/caltech_pod_graph_metadata.json
```

## Step 9 — Prepare split and RSPS

```bash
python real_validation/prepare_real_validation.py
```

This original preparation script writes diagnostics and frozen controls to `results_real/`. Preserve existing files before an intentional rerun.

```text
Hourly matrix: (16080, 37)
Daily matrix:  (670, 37)
```

| Split | Dates, UTC | Days | Forecast windows |
|---|---|---:|---:|
| Training | 2018-05-01 to 2019-08-12 | 469 | 462 |
| Validation | 2019-08-13 to 2019-11-20 | 100 | 100 |
| Test | 2019-11-21 to 2020-02-29 | 101 | 101 |

The neural task uses 168 hours of observed history to predict the next 24 hours, with daily forecast origins. Validation/test windows can use earlier observed history while keeping their forecast targets within their own split. Normalization is fitted only on training observations.

### RSPS configuration

The diagnostic uses standardized daily energy, weekly differencing, one-day residual lags, and pooled ridge regression:

```text
Weekly residual: r[i,k] = z[i,k] - z[i,k-7]
Self predictor: r[i,k-1]
Spatial predictors: r[i,k-1], sum_j W[i,j] r[j,k-1]
Ridge alpha: 1.0
Intercept: unpenalized
Fit: training target days only
Evaluate: validation target days only
```

The original preparation uses 100 graph permutations for each node set with permutation seed 777. The first 20 permutations are saved for neural controls.

Outputs:

```text
results_real/real_rsps_summary.csv
results_real/real_rsps_permutations_all37.csv
results_real/real_rsps_permutations_pod16.csv
results_real/real_graph_permutations.json
results_real/real_validation_split.json
```

| Original diagnostic | All 37 nodes | Pod-connected 16 nodes |
|---|---:|---:|
| Aligned RSPS | −0.000357 | −0.000415 |
| Mean shuffled RSPS | −0.001123 | −0.002055 |
| Shuffled RSPS standard deviation | 0.001071 | 0.002547 |
| Alignment difference | +0.000766 | +0.001640 |
| Shuffled scores exceeded | 78.0% | 72.0% |

Both aligned scores are slightly negative: adding the spatial predictor did not improve validation residual MSE over the self-history diagnostic. The positive alignment differences mean the aligned graph did better than the average shuffled graph. These are different comparisons.

## Step 10 — Run original neural validation

```bash
python -m real_validation.run_real_graph_validation --device cuda --batch-size 4 --n-permutations 20 --verbose
```

This original runner writes to `results_real/`; it is not the isolated additional-seeds runner.

```text
Nodes:                 16
Undirected real edges: 56
Training seeds:        42, 52, 62
Permutations per seed: 20
Lookback:              168 hours
Forecast horizon:      24 hours
```

The matched branches are a shared GRU, GRU-FT, the real pod-GNN, and permuted graph branches. Graph branches share the same temporal checkpoint and matched initialization within each seed. Training and checkpoint selection use training and validation data, respectively.

Original three-seed aggregate:

| Measure | Mean ± sample SD |
|---|---:|
| GRU-FT MAE | 0.3440348 ± 0.0043554 |
| Real pod-GNN MAE | 0.3430297 ± 0.0016157 |
| Real gain over GRU-FT | +0.2845% ± 0.9551% |
| Real gain over mean permutation | +1.2019% ± 2.5185% |
| Mean percentage of permutations beaten | 65.0% |
| Final graph gate | 0.01797 ± 0.01575 |

| Seed | Real gain over GRU-FT | Permutations beaten |
|---|---:|---:|
| 42 | +0.8817% | 55% |
| 52 | +0.7889% | 95% |
| 62 | −0.8171% | 45% |

These saved values describe the original experiment; they are not guarantees of bitwise reproduction across environments.

## Camera-ready additions

These root-level scripts use the prepared data and store their outputs in new `revision_runs/` directories. They do not require repeating Steps 1–9.

| Script | Addition |
|---|---|
| `run_seasonal_naive.py` | Weekly baseline on the same Caltech test targets |
| `run_dlinear.py` | Three-seed shared-channel DLinear baseline |
| `run_additional_caltech_seeds.py` | Original-model seeds 72/82; separate combined five-seed summary |
| `run_stgcn.py` | Adapted STGCN with three seeds and 20 shuffled graphs per seed |

```bash
python -u run_seasonal_naive.py
python -u run_dlinear.py
python -u run_additional_caltech_seeds.py
python -u run_stgcn.py
```

For a new environment, use `python -u run_stgcn.py --smoke` before a full STGCN run. Do not report smoke-test scores as experiment results.

The completed additions show that weekly seasonal naive has lower Caltech MAE but higher RMSE than the tested neural models, and that alignment gains remain small or seed-dependent across the five-seed original-model extension and adapted STGCN. Full values are documented in [results_real/README.md](../results_real/README.md).

### Separate UrbanEV validation

The real geographic experiment is implemented separately at the repository root:

```text
download_urbanev.py
run_urbanev.py
run_urbanev_rsps.py
```

It uses 275 regions and a distance-based geographic graph, rather than the Caltech electrical pod graph. Its target is published estimated energy, with upstream preprocessing limitations. UrbanEV RSPS has negligible linear association with neural error across graph permutations, so it should not be presented as confirming universal diagnostic reliability. See the [root README](../README.md) for commands, protocols, and findings.

## Figures, manuscript preparation, and preservation

To export completed revision results without retraining:

```bash
python -u export_paper_results.py
```

The exporter also expects the completed UrbanEV/RSPS outputs. It produces figures, full-precision comparison tables, and a small evidence package under `paper_exports/`; it does not edit the manuscript or constitute a complete data backup.

Keep original three-seed outputs and new results separately identifiable. Use full-precision CSVs for final tables, label seed counts, and distinguish standard deviations from confidence intervals. Preserve full experiment folders locally even when large files are excluded from Git.

The overall conclusion remains cautious: graph-model improvements do not necessarily imply substantial value from the particular graph alignment, and the usefulness of the RSPS diagnostic must be assessed within the tested setting.
