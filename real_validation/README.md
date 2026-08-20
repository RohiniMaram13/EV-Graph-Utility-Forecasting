# Real-Data Validation Pipeline

This directory contains the Caltech ACN real-data validation pipeline.

Run commands from the repository root.

## Pipeline

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

## Step 1 — Verify API access

```bash
python real_validation/test_acn.py
```

Expected:

```text
HTTP status: 200
```

Keep API tokens local.

## Step 2 — Inspect dataset

```bash
python real_validation/inspect_acn.py
```

Expected:

```text
Total sessions:   31,424
Unique stations:  55
```

## Step 3 — Inspect timeline

```bash
python real_validation/inspect_timeline.py
```

Frozen paper interval:

```text
2018-05-01 through 2020-02-29
```

## Step 4 — Select real dataset

```bash
python real_validation/select_real_dataset.py
```

Selection rule:

```text
minimum active days = 300
```

Expected:

```text
Selected stations: 37
Selected sessions: 24,985
Total energy:      223,827.48 kWh
```

Outputs:

```text
data/real_acn/caltech_selected_stations.csv
data/real_acn/caltech_selected_sessions.csv
```

## Step 5 — Download selected time series

```bash
python real_validation/download_selected_timeseries.py
```

Expected JSON records:

```text
24,985
```

## Step 6 — Build hourly matrix

```bash
python real_validation/build_hourly_matrix.py
```

Expected:

```text
Shape:                       (16080, 37)
Usable time-series sessions: 24,251
Missing chargingCurrent:        694
Bad time series:                  0
Zero-current + positive kWh:     40
Input energy:            218,229.383 kWh
Represented energy:      218,216.734 kWh
Energy retention:              99.9942%
Charging hours:                12,486
All-zero hours:                 3,594
Charging-hour fraction:         77.65%
```

Outputs:

```text
data/real_acn/caltech_hourly_energy.csv
data/real_acn/caltech_hourly_energy.npz
```

## Step 7 — Build station mapping

```bash
python real_validation/build_station_mapping.py
```

Expected:

```text
37 selected stations
37 with exactly one spaceID
0 with multiple spaceIDs
37 / 37 mapped into ACN-Sim
```

Output:

```text
data/real_acn/caltech_station_mapping.csv
```

## Step 8 — Build physical graph

```bash
python real_validation/build_real_graph.py
```

Expected:

```text
Nodes:                37
Pod-connected nodes:  16
Isolated nodes:       21
Undirected edges:     56
Degree 0:             21
Degree 7:             16
```

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

Expected matrices:

```text
Hourly: (16080, 37)
Daily:  (670, 37)
```

Split:

```text
Train: 469 days / 462 windows
Validation: 100 days / 100 windows
Test: 101 days / 101 windows
```

RSPS ridge:

```text
alpha = 1.0
```

Expected diagnostic outputs:

```text
results_real/real_rsps_summary.csv
results_real/real_rsps_permutations_all37.csv
results_real/real_rsps_permutations_pod16.csv
results_real/real_graph_permutations.json
results_real/real_validation_split.json
```

Expected all-37:

```text
Real RSPS:                -0.000357
Permutation mean:         -0.001123
Permutation std:           0.001071
Alignment advantage:       0.000766
Real graph beats:         78.0%
```

Expected pod-16:

```text
Real RSPS:                -0.000415
Permutation mean:         -0.002055
Permutation std:           0.002547
Alignment advantage:       0.001640
Real graph beats:         72.0%
```

## Step 10 — Run neural validation

```bash
python -m real_validation.run_real_graph_validation     --n-permutations 20     --verbose
```

Frozen configuration:

```text
Nodes:                  16
Real edges:             56
Training seeds:         42, 52, 62
Permutations per seed:  20
Lookback:               168 hours
Forecast horizon:       24 hours
```

Expected aggregate:

```text
GRU-FT MAE:
0.3440348 +/- 0.0043554

Real Pod-GNN MAE:
0.3430297 +/- 0.0016157

Real gain over GRU-FT:
+0.2845% +/- 0.9551%

Real vs mean permutation:
+1.2019% +/- 2.5185%

Mean permutations beaten:
65.0%

Final graph gate:
0.01797 +/- 0.01575
```

Expected seed behavior:

```text
Seed 42: +0.8817%, 55% permutations beaten
Seed 52: +0.7889%, 95% permutations beaten
Seed 62: -0.8171%, 45% permutations beaten
```

The real-data result is intentionally cautious.