# Caltech Real-Data Results

This directory contains outputs from the observed Caltech ACN validation.

See:

```text
../data/real_acn/README.md
../real_validation/README.md
```

## RSPS outputs

```text
real_rsps_summary.csv
real_rsps_permutations_all37.csv
real_rsps_permutations_pod16.csv
real_graph_permutations.json
real_validation_split.json
```

Expected all-37:

```text
Real RSPS:                -0.000357
Permutation mean:         -0.001123
Permutation std:           0.001071
Alignment advantage:       0.000766
Permutations beaten:      78.0%
```

Expected pod-16:

```text
Real RSPS:                -0.000415
Permutation mean:         -0.002055
Permutation std:           0.002547
Alignment advantage:       0.001640
Permutations beaten:      72.0%
```

## Neural outputs

```text
real_gnn_seed_summary.csv
real_gnn_final_summary.csv
real_gnn_permutation_runs.csv
real_gnn_permutations_seed42.csv
real_gnn_permutations_seed52.csv
real_gnn_permutations_seed62.csv
real_pod16_station_list.csv
```

Expected final configuration:

```text
n_nodes = 16
n_real_edges = 56
n_train_seeds = 3
n_permutations_per_seed = 20
```

Expected aggregate:

```text
GRU-FT MAE:
0.3440348 +/- 0.0043554

Real-GNN MAE:
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
Seed 42:
Real gain over GRU-FT = +0.8817%
Permutations beaten   = 55%

Seed 52:
Real gain over GRU-FT = +0.7889%
Permutations beaten   = 95%

Seed 62:
Real gain over GRU-FT = -0.8171%
Permutations beaten   = 45%
```

## Reproduction checks

Verify:

```text
37 selected stations
37 / 37 mapped into ACN-Sim
16 pod-connected nodes
21 isolated nodes
56 physical edges
469 / 100 / 101 chronological days
462 / 100 / 101 forecast windows
3 training seeds
20 neural graph permutations per seed
```

Station ordering must remain consistent across the hourly matrix, mapping, adjacency, edge list, and model indices.

## Interpretation

These results do **not** support a claim of a large or uniformly positive real-world GNN gain.

The reproducible conclusion is:

> Under the frozen Caltech station selection, pod-based graph, chronological split, 24-hour forecasting task, and residual GRU+GCN architecture, the physical graph shows weak residual predictive signal and a small, seed-dependent graph-specific forecasting advantage.


