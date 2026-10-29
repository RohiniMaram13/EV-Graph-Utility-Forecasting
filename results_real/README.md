# Caltech Real-Data Results

This directory contains the **original observed Caltech ACN validation outputs**, including the three-seed neural experiment and frozen graph permutations. Keep these results intact. Camera-ready additions are saved separately under `../revision_runs/`.

Related documentation:

- [Data preparation](../data/real_acn/README.md)
- [Original validation code](../real_validation/README.md)
- [Repository overview and revision results](../README.md)
- [Reproduction and figure commands](../scripts/README.md)

Reported values below are rounded. Use the full-precision CSVs for final tables and calculations. `±` denotes sample standard deviation, not a confidence interval.

## Original RSPS outputs

```text
real_rsps_summary.csv
real_rsps_permutations_all37.csv
real_rsps_permutations_pod16.csv
real_graph_permutations.json
real_validation_split.json
```

The original diagnostic uses 100 graph permutations for each station set. The saved neural controls use the first 20 permutations for the corresponding node set.

| Diagnostic | All 37 EVSEs | Pod-connected 16 EVSEs |
|---|---:|---:|
| Aligned RSPS | −0.000357 | −0.000415 |
| Mean shuffled RSPS | −0.001123 | −0.002055 |
| Shuffled RSPS standard deviation | 0.001071 | 0.002547 |
| RSPS alignment difference | +0.000766 | +0.001640 |
| Percentage of shuffled scores exceeded | 78.0% | 72.0% |

RSPS is the relative reduction in validation residual MSE when a lagged neighbor predictor is added to an own-lag predictor. A negative RSPS means the spatial diagnostic performed worse than the self-history diagnostic. A positive alignment difference means the aligned graph performed better than the average shuffled graph; it does **not** necessarily mean that adding the spatial predictor improved over self-history. Here, both aligned RSPS scores are slightly negative.

The diagnostic is linear, single-lag, and noncausal. It does not exhaust the nonlinear or multi-lag information a neural model might exploit.

## Original neural outputs: three seeds

```text
real_gnn_seed_summary.csv
real_gnn_final_summary.csv
real_gnn_permutation_runs.csv
real_gnn_permutations_seed42.csv
real_gnn_permutations_seed52.csv
real_gnn_permutations_seed62.csv
real_pod16_station_list.csv
```

### Configuration

```text
Nodes:                         16 pod-connected EVSEs
Undirected physical edges:     56
Training seeds:                42, 52, 62
Permuted graphs per seed:      20
Lookback / forecast horizon:   168 / 24 hours
Forecast stride:               24 hours
```

### Original aggregate

| Measure | Mean ± sample SD |
|---|---:|
| GRU-FT MAE | 0.3440348 ± 0.0043554 |
| Real pod-GNN MAE | 0.3430297 ± 0.0016157 |
| Real gain over GRU-FT | +0.2845% ± 0.9551% |
| Real gain over mean permutation | +1.2019% ± 2.5185% |
| Mean percentage of permutations beaten | 65.0% |
| Final graph gate | 0.01797 ± 0.01575 |

### Original seed results

| Seed | Real gain over GRU-FT | Real gain over mean permutation | Permutations beaten |
|---|---:|---:|---:|
| 42 | +0.8817% | +0.0853% | 55% |
| 52 | +0.7889% | +4.0857% | 95% |
| 62 | −0.8171% | −0.5651% | 45% |

A positive gain denotes lower MAE for the real graph. Per-seed alignment gain is calculated as:

```text
100 * (mean_permuted_MAE - real_graph_MAE) / mean_permuted_MAE
```

Aggregate gains are the mean of the per-seed percentages. They need not equal a percentage calculated from aggregate MAEs.

## Camera-ready additions: separate output directories

These experiments do not replace the original files above. Paths are relative to the repository root; generated run suffixes vary.

| Addition | Directory | Main result files |
|---|---|---|
| Weekly seasonal naive | `revision_runs/seasonal_naive_*/` | `metrics.csv`, `predictions.npz`, `run_details.json` |
| DLinear, seeds 42/52/62 | `revision_runs/dlinear_*/` | `seed_metrics.csv`, `summary_metrics.csv`, `run_details.json` |
| Added original-model seeds 72/82 | `revision_runs/caltech_extra_seeds_*/` | `results_real/real_gnn_seed_summary.csv`, `combined_five_seed_metrics.csv`, `combined_five_seed_summary.csv` |
| Adapted STGCN, seeds 42/52/62 | `revision_runs/stgcn_full_*/` | `all_runs.csv`, `seed_summary.csv`, `summary_metrics.csv`, `run_details.json` |

### Original model extended to five seeds

The extension adds seeds **72 and 82**, each with all 20 frozen shuffled graphs. The combined analysis uses **42, 52, 62, 72, 82**. Preserve labels distinguishing the original and added runs and disclose execution-environment differences.

| Measure | Five-seed mean ± sample SD |
|---|---:|
| GRU-FT MAE | 0.346089 ± 0.006573 |
| Real pod-GNN MAE | 0.338716 ± 0.008242 |
| Real pod-GNN RMSE | 0.806499 ± 0.003083 |
| Real gain over GRU-FT | +2.112172% ± 2.602249% |
| Real gain over mean permutation | +0.629533% ± 1.947556% |
| Percentage of permutations beaten | 54.000000% ± 24.596748% |

The two added seeds have alignment gains of approximately −0.35% and −0.11%. The five-seed results therefore retain the finding of small, seed-dependent graph-alignment value.

### Additional forecasting baselines

All rows below use the same 16 EVSEs and 101 daily test windows. MAE and RMSE are evaluated in hourly energy units (kWh), after reversing training-only normalization.

| Model | Seeds | Test MAE | Test RMSE |
|---|---:|---:|---:|
| Weekly seasonal naive | Deterministic | 0.304991 | 1.122903 |
| Shared-channel DLinear | 3 | 0.357444 ± 0.004120 | 0.817999 ± 0.000891 |
| Original pod-GNN, five-seed extension | 5 | 0.338716 ± 0.008242 | 0.806499 ± 0.003083 |
| Adapted STGCN with aligned graph | 3 | 0.331638 ± 0.016670 | 0.806595 ± 0.001282 |

Weekly seasonal naive has lower MAE than the neural models in this table, but higher RMSE. Do not describe the graph model as uniformly superior. Different seed counts are explicitly labeled; these descriptive comparisons are not significance tests.

The adapted STGCN comparison uses 20 shuffled graphs per seed. Its mean alignment gain is **−0.113776% ± 4.798527%**. Seed-level gains are approximately −5.655%, +2.660%, and +2.653%: two positive cases and one negative case. The implementation adapts the standard STGCN output head to direct 24-hour prediction and uses the repository's training protocol; it is not an exact reproduction of the original STGCN paper.

## Reproduction and consistency checks

Original Caltech inputs and split:

```text
Selected stations:                 37
Stations mapped into ACN-Sim:       37 / 37
Pod-connected nodes:               16
Isolated nodes in the full graph:   21
Undirected physical edges:         56
Train / validation / test days:    469 / 100 / 101
Train / validation / test windows: 462 / 100 / 101
Original training seeds:           42, 52, 62
Original-model extension seeds:    72, 82
Neural permutations per seed:      20
```

| Split | Dates, UTC |
|---|---|
| Training | 2018-05-01 through 2019-08-12 |
| Validation | 2019-08-13 through 2019-11-20 |
| Test | 2019-11-21 through 2020-02-29 |

Station ordering must remain consistent across the hourly matrix, mapping, adjacency, edge list, and model indices. Scaling is fitted only on training observations. Validation/test forecast windows may use preceding observed history, while their targets remain within the designated split.

The graph has two eight-node cliques and 21 isolated nodes in the full 37-node representation. The primary neural analysis is restricted to the 16 connected nodes. Findings should not be generalized to richer electrical or geographic networks without further evidence.

From the repository root, reproduce additions only when needed:

```bash
python -u run_seasonal_naive.py
python -u run_dlinear.py
python -u run_additional_caltech_seeds.py
python -u run_stgcn.py
```

For a new environment, use `python -u run_stgcn.py --smoke` before the full STGCN run. Smoke-test scores are not final experimental results.

To export existing results into figures and tables without retraining:

```bash
python -u export_paper_results.py
```

The exporter also expects the completed UrbanEV and RSPS revision outputs. See the root README for those experiments. UrbanEV results are not stored in this original Caltech directory.

## Interpretation

Under the frozen Caltech selection, pod graph, chronological split, and 24-hour task, the original residual GRU+GCN model shows a small, seed-dependent alignment advantage. Expanding to five seeds and testing adapted STGCN does not establish a consistently beneficial physical alignment. The seasonal-naive comparison also shows that forecasting superiority depends on the metric.

This supports the need for aligned-versus-shuffled controls; it does not establish that physical graphs are generally useless or that RSPS reliably predicts neural benefits in every setting. A near-zero or negative diagnostic result is a finding, not a reason to omit an experiment.

The reported standard deviations are descriptive. No significance claim is made, and the 20 shuffled graphs within a seed must not be counted as 20 independent training seeds.

## Preserve the evidence

Keep original three-seed CSVs and frozen permutations unchanged, alongside the separate revision outputs. Track final CSVs, protocols, summaries, and small graph artifacts in Git when appropriate. Keep large prediction arrays and model checkpoints in a separate backup if excluded by `.gitignore`.

Console values in this README are rounded summaries. Before final manuscript submission, use the saved full-precision files, verify figure/table cross-references, and retain the distinction between original and added experiments.
