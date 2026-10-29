# The Spatial Illusion: Beyond Correlation in GNN-Based EV Forecasting

## Overview

This repository studies whether a proposed graph's **specific alignment with node histories** contributes to EV charging forecasts. An improvement from adding a graph branch does not, by itself, establish that the proposed connections are responsible.

The evaluation combines:

- **Topology-preserving node permutations:** compare aligned and shuffled graphs while retaining graph topology and the degree distribution. Individual named nodes may receive different degrees after permutation.
- **Matched training controls:** compare the temporal baseline, its fine-tuned counterpart, and graph branches initialized from the same temporal checkpoint.
- **Residual Spatial Predictability Score (RSPS):** a lightweight, linear, single-lag diagnostic of whether neighboring residuals improve next-day residual prediction beyond the node's own lag.

The evidence covers a controlled synthetic Washington benchmark, observed Caltech ACN charging data with an electrical pod graph, and published UrbanEV regional data with a geographic graph. UrbanEV's charging-volume target is **estimated energy**, not directly metered energy.

RSPS is a complementary diagnostic, not a replacement for direct forecasting controls. It tracked alignment benefits strongly across controlled benchmark conditions, but did not meaningfully rank neural performance across UrbanEV graph permutations.

## Experiment status and provenance

The original Washington and three-seed Caltech results are preserved. Camera-ready revision experiments are stored separately in `revision_runs/`.

| Experiment | Status | Evidence |
|---|---|---|
| Original Washington benchmark | Existing results retained | `results_v2/` |
| Original Caltech experiment, seeds 42/52/62 | Existing results retained | `results_real/` |
| Caltech weekly seasonal naive | Completed | `revision_runs/seasonal_naive_*/` |
| Caltech shared-channel DLinear, three seeds | Completed | `revision_runs/dlinear_*/` |
| Additional Caltech seeds 72/82 | Completed; five-seed summary available | `revision_runs/caltech_extra_seeds_*/` |
| Adapted STGCN, three seeds and 20 permutations per seed | Completed | `revision_runs/stgcn_full_*/` |
| UrbanEV geographic graph, three seeds and 20 permutations per seed | Completed | `revision_runs/urbanev_full_*/` |
| UrbanEV RSPS on the same 20 permutations | Completed | `revision_runs/urbanev_rsps_*/` |
| Revision figures and paper handoff package | Export script supplied; generation/visual review pending confirmation | `paper_exports/` |
| Manuscript integration of the new experiments | Pending | Update tables, claims, limitations, and reviewer responses |

Revision numbers below are rounded from completed Ubuntu run summaries supplied by the researcher. Full-precision CSVs, protocols, and available prediction/checkpoint artifacts in the run folders are the authoritative evidence. Not every runner saves final model checkpoints or neural predictions; preserve the complete outputs it does produce.

## Main findings

### Controlled Washington benchmark — original experiments

The benchmark uses 581 geographic units with synthetic hourly charging trajectories. It varies true lagged spatial coupling and contemporaneous shared shocks independently.

- At zero coupling, GEO-GNN and PERM-GNN are approximately indistinguishable, despite GEO-GNN improving over GRU-FT.
- Greater true coupling generally produces a larger alignment advantage within the tested generator. This is not a claim of strict monotonicity in every run.
- Shared shocks can produce high raw neighbor correlation without a comparable graph-alignment advantage.
- Across nine pooled conditions, the original reported correlations with forecasting alignment advantage are **0.968 for RSPS alignment** and **0.791 for raw-correlation alignment**.

These controlled correlations concern variation across benchmark conditions. They are not the same analysis as the within-dataset UrbanEV graph-permutation correlation reported below.

### Caltech — original three-seed results

The original results use seeds **42, 52, 62**, with 20 permuted graphs per seed.

| Measure | Original mean ± sample SD |
|---|---:|
| GRU-FT MAE | 0.3440348 ± 0.0043554 |
| Real pod-GNN MAE | 0.3430297 ± 0.0016157 |
| Real gain over GRU-FT | +0.2845% ± 0.9551% |
| Real gain over mean permuted graph | +1.2019% ± 2.5185% |
| Mean percentage of permutations beaten | 65.0% |
| Final graph gate | 0.01797 ± 0.01575 |

### Caltech — added camera-ready evidence

The original model was extended to seeds **42, 52, 62, 72, 82** by running only the two additional seeds. The combined five-seed summary mixes original and newly executed runs; retain their separate provenance and record environment differences.

| Measure | Five-seed mean ± sample SD |
|---|---:|
| GRU-FT MAE | 0.346089 ± 0.006573 |
| Real pod-GNN MAE | 0.338716 ± 0.008242 |
| Real pod-GNN RMSE | 0.806499 ± 0.003083 |
| Real gain over GRU-FT | +2.112172% ± 2.602249% |
| Real gain over mean permuted graph | +0.629533% ± 1.947556% |
| Percentage of permutations beaten | 54.000000% ± 24.596748% |

Additional forecasting baselines use the same Caltech test targets:

| Model | Seeds | Test MAE | Test RMSE |
|---|---:|---:|---:|
| Weekly seasonal naive | Deterministic | **0.304991** | 1.122903 |
| Shared-channel DLinear | 3 | 0.357444 ± 0.004120 | 0.817999 ± 0.000891 |
| Real pod-GNN, five-seed extension | 5 | 0.338716 ± 0.008242 | **0.806499 ± 0.003083** |
| Adapted STGCN with aligned pod graph | 3 | 0.331638 ± 0.016670 | 0.806595 ± 0.001282 |

Bold entries mark the lowest reported mean in each metric column, not statistically established superiority. Seed counts differ. Weekly seasonal naive has lower MAE than these neural models, while neural models have lower RMSE. The paper must not claim universal forecasting superiority.

For adapted STGCN, alignment gains over the mean shuffled graph were approximately **−5.655%, +2.660%, and +2.653%** for seeds 42, 52, and 62. The mean is **−0.113776% ± 4.798527%**: alignment benefits vary substantially across seeds.

### UrbanEV — observed geographic validation

The geographic GNN uses the original matched GRU/graph-branch protocol on 275 regions. Its graph is the symmetric union of five-nearest-neighbor connections from the published distance matrix.

| Measure | Three-seed mean ± sample SD |
|---|---:|
| GRU-FT MAE | 72.579956 ± 0.775026 |
| Geographic GNN MAE | 72.451553 ± 0.750757 |
| Geographic GNN RMSE | 291.351359 ± 3.049993 |
| Mean permuted-GNN MAE | 72.487692 ± 0.739126 |
| Geographic gain over GRU-FT | +0.176646% ± 0.110586% |
| Geographic gain over mean permuted graph | +0.049956% ± 0.054612% |

The aligned geographic graph has a small positive advantage in each seed: approximately **+0.112%, +0.030%, and +0.008%**. This is not zero benefit, but neither its practical importance nor statistical significance should be overstated.

MAE and RMSE here refer to the published regional estimated-energy target. Their absolute values should not be compared directly with per-EVSE Caltech errors.

### UrbanEV — RSPS extension and its limitation

The extension reuses the exact graph and 20 permutations from the completed neural experiment. It fits the original linear, single-lag diagnostic on training days and evaluates it on validation days.

| Diagnostic | Value |
|---|---:|
| Aligned RSPS | 0.03105048 |
| Mean shuffled RSPS | 0.02974016 |
| RSPS alignment difference | +0.00131032 |
| Shuffled scores exceeded by aligned RSPS | 14 / 20 |
| Exploratory Pearson correlation: RSPS vs mean neural test MAE across 21 graphs | −0.042642 |

The positive diagnostic alignment difference agrees in direction with the small positive neural alignment gain. However, the near-zero correlation does **not** demonstrate useful linear ranking of neural forecasting performance by RSPS in this setting.

RSPS is a dimensionless reduction in validation residual MSE. Its value of 0.03105048 corresponds to a 3.105048% reduction relative to the own-lag diagnostic, not a 3.105048% improvement in neural forecasting. The aligned-versus-shuffled RSPS difference is 0.131032 percentage points of residual-MSE reduction.

This was a retrospective extension with fixed diagnostic settings. Graphs share data and topology; the correlation is descriptive, not an independent-sample significance test or a general validation across datasets.

## Repository layout

Place the supplied standalone revision scripts at the repository root before using the commands below.

```text
.
├── README.md
├── requirements.txt
├── setup_linux_gpu.sh
├── check_gpu.py
├── experiments_v2/                # Original controlled benchmark and model code
├── real_validation/               # Original ACN preparation and evaluation
├── data/
│   ├── processed_v2/
│   └── real_acn/                   # Prepared ACN inputs and local raw downloads
├── results_v2/                    # Original Washington results
├── results_real/                  # Original Caltech results and frozen controls
├── scripts/                       # Original reproduction/figure scripts
├── run_seasonal_naive.py
├── run_dlinear.py
├── run_additional_caltech_seeds.py
├── run_stgcn.py
├── download_urbanev.py
├── run_urbanev.py
├── run_urbanev_rsps.py
├── export_paper_results.py
├── revision_inputs/               # Versioned UrbanEV snapshots and manifests
├── revision_runs/                 # Separately named revision experiment outputs
└── paper_exports/                 # Generated figure/table handoff packages
```

Run suffixes are generated automatically. The scripts retain existing outputs and create new revision directories. Where a script expects exactly one completed run and finds several, select the intended run explicitly when its CLI supports that option; do not delete runs simply to satisfy discovery.

## Environment

The Ubuntu revision runs reported:

```text
Python: 3.12 environment
PyTorch: 2.11.0+cu128
CUDA runtime: 12.8
GPU: NVIDIA GeForce RTX 5090
```

CUDA tensor operations and graph-convolution forward/backward checks passed. Hardware and software differences can change numerical results.

For an existing configured environment:

```bash
source .venv/bin/activate
python check_gpu.py
```

For a fresh installation, inspect and use the repository setup script:

```bash
chmod +x setup_linux_gpu.sh
./setup_linux_gpu.sh
```

## Original experiments

The original paper reproduction script is:

```bash
chmod +x scripts/run_all_paper_experiments.sh
./scripts/run_all_paper_experiments.sh
```

This is the **original** experiment pipeline, not a wrapper for the standalone camera-ready additions. Original runners can write to the original result directories; preserve their existing evidence before deliberate reproduction. No original experiments need to be rerun merely to generate revision figures.

### Washington benchmark protocol

Washington geographic units, coordinates, and EV-registration-derived scale information provide context; hourly charging trajectories are synthetic.

| Setting | Value |
|---|---|
| Nodes / duration | 581 / 365 days |
| Lookback / horizon / stride | 168 / 24 / 24 hours |
| Train / validation / test | 250 / 50 / 65 days |
| Train / validation / test windows | 243 / 50 / 65 |
| Data seeds / training seeds | 42, 52, 62 / 42, 52, 62 |
| Coupling levels | 0, 0.25, 0.50, 0.75, 1.00 |
| Shared-shock strengths at zero coupling | 0.05, 0.10, 0.15, 0.20 |
| Geographic graph | Symmetric 6-nearest-neighbor graph |
| Undirected edges | 2,179 |
| Median edge distance | 12.54 km |
| Frozen graph seed | 777 |

With row-normalized adjacency `W`, the controlled mechanism is:

```text
s_k = 0.70 * [(1-c) s_(k-1) + c W s_(k-1)] + eta_k
```

Shared shocks increase contemporaneous similarity without deliberately introducing this lagged spatial mechanism. The original approximate GEO-versus-PERM advantages are −0.01% at zero coupling and +4.92% at maximum coupling; corresponding gains over GRU-FT are +3.78% and +11.25%.

Relevant files:

```text
experiments_v2/generator.py
experiments_v2/graphs.py
experiments_v2/novelty_diagnostics.py
experiments_v2/models.py
experiments_v2/run_full_graph_control.py
results_v2/paper_all_runs_gs777.csv
results_v2/paper_scenario_summary_gs777.csv
results_v2/paper_main_table_gs777.csv
results_v2/paper_association_by_data_seed_gs777.csv
results_v2/paper_association_pooled_gs777.csv
```

### Caltech inputs and protocol

The original data preparation reports 31,424 downloaded sessions, 24,985 selected sessions, 24,251 usable measured-current sessions, and 37 selected EVSEs. The primary neural analysis uses the **16 pod-connected EVSEs**, with 56 undirected edges forming two eight-node cliques. The other 21 selected EVSEs are isolated in the pod graph.

| Split | Dates, UTC | Forecast windows |
|---|---|---:|
| Train | 2018-05-01 to 2019-08-12 | 462 |
| Validation | 2019-08-13 to 2019-11-20 | 100 |
| Test | 2019-11-21 to 2020-02-29 | 101 |

There are 16,080 hourly observations. Models use 168 hours of history to predict 24 hours, with daily forecast origins and normalization fitted only on training data. Earlier observations from a validation/test period can serve as history at later forecast origins.

Use the existing prepared hourly matrix, adjacency, and station mapping for the revision scripts. The original raw-data pipeline is documented under `real_validation/` and `data/real_acn/`. Rebuilding raw ACN inputs is not required for these additions; independently verify timestamp units and preprocessing equivalence if rebuilding with a different pandas version.

### Matched GRU/graph training

The graph model preserves the local temporal representation:

```text
h_fused = h_local + sigmoid(graph_gate_logit) * h_spatial
graph_gate_logit initially = -5.0
```

The original matched protocol uses a shared GRU checkpoint, up to 100 baseline epochs, 30 graph-only epochs, and 70 joint fine-tuning epochs. Learning rates are 0.001 for baseline/graph stages and 0.0005 for fine-tuning. Batch size is four, weight decay is 0.0001, and gradient clipping is 1.0. Checkpoint selection uses validation loss. The temporal fine-tuning and graph branches retain eligible incoming checkpoints.

## Reproduce the camera-ready additions

All commands below run from the repository root with the configured environment active. Completed runs need not be repeated to export figures.

### Caltech baselines and extra seeds

```bash
python -u run_seasonal_naive.py
python -u run_dlinear.py
python -u run_additional_caltech_seeds.py
```

- Weekly seasonal naive predicts the corresponding hour one week earlier and requires no training.
- DLinear uses a 25-hour endpoint-padded moving average and separate shared-channel trend/residual projections from 168 to 24 hours. Seeds are 42/52/62, with up to 100 epochs and validation early stopping. This is a tested configuration, not a claim about every DLinear variant.
- The additional-seeds script runs **only 72 and 82**, with the original 20 frozen permutations, then combines them with the original three-seed CSV in a separate output directory.

### Additional STGNN architecture

For a fresh implementation/environment check:

```bash
python -u run_stgcn.py --smoke
```

For the reportable experiment:

```bash
python -u run_stgcn.py
```

The implementation follows the [original STGCN architecture](https://github.com/VeritasYin/STGCN_IJCAI-18), with Chebyshev spatial convolutions and temporal convolution blocks. It adapts the head to direct 24-hour output and uses the repository's training protocol; it is not an exact replication of the original STGCN paper's training setup. It has 5,312,832 parameters.

Aligned and permuted treatments within each seed share initial learned parameters and training budgets. Unlike the original residual GRU-GCN model, STGCN does not use a GRU warm start. The full run includes three seeds and 20 permutations per seed. The mathematical consistency check uses double precision; actual training retains float32/AMP settings.

### UrbanEV download and geographic evaluation

Dataset source: [UrbanEV](https://github.com/IntelligentSystemsLab/UrbanEV), published under CC0. Cite [Li et al., Scientific Data (2025)](https://doi.org/10.1038/s41597-025-04874-4) when using the dataset.

The completed experiments used source commit:

```text
44f2aa0c8d89f192bce00bafb0def74a21b39c68
```

For initial acquisition:

```bash
python -u download_urbanev.py
```

The downloader resolves the current source commit and records it; it does not guarantee the same revision on future runs. Preserve the completed experiment's snapshot and checksum manifest for exact reproduction. If multiple snapshots exist, pass the intended one with `--snapshot PATH`.

```bash
python -u run_urbanev.py --smoke
python -u run_urbanev.py
```

| Setting | Frozen revision protocol |
|---|---|
| Target | Published `volume.csv`, estimated regional energy |
| Nodes / observations | 275 regions / 4,344 hourly observations |
| Training | 2022-09-01 to 2023-01-04: 126 days, 119 forecast windows |
| Validation | 2023-01-05 to 2023-01-31: 27 days/windows |
| Test | 2023-02-01 to 2023-02-28: 28 days/windows |
| Lookback / horizon / stride | 168 / 24 / 24 hours |
| Graph | Binary symmetric union of five nearest neighbors from published distances |
| Graph size | 843 undirected edges; degrees 5–10 |
| Training seeds | 42, 52, 62 |
| Permutations | 20 per seed; permutation RNG seed 20261029 |

All 275 published regions are retained. Graph construction does not use charging targets. The split and graph rule were recorded before the new neural evaluation. Timestamps remain in their published form and are not converted to UTC.

Important limitations:

- The target is estimated rather than directly metered energy.
- Published preprocessing includes bidirectional imputation and precedes our train/validation/test split. This is not an independently leakage-audited raw-data evaluation.
- The distance matrix has labeled columns but no explicit row IDs. Row order is assumed to follow published column order; order, symmetry, and zero diagonal are checked, but these checks cannot independently prove the geographic mapping.
- The evaluation includes only 28 daily test windows and three training seeds.

### UrbanEV RSPS

After the full geographic experiment:

```bash
python -u run_urbanev_rsps.py
```

If needed, select artifacts explicitly:

```bash
python -u run_urbanev_rsps.py --neural-run /path/to/urbanev_full_RUN --snapshot /path/to/urbanev_SNAPSHOT
```

This uses daily energy totals, training-only daily normalization, weekly differencing, and pooled ridge regression with alpha 1.0 and an unpenalized intercept:

```text
self predictor:    r[i,k-1]
spatial predictors: r[i,k-1], sum_j W[i,j] r[j,k-1]
RSPS = (MSE_self - MSE_spatial) / (MSE_self + 1e-12)
```

Training target days are 8–125 and validation target days are 126–152, using zero-based day indices. No test target values enter RSPS fitting or scoring. The saved neural test errors are used only in the retrospective comparison. No neural retraining or GPU is required.

## Figures and paper handoff

Original figures:

```bash
chmod +x scripts/make_paper_figures.sh
./scripts/make_paper_figures.sh
```

New revision figures and result package:

```bash
python -u export_paper_results.py
```

The revision export creates PDF/PNG/SVG figures, comparison tables, small evidence files, and a ZIP under `paper_exports/`. It reads full-precision CSVs rather than rounded console logs. It does not rerun training or edit the manuscript. Visually inspect exported figures before inserting them into LaTeX.

Error bars represent sample standard deviations across seeds, not confidence intervals. Alignment panels use separate y-axis scales; disclose this in captions. Figure captions belong in LaTeX. Keep the original Washington figures alongside the new revision figures.

The package excludes raw/input data, large prediction arrays, and model checkpoints. It is a handoff package, **not a complete backup**.

## Version control and local backups

Keep scripts, manuscript sources, final CSVs, JSON protocols/manifests, text summaries, and small frozen graph/permutation artifacts trackable. Do not ignore the entire `revision_runs/` directory.

Large raw downloads, prediction tensors, model checkpoints, virtual environments, caches, and smoke-test outputs may be excluded from Git, but should remain in separate backups when needed for reproducibility. `.gitignore` does not remove already tracked files or old history.

The raw ACN `timeseries_sessions/` filenames in the existing repository history include characters incompatible with Windows checkout. Ignoring the folder does not repair that history. Preserve the Ubuntu copy or a suitable archive when transferring those raw files.
