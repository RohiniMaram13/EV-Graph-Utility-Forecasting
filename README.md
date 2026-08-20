# The Spatial Illusion: Beyond Correlation in GNN-Based EV Forecasting


> **The Spatial Illusion: Beyond Correlation in GNN-Based EV Forecasting**

## Overview

This repository studies a specific attribution question in graph-based EV charging forecasting:

> **Does a proposed graph contain predictive information beyond what is already available from each node's temporal history?**

The work combines two complementary evaluations:

1. **Controlled Washington benchmark** — a 581-node benchmark in which lagged graph coupling is known and manipulated by construction.
2. **Observed Caltech ACN validation** — a real charging-data experiment using measured sessions from the Caltech Adaptive Charging Network and a physical electrical-pod graph.

The paper does not assume that high spatial correlation implies that a graph is useful. Instead, it evaluates whether the **specific node-to-graph alignment** matters.

Two ideas are central:

- **Residual Spatial Predictability Score (RSPS)**: tests whether lagged neighbor residuals improve next-day residual prediction beyond self-history.
- **Topology-preserving node permutations**: preserve graph topology and degree distribution while breaking node alignment, allowing graph-specific value to be separated from generic graph-model effects.

## Main findings

### Controlled benchmark

The controlled benchmark varies:

```text
True coupling:
c = 0.00, 0.25, 0.50, 0.75, 1.00

False-signal shared shock:
sigma_g = 0.05, 0.10, 0.15, 0.20
with c = 0
```

Across three data seeds and three training seeds:

- At zero coupling, GEO-GNN and PERM-GNN are approximately indistinguishable.
- As true coupling increases, GEO-GNN increasingly separates from PERM-GNN.
- Under false-signal conditions, raw neighbor correlation can become very high while graph-alignment advantage stays near zero.
- Across the nine pooled conditions:

```text
RSPS-alignment vs graph-specific forecasting advantage:
r = 0.968

Raw-correlation alignment vs graph-specific forecasting advantage:
r = 0.791
```

### Caltech ACN validation

The real-data experiment uses:

```text
Downloaded Caltech sessions:       31,424
Selected EVSEs:                    37
Selected sessions:                 24,985
Usable measured-current sessions:  24,251
Hourly observations:               16,080
Pod-connected EVSEs:               16
Physical pod edges:                56
Training seeds:                    42, 52, 62
Neural graph permutations / seed:  20
```

The real graph shows weak graph-specific predictive value:

```text
GRU-FT MAE:
0.3440348 +/- 0.0043554

Real Pod-GNN MAE:
0.3430297 +/- 0.0016157

Real gain over GRU-FT:
+0.2845% +/- 0.9551%

Real vs mean topology-matched permutation:
+1.2019% +/- 2.5185%

Mean permutations beaten:
65.0%

Final graph gate:
0.01797 +/- 0.01575
```

The paper therefore does **not** claim a large real-world GNN gain. The conclusion is that a physically meaningful graph can still have only weak predictive value for a particular forecasting task and horizon.

---

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── setup_linux_gpu.sh
├── check_gpu.py
│
├── data/
│   ├── README.md
│   ├── processed/
│   ├── processed_v2/
│   └── real_acn/
│       ├── README.md
│       ├── caltech_hourly_energy.csv
│       ├── caltech_hourly_energy.npz
│       ├── caltech_pod_adjacency.npy
│       ├── caltech_pod_edges.csv
│       ├── caltech_pod_graph_metadata.json
│       ├── caltech_selected_sessions.csv
│       ├── caltech_selected_stations.csv
│       ├── caltech_station_mapping.csv
│       └── ...
│
├── experiments_v2/
│   ├── README.md
│   ├── config.py
│   ├── data.py
│   ├── generator.py
│   ├── graphs.py
│   ├── models.py
│   ├── train_eval.py
│   ├── metrics.py
│   ├── novelty_diagnostics.py
│   ├── run_full_graph_control.py
│   ├── aggregate_paper_results.py
│   └── make_paper_figures.py
│
├── real_validation/
│   ├── README.md
│   ├── test_acn.py
│   ├── inspect_acn.py
│   ├── inspect_timeline.py
│   ├── select_real_dataset.py
│   ├── download_selected_timeseries.py
│   ├── build_hourly_matrix.py
│   ├── build_station_mapping.py
│   ├── build_real_graph.py
│   ├── prepare_real_validation.py
│   └── run_real_graph_validation.py
│
├── results_v2/
│   └── controlled-paper outputs
│
├── results_real/
│   ├── README.md
│   └── real-data outputs
│
└── scripts/
    ├── README.md
    ├── run_all_paper_experiments.sh
    └── make_paper_figures.sh
```

---

## Environment setup

The final experiments were run on Linux with a CUDA-enabled NVIDIA GPU.

The tested environment included:

```text
PyTorch: 2.11.0
CUDA: 12.8
GPU: NVIDIA GeForce RTX 5090
```

Recommended setup:

```bash
chmod +x setup_linux_gpu.sh
./setup_linux_gpu.sh
```

If a local virtual environment is created:

```bash
source .venv/bin/activate
```

Verify the GPU:

```bash
python check_gpu.py
```

Small numerical differences may occur across hardware, CUDA versions, and PyTorch versions.

---

## End-to-end paper reproduction

The primary top-level reproduction command is:

```bash
chmod +x scripts/run_all_paper_experiments.sh
./scripts/run_all_paper_experiments.sh
```

Use the checked-in script rather than manually changing paper parameters if the goal is to reproduce the frozen manuscript results.

---

# Controlled Washington benchmark

## What is observed and what is synthetic

The controlled benchmark uses real Washington geographic units, coordinates, and EV-registration-derived scale information.

The **hourly charging trajectories are synthetic**.

This distinction is intentional because the benchmark requires known ground truth about whether lagged graph-linked predictive dependence exists.

Generator:

```text
experiments_v2/generator.py
```

Graph construction:

```text
experiments_v2/graphs.py
```

### Frozen setup

```text
Nodes:                    581
Duration:                 365 days
Hourly observations:      8,760 per node
Lookback:                 168 hours
Forecast horizon:         24 hours
Window stride:            24 hours

Training:                 250 days
Validation:                50 days
Test:                      65 days

Training windows:         243
Validation windows:        50
Test windows:              65

Data seeds:               42, 52, 62
Training seeds:           42, 52, 62
```

### Controlled latent-state mechanism

Let `W` be the row-normalized geographic adjacency:

```text
s_k = phi [(1-c)s_(k-1) + c W s_(k-1)] + eta_k
```

with:

```text
phi = 0.70
```

At `c = 0`, persistence is local. As `c` increases, lagged geographic-neighbor state contributes more strongly.

### False-signal construction

The false-signal experiment adds:

```text
g_k ~ Normal(0, sigma_g^2)
```

to all nodes while fixing coupling to zero.

This increases contemporaneous similarity without deliberately creating the same lagged neighbor-predictive mechanism.

### Geographic graph

```text
Symmetric geographic k-NN
k = 6
Nodes = 581
Undirected edges = 2,179
Directed message-passing edges = 4,358
Median edge distance = 12.54 km
```

The matched permuted graph preserves topology and node degree while breaking geographic alignment.

Frozen graph seed:

```text
777
```

Approximate undirected edge overlap with the aligned graph:

```text
0.78%
```

### RSPS

Implemented in:

```text
experiments_v2/novelty_diagnostics.py
```

Weekly residual:

```text
r[i,k] = z[i,k] - z[i,k-7]
```

Lagged neighbor residual:

```text
n[i,k-1] = sum_j W[i,j] r[j,k-1]
```

Ridge configuration:

```text
Ridge(alpha=1.0, fit_intercept=True)
```

RSPS:

```text
RSPS = (MSE_self - MSE_spatial) / MSE_self
```

RSPS is a predictive diagnostic, not a causal estimator.

### Forecasting models

Implemented in:

```text
experiments_v2/models.py
```

The temporal model is `GRUForecast`.

The graph model is `GRUGCNForecast`, which preserves the local GRU representation and adds:

```text
h_fused = h_local + gate * h_spatial
```

The graph gate begins near zero:

```text
graph_gate_logit = -5.0
sigmoid(-5) ~= 0.0067
```

### Matched training

Main runner:

```text
experiments_v2/run_full_graph_control.py
```

Branches:

```text
A. Shared GRU baseline
B. GRU-FT
C. GEO-GNN
D. PERM-GNN
```

Frozen training settings:

```text
Baseline epochs:           up to 100
Graph-only epochs:         up to 30
Joint fine-tune epochs:    up to 70
Baseline LR:               1e-3
Graph-stage LR:            1e-3
Fine-tune LR:              5e-4
Weight decay:              1e-4
Gradient clipping:         1.0
Batch size:                4
```

Inspect individual CLI options:

```bash
python -m experiments_v2.run_full_graph_control --help
```

For exact paper reproduction, prefer:

```bash
./scripts/run_all_paper_experiments.sh
```

### Controlled output files

Final controlled outputs are stored under:

```text
results_v2/
```

Important files include:

```text
results_v2/paper_all_runs_gs777.csv
results_v2/paper_scenario_summary_gs777.csv
results_v2/paper_main_table_gs777.csv
results_v2/paper_association_by_data_seed_gs777.csv
results_v2/paper_association_pooled_gs777.csv
```

### Expected qualitative controlled result

```text
At c = 0:
GEO approximately equals PERM

As true coupling increases:
GEO increasingly separates from PERM

As shared-shock strength increases:
raw correlation rises strongly

Under false signal:
GEO does not develop a comparable alignment advantage
```

Key pooled values:

```text
RSPS alignment vs GEO-vs-PERM:
r = 0.968

Raw-correlation alignment vs GEO-vs-PERM:
r = 0.791
```

At maximum coupling:

```text
GEO gain over GRU-FT ≈ +11.25%
GEO vs PERM ≈ +4.92%
```

At zero coupling:

```text
GEO gain over GRU-FT ≈ +3.78%
GEO vs PERM ≈ -0.01%
```

---

# Caltech ACN validation

Detailed documentation is in:

```text
data/real_acn/README.md
real_validation/README.md
results_real/README.md
```

The exact ordered pipeline is:

```bash
python real_validation/test_acn.py
python real_validation/inspect_acn.py
python real_validation/inspect_timeline.py
python real_validation/select_real_dataset.py
python real_validation/download_selected_timeseries.py
python real_validation/build_hourly_matrix.py
python real_validation/build_station_mapping.py
python real_validation/build_real_graph.py
python real_validation/prepare_real_validation.py
python -m real_validation.run_real_graph_validation --n-permutations 20 --verbose
```

---

## Paper figures

Generate figures with:

```bash
chmod +x scripts/make_paper_figures.sh
./scripts/make_paper_figures.sh
```

Figures should be generated from result files rather than manually edited.

---

## Interpretation rules

This repository does **not** claim:

- that high spatial correlation proves graph utility;
- that RSPS is causal;
- that geographic KNN is universally appropriate;
- that every physical electrical relationship should become a GNN edge;
- that the Caltech physical graph provides a large forecasting gain;
- or that a universal RSPS threshold determines whether a graph should be used.

The controlled benchmark establishes mechanism-level behavior.

The Caltech experiment provides observed-data validation and produces a weaker, more cautious result.

---


---

