# Controlled Washington Benchmark

This directory contains the controlled experiments for:

> **The Spatial Illusion: Beyond Correlation in GNN-Based EV Forecasting**

## Purpose

The benchmark separates true lagged graph coupling from shared contemporaneous signal that raises raw correlation.

## Core files

```text
config.py
data.py
device_utils.py
generator.py
graphs.py
metrics.py
models.py
novelty_diagnostics.py
train_eval.py
run_full_graph_control.py
aggregate_paper_results.py
aggregate_results.py
make_paper_figures.py
```

Main frozen paper experiment:

```text
run_full_graph_control.py
```

## Data generator

```text
581 nodes
365 days
8,760 hourly observations per node
Data seeds: 42, 52, 62
```

True coupling:

```text
0.00, 0.25, 0.50, 0.75, 1.00
```

False signal:

```text
sigma_g = 0.05, 0.10, 0.15, 0.20
with coupling = 0
```

## Graph construction

```text
symmetric geographic 6-nearest-neighbor graph
581 nodes
2,179 undirected edges
4,358 directed message-passing edges
median edge distance = 12.54 km
```

Frozen graph seed:

```text
777
```

## RSPS

Implemented in `novelty_diagnostics.py`.

```text
alpha = 1.0
fit_intercept = True
```

RSPS compares self-only residual prediction with self-plus-lagged-neighbor residual prediction.

## Models

```text
GRUForecast
GRUGCNForecast
```

Hidden dimension:

```text
64
```

Forecast horizon:

```text
24
```

Residual fusion:

```text
h_fused = h_local + gate * h_spatial
```

Initial graph gate:

```text
sigmoid(-5) ~= 0.0067
```

## Matched training

Branches:

```text
GRU baseline
GRU-FT
GEO-GNN
PERM-GNN
```

Frozen settings:

```text
Baseline training:      up to 100 epochs
Graph-only stage:       up to 30 epochs
Joint fine-tune stage:  up to 70 epochs
Baseline / graph LR:    1e-3
Fine-tune LR:           5e-4
Weight decay:           1e-4
Gradient clipping:      1.0
Batch size:             4
Training seeds:         42, 52, 62
```

## Run

```bash
python -m experiments_v2.run_full_graph_control --help
```

For exact paper reproduction:

```bash
./scripts/run_all_paper_experiments.sh
```

## Output

Stored under:

```text
results_v2/
```

Important files:

```text
paper_all_runs_gs777.csv
paper_scenario_summary_gs777.csv
paper_main_table_gs777.csv
paper_association_by_data_seed_gs777.csv
paper_association_pooled_gs777.csv
```

## Expected results

At zero coupling:

```text
GEO approximately equals PERM
```

At maximum coupling:

```text
GEO gain over GRU-FT ≈ +11.25%
GEO vs PERM ≈ +4.92%
```

Under strongest false signal:

```text
raw neighbor correlation ≈ 0.860
GEO vs PERM ≈ -0.14%
```

Pooled association:

```text
RSPS alignment vs GEO-vs-PERM: r = 0.968
Raw-correlation alignment vs GEO-vs-PERM: r = 0.791
```