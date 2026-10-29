# Controlled Washington Benchmark

This directory contains the original controlled experiments for **The Spatial Illusion: Beyond Correlation in GNN-Based EV Forecasting** and shared model/training utilities used by the real-data evaluations.

The original Washington experiment settings and measured results are retained. Camera-ready additions are implemented in separate root-level scripts and stored under `revision_runs/`; they do not replace this benchmark.

## Purpose and scope

The benchmark separates **true lagged graph coupling** from **shared contemporaneous shocks** that can raise raw spatial correlation without adding the same lagged predictive mechanism.

Washington geographic units, coordinates, and EV-registration-derived scale information provide the setting. The **hourly charging trajectories are synthetic**. This allows the experiment to control whether graph-linked predictive dependence exists.

The benchmark tests behavior under a particular generator and forecasting architecture. It does not establish that the same diagnostic relationship must hold across all real datasets or models.

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

The main frozen paper experiment is `run_full_graph_control.py`. Use `aggregate_paper_results.py` for the paper-specific aggregation workflow; inspect script settings rather than assuming every historical runner uses the same protocol.

## Generator and chronological split

| Setting | Frozen value |
|---|---|
| Nodes | 581 |
| Duration | 365 days |
| Hourly observations | 8,760 per node |
| Data seeds | 42, 52, 62 |
| Training seeds | 42, 52, 62 |
| Lookback | 168 hours |
| Forecast horizon | 24 hours |
| Window stride | 24 hours |
| Train / validation / test | 250 / 50 / 65 days |
| Train / validation / test windows | 243 / 50 / 65 |

The 168-hour history covers one week. Forecast targets remain inside their designated chronological split; preceding observed history can cross a split boundary. Normalization is fitted on training data.

### True-coupling conditions

```text
c = 0.00, 0.25, 0.50, 0.75, 1.00
```

The latent-state mechanism is:

```text
s_k = phi * [(1-c) * s_(k-1) + c * W * s_(k-1)] + eta_k
phi = 0.70
```

Here `s_k` is the vector of latent node states on day `k`, `eta_k` is the innovation vector, and `W` is the row-normalized geographic adjacency. For positive-degree nodes, `W[i,j] = A[i,j] / sum_j A[i,j]`; a zero-degree row is defined as zero.

At zero coupling, latent persistence is local. Increasing `c` increases the contribution of neighbors' previous-day states. The persistence coefficient 0.70 is a fixed benchmark setting, not an empirically established universal EV-charging parameter. Generator-sensitivity experiments were not added in the camera-ready extension.

### False-signal conditions

```text
sigma_g = 0.05, 0.10, 0.15, 0.20
c = 0
```

Shared contemporaneous shocks increase cross-node similarity without deliberately introducing the same lagged neighbor-predictive mechanism. These four conditions and the five coupling conditions form the nine reported benchmark conditions.

## Geographic graph and permutation control

```text
Graph:                          symmetric geographic 6-nearest-neighbor graph
Nodes:                          581
Undirected edges:               2,179
Directed message-passing edges: 4,358
Median edge distance:           12.54 km
Frozen graph/permutation seed:  777
```

The graph defines which histories can exchange messages. The permutation control keeps the same graph topology, edge count, and degree distribution while changing its alignment with the node histories:

```text
A_perm = A[p, p]  # indexed over both axes by the same permutation
```

In NumPy, use `A[np.ix_(p, p)]` for this operation. A named data node can receive a different degree after relabeling; the degree distribution is preserved, not necessarily each data node's original degree.

The geographic graph is an experimental hypothesis, not an assumed optimal adjacency. Its usefulness must be assessed against the matched controls.

## Residual Spatial Predictability Score

Implemented in `novelty_diagnostics.py`.

```text
Ridge alpha: 1.0
Fit intercept: True
```

The diagnostic compares self-only residual prediction with self-plus-lagged-neighbor residual prediction:

```text
r[i,k] = z[i,k] - z[i,k-7]
neighbor[i,k-1] = sum_j W[i,j] * r[j,k-1]

self predictors:    r[i,k-1]
spatial predictors: r[i,k-1], neighbor[i,k-1]

RSPS = (MSE_self - MSE_spatial) / MSE_self
```

`z` denotes standardized daily demand. RSPS is dimensionless and measures relative residual-MSE reduction. Its alignment difference compares the aligned graph's RSPS with that of the permutation control. This is distinct from the neural model's percentage MAE improvement.

RSPS is linear, single-lag, and noncausal. A low score does not rule out nonlinear, multi-lag, or multi-hop predictive information. The strong controlled-benchmark association reported below is scoped to these conditions, not a universal reliability guarantee.

## Forecasting models

Implemented in `models.py`:

```text
GRUForecast
GRUGCNForecast
```

Both use hidden dimension 64 and a 24-hour forecast horizon. GRU denotes a gated recurrent unit; GCN denotes graph convolutional network layers.

The graph model retains the local temporal representation and adds a gated spatial correction:

```text
h_fused = h_local + gate * h_spatial
gate = sigmoid(graph_gate_logit)
initial graph_gate_logit = -5.0
initial gate ≈ 0.0067
```

Starting with a small graph gate supports warm-starting from the temporal model. It does not itself establish that the chosen graph carries useful predictive information.

## Matched training

The branches are:

1. Shared GRU baseline.
2. GRU-FT: matched temporal fine-tuning control.
3. GEO-GNN: aligned geographic graph.
4. PERM-GNN: topology-preserving permuted graph.

Graph branches start from the same GRU checkpoint and share matched graph-parameter initialization within each training seed. The comparison separates improvement over the temporal control from improvement attributable to graph alignment.

```text
Baseline training:      up to 100 epochs
Graph-only stage:       up to 30 epochs
Joint fine-tune stage:  up to 70 epochs
Baseline / graph LR:    1e-3
Fine-tune LR:           5e-4
Weight decay:          1e-4
Gradient clipping:     1.0
Batch size:            4
Training seeds:        42, 52, 62
```

Checkpoint selection uses validation loss. Warm-start stages retain the incoming model as an eligible checkpoint, so they need not accept a worse validation solution merely because additional training occurred.

## Run the original benchmark

Run from the repository root with the configured environment active:

```bash
python -m experiments_v2.run_full_graph_control --help
```

For the original paper reproduction workflow:

```bash
chmod +x scripts/run_all_paper_experiments.sh
./scripts/run_all_paper_experiments.sh
```

Inspect the runner before launching a long job. Preserve existing results before a deliberate rerun: original scripts can write to the original output locations. Do not modify frozen parameters based on test results.

The original runner does not automatically execute the newer standalone revision scripts. Completed original experiments need not be repeated to prepare the camera-ready manuscript.

## Original outputs

Outputs are stored under `results_v2/` at the repository root:

```text
paper_all_runs_gs777.csv
paper_scenario_summary_gs777.csv
paper_main_table_gs777.csv
paper_association_by_data_seed_gs777.csv
paper_association_pooled_gs777.csv
```

The original controlled results cover nine conditions × three data seeds × three training seeds, giving 81 experiment records. Retain seed-level outputs rather than only aggregate tables.

## Reported controlled results

Values below summarize the original saved results and are rounded. Use full-precision CSVs for final tables.

| Condition / comparison | Reported result |
|---|---:|
| Zero coupling: GEO gain over GRU-FT | approximately +3.78% |
| Zero coupling: GEO gain over PERM | approximately −0.01% |
| Maximum coupling: GEO gain over GRU-FT | approximately +11.25% |
| Maximum coupling: GEO gain over PERM | approximately +4.92% |
| Strongest shared shock: raw neighbor correlation | approximately 0.860 |
| Strongest shared shock: GEO gain over PERM | approximately −0.14% |

At zero coupling, GEO and PERM perform approximately alike even though adding the graph branch can improve over GRU-FT. Under stronger true coupling, aligned-graph benefits generally increase; this does not imply strict monotonicity in every individual run. Shared shocks can yield high raw correlation without comparable alignment value.

Across the nine pooled conditions:

```text
RSPS alignment vs forecasting alignment advantage:            r = 0.968
Raw-correlation alignment vs forecasting alignment advantage: r = 0.791
```

These are descriptive associations under the controlled generator. They are not causal estimates or guarantees of transfer to new datasets.

## Relationship to the camera-ready additions

The revision adds the following experiments outside this directory:

| Addition | Root-level script |
|---|---|
| Caltech weekly seasonal naive | `run_seasonal_naive.py` |
| Caltech shared-channel DLinear | `run_dlinear.py` |
| Original Caltech model extended from three to five seeds | `run_additional_caltech_seeds.py` |
| Adapted STGCN with graph-permutation controls | `run_stgcn.py` |
| UrbanEV geographic graph validation | `download_urbanev.py`, `run_urbanev.py` |
| UrbanEV RSPS extension | `run_urbanev_rsps.py` |

The real-data findings qualify the broader interpretation without replacing the controlled results:

- Weekly seasonal naive has lower Caltech MAE than the tested neural baselines, although its RMSE is worse.
- Caltech alignment benefits remain small or seed-dependent across additional seeds and adapted STGCN.
- UrbanEV shows a small positive mean geographic alignment gain of approximately 0.05%.
- UrbanEV RSPS has a positive aligned-versus-shuffled difference, but its exploratory correlation with mean neural test MAE across graphs is approximately −0.043, providing no meaningful linear ranking evidence in that setting.

The UrbanEV correlation compares different graphs within one dataset, whereas the Washington correlation compares benchmark conditions. Do not present them as identical analyses or claim universal RSPS performance from the controlled correlation.

The new experiments do not include a nonlinear/multi-lag RSPS variant or a generator-sensitivity sweep. See the [root README](../README.md) for full revision results and limitations.

## Figures and manuscript preparation

Original figures:

```bash
./scripts/make_paper_figures.sh
```

Revision figures and evidence export, after all required revision outputs exist:

```bash
python -u export_paper_results.py
```

Generate plots from result files, visually inspect them, and preserve the distinction between original and added experiments. Error bars reported as sample standard deviations are not confidence intervals. Do not count graph permutations as independent training seeds or describe descriptive results as statistically significant without an appropriate analysis.

The revised paper should present graph alignment as a testable hypothesis and RSPS as a complementary diagnostic whose utility must be assessed in the setting of interest.
