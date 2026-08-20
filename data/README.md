# Data Documentation

This directory contains data artifacts used by:

> **The Spatial Illusion: Beyond Correlation in GNN-Based EV Forecasting**

The repository contains two distinct data components:

1. a controlled Washington benchmark; and
2. an observed Caltech ACN validation dataset.

They should not be described as the same type of data.

## Controlled Washington benchmark

The controlled benchmark uses:

```text
581 geographic nodes
365 days
8,760 hourly observations per node
```

Observed components include geographic units, coordinates/centroids, EV-registration counts, and node-level relative demand scale.

The hourly EV charging-demand trajectories are **synthetically generated**.

Generator:

```text
experiments_v2/generator.py
```

Graph construction:

```text
experiments_v2/graphs.py
```

Controlled conditions:

```text
True coupling:
0.00, 0.25, 0.50, 0.75, 1.00

False-signal shared shock:
0.05, 0.10, 0.15, 0.20
with coupling = 0
```

Data seeds:

```text
42, 52, 62
```

## Caltech ACN observed data

The real-data validation uses observed EV charging sessions from the Caltech Adaptive Charging Network.

Prepared real-data artifacts are under:

```text
data/real_acn/
```

See `data/real_acn/README.md` for complete details.

## Chronological splitting

Controlled benchmark:

```text
Train:       250 days
Validation:   50 days
Test:         65 days
```

Caltech:

```text
Train:
2018-05-01 through 2019-08-12

Validation:
2019-08-13 through 2019-11-20

Test:
2019-11-21 through 2020-02-29
```

No random temporal split is used for the paper experiments.

Training-period statistics are used for normalization.

## Files that should normally not be committed

Avoid committing:

```text
raw ACN time-series JSON downloads
private credentials
API tokens
.venv/
__pycache__/
temporary logs
machine-specific caches
large regenerated intermediate arrays
```

Third-party raw data should only be redistributed when its license or terms permit it.