# Caltech ACN Real-Data Validation Dataset

This directory contains prepared artifacts for the observed-data validation in:

> **The Spatial Illusion: Beyond Correlation in GNN-Based EV Forecasting**

Unlike the controlled Washington benchmark, the Caltech charging sessions are observed data.

## Raw dataset summary

```text
Total sessions:        31,424
Unique station IDs:    55
Dataset period:
2018-04-25 through 2021-09-14
```

The final paper freezes the study interval to:

```text
2018-05-01 through 2020-02-29
```

to avoid the major COVID-era regime shift.

## Station-selection rule

```text
minimum active charging days = 300
```

Expected selection:

```text
Selected stations: 37
Selected sessions: 24,985
Selected energy:   223,827.48 kWh
```

Files:

```text
caltech_selected_stations.csv
caltech_selected_sessions.csv
```

## Measured-current time series

```text
Non-null chargingCurrent: 24,291
Usable for reconstruction: 24,251
Missing chargingCurrent:      694
Positive-kWh sessions with zero usable current: 40
```

## Hourly reconstruction

Measured current traces determine the timing of delivered energy. The pipeline does not assume a fixed voltage.

Expected matrix:

```text
Shape:                    (16080, 37)
Stations:                 37
Hours:                    16,080
Usable sessions:          24,251
Input usable-session kWh: 218,229.383
Energy represented:       218,216.734 kWh
Energy retention:         99.9942%
Hours with charging:      12,486
All-zero hours:           3,594
Fraction nonzero:         77.65%
```

Files:

```text
caltech_hourly_energy.csv
caltech_hourly_energy.npz
```

## Station-to-space mapping

```text
Selected stations:                  37
Stations with exactly one spaceID: 37
Stations with multiple spaceIDs:    0
Mapped into ACN-Sim:                37 / 37
Missing from ACN-Sim:               NONE
```

Mapping file:

```text
caltech_station_mapping.csv
```

Verified mapping:

| stationID | spaceID |
|---|---|
| 2-39-123-23 | CA-313 |
| 2-39-124-22 | CA-312 |
| 2-39-125-21 | CA-311 |
| 2-39-126-20 | CA-310 |
| 2-39-127-19 | CA-309 |
| 2-39-128-18 | CA-308 |
| 2-39-129-17 | CA-307 |
| 2-39-130-31 | CA-306 |
| 2-39-131-30 | CA-305 |
| 2-39-138-29 | CA-304 |
| 2-39-139-28 | CA-303 |
| 2-39-78-360 | CA-322 |
| 2-39-78-361 | CA-493 |
| 2-39-78-362 | CA-496 |
| 2-39-78-363 | CA-320 |
| 2-39-78-364 | CA-495 |
| 2-39-78-365 | CA-321 |
| 2-39-78-366 | CA-323 |
| 2-39-78-367 | CA-494 |
| 2-39-79-376 | CA-324 |
| 2-39-79-377 | CA-325 |
| 2-39-79-378 | CA-326 |
| 2-39-79-379 | CA-327 |
| 2-39-79-380 | CA-489 |
| 2-39-79-381 | CA-490 |
| 2-39-79-382 | CA-491 |
| 2-39-79-383 | CA-492 |
| 2-39-88-24 | CA-314 |
| 2-39-89-25 | CA-315 |
| 2-39-90-26 | CA-316 |
| 2-39-90-440 | CA-500 |
| 2-39-91-437 | CA-317 |
| 2-39-91-441 | CA-499 |
| 2-39-92-436 | CA-318 |
| 2-39-92-442 | CA-498 |
| 2-39-95-27 | CA-319 |
| 2-39-95-444 | CA-497 |

## ACN-Sim electrical structure

```text
54 EVSEs
8 electrical constraint rows
```

Constraints:

```text
CC Pod
AV Pod
Secondary A
Secondary B
Secondary C
Primary A
Primary B
Primary C
```

The broader secondary/primary constraints include signed/fractional coefficients such as `1.0`, `-1.0`, `0.25`, and `-0.5`. These are not automatically treated as ordinary positive pairwise GNN edges.

## Physical pod graph

### CC Pod

```text
CA-322
CA-493
CA-496
CA-320
CA-495
CA-321
CA-323
CA-494
```

### AV Pod

```text
CA-324
CA-325
CA-326
CA-327
CA-489
CA-490
CA-491
CA-492
```

Each pod is an 8-node clique.

```text
Nodes:                37
Pod-connected nodes:  16
Isolated nodes:       21
Undirected edges:     56
Degree 0:             21 nodes
Degree 7:             16 nodes
```

Files:

```text
caltech_pod_adjacency.npy
caltech_pod_edges.csv
caltech_pod_graph_metadata.json
```

## Chronological split

```text
Complete days: 670
```

Training:

```text
2018-05-01 00:00 UTC
to
2019-08-12 23:00 UTC
469 days
462 forecast windows
```

Validation:

```text
2019-08-13 00:00 UTC
to
2019-11-20 23:00 UTC
100 days
100 forecast windows
```

Test:

```text
2019-11-21 00:00 UTC
to
2020-02-29 23:00 UTC
101 days
101 forecast windows
```

Lookback / horizon:

```text
168 hours / 24 hours
```

## Reproduction order

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

## Important ordering rule

Preserve the same node ordering across:

```text
hourly matrix
daily matrix
station mapping
adjacency matrix
edge list
model node dimension
```

A correct graph attached to the wrong station ordering is a misaligned graph.