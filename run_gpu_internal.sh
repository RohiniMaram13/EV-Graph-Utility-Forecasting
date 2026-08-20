#!/usr/bin/env bash
# Internal-review package: full diagnostic (single seed), graph construction,
# and the classical baseline/ablation suite. Results are crash-safe CSVs.
set -euo pipefail
export PYTHONUNBUFFERED=1
python check_gpu.py
python -m experiments_v2.graphs

python -m experiments_v2.run_novelty \
  --suite diagnostic --mode full \
  --data-seed 42 --train-seed 42 \
  --device cuda --batch-size 4 --epochs 100 --patience 15

python -m experiments_v2.run_novelty \
  --suite graphs --mode full \
  --data-seed 42 --train-seed 42 \
  --graph-coupling 0.75 \
  --device cuda --batch-size 4 --epochs 100 --patience 15

python -m experiments_v2.run_experiments \
  --suite core --data-seeds 42 --train-seeds 42 \
  --reference-coupling 0.50 \
  --device cuda --batch-size 4 --epochs 100 --patience 15

python -m experiments_v2.aggregate_results
