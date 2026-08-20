#!/usr/bin/env bash
# Stronger paper run: 3 training seeds for the main novelty diagnostic and
# residual-graph experiment, plus classical core and coupling sweep.
set -euo pipefail
export PYTHONUNBUFFERED=1
python check_gpu.py
python -m experiments_v2.graphs

for TS in 42 52 62; do
  python -m experiments_v2.run_novelty \
    --suite diagnostic --mode full \
    --data-seed 42 --train-seed "$TS" \
    --device cuda --batch-size 4 --epochs 100 --patience 15

  python -m experiments_v2.run_novelty \
    --suite graphs --mode full \
    --data-seed 42 --train-seed "$TS" \
    --graph-coupling 0.75 \
    --device cuda --batch-size 4 --epochs 100 --patience 15
done

python -m experiments_v2.run_experiments \
  --suite core --data-seeds 42 --train-seeds 42 \
  --reference-coupling 0.50 \
  --device cuda --batch-size 4 --epochs 100 --patience 15

python -m experiments_v2.run_experiments \
  --suite coupling --data-seeds 42 --train-seeds 42 52 62 \
  --device cuda --batch-size 4 --epochs 100 --patience 15

python -m experiments_v2.aggregate_results
