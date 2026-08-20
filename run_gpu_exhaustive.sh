#!/usr/bin/env bash
# EXHAUSTIVE robustness run. This is intentionally large. Run only after the
# single-seed/internal results make scientific sense.
set -euo pipefail
export PYTHONUNBUFFERED=1
python check_gpu.py
python -m experiments_v2.graphs

for DS in 42 52 62; do
  for TS in 42 52 62; do
    python -m experiments_v2.run_novelty \
      --suite diagnostic --mode full \
      --data-seed "$DS" --train-seed "$TS" \
      --device cuda --batch-size 4 --epochs 100 --patience 15
  done
done

for TS in 42 52 62; do
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
  --suite coupling --data-seeds 42 52 62 --train-seeds 42 52 62 \
  --device cuda --batch-size 4 --epochs 100 --patience 15

python -m experiments_v2.aggregate_results
