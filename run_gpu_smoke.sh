#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
python check_gpu.py
python -m experiments_v2.graphs
python -m experiments_v2.run_novelty \
  --suite diagnostic --mode smoke \
  --data-seed 42 --train-seed 42 \
  --device cuda --batch-size 4 --epochs 3 --patience 2
