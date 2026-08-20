#!/usr/bin/env bash
set -euo pipefail

python check_gpu.py
python -m experiments_v2.run_matched_control \
  --data-seed 42 \
  --device auto \
  --batch-size 4 \
  --gru-epochs 100 \
  --graph-epochs 30 \
  --finetune-epochs 70 \
  --patience 15 \
  --graph-patience 10 \
  --graph-lr 1e-3 \
  --finetune-lr 5e-4
