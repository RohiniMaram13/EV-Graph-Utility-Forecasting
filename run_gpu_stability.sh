#!/usr/bin/env bash
set -euo pipefail
python check_gpu.py
python -m experiments_v2.run_stability --data-seed 42 --device auto --batch-size 4 --epochs 100 --patience 15
