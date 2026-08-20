#!/usr/bin/env bash
set -euo pipefail

# Robustness run: repeat the frozen full graph-control experiment on
# independent synthetic data seeds 52 and 62. Training seeds remain
# 42, 52, 62 inside run_full_graph_control.py.

mkdir -p logs

python check_gpu.py

COMMON_ARGS=(
  --graph-seed 777
  --device cuda
  --batch-size 4
  --gru-epochs 100
  --graph-epochs 30
  --finetune-epochs 70
  --patience 15
  --graph-patience 10
  --graph-lr 1e-3
  --finetune-lr 5e-4
  --perm-tries 64
)

for DATA_SEED in 52 62; do
  echo "============================================================"
  echo "Starting frozen full graph-control experiment: data seed ${DATA_SEED}"
  echo "Training seeds inside experiment: 42, 52, 62"
  echo "============================================================"

  python -m experiments_v2.run_full_graph_control \
    --data-seed "${DATA_SEED}" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "logs/full_graph_control_ds${DATA_SEED}.log"

  echo "Completed data seed ${DATA_SEED}."
  echo
 done

echo "============================================================"
echo "ALL ROBUSTNESS RUNS COMPLETED"
echo "Expected result files:"
echo "  results_v2/full_graph_control_diagnostic_ds52_gs777.csv"
echo "  results_v2/full_graph_control_summary_ds52_gs777.csv"
echo "  results_v2/full_graph_control_association_ds52_gs777.csv"
echo "  results_v2/full_graph_control_diagnostic_ds62_gs777.csv"
echo "  results_v2/full_graph_control_summary_ds62_gs777.csv"
echo "  results_v2/full_graph_control_association_ds62_gs777.csv"
echo "============================================================"
