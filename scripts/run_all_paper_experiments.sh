#!/usr/bin/env bash
set -euo pipefail

GRAPH_SEED="${GRAPH_SEED:-777}"
DATA_SEEDS=(42 52 62)
mkdir -p logs results_v2

for ds in "${DATA_SEEDS[@]}"; do
  echo "============================================================"
  echo "Running paper experiment: data seed ${ds}, graph seed ${GRAPH_SEED}"
  echo "============================================================"
  python -m experiments_v2.run_full_graph_control \
    --data-seed "${ds}" \
    --graph-seed "${GRAPH_SEED}" \
    2>&1 | tee "logs/full_graph_control_ds${ds}_gs${GRAPH_SEED}.log"
done

python -m experiments_v2.aggregate_paper_results --graph-seed "${GRAPH_SEED}"
python -m experiments_v2.make_paper_figures --graph-seed "${GRAPH_SEED}"

echo
echo "All paper experiments completed."
echo "Raw per-seed CSVs: results_v2/full_graph_control_*"
echo "Aggregated tables: results_v2/paper_*"
echo "Figures: results_v2/figures/"
