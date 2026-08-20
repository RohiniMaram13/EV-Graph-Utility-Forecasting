#!/usr/bin/env bash
set -euo pipefail
GRAPH_SEED="${GRAPH_SEED:-777}"
python -m experiments_v2.aggregate_paper_results --graph-seed "${GRAPH_SEED}"
python -m experiments_v2.make_paper_figures --graph-seed "${GRAPH_SEED}"
