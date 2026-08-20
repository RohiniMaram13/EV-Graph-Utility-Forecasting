NEXT ROBUSTNESS EXPERIMENT
==========================

Purpose
-------
The architecture and training procedure are now frozen. This run repeats the
exact full graph-control experiment on two new independently generated data
seeds (52 and 62). Each data seed is evaluated with training seeds 42, 52,
and 62.

Do not change model settings, learning rates, graph seed, epochs, or scenario
grid for these runs.

Run
---
From the project root, with the CUDA environment activated:

  chmod +x run_gpu_data_seeds_52_62.sh
  ./run_gpu_data_seeds_52_62.sh

To keep one outer log as well:

  ./run_gpu_data_seeds_52_62.sh 2>&1 | tee overnight_data_seed_run.log

The script already writes one detailed log per data seed under logs/.

What it runs
------------
Data seeds: 52, 62
Training seeds within each data seed: 42, 52, 62
Conditions per data seed: 9
  - true coupling: 0.00, 0.25, 0.50, 0.75, 1.00
  - false signal shocks: 0.05, 0.10, 0.15, 0.20
Models per condition/seed:
  - shared GRU baseline
  - matched GRU fine-tuning control
  - geographic residual GNN
  - permuted-graph residual GNN

Expected output files
---------------------
results_v2/full_graph_control_diagnostic_ds52_gs777.csv
results_v2/full_graph_control_summary_ds52_gs777.csv
results_v2/full_graph_control_association_ds52_gs777.csv
results_v2/full_graph_control_diagnostic_ds62_gs777.csv
results_v2/full_graph_control_summary_ds62_gs777.csv
results_v2/full_graph_control_association_ds62_gs777.csv

After completion, upload the six CSV files above for cross-data-seed analysis.
