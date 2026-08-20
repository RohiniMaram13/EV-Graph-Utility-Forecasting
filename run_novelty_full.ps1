python -m experiments_v2.graphs
python -m experiments_v2.run_novelty --suite diagnostic --mode full --data-seed 42 --train-seed 42
python -m experiments_v2.run_novelty --suite graphs --data-seed 42 --train-seed 42 --graph-coupling 0.75 --graph-shared-shock 0.0
