python -m experiments_v2.graphs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m experiments_v2.run_experiments --suite core --data-seeds 42 --train-seeds 42
