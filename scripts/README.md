# Repository Scripts

This directory contains repository-level convenience scripts.

## `run_all_paper_experiments.sh`

Primary end-to-end experiment runner.

```bash
chmod +x scripts/run_all_paper_experiments.sh
./scripts/run_all_paper_experiments.sh
```

Before launching a long run:

```bash
sed -n '1,260p' scripts/run_all_paper_experiments.sh
```

For exact reproduction, do not modify frozen seeds or experiment parameters.

## `make_paper_figures.sh`

```bash
chmod +x scripts/make_paper_figures.sh
./scripts/make_paper_figures.sh
```

Underlying implementation:

```text
experiments_v2/make_paper_figures.py
```

Figures should be generated from result CSV files rather than manually edited.

## Environment setup

From repository root:

```bash
chmod +x setup_linux_gpu.sh
./setup_linux_gpu.sh
python check_gpu.py
```

## Long-running jobs

A persistent shell such as `tmux` or `screen` is recommended.

Example:

```bash
nohup ./scripts/run_all_paper_experiments.sh     > paper_experiments.log 2>&1 &
```

Monitor with:

```bash
tail -f paper_experiments.log
```
