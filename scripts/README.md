# Repository Scripts

This directory contains convenience scripts for the original paper experiments and figures. The camera-ready revision scripts are separate Python files located at the **repository root**.

Run all commands below from the repository root, with the configured environment active.

## Original experiment runner

`scripts/run_all_paper_experiments.sh` runs the original experiment pipeline. It does **not** automatically run the standalone camera-ready additions listed below.

Inspect it before launching:

```bash
sed -n '1,260p' scripts/run_all_paper_experiments.sh
```

To deliberately reproduce the original experiments:

```bash
chmod +x scripts/run_all_paper_experiments.sh
./scripts/run_all_paper_experiments.sh
```

For exact reproduction, retain the frozen seeds and parameters. Preserve existing outputs before running scripts that write to `results_v2/` or `results_real/`.

**Completed experiments do not need to be rerun to generate figures or refine the manuscript.**

## Original paper figures

```bash
chmod +x scripts/make_paper_figures.sh
./scripts/make_paper_figures.sh
```

Underlying implementation:

```text
experiments_v2/make_paper_figures.py
```

Generate figures from result files rather than manually editing plotted values. This command is separate from the revision figure export below.

## Camera-ready revision experiments

The following files belong at the repository root, not inside `scripts/`. These commands document reproduction; do not rerun completed experiments just to create the paper package.

| Root-level script | Purpose |
|---|---|
| `run_seasonal_naive.py` | Weekly seasonal-naive baseline on Caltech Pod-16 |
| `run_dlinear.py` | Shared-channel DLinear on Caltech, seeds 42/52/62 |
| `run_additional_caltech_seeds.py` | Add seeds 72/82 to the original model and save a separate five-seed summary |
| `run_stgcn.py` | Adapted STGCN with aligned and shuffled Caltech pod graphs |
| `download_urbanev.py` | Download and inspect a versioned UrbanEV snapshot |
| `run_urbanev.py` | UrbanEV geographic graph evaluation and temporal controls |
| `run_urbanev_rsps.py` | RSPS diagnostic using the saved UrbanEV graph permutations and neural results |
| `export_paper_results.py` | Generate revision figures and tables and bundle small evidence files |

### Caltech baselines and additional seeds

```bash
python -u run_seasonal_naive.py
python -u run_dlinear.py
python -u run_additional_caltech_seeds.py
```

The additional-seeds runner expects the original three-seed summary and frozen permutations in `results_real/`. It runs only the two new seeds, preserving the original outputs.

### Adapted STGCN

On a new setup, check the implementation first:

```bash
python -u run_stgcn.py --smoke
```

Then run the full experiment if reproduction is needed:

```bash
python -u run_stgcn.py
```

The full experiment uses three seeds and 20 shuffled graphs per seed. Smoke-test scores are not paper results.

### UrbanEV geographic experiment

For initial data acquisition only:

```bash
python -u download_urbanev.py
```

Preserve the existing snapshot for exact reproduction: the downloader resolves the current upstream commit, which may change. The completed revision experiment used commit `44f2aa0c8d89f192bce00bafb0def74a21b39c68`.

```bash
python -u run_urbanev.py --smoke
python -u run_urbanev.py
```

If multiple snapshots exist, select the intended one with `--snapshot PATH`. The full run records its data split, geographic graph rule, permutations, and settings before training.

After the full geographic run, evaluate RSPS without neural retraining:

```bash
python -u run_urbanev_rsps.py
```

If necessary, identify the saved artifacts explicitly:

```bash
python -u run_urbanev_rsps.py --neural-run /path/to/urbanev_full_RUN --snapshot /path/to/urbanev_SNAPSHOT
```

See the root README for the experimental protocols, target definitions, results, and limitations. In particular, UrbanEV volume is estimated energy and RSPS is a complementary diagnostic, not a universally reliable predictor of neural performance.

## Revision figures and paper handoff

Once the completed revision outputs are available:

```bash
python -u export_paper_results.py
```

This creates a new directory and ZIP under `paper_exports/`, containing:

- Caltech and UrbanEV forecasting figures in PDF, PNG, and SVG formats.
- Graph-alignment results by training seed.
- The UrbanEV RSPS-versus-neural-error scatter plot.
- Full-precision comparison tables, small result files, protocols, and source copies.

The exporter does not train models or edit the manuscript. It expects one completed run for each experiment family. If several runs exist, retain them and resolve the selection rather than deleting evidence.

Visually inspect the figures before using them in LaTeX. Error bars represent sample standard deviations, not confidence intervals. Alignment panels use different vertical scales; describe this in the caption. Keep captions in LaTeX rather than embedding them in the graphics.

The handoff ZIP excludes raw/input data, large prediction arrays, and model checkpoints. It is **not a complete backup**. Preserve the complete `revision_runs/` and `revision_inputs/` folders separately.

## Environment setup

For an existing configured environment:

```bash
source .venv/bin/activate
python check_gpu.py
```

For a fresh installation:

```bash
chmod +x setup_linux_gpu.sh
./setup_linux_gpu.sh
source .venv/bin/activate
python check_gpu.py
```

The reported Ubuntu revision runs used PyTorch `2.11.0+cu128`, CUDA runtime `12.8`, and an NVIDIA GeForce RTX 5090. Do not reinstall the environment merely to export completed results. Preserve software-version records because numerical results can vary across environments.

## Long-running jobs

A persistent shell such as `tmux` or `screen` can keep a run active after disconnecting. Alternatively, use `nohup` with the configured Python interpreter.

Example for an intentional full UrbanEV run:

```bash
nohup .venv/bin/python -u run_urbanev.py > "urbanev_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

Monitor the actual log filename created by that command:

```bash
tail -f urbanev_YYYYMMDD_HHMMSS.log
```

For an intentional original-pipeline reproduction, after activating the environment:

```bash
nohup ./scripts/run_all_paper_experiments.sh > "paper_experiments_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

Launch one experiment job at a time unless GPU memory and concurrent execution have been checked. Preserve final CSVs and settings even when logs are ignored by Git.

## Output locations

| Directory | Contents |
|---|---|
| `results_v2/` | Original Washington results |
| `results_real/` | Original Caltech results and frozen permutations |
| `revision_inputs/` | Downloaded UrbanEV snapshots and manifests |
| `revision_runs/` | New experiment outputs, including separately labeled smoke runs |
| `paper_exports/` | Generated revision figures, tables, and handoff ZIPs |

Revision runners create separate output directories. Keep final metrics, protocols, summaries, and small graph/permutation artifacts trackable in Git. Keep ignored large files in a separate local backup; `.gitignore` does not back up files or remove already tracked history.
