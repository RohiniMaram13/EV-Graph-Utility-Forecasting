"""Add Caltech seeds 72 and 82 using the original runner and all 20 controls.
Run from the repository root: python -u run_additional_caltech_seeds.py
Preserves original outputs. New results and a labeled five-seed summary are
written to revision_runs/caltech_extra_seeds_*/. Keep all seeds regardless of
their results. The combined summary mixes original runs and added runs;
software/hardware differences should be disclosed when reporting it.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repo.resolve()
    script = Path(__file__).resolve()
    files = ["real_validation/run_real_graph_validation.py",
             "experiments_v2/run_full_graph_control.py", "experiments_v2/models.py",
             "experiments_v2/train_eval.py", "experiments_v2/config.py",
             "experiments_v2/metrics.py", "experiments_v2/device_utils.py",
             "data/real_acn/caltech_hourly_energy.npz",
             "data/real_acn/caltech_pod_adjacency.npy",
             "data/real_acn/caltech_station_mapping.csv",
             "results_real/real_graph_permutations.json",
             "results_real/real_gnn_seed_summary.csv"]
    for name in files:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Missing {root / name}; run from the repository root or use --repo PATH")
    import pandas as pd
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; activate the tested environment")
    original = pd.read_csv(root / "results_real/real_gnn_seed_summary.csv")
    assert len(original) == 3 and set(original.train_seed) == {42, 52, 62}, \
        "Expected exactly the original seeds 42, 52 and 62; inspect before proceeding"
    perms = json.loads((root / "results_real/real_graph_permutations.json").read_text())
    assert len(perms) == 20, "Expected the original 20 frozen controls"
    parent = root / "revision_runs"
    parent.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="caltech_extra_seeds_", dir=parent))
    (output / "results_real").mkdir()
    shutil.copy2(root / "results_real/real_graph_permutations.json",
                 output / "results_real/real_graph_permutations.json")
    shutil.copy2(root / "results_real/real_gnn_seed_summary.csv",
                 output / "original_three_seed_summary.csv")
    shutil.copy2(script, output / script.name)
    sys.path.insert(0, str(root))
    # Import and run in an isolated directory because the original runner uses
    # relative output paths. Point its three data files to the existing inputs.
    os.chdir(output)
    import real_validation.run_real_graph_validation as runner
    runner.ENERGY_FILE = root / "data/real_acn/caltech_hourly_energy.npz"
    runner.ADJ_FILE = root / "data/real_acn/caltech_pod_adjacency.npy"
    runner.MAPPING_FILE = root / "data/real_acn/caltech_station_mapping.csv"
    runner.TRAIN_SEEDS = [72, 82]
    details = {"added_seeds": [72, 82], "original_seeds": [42, 52, 62],
               "permutations_per_seed": 20, "batch_size": 4, "amp": True,
               "torch": str(torch.__version__), "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0),
               "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                 for name in files}}
    (output / "run_details.json").write_text(json.dumps(details, indent=2) + "\n")
    print(f"\nNEW RESULTS: {output}\nRunning ONLY seeds 72 and 82, each with 20 shuffled graphs.", flush=True)
    start = time.perf_counter()
    sys.argv = ["run_real_graph_validation.py", "--device", "cuda",
                "--batch-size", "4", "--n-permutations", "20", "--verbose"]
    runner.main()
    added = pd.read_csv(output / "results_real/real_gnn_seed_summary.csv")
    assert len(added) == 2 and set(added.train_seed) == {72, 82}
    combined = pd.concat([original.assign(run_group="original"),
                          added.assign(run_group="added")], ignore_index=True)
    combined.to_csv(output / "combined_five_seed_metrics.csv", index=False)
    columns = ["gru_ft_mae", "real_gnn_mae", "real_gnn_rmse",
               "real_gain_over_gru_ft_pct", "real_vs_mean_perm_gain_pct",
               "percent_permutations_beaten"]
    summary = combined[columns].agg(["mean", "std"])
    summary.to_csv(output / "combined_five_seed_summary.csv", index_label="statistic")
    report = ("COMBINED FIVE SEEDS: 42, 52, 62, 72, 82\n"
              "Means and sample standard deviations; original and added runs combined.\n"
              + summary.to_string()
              + f"\nAdded-run runtime: {(time.perf_counter()-start)/60:.2f} minutes\n")
    (output / "summary.txt").write_text(report)
    print("\n" + report + f"Saved to: {output}", flush=True)


if __name__ == "__main__":
    main()
