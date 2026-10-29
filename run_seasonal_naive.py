"""Run from the repository root: python revision_experiments/run_seasonal_naive.py"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repo.resolve()
    required = [
        "real_validation/run_real_graph_validation.py",
        "experiments_v2/metrics.py",
        "data/real_acn/caltech_hourly_energy.npz",
        "data/real_acn/caltech_pod_adjacency.npy",
    ]
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Missing {root / name}; run from the repository root or use --repo PATH")
    sys.path.insert(0, str(root))
    os.chdir(root)
    import numpy as np
    import pandas as pd
    from real_validation.run_real_graph_validation import load_real_pod_dataset
    from experiments_v2.metrics import metrics

    dataset = load_real_pod_dataset()
    x, y = (tensor.numpy() for tensor in dataset["test"])
    assert x.shape == (101, 16, 168), x.shape
    assert y.shape == (101, 16, 24), y.shape
    mean, std = dataset["scaler"]["mean"], dataset["scaler"]["std"]
    # The first 24 hours of the 168-hour history are exactly one week earlier.
    prediction = x[:, :, :24] * std + mean
    target = y * std + mean
    assert np.isfinite(prediction).all() and np.isfinite(target).all()
    # Independent raw-data check of the lag and original test boundaries.
    energy = dataset["energy"]
    starts = range(569 * 24, 670 * 24, 24)
    expected = np.stack([energy[t-168:t-144].T for t in starts])
    truth = np.stack([energy[t:t+24].T for t in starts])
    np.testing.assert_allclose(prediction, expected, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(target, truth, atol=1e-5, rtol=1e-5)

    parent = root / "revision_runs"
    parent.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="seasonal_naive_", dir=parent))
    row = {"model": "Weekly seasonal naive", "dataset": "Caltech Pod-16",
           "test_windows": 101, **metrics(prediction, target)}
    pd.DataFrame([row]).to_csv(output / "metrics.csv", index=False)
    np.savez_compressed(output / "predictions.npz", prediction=prediction,
                        target=target, station_ids=dataset["station_ids"])
    provenance = {"method": "Same hour 168 hours earlier; rolling daily 24-hour forecast",
                  "units": "kWh per hourly bin (MAE and RMSE)",
                  "test_day_indices": "569 through 669, zero based",
                  "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                    for name in required}}
    (output / "run_details.json").write_text(json.dumps(provenance, indent=2) + "\n")
    shutil.copy2(Path(__file__).resolve(), output / "run_seasonal_naive.py")
    summary = pd.DataFrame([row]).to_string(index=False)
    (output / "summary.txt").write_text(summary + "\n")
    print("\n" + summary)
    print("\nSaved to:", output)


if __name__ == "__main__":
    main()
