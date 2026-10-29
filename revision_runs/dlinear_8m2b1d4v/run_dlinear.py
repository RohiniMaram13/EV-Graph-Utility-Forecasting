"""Caltech Pod-16 DLinear baseline. Run from your repository root:
    python -u run_dlinear.py

Architecture: shared-channel DLinear, endpoint-padded moving average (width 25),
separate linear projections of trend and residual from 168 to 24 hours.
Reference: https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py
Uses the repository's training-only normalization, MSE training/validation,
Adam, scheduler, early stopping, and unscaled evaluation without clipping.
This is an added baseline, not a reproduction of the DLinear paper's results.
Default: seeds 42/52/62, 100 epochs maximum each, patience 15, batch size 4,
learning rate 0.001, existing repository weight decay, AMP enabled on CUDA.
No hyperparameter selection is performed on the test set.
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--self-check", action="store_true", help="Check model mechanics only; do not train")
    args = parser.parse_args()
    root = args.repo.resolve()
    source = Path(__file__).resolve()
    required = ["real_validation/run_real_graph_validation.py",
                "experiments_v2/train_eval.py", "experiments_v2/metrics.py",
                "experiments_v2/config.py", "experiments_v2/device_utils.py",
                "data/real_acn/caltech_hourly_energy.npz",
                "data/real_acn/caltech_pod_adjacency.npy"]
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Missing {root / name}; use --repo PATH or run from the project root")
    os.chdir(root)
    sys.path.insert(0, str(root))
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from torch.nn import functional as F
    from real_validation.run_real_graph_validation import load_real_pod_dataset
    from experiments_v2.train_eval import train_model, predict, set_seed
    from experiments_v2.metrics import metrics
    from experiments_v2.device_utils import resolve_device
    from experiments_v2.config import WEIGHT_DECAY

    class DLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.seasonal = nn.Linear(168, 24)
            self.trend = nn.Linear(168, 24)

        def forward(self, x, x_static=None, edge_index=None, edge_weight=None):
            # Repository tensors are [batch, stations, hours].
            trend = F.avg_pool1d(F.pad(x, (12, 12), mode="replicate"), 25, stride=1)
            return self.seasonal(x - trend) + self.trend(trend)

    device = resolve_device(args.device)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA requested but unavailable")
    # Check endpoint padding against an explicit independent moving average,
    # and ensure both linear paths support finite gradients on the device.
    set_seed(0)
    probe = DLinear().to(device)
    sample = torch.randn(2, 16, 168, device=device)
    padded = torch.cat([sample[..., :1].expand(-1, -1, 12), sample,
                        sample[..., -1:].expand(-1, -1, 12)], dim=-1)
    trend_reference = padded.unfold(-1, 25, 1).mean(dim=-1)
    expected = probe.seasonal(sample - trend_reference) + probe.trend(trend_reference)
    actual = probe(sample)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    assert actual.shape == (2, 16, 24)
    actual.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probe.parameters())
    print(f"DLinear shape, moving-average and backward checks PASSED on {device}", flush=True)
    if args.self_check:
        return
    del probe, sample, padded, trend_reference, expected, actual

    data = load_real_pod_dataset()
    assert tuple(data["train"][0].shape) == (462, 16, 168)
    assert tuple(data["validation"][0].shape) == (100, 16, 168)
    assert tuple(data["test"][0].shape) == (101, 16, 168)
    parent = root / "revision_runs"
    parent.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="dlinear_", dir=parent))
    shutil.copy2(source, output / "run_dlinear.py")
    settings = dict(model="DLinear shared-channel", kernel_size=25,
                    lookback=168, horizon=24, seeds=[42, 52, 62],
                    max_epochs=100, patience=15, batch_size=4, lr=0.001,
                    weight_decay=WEIGHT_DECAY, amp=True, loss="normalized MSE",
                    selection="minimum validation normalized MSE", device=str(device),
                    torch_version=str(torch.__version__), test_windows=101,
                    gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                    source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                   for name in required})
    (output / "run_details.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"\nResults folder: {output}", flush=True)
    rows = []
    start = time.perf_counter()
    for seed in settings["seeds"]:
        print(f"\nDLINEAR SEED {seed}", flush=True)
        set_seed(seed)
        model = DLinear()
        tick = time.perf_counter()
        model, best_val, epochs = train_model(
            model, data["train"], data["validation"], seed=seed,
            verbose=True, device=device, batch_size=4, max_epochs=100,
            patience=15, amp=True, lr=0.001,
        )
        prediction, target = predict(model, data["test"], data["scaler"],
                                     device=device, batch_size=4, amp=True)
        assert np.isfinite(prediction).all() and np.isfinite(target).all()
        row = dict(seed=seed, best_validation_mse=best_val, epochs_run=epochs,
                   runtime_minutes=(time.perf_counter()-tick)/60,
                   **metrics(prediction, target))
        rows.append(row)
        pd.DataFrame(rows).to_csv(output / "seed_metrics.csv", index=False)
        np.savez_compressed(output / f"predictions_seed_{seed}.npz",
                            prediction=prediction, target=target,
                            station_ids=data["station_ids"])
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "seed": seed, "settings": settings}, output / f"checkpoint_seed_{seed}.pt")
        print(pd.DataFrame([row]).to_string(index=False), flush=True)

    frame = pd.DataFrame(rows)
    keys = ["mae", "rmse", "nmae", "r2", "peak_mae", "offpeak_mae"]
    summary = frame[keys].agg(["mean", "std"])
    summary.to_csv(output / "summary_metrics.csv", index_label="statistic")
    report = ("DLinear: mean and sample standard deviation across three seeds\n"
              + summary.to_string() + f"\nTotal minutes: {(time.perf_counter()-start)/60:.2f}\n")
    (output / "summary.txt").write_text(report)
    print("\n" + report + f"Saved to: {output}", flush=True)


if __name__ == "__main__":
    main()
