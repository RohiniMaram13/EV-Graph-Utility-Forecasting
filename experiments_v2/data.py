import numpy as np
import pandas as pd
import torch
from functools import lru_cache

from experiments_v2.config import ROOT, LOOKBACK, HORIZON, STEP, TRAIN_DAYS, VAL_DAYS


def split_starts(total_hours, step=STEP):
    train_end = TRAIN_DAYS * 24
    val_end = (TRAIN_DAYS + VAL_DAYS) * 24
    max_t = total_hours - LOOKBACK - HORIZON
    all_starts = list(range(0, max_t + 1, step))
    train, val, test = [], [], []
    for t in all_starts:
        target_start = t + LOOKBACK
        target_end = target_start + HORIZON
        if target_end <= train_end:
            train.append(t)
        elif target_start >= train_end and target_end <= val_end:
            val.append(t)
        elif target_start >= val_end:
            test.append(t)
    return train, val, test


def normalize_ts(ts):
    """Normalize each ZIP independently using TRAINING data only.

    A single global scaler makes high-load ZIPs dominate the loss and can make
    graph smoothing artificially destructive. Node-wise scaling gives each
    region equal numerical footing while preserving physical-unit evaluation
    after inverse transformation.
    """
    train_end = TRAIN_DAYS * 24
    train = ts[:, :train_end]
    mean = train.mean(axis=1, keepdims=True).astype(np.float32)   # [N, 1]
    std = train.std(axis=1, keepdims=True).astype(np.float32)    # [N, 1]
    std = np.maximum(std, 1e-6)
    norm = ((ts - mean) / std).astype(np.float32)
    return norm, {'mean': mean, 'std': std, 'mode': 'per_node'}


def static_features(columns):
    if not columns:
        return None
    df = pd.read_csv(ROOT / 'data' / 'processed' / 'node_features.csv')
    x = df[columns].to_numpy(dtype=np.float32)
    x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-8)
    return torch.tensor(x, dtype=torch.float32)


def stack_windows(ts_norm, starts):
    xs, ys = [], []
    for t in starts:
        xs.append(torch.tensor(ts_norm[:, t:t + LOOKBACK], dtype=torch.float32))
        ys.append(torch.tensor(
            ts_norm[:, t + LOOKBACK:t + LOOKBACK + HORIZON],
            dtype=torch.float32,
        ))
    return torch.stack(xs), torch.stack(ys)


@lru_cache(maxsize=16)
def load_dataset(path):
    path = str(path)
    ts = np.load(path)
    starts = split_starts(ts.shape[1])
    ts_norm, scaler = normalize_ts(ts)
    sets = [stack_windows(ts_norm, s) for s in starts]
    return ts, ts_norm, scaler, starts, sets
