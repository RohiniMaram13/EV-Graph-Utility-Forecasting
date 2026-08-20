"""Controlled regional EV-demand generator.

Two controls are intentionally separated:

1. coupling in [0, 1]
   Controls lagged cross-region dependence. At coupling=0, persistent daily
   deviations are driven only by each node's own previous state. At coupling=1,
   persistence is driven by geographic neighbors' previous states.

2. shared_shock_sigma >= 0
   Adds an IID same-day shock shared by every node. This raises raw spatial
   correlation but is not predictable from the previous day. It is used as a
   false-spatial-signal stress test: raw correlation can be high even when a
   graph has little incremental predictive value.
"""
import argparse
import datetime
import json
import numpy as np
import pandas as pd

from experiments_v2.config import ROOT, DATA_DIR, TOTAL_DAYS
from experiments_v2.graphs import build_geographic_knn, row_normalized_adjacency

DAILY_KWH_PER_EV = 10.0
MONTHLY_FACTORS = {1:1.15,2:1.12,3:1.00,4:0.98,5:0.95,6:0.90,7:0.88,8:0.90,9:0.97,10:1.02,11:1.08,12:1.15}
DOW_FACTORS = {0:1.00,1:1.00,2:1.00,3:1.00,4:1.00,5:0.82,6:0.80}


def scenario_tag(coupling, shared_shock_sigma, seed):
    return f'c{coupling:.2f}_g{shared_shock_sigma:.2f}_s{seed}'


def _shape(peak_home, peak_work, std_home, std_work, home_weight):
    h = np.arange(24, dtype=float)
    home = np.exp(-0.5 * ((h - peak_home) / std_home) ** 2)
    work = np.exp(-0.5 * ((h - peak_work) / std_work) ** 2)
    s = home_weight * home + (1.0 - home_weight) * work
    return s / s.sum()


def generate(coupling=0.5, seed=42, shared_shock_sigma=0.0,
             phi=0.70, sigma_state=0.07, sigma_hour=0.01):
    if not 0 <= coupling <= 1:
        raise ValueError('coupling must be in [0,1]')
    if shared_shock_sigma < 0:
        raise ValueError('shared_shock_sigma must be >= 0')

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    df = pd.read_csv(ROOT / 'data' / 'processed' / 'node_features.csv')
    n = len(df)
    ev = df.ev_count.to_numpy(dtype=float)

    geo, _, _ = build_geographic_knn(k=6)
    W = row_normalized_adjacency(geo, n)

    # Stable node-specific profile heterogeneity. Socioeconomic variables are not
    # used to generate labels, so their value can be tested rather than baked in.
    home_peak = np.clip(rng.normal(21.0, 0.9, n), 18.0, 23.0)
    work_peak = np.clip(rng.normal(9.0, 0.7, n), 6.0, 12.0)
    home_weight = np.clip(rng.normal(0.75, 0.07, n), 0.55, 0.90)
    weekday_shapes = np.stack([
        _shape(home_peak[i], work_peak[i], 2.0, 1.5, home_weight[i])
        for i in range(n)
    ])
    weekend_shapes = np.stack([
        _shape(home_peak[i]-1.0, work_peak[i]+1.0, 2.1, 1.7,
               min(0.95, home_weight[i]+0.12))
        for i in range(n)
    ])

    # Lagged latent state. This is the ground-truth predictive cross-region signal.
    state = np.zeros((n, TOTAL_DAYS), dtype=np.float32)
    state[:, 0] = rng.normal(0.0, sigma_state, n)
    for d in range(1, TOTAL_DAYS):
        prev = state[:, d-1]
        neighbor_prev = W @ prev
        persistent = (1.0 - coupling) * prev + coupling * neighbor_prev
        state[:, d] = phi * persistent + rng.normal(0.0, sigma_state, n)

    # IID global shocks raise same-day correlation without introducing lagged
    # neighbor predictability. This is the key false-correlation stress test.
    global_shock = rng.normal(0.0, shared_shock_sigma, TOTAL_DAYS).astype(np.float32)

    day_factor = np.clip(1.0 + state + global_shock[None, :], 0.55, 1.45)

    ts = np.zeros((n, TOTAL_DAYS * 24), dtype=np.float32)
    ref = datetime.date(2024, 1, 1)
    for d in range(TOTAL_DAYS):
        date = ref + datetime.timedelta(days=d)
        dow = date.weekday()
        shape = weekend_shapes if dow >= 5 else weekday_shapes
        scale = DAILY_KWH_PER_EV * DOW_FACTORS[dow] * MONTHLY_FACTORS[date.month]
        base = ev[:, None] * scale * shape / 1000.0
        hour_jitter = rng.normal(0.0, sigma_hour, size=(n, 24))
        hourly_factor = np.clip(day_factor[:, d, None] * (1.0 + hour_jitter), 0.50, 1.50)
        ts[:, d*24:(d+1)*24] = (base * hourly_factor).astype(np.float32)

    tag = scenario_tag(coupling, shared_shock_sigma, seed)
    np.save(DATA_DIR / f'timeseries_{tag}.npy', ts)
    np.save(DATA_DIR / f'latent_state_{tag}.npy', state)
    np.save(DATA_DIR / f'global_shock_{tag}.npy', global_shock)
    meta = {
        'coupling': coupling,
        'shared_shock_sigma': shared_shock_sigma,
        'seed': seed,
        'phi': phi,
        'sigma_state': sigma_state,
        'sigma_hour': sigma_hour,
        'daily_kwh_per_ev': DAILY_KWH_PER_EV,
        'nodes': n,
        'hours': int(ts.shape[1]),
        'interpretation': (
            'coupling controls lagged neighbor predictive dependence; '
            'shared_shock_sigma controls contemporaneous shared correlation '
            'without lagged cross-region causation.'
        )
    }
    (DATA_DIR / f'meta_{tag}.json').write_text(json.dumps(meta, indent=2))
    print(f'Saved {tag}: {ts.shape}, mean={ts.mean():.6f} MW, max={ts.max():.3f} MW')
    return ts, state


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--coupling', type=float, default=0.5)
    p.add_argument('--shared-shock-sigma', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    generate(args.coupling, args.seed, args.shared_shock_sigma)
