import numpy as np


def mean_edge_correlation(ts, edge_index):
    # Daily total, then z-score each node over days. Correlation is evaluated only
    # once per undirected edge (src < dst) to avoid double counting.
    n, hours = ts.shape
    daily = ts.reshape(n, hours//24, 24).sum(axis=2)
    daily = (daily - daily.mean(axis=1, keepdims=True)) / (daily.std(axis=1, keepdims=True) + 1e-8)
    src = edge_index[0].cpu().numpy(); dst = edge_index[1].cpu().numpy()
    vals = []
    for a, b in zip(src, dst):
        if a < b:
            vals.append(float(np.mean(daily[a] * daily[b])))
    return float(np.mean(vals))


def mean_random_pair_correlation(ts, n_pairs=1000, seed=123):
    rng = np.random.default_rng(seed)
    n, hours = ts.shape
    daily = ts.reshape(n, hours//24, 24).sum(axis=2)
    daily = (daily - daily.mean(axis=1, keepdims=True)) / (daily.std(axis=1, keepdims=True) + 1e-8)
    vals=[]
    for _ in range(n_pairs):
        a,b = rng.choice(n, 2, replace=False)
        vals.append(float(np.mean(daily[a]*daily[b])))
    return float(np.mean(vals))
