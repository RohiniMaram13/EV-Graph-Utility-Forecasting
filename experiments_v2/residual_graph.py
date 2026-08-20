"""Build graph candidates from training data only.

- raw-correlation graph: top-k relationships by contemporaneous training load correlation.
- residual-predictive graph: top-k lagged partial correlations after removing weekly
  seasonality and conditioning on each target node's own previous residual.

The residual graph is the research candidate: it attempts to encode predictive
relationships not already explained by local temporal history.
"""
import numpy as np
import torch

from experiments_v2.config import TRAIN_DAYS
from experiments_v2.novelty_diagnostics import standardized_daily, weekly_residuals


def _symmetrize_topk(score, k=6, use_abs=True):
    n = score.shape[0]
    edges = set()
    for i in range(n):
        s = score[i].copy()
        s[i] = -np.inf
        rank_score = np.abs(s) if use_abs else s
        rank_score[i] = -np.inf
        idx = np.argpartition(rank_score, -min(k, n-1))[-min(k, n-1):]
        for j in idx:
            if not np.isfinite(rank_score[j]):
                continue
            a, b = sorted((int(i), int(j)))
            if a != b:
                edges.add((a, b))
    directed = []
    for a, b in sorted(edges):
        directed.extend([(a, b), (b, a)])
    return torch.tensor(directed, dtype=torch.long).t().contiguous()


def raw_correlation_graph(ts, k=6):
    z = standardized_daily(ts)[:, :TRAIN_DAYS]
    corr = np.corrcoef(z)
    np.fill_diagonal(corr, 0.0)
    return _symmetrize_topk(corr, k=k, use_abs=True), corr


def residual_predictive_graph(ts, k=6):
    r = weekly_residuals(ts)
    # Only residuals whose original days are in the training period.
    r = r[:, :TRAIN_DAYS-7]
    X = r[:, :-1]
    Y = r[:, 1:]

    # Standardize each series across time.
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    Y = (Y - Y.mean(axis=1, keepdims=True)) / (Y.std(axis=1, keepdims=True) + 1e-8)

    # Correlations needed for partial corr(Y_i, X_j | X_i).
    r_yx = (Y @ X.T) / X.shape[1]       # i,j: corr(Y_i, X_j)
    r_xx = (X @ X.T) / X.shape[1]       # i,j: corr(X_i, X_j)
    own = np.diag(r_yx)                  # i: corr(Y_i, X_i)

    n = X.shape[0]
    partial = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        num = r_yx[i] - own[i] * r_xx[i]
        den = np.sqrt(np.maximum(1.0 - own[i] ** 2, 1e-8) * np.maximum(1.0 - r_xx[i] ** 2, 1e-8))
        partial[i] = num / den
        partial[i, i] = 0.0

    return _symmetrize_topk(partial, k=k, use_abs=True), partial


def graph_dict(edge_index):
    return {'edge_index': edge_index, 'edge_weight': None}
