"""Diagnostics for deciding whether graph learning is likely to add value.

RSPS = Residual Spatial Predictability Score.

The score is deliberately computed from TRAIN/VALIDATION data only. It asks
whether lagged geographic-neighbor residuals improve prediction of a node's
next-day residual beyond the node's own lagged residual.

Residuals are defined using a weekly seasonal difference on daily total load:
    r[i,d] = z[i,d] - z[i,d-7]
where z is node-wise standardized daily total using training statistics.
This removes a large amount of shared weekly structure before measuring
incremental spatial predictability.
"""
import numpy as np
from sklearn.linear_model import Ridge

from experiments_v2.config import TRAIN_DAYS, VAL_DAYS
from experiments_v2.graphs import row_normalized_adjacency


def daily_totals(ts):
    n, h = ts.shape
    if h % 24 != 0:
        raise ValueError('Hourly series length must be divisible by 24.')
    return ts.reshape(n, h // 24, 24).sum(axis=2).astype(np.float64)


def standardized_daily(ts):
    d = daily_totals(ts)
    mu = d[:, :TRAIN_DAYS].mean(axis=1, keepdims=True)
    sd = d[:, :TRAIN_DAYS].std(axis=1, keepdims=True) + 1e-8
    return (d - mu) / sd


def weekly_residuals(ts):
    z = standardized_daily(ts)
    return z[:, 7:] - z[:, :-7]


def _samples(resid, W, day_start, day_end):
    # resid day r corresponds to original day r+7. Predict day d from d-1.
    # day_start/day_end use original-day indexing.
    xs_self = []
    xs_spatial = []
    ys = []
    for d in range(max(day_start, 8), day_end):
        ridx = d - 7
        prev = resid[:, ridx - 1]
        target = resid[:, ridx]
        neigh = W @ prev
        xs_self.append(prev[:, None])
        xs_spatial.append(np.stack([prev, neigh], axis=1))
        ys.append(target)
    return (
        np.concatenate(xs_self, axis=0),
        np.concatenate(xs_spatial, axis=0),
        np.concatenate(ys, axis=0),
    )


def rsps_score(ts, edge_index, alpha=1.0):
    """Return RSPS and the underlying train/validation MSEs.

    RSPS = (MSE_self - MSE_self_plus_neighbors) / MSE_self on validation.
    Positive values mean neighbor residuals add predictive value beyond self
    residual history. Negative values mean they do not.
    """
    n = ts.shape[0]
    W = row_normalized_adjacency(edge_index, n)
    resid = weekly_residuals(ts)

    Xs_tr, Xsp_tr, y_tr = _samples(resid, W, 8, TRAIN_DAYS)
    Xs_va, Xsp_va, y_va = _samples(resid, W, TRAIN_DAYS, TRAIN_DAYS + VAL_DAYS)

    self_model = Ridge(alpha=alpha, fit_intercept=True)
    spatial_model = Ridge(alpha=alpha, fit_intercept=True)
    self_model.fit(Xs_tr, y_tr)
    spatial_model.fit(Xsp_tr, y_tr)

    p_self = self_model.predict(Xs_va)
    p_sp = spatial_model.predict(Xsp_va)
    mse_self = float(np.mean((p_self - y_va) ** 2))
    mse_sp = float(np.mean((p_sp - y_va) ** 2))
    score = (mse_self - mse_sp) / (mse_self + 1e-12)
    return {
        'rsps': float(score),
        'rsps_self_mse': mse_self,
        'rsps_spatial_mse': mse_sp,
        'rsps_neighbor_coef': float(spatial_model.coef_[1]),
    }


def raw_neighbor_correlation(ts, edge_index):
    d = standardized_daily(ts)
    # Use training period only for the diagnostic.
    d = d[:, :TRAIN_DAYS]
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    vals = []
    for a, b in zip(src, dst):
        if a < b:
            vals.append(float(np.corrcoef(d[a], d[b])[0, 1]))
    return float(np.nanmean(vals))
