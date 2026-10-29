import numpy as np
from experiments_v2.config import PEAK_HOURS


def metrics(pred, target):
    pred = np.asarray(pred); target = np.asarray(target)
    err = pred - target
    ae = np.abs(err)
    se = err ** 2
    mae = float(ae.mean())
    rmse = float(np.sqrt(se.mean()))
    denom = float(np.mean(np.abs(target))) + 1e-12
    nmae = mae / denom
    y = target.reshape(-1); p = pred.reshape(-1)
    r2 = 1.0 - float(((y-p)**2).sum()) / (float(((y-y.mean())**2).sum()) + 1e-12)
    peak = ae[:, :, PEAK_HOURS].mean()
    off = [h for h in range(24) if h not in PEAK_HOURS]
    offpeak = ae[:, :, off].mean()
    return {'mae':mae,'rmse':rmse,'nmae':float(nmae),'r2':float(r2),'peak_mae':float(peak),'offpeak_mae':float(offpeak)}


def subgroup_metrics(pred, target, ev_counts):
    qs = np.quantile(ev_counts, [0, .25, .5, .75, 1.0])
    out = []
    for q in range(4):
        lo, hi = qs[q], qs[q+1]
        mask = (ev_counts >= lo) & (ev_counts <= hi if q == 3 else ev_counts < hi)
        m = metrics(pred[:, mask, :], target[:, mask, :])
        m.update({'quartile': q+1, 'n_nodes': int(mask.sum()), 'ev_min': float(lo), 'ev_max': float(hi)})
        out.append(m)
    return out
