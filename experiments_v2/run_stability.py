"""Optimization-stability experiment for the residual GRU+GCN model.

Tests whether GNN conclusions are robust to initialization and learning rate.
The synthetic data seed is held fixed while training seed, coupling, and GNN
learning rate are varied. GRU is trained once per coupling/seed at the default
LR and reused as the baseline for the three GNN learning rates.
"""
import argparse
import time
import numpy as np
import pandas as pd
import torch

from experiments_v2.config import DATA_DIR, RESULTS_DIR, HIDDEN_DIM, HORIZON, DROPOUT, LR
from experiments_v2.generator import generate, scenario_tag
from experiments_v2.graphs import save_graphs
from experiments_v2.data import load_dataset
from experiments_v2.models import GRUForecast, GRUGCNForecast
from experiments_v2.train_eval import train_model, predict
from experiments_v2.metrics import metrics
from experiments_v2.novelty_diagnostics import rsps_score, raw_neighbor_correlation
from experiments_v2.device_utils import resolve_device, device_report

COUPLINGS = [0.00, 0.50, 1.00]
GNN_LRS = [1e-3, 5e-4, 2.5e-4]
TRAIN_SEEDS = [42, 52, 62]


def load_geo():
    p = DATA_DIR / 'graph_geo.pt'
    if not p.exists():
        save_graphs()
    return torch.load(p, weights_only=False)


def ensure_scenario(coupling, data_seed):
    tag = scenario_tag(coupling, 0.0, data_seed)
    p = DATA_DIR / f'timeseries_{tag}.npy'
    if not p.exists():
        generate(coupling=coupling, seed=data_seed, shared_shock_sigma=0.0)
    return p, tag


def fit_eval(ts_path, kind, graph, seed, lr, runtime):
    _, _, scaler, _, sets = load_dataset(str(ts_path))
    tr, va, te = sets
    if kind == 'gru':
        model, graph_arg = GRUForecast(HIDDEN_DIM, HORIZON), None
    else:
        model, graph_arg = GRUGCNForecast(HIDDEN_DIM, HORIZON, DROPOUT), graph
    model, best_val, epochs = train_model(
        model, tr, va, None, graph_arg, seed=seed,
        verbose=runtime['verbose'], device=runtime['device'],
        batch_size=runtime['batch_size'], max_epochs=runtime['max_epochs'],
        patience=runtime['patience'], amp=runtime['amp'], lr=lr,
    )
    pred, target = predict(
        model, te, scaler, None, graph_arg, device=runtime['device'],
        batch_size=runtime['batch_size'], amp=runtime['amp'],
    )
    out = metrics(pred, target)
    out['best_val_mse'] = best_val
    out['epochs'] = epochs
    out['graph_gate'] = model.graph_gate() if hasattr(model, 'graph_gate') else np.nan
    return out


def run(data_seed, runtime):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_graphs(seed=42)
    geo = load_geo()
    rows = []
    out_csv = RESULTS_DIR / f'gnn_stability_ds{data_seed}.csv'

    total_groups = len(COUPLINGS) * len(TRAIN_SEEDS)
    group = 0
    for coupling in COUPLINGS:
        p, tag = ensure_scenario(coupling, data_seed)
        ts = np.load(p)
        diag = rsps_score(ts, geo['edge_index'])
        raw_corr = raw_neighbor_correlation(ts, geo['edge_index'])
        for seed in TRAIN_SEEDS:
            group += 1
            t0 = time.perf_counter()
            print(f'\n[{group}/{total_groups}] coupling={coupling:.2f}, train_seed={seed}', flush=True)
            print(f'  diagnostics: raw_corr={raw_corr:.4f}, RSPS={diag["rsps"]:.4f}', flush=True)
            print(f'  GRU baseline lr={LR:.2e}', flush=True)
            gru = fit_eval(p, 'gru', None, seed, LR, runtime)
            print(f'  GRU MAE={gru["mae"]:.5f}, best_val={gru["best_val_mse"]:.6f}', flush=True)

            for gnn_lr in GNN_LRS:
                print(f'  GNN lr={gnn_lr:.2e}', flush=True)
                gnn = fit_eval(p, 'gcn', geo, seed, gnn_lr, runtime)
                gain = 100.0 * (gru['mae'] - gnn['mae']) / (gru['mae'] + 1e-12)
                row = {
                    'scenario': tag, 'coupling': coupling, 'data_seed': data_seed,
                    'train_seed': seed, 'gnn_lr': gnn_lr,
                    'raw_neighbor_corr': raw_corr, **diag,
                    'gru_lr': LR, 'gru_mae': gru['mae'], 'gru_rmse': gru['rmse'],
                    'gru_r2': gru['r2'], 'gru_best_val_mse': gru['best_val_mse'],
                    'gru_epochs': gru['epochs'],
                    'gnn_mae': gnn['mae'], 'gnn_rmse': gnn['rmse'], 'gnn_r2': gnn['r2'],
                    'gnn_best_val_mse': gnn['best_val_mse'], 'gnn_epochs': gnn['epochs'],
                    'gnn_graph_gate': gnn['graph_gate'], 'gnn_mae_gain_pct': gain,
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                print(f'    RESULT MAE={gnn["mae"]:.5f} gain={gain:+.2f}% gate={gnn["graph_gate"]:.3f}', flush=True)
            print(f'  group_time={(time.perf_counter()-t0)/60:.1f} min', flush=True)

    df = pd.DataFrame(rows)
    summary = df.groupby(['coupling', 'gnn_lr'], as_index=False).agg(
        mean_gnn_gain_pct=('gnn_mae_gain_pct', 'mean'),
        sd_gnn_gain_pct=('gnn_mae_gain_pct', 'std'),
        min_gnn_gain_pct=('gnn_mae_gain_pct', 'min'),
        max_gnn_gain_pct=('gnn_mae_gain_pct', 'max'),
        mean_gnn_mae=('gnn_mae', 'mean'),
        sd_gnn_mae=('gnn_mae', 'std'),
        mean_gate=('gnn_graph_gate', 'mean'),
    )
    summary.to_csv(RESULTS_DIR / f'gnn_stability_summary_ds{data_seed}.csv', index=False)

    lr_summary = df.groupby('gnn_lr', as_index=False).agg(
        mean_gain_pct=('gnn_mae_gain_pct', 'mean'),
        sd_gain_pct=('gnn_mae_gain_pct', 'std'),
        worst_gain_pct=('gnn_mae_gain_pct', 'min'),
        best_gain_pct=('gnn_mae_gain_pct', 'max'),
        mean_gnn_mae=('gnn_mae', 'mean'),
    )
    lr_summary.to_csv(RESULTS_DIR / f'gnn_stability_lr_summary_ds{data_seed}.csv', index=False)
    print('\n=== Stability summary by coupling and LR ===', flush=True)
    print(summary.to_string(index=False), flush=True)
    print('\n=== Overall LR summary ===', flush=True)
    print(lr_summary.to_string(index=False), flush=True)
    print(f'\nSaved: {out_csv}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-seed', type=int, default=42)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--no-amp', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    runtime = {
        'device': resolve_device(a.device), 'batch_size': a.batch_size,
        'max_epochs': a.epochs, 'patience': a.patience,
        'amp': not a.no_amp, 'verbose': not a.quiet,
    }
    print(device_report(runtime['device']), flush=True)
    print('Stability grid: couplings=0,0.5,1.0; GNN LRs=1e-3,5e-4,2.5e-4; train seeds=42,52,62', flush=True)
    run(a.data_seed, runtime)


if __name__ == '__main__':
    main()
