"""Warm-start residual-GNN experiment.

Scientific purpose
------------------
The temporal GRU is trained first. The residual GRU+GCN then copies the exact
GRU temporal encoder and output head, starts with a near-zero graph gate, and
learns only the incremental spatial correction before a short joint fine-tune.
This makes the comparison directly answer: does graph information add value
beyond an already-trained temporal forecaster?

Default grid: coupling={0.0, 0.5, 1.0} x train_seed={42,52,62}, data_seed=42.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments_v2.config import DATA_DIR, RESULTS_DIR, HIDDEN_DIM, HORIZON, DROPOUT, LR
from experiments_v2.generator import generate, scenario_tag
from experiments_v2.graphs import save_graphs
from experiments_v2.data import load_dataset
from experiments_v2.models import GRUForecast, GRUGCNForecast
from experiments_v2.train_eval import train_model, predict, set_seed
from experiments_v2.metrics import metrics
from experiments_v2.novelty_diagnostics import rsps_score, raw_neighbor_correlation
from experiments_v2.device_utils import resolve_device, device_report

COUPLINGS = [0.00, 0.50, 1.00]
TRAIN_SEEDS = [42, 52, 62]


def load_geo():
    p = DATA_DIR / 'graph_geo.pt'
    if not p.exists():
        save_graphs(seed=42)
    return torch.load(p, weights_only=False)


def ensure_scenario(coupling, data_seed):
    tag = scenario_tag(coupling, 0.0, data_seed)
    p = DATA_DIR / f'timeseries_{tag}.npy'
    if not p.exists():
        generate(coupling=coupling, seed=data_seed, shared_shock_sigma=0.0)
    return p, tag


def evaluate(model, te, scaler, graph, runtime):
    pred, target = predict(
        model, te, scaler, None, graph,
        device=runtime['device'], batch_size=runtime['batch_size'], amp=runtime['amp'],
    )
    return metrics(pred, target)


def run_one(ts_path, geo, train_seed, runtime):
    _, _, scaler, _, sets = load_dataset(str(ts_path))
    tr, va, te = sets

    # 1) Temporal baseline.
    print('  Stage 1/3: train GRU baseline', flush=True)
    set_seed(train_seed)
    gru = GRUForecast(HIDDEN_DIM, HORIZON)
    gru, gru_best, gru_epochs = train_model(
        gru, tr, va, None, None, seed=train_seed,
        verbose=runtime['verbose'], device=runtime['device'],
        batch_size=runtime['batch_size'], max_epochs=runtime['gru_epochs'],
        patience=runtime['patience'], amp=runtime['amp'], lr=LR,
    )
    gru_m = evaluate(gru, te, scaler, None, runtime)
    print(f'    GRU test MAE={gru_m["mae"]:.5f}, best_val={gru_best:.6f}', flush=True)

    # 2) Exact GRU warm-start; only graph branch can move.
    print('  Stage 2/3: warm-start GNN; train graph correction only', flush=True)
    set_seed(train_seed + 10000)
    gnn = GRUGCNForecast(HIDDEN_DIM, HORIZON, DROPOUT)
    gnn.initialize_from_gru(gru)
    initial_gate = gnn.graph_gate()
    gnn.set_graph_only_trainable()
    gnn, graph_best, graph_epochs = train_model(
        gnn, tr, va, None, geo, seed=train_seed,
        verbose=runtime['verbose'], device=runtime['device'],
        batch_size=runtime['batch_size'], max_epochs=runtime['graph_epochs'],
        patience=runtime['graph_patience'], amp=runtime['amp'], lr=runtime['graph_lr'],
        include_initial=True,
    )
    graph_only_gate = gnn.graph_gate()

    # 3) Joint fine-tune from the graph-corrected warm-start.
    print('  Stage 3/3: joint fine-tune', flush=True)
    gnn.set_all_trainable()
    gnn, ft_best, ft_epochs = train_model(
        gnn, tr, va, None, geo, seed=train_seed + 1000,
        verbose=runtime['verbose'], device=runtime['device'],
        batch_size=runtime['batch_size'], max_epochs=runtime['finetune_epochs'],
        patience=runtime['patience'], amp=runtime['amp'], lr=runtime['finetune_lr'],
        include_initial=True,
    )
    gnn_m = evaluate(gnn, te, scaler, geo, runtime)
    gain = 100.0 * (gru_m['mae'] - gnn_m['mae']) / (gru_m['mae'] + 1e-12)
    return {
        'gru_mae': gru_m['mae'], 'gru_rmse': gru_m['rmse'], 'gru_r2': gru_m['r2'],
        'gru_best_val_mse': gru_best, 'gru_epochs': gru_epochs,
        'gnn_mae': gnn_m['mae'], 'gnn_rmse': gnn_m['rmse'], 'gnn_r2': gnn_m['r2'],
        'gnn_best_val_mse': ft_best, 'graph_stage_best_val_mse': graph_best,
        'graph_stage_epochs': graph_epochs, 'finetune_epochs': ft_epochs,
        'initial_graph_gate': initial_gate, 'graph_stage_gate': graph_only_gate,
        'final_graph_gate': gnn.graph_gate(), 'gnn_mae_gain_pct': gain,
    }


def run(data_seed, runtime):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_graphs(seed=42)
    geo = load_geo()
    rows = []
    out_csv = RESULTS_DIR / f'warmstart_diagnostic_ds{data_seed}.csv'
    total = len(COUPLINGS) * len(TRAIN_SEEDS)
    k = 0

    for coupling in COUPLINGS:
        p, tag = ensure_scenario(coupling, data_seed)
        ts = np.load(p)
        diag = rsps_score(ts, geo['edge_index'])
        raw_corr = raw_neighbor_correlation(ts, geo['edge_index'])
        for seed in TRAIN_SEEDS:
            k += 1
            t0 = time.perf_counter()
            print(f'\n[{k}/{total}] coupling={coupling:.2f}, train_seed={seed}', flush=True)
            print(f'  diagnostics: raw_corr={raw_corr:.4f}, RSPS={diag["rsps"]:.4f}', flush=True)
            out = run_one(p, geo, seed, runtime)
            row = {
                'scenario': tag, 'coupling': coupling, 'data_seed': data_seed,
                'train_seed': seed, 'raw_neighbor_corr': raw_corr, **diag,
                **out,
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(
                f'  RESULT GRU_MAE={out["gru_mae"]:.5f} GNN_MAE={out["gnn_mae"]:.5f} '
                f'gain={out["gnn_mae_gain_pct"]:+.2f}% gate={out["final_graph_gate"]:.4f} '
                f'time={(time.perf_counter()-t0)/60:.1f} min',
                flush=True,
            )

    df = pd.DataFrame(rows)
    summary = df.groupby('coupling', as_index=False).agg(
        rsps=('rsps', 'first'),
        raw_neighbor_corr=('raw_neighbor_corr', 'first'),
        mean_gru_mae=('gru_mae', 'mean'),
        sd_gru_mae=('gru_mae', 'std'),
        mean_gnn_mae=('gnn_mae', 'mean'),
        sd_gnn_mae=('gnn_mae', 'std'),
        mean_gnn_gain_pct=('gnn_mae_gain_pct', 'mean'),
        sd_gnn_gain_pct=('gnn_mae_gain_pct', 'std'),
        min_gnn_gain_pct=('gnn_mae_gain_pct', 'min'),
        max_gnn_gain_pct=('gnn_mae_gain_pct', 'max'),
        mean_final_gate=('final_graph_gate', 'mean'),
    )
    summary_path = RESULTS_DIR / f'warmstart_summary_ds{data_seed}.csv'
    summary.to_csv(summary_path, index=False)
    print('\n=== Warm-start summary ===', flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f'\nSaved: {out_csv}', flush=True)
    print(f'Saved: {summary_path}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-seed', type=int, default=42)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--gru-epochs', type=int, default=100)
    ap.add_argument('--graph-epochs', type=int, default=30)
    ap.add_argument('--finetune-epochs', type=int, default=70)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--graph-patience', type=int, default=10)
    ap.add_argument('--graph-lr', type=float, default=1e-3)
    ap.add_argument('--finetune-lr', type=float, default=5e-4)
    ap.add_argument('--no-amp', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    runtime = {
        'device': resolve_device(a.device), 'batch_size': a.batch_size,
        'gru_epochs': a.gru_epochs, 'graph_epochs': a.graph_epochs,
        'finetune_epochs': a.finetune_epochs, 'patience': a.patience,
        'graph_patience': a.graph_patience, 'graph_lr': a.graph_lr,
        'finetune_lr': a.finetune_lr, 'amp': not a.no_amp,
        'verbose': not a.quiet,
    }
    print(device_report(runtime['device']), flush=True)
    print(
        'Warm-start grid: couplings=0,0.5,1.0; train seeds=42,52,62; '
        f'graph_lr={a.graph_lr:g}; finetune_lr={a.finetune_lr:g}',
        flush=True,
    )
    run(a.data_seed, runtime)


if __name__ == '__main__':
    main()
