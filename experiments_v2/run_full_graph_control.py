"""Full matched-control experiment with geographic and permuted-graph controls.

Purpose
-------
This is the decisive controlled experiment for the RSPS paper direction.
For every scenario and training seed, all treatment branches start from the
same trained GRU checkpoint and receive matched fine-tuning budgets.

Models compared
---------------
1) GRU-FT: temporal-only matched fine-tuning control.
2) GEO-GNN: warm-start residual GNN using the true geographic KNN graph.
3) PERM-GNN: same warm-start residual GNN, same initialization/training seed,
   but with a node-label-permuted version of the geographic graph. This keeps
   the exact topology and degree distribution while breaking geographic node
   alignment.

Scenario grid
-------------
True coupling: 0.00, 0.25, 0.50, 0.75, 1.00
False signal:  coupling=0.00 with shared shock sigma 0.05, 0.10, 0.15, 0.20
Training seeds: 42, 52, 62
Data seed: 42 by default

Primary estimands
-----------------
geo_gain_over_gru_ft_pct
perm_gain_over_gru_ft_pct
geo_gain_over_perm_pct

A positive geo_gain_over_perm_pct means the geographically aligned graph beats
an equally structured but misaligned graph.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments_v2.config import DATA_DIR, RESULTS_DIR, HIDDEN_DIM, HORIZON, DROPOUT, LR
from experiments_v2.generator import generate, scenario_tag
from experiments_v2.graphs import save_graphs, build_permuted_graph_like, edge_overlap_fraction
from experiments_v2.data import load_dataset
from experiments_v2.models import GRUForecast, GRUGCNForecast
from experiments_v2.train_eval import train_model, predict, set_seed
from experiments_v2.metrics import metrics
from experiments_v2.novelty_diagnostics import rsps_score, raw_neighbor_correlation
from experiments_v2.device_utils import resolve_device, device_report

TRUE_COUPLINGS = [0.00, 0.25, 0.50, 0.75, 1.00]
FALSE_SHOCKS = [0.05, 0.10, 0.15, 0.20]
TRAIN_SEEDS = [42, 52, 62]


def scenario_grid():
    rows = [(c, 0.0, 'true_coupling') for c in TRUE_COUPLINGS]
    rows += [(0.0, g, 'false_signal') for g in FALSE_SHOCKS]
    return rows


def load_geo():
    p = DATA_DIR / 'graph_geo.pt'
    if not p.exists():
        save_graphs(seed=42)
    return torch.load(p, weights_only=False)


def ensure_scenario(coupling, shared_shock_sigma, data_seed):
    tag = scenario_tag(coupling, shared_shock_sigma, data_seed)
    p = DATA_DIR / f'timeseries_{tag}.npy'
    if not p.exists():
        generate(coupling=coupling, seed=data_seed, shared_shock_sigma=shared_shock_sigma)
    return p, tag


def evaluate(model, te, scaler, graph, runtime):
    pred, target = predict(
        model, te, scaler, None, graph,
        device=runtime['device'], batch_size=runtime['batch_size'], amp=runtime['amp'],
    )
    return metrics(pred, target)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def make_graph_dict(edge_index):
    return {'edge_index': edge_index.contiguous(), 'edge_weight': None}


def train_gnn_branch(shared_state, tr, va, te, scaler, graph, train_seed, runtime, label):
    """Train one graph treatment branch from a fixed shared GRU checkpoint."""
    # Identical model initialization for GEO and PERM branches.
    init_seed = train_seed + 10000
    set_seed(init_seed)
    shared_gru = GRUForecast(HIDDEN_DIM, HORIZON)
    shared_gru.load_state_dict(shared_state)
    gnn = GRUGCNForecast(HIDDEN_DIM, HORIZON, DROPOUT)
    gnn.initialize_from_gru(shared_gru)
    initial_state = cpu_state(gnn)

    # Re-load to make it explicit that every branch begins from exactly the
    # same parameter values before seeing a different graph.
    gnn.load_state_dict(initial_state)
    initial_gate = gnn.graph_gate()

    print(f'    {label}: graph-only warm-start stage', flush=True)
    gnn.set_graph_only_trainable()
    gnn, graph_best, graph_epochs = train_model(
        gnn, tr, va, None, graph, seed=train_seed,
        verbose=runtime['verbose'], device=runtime['device'],
        batch_size=runtime['batch_size'], max_epochs=runtime['graph_epochs'],
        patience=runtime['graph_patience'], amp=runtime['amp'], lr=runtime['graph_lr'],
        include_initial=True,
    )
    graph_gate = gnn.graph_gate()

    print(f'    {label}: matched joint fine-tune', flush=True)
    gnn.set_all_trainable()
    gnn, best, ft_epochs = train_model(
        gnn, tr, va, None, graph, seed=train_seed + 1000,
        verbose=runtime['verbose'], device=runtime['device'],
        batch_size=runtime['batch_size'], max_epochs=runtime['finetune_epochs'],
        patience=runtime['patience'], amp=runtime['amp'], lr=runtime['finetune_lr'],
        include_initial=True,
    )
    m = evaluate(gnn, te, scaler, graph, runtime)
    return {
        'mae': m['mae'], 'rmse': m['rmse'], 'r2': m['r2'],
        'graph_stage_best_val_mse': graph_best,
        'graph_stage_epochs': graph_epochs,
        'finetune_best_val_mse': best,
        'finetune_epochs': ft_epochs,
        'initial_gate': initial_gate,
        'graph_stage_gate': graph_gate,
        'final_gate': gnn.graph_gate(),
    }


def run_one(ts_path, geo, perm_graph, train_seed, runtime):
    _, _, scaler, _, sets = load_dataset(str(ts_path))
    tr, va, te = sets

    # Stage A: shared baseline.
    print('  A/4 shared GRU baseline', flush=True)
    set_seed(train_seed)
    gru = GRUForecast(HIDDEN_DIM, HORIZON)
    gru, gru_best, gru_epochs = train_model(
        gru, tr, va, None, None, seed=train_seed,
        verbose=runtime['verbose'], device=runtime['device'],
        batch_size=runtime['batch_size'], max_epochs=runtime['gru_epochs'],
        patience=runtime['patience'], amp=runtime['amp'], lr=LR,
    )
    baseline_m = evaluate(gru, te, scaler, None, runtime)
    shared_state = cpu_state(gru)

    # Stage B: matched temporal-only fine-tuning control.
    print('  B/4 matched GRU fine-tuning control', flush=True)
    set_seed(train_seed + 1000)
    gru_ft = GRUForecast(HIDDEN_DIM, HORIZON)
    gru_ft.load_state_dict(shared_state)
    gru_ft, gru_ft_best, gru_ft_epochs = train_model(
        gru_ft, tr, va, None, None, seed=train_seed + 1000,
        verbose=runtime['verbose'], device=runtime['device'],
        batch_size=runtime['batch_size'], max_epochs=runtime['finetune_epochs'],
        patience=runtime['patience'], amp=runtime['amp'], lr=runtime['finetune_lr'],
        include_initial=True,
    )
    gru_ft_m = evaluate(gru_ft, te, scaler, None, runtime)

    # Stages C/D: geographically aligned and permuted graph treatments.
    print('  C/4 geographic residual GNN', flush=True)
    geo_out = train_gnn_branch(shared_state, tr, va, te, scaler, geo, train_seed, runtime, 'GEO')
    print('  D/4 permuted-graph residual GNN', flush=True)
    perm_out = train_gnn_branch(shared_state, tr, va, te, scaler, perm_graph, train_seed, runtime, 'PERM')

    eps = 1e-12
    extra_train_gain = 100.0 * (baseline_m['mae'] - gru_ft_m['mae']) / (baseline_m['mae'] + eps)
    geo_gain = 100.0 * (gru_ft_m['mae'] - geo_out['mae']) / (gru_ft_m['mae'] + eps)
    perm_gain = 100.0 * (gru_ft_m['mae'] - perm_out['mae']) / (gru_ft_m['mae'] + eps)
    geo_vs_perm = 100.0 * (perm_out['mae'] - geo_out['mae']) / (perm_out['mae'] + eps)

    out = {
        'gru_baseline_mae': baseline_m['mae'],
        'gru_baseline_rmse': baseline_m['rmse'],
        'gru_baseline_r2': baseline_m['r2'],
        'gru_baseline_best_val_mse': gru_best,
        'gru_baseline_epochs': gru_epochs,
        'gru_ft_mae': gru_ft_m['mae'],
        'gru_ft_rmse': gru_ft_m['rmse'],
        'gru_ft_r2': gru_ft_m['r2'],
        'gru_ft_best_val_mse': gru_ft_best,
        'gru_ft_epochs': gru_ft_epochs,
        'extra_train_gain_pct': extra_train_gain,
        'geo_gnn_mae': geo_out['mae'],
        'geo_gnn_rmse': geo_out['rmse'],
        'geo_gnn_r2': geo_out['r2'],
        'geo_gain_over_gru_ft_pct': geo_gain,
        'geo_initial_gate': geo_out['initial_gate'],
        'geo_graph_stage_gate': geo_out['graph_stage_gate'],
        'geo_final_gate': geo_out['final_gate'],
        'geo_graph_stage_best_val_mse': geo_out['graph_stage_best_val_mse'],
        'geo_finetune_best_val_mse': geo_out['finetune_best_val_mse'],
        'geo_graph_stage_epochs': geo_out['graph_stage_epochs'],
        'geo_finetune_epochs': geo_out['finetune_epochs'],
        'perm_gnn_mae': perm_out['mae'],
        'perm_gnn_rmse': perm_out['rmse'],
        'perm_gnn_r2': perm_out['r2'],
        'perm_gain_over_gru_ft_pct': perm_gain,
        'perm_initial_gate': perm_out['initial_gate'],
        'perm_graph_stage_gate': perm_out['graph_stage_gate'],
        'perm_final_gate': perm_out['final_gate'],
        'perm_graph_stage_best_val_mse': perm_out['graph_stage_best_val_mse'],
        'perm_finetune_best_val_mse': perm_out['finetune_best_val_mse'],
        'perm_graph_stage_epochs': perm_out['graph_stage_epochs'],
        'perm_finetune_epochs': perm_out['finetune_epochs'],
        'geo_gain_over_perm_pct': geo_vs_perm,
    }
    return out


def run(data_seed, graph_seed, runtime):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_graphs(seed=42)
    geo = load_geo()
    n_nodes = int(geo['edge_index'].max().item()) + 1
    perm_edge, perm, overlap = build_permuted_graph_like(
        geo['edge_index'], n_nodes, seed=graph_seed, tries=runtime['perm_tries']
    )
    perm_graph = make_graph_dict(perm_edge)
    print(
        f'Permuted graph control: seed={graph_seed}, exact topology/degree preserved, '
        f'undirected edge overlap with geographic graph={100.0*overlap:.2f}%',
        flush=True,
    )

    rows = []
    out_csv = RESULTS_DIR / f'full_graph_control_diagnostic_ds{data_seed}_gs{graph_seed}.csv'
    grid = scenario_grid()
    total = len(grid) * len(TRAIN_SEEDS)
    k = 0

    for coupling, shock, scenario_type in grid:
        p, tag = ensure_scenario(coupling, shock, data_seed)
        ts = np.load(p)
        geo_diag = rsps_score(ts, geo['edge_index'])
        geo_raw = raw_neighbor_correlation(ts, geo['edge_index'])
        perm_diag = rsps_score(ts, perm_edge)
        perm_raw = raw_neighbor_correlation(ts, perm_edge)

        for train_seed in TRAIN_SEEDS:
            k += 1
            t0 = time.perf_counter()
            print(
                f'\n[{k}/{total}] {scenario_type} coupling={coupling:.2f} '
                f'shared_shock={shock:.2f} train_seed={train_seed}', flush=True,
            )
            print(
                f'  GEO diagnostics raw_corr={geo_raw:.4f} RSPS={geo_diag["rsps"]:.4f} | '
                f'PERM raw_corr={perm_raw:.4f} RSPS={perm_diag["rsps"]:.4f}', flush=True,
            )

            out = run_one(p, geo, perm_graph, train_seed, runtime)
            row = {
                'scenario': tag,
                'scenario_type': scenario_type,
                'coupling': coupling,
                'shared_shock_sigma': shock,
                'data_seed': data_seed,
                'train_seed': train_seed,
                'graph_seed': graph_seed,
                'geo_perm_edge_overlap_fraction': overlap,
                'geo_raw_neighbor_corr': geo_raw,
                'geo_rsps': geo_diag['rsps'],
                'geo_rsps_self_mse': geo_diag['rsps_self_mse'],
                'geo_rsps_spatial_mse': geo_diag['rsps_spatial_mse'],
                'geo_rsps_neighbor_coef': geo_diag['rsps_neighbor_coef'],
                'perm_raw_neighbor_corr': perm_raw,
                'perm_rsps': perm_diag['rsps'],
                'perm_rsps_self_mse': perm_diag['rsps_self_mse'],
                'perm_rsps_spatial_mse': perm_diag['rsps_spatial_mse'],
                'perm_rsps_neighbor_coef': perm_diag['rsps_neighbor_coef'],
                **out,
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(
                f'  RESULT GRU_FT={out["gru_ft_mae"]:.5f} '
                f'GEO={out["geo_gnn_mae"]:.5f} ({out["geo_gain_over_gru_ft_pct"]:+.2f}%) '
                f'PERM={out["perm_gnn_mae"]:.5f} ({out["perm_gain_over_gru_ft_pct"]:+.2f}%) '
                f'GEO_vs_PERM={out["geo_gain_over_perm_pct"]:+.2f}% '
                f'time={(time.perf_counter()-t0)/60:.1f} min', flush=True,
            )

    df = pd.DataFrame(rows)
    group_cols = ['scenario', 'scenario_type', 'coupling', 'shared_shock_sigma']
    summary = df.groupby(group_cols, as_index=False).agg(
        geo_rsps=('geo_rsps', 'first'),
        geo_raw_neighbor_corr=('geo_raw_neighbor_corr', 'first'),
        perm_rsps=('perm_rsps', 'first'),
        perm_raw_neighbor_corr=('perm_raw_neighbor_corr', 'first'),
        mean_gru_ft_mae=('gru_ft_mae', 'mean'),
        sd_gru_ft_mae=('gru_ft_mae', 'std'),
        mean_geo_gnn_mae=('geo_gnn_mae', 'mean'),
        sd_geo_gnn_mae=('geo_gnn_mae', 'std'),
        mean_perm_gnn_mae=('perm_gnn_mae', 'mean'),
        sd_perm_gnn_mae=('perm_gnn_mae', 'std'),
        mean_geo_gain_over_gru_ft_pct=('geo_gain_over_gru_ft_pct', 'mean'),
        sd_geo_gain_over_gru_ft_pct=('geo_gain_over_gru_ft_pct', 'std'),
        min_geo_gain_over_gru_ft_pct=('geo_gain_over_gru_ft_pct', 'min'),
        max_geo_gain_over_gru_ft_pct=('geo_gain_over_gru_ft_pct', 'max'),
        mean_perm_gain_over_gru_ft_pct=('perm_gain_over_gru_ft_pct', 'mean'),
        sd_perm_gain_over_gru_ft_pct=('perm_gain_over_gru_ft_pct', 'std'),
        mean_geo_gain_over_perm_pct=('geo_gain_over_perm_pct', 'mean'),
        sd_geo_gain_over_perm_pct=('geo_gain_over_perm_pct', 'std'),
        mean_geo_gate=('geo_final_gate', 'mean'),
        mean_perm_gate=('perm_final_gate', 'mean'),
    )
    summary['rsps_alignment_advantage'] = summary['geo_rsps'] - summary['perm_rsps']
    summary['rawcorr_alignment_advantage'] = summary['geo_raw_neighbor_corr'] - summary['perm_raw_neighbor_corr']

    summary_path = RESULTS_DIR / f'full_graph_control_summary_ds{data_seed}_gs{graph_seed}.csv'
    summary.to_csv(summary_path, index=False)

    assoc = pd.DataFrame([{
        'data_seed': data_seed,
        'graph_seed': graph_seed,
        'n_conditions': len(summary),
        'geo_perm_edge_overlap_fraction': overlap,
        'pearson_geo_rsps_vs_geo_gain': summary['geo_rsps'].corr(summary['mean_geo_gain_over_gru_ft_pct']),
        'pearson_geo_rawcorr_vs_geo_gain': summary['geo_raw_neighbor_corr'].corr(summary['mean_geo_gain_over_gru_ft_pct']),
        'pearson_geo_rsps_vs_geo_vs_perm_gain': summary['geo_rsps'].corr(summary['mean_geo_gain_over_perm_pct']),
        'pearson_rsps_alignment_advantage_vs_geo_vs_perm_gain': summary['rsps_alignment_advantage'].corr(summary['mean_geo_gain_over_perm_pct']),
        'pearson_rawcorr_alignment_advantage_vs_geo_vs_perm_gain': summary['rawcorr_alignment_advantage'].corr(summary['mean_geo_gain_over_perm_pct']),
    }])
    assoc_path = RESULTS_DIR / f'full_graph_control_association_ds{data_seed}_gs{graph_seed}.csv'
    assoc.to_csv(assoc_path, index=False)

    print('\n=== Full graph-control summary ===', flush=True)
    print(summary.to_string(index=False), flush=True)
    print('\n=== Descriptive associations across 9 conditions ===', flush=True)
    print(assoc.to_string(index=False), flush=True)
    print(f'\nSaved: {out_csv}', flush=True)
    print(f'Saved: {summary_path}', flush=True)
    print(f'Saved: {assoc_path}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-seed', type=int, default=42)
    ap.add_argument('--graph-seed', type=int, default=777)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--gru-epochs', type=int, default=100)
    ap.add_argument('--graph-epochs', type=int, default=30)
    ap.add_argument('--finetune-epochs', type=int, default=70)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--graph-patience', type=int, default=10)
    ap.add_argument('--graph-lr', type=float, default=1e-3)
    ap.add_argument('--finetune-lr', type=float, default=5e-4)
    ap.add_argument('--perm-tries', type=int, default=64)
    ap.add_argument('--no-amp', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()

    runtime = {
        'device': resolve_device(a.device),
        'batch_size': a.batch_size,
        'gru_epochs': a.gru_epochs,
        'graph_epochs': a.graph_epochs,
        'finetune_epochs': a.finetune_epochs,
        'patience': a.patience,
        'graph_patience': a.graph_patience,
        'graph_lr': a.graph_lr,
        'finetune_lr': a.finetune_lr,
        'perm_tries': a.perm_tries,
        'amp': not a.no_amp,
        'verbose': not a.quiet,
    }
    print(device_report(runtime['device']), flush=True)
    print(
        'Full graph-control grid: 9 conditions x 3 train seeds; '
        'models=matched GRU-FT + GEO residual GNN + PERM residual GNN; '
        f'graph_seed={a.graph_seed}; graph_lr={a.graph_lr:g}; '
        f'finetune_lr={a.finetune_lr:g}', flush=True,
    )
    run(a.data_seed, a.graph_seed, runtime)


if __name__ == '__main__':
    main()
