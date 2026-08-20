"""Novelty-focused experiments, CUDA-ready.

Primary hypothesis:
A training-only Residual Spatial Predictability Score (RSPS) is more informative
about actual GNN forecasting gain than raw spatial correlation.

The test split is NEVER used to build RSPS or graph candidates.
"""
import argparse
import time
import numpy as np
import pandas as pd
import torch

from experiments_v2.config import DATA_DIR, RESULTS_DIR, HIDDEN_DIM, HORIZON, DROPOUT
from experiments_v2.generator import generate, scenario_tag
from experiments_v2.graphs import save_graphs
from experiments_v2.data import load_dataset
from experiments_v2.models import GRUForecast, GRUGCNForecast
from experiments_v2.train_eval import train_model, predict
from experiments_v2.metrics import metrics
from experiments_v2.novelty_diagnostics import rsps_score, raw_neighbor_correlation
from experiments_v2.residual_graph import raw_correlation_graph, residual_predictive_graph, graph_dict
from experiments_v2.plots import make_novelty_plots
from experiments_v2.device_utils import resolve_device, device_report


def load_geo():
    p = DATA_DIR / 'graph_geo.pt'
    if not p.exists():
        save_graphs()
    return torch.load(p, weights_only=False)


def ensure_scenario(coupling, shared_shock_sigma, data_seed):
    tag = scenario_tag(coupling, shared_shock_sigma, data_seed)
    p = DATA_DIR / f'timeseries_{tag}.npy'
    if not p.exists():
        generate(coupling=coupling, seed=data_seed, shared_shock_sigma=shared_shock_sigma)
    return p, tag


def run_model(ts_path, model_kind, graph, train_seed, runtime):
    _, _, scaler, _, sets = load_dataset(str(ts_path))
    tr, va, te = sets
    if model_kind == 'gru':
        model = GRUForecast(HIDDEN_DIM, HORIZON)
        graph_arg = None
    elif model_kind == 'gcn':
        model = GRUGCNForecast(HIDDEN_DIM, HORIZON, DROPOUT)
        graph_arg = graph
    else:
        raise ValueError(model_kind)

    model, best_val, epochs = train_model(
        model, tr, va, None, graph_arg, seed=train_seed,
        verbose=runtime['verbose'], device=runtime['device'],
        batch_size=runtime['batch_size'], max_epochs=runtime['max_epochs'],
        patience=runtime['patience'], amp=runtime['amp'],
    )
    pred, target = predict(
        model, te, scaler, None, graph_arg, device=runtime['device'],
        batch_size=runtime['batch_size'], amp=runtime['amp'],
    )
    out = metrics(pred, target)
    out['best_val_mse'] = best_val
    out['epochs'] = epochs
    if hasattr(model, 'graph_gate'):
        out['graph_gate'] = model.graph_gate()
    return out


def scenario_grid(mode):
    if mode == 'smoke':
        return [
            (0.00, 0.00, 'true_coupling'),
            (1.00, 0.00, 'true_coupling'),
        ]
    if mode == 'quick':
        return [
            (0.00, 0.00, 'true_coupling'),
            (0.50, 0.00, 'true_coupling'),
            (1.00, 0.00, 'true_coupling'),
            (0.00, 0.20, 'false_signal'),
        ]
    out = []
    for c in [0.00, 0.25, 0.50, 0.75, 1.00]:
        out.append((c, 0.00, 'true_coupling'))
    for g in [0.05, 0.10, 0.15, 0.20]:
        out.append((0.00, g, 'false_signal'))
    return out


def run_diagnostic_suite(data_seed=42, train_seed=42, mode='full', runtime=None):
    runtime = runtime or default_runtime()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_graphs(seed=train_seed)
    geo = load_geo()
    rows = []
    scenarios = scenario_grid(mode)

    print(f'\nDiagnostic suite: {len(scenarios)} scenarios, 2 neural models each', flush=True)
    for si, (coupling, shared_sigma, scenario_type) in enumerate(scenarios, 1):
        t0 = time.perf_counter()
        print(f'\n[{si}/{len(scenarios)}] scenario coupling={coupling:.2f}, shared_shock={shared_sigma:.2f}, type={scenario_type}', flush=True)
        p, tag = ensure_scenario(coupling, shared_sigma, data_seed)
        ts = np.load(p)

        diag = rsps_score(ts, geo['edge_index'])
        raw_corr = raw_neighbor_correlation(ts, geo['edge_index'])
        print(f'  training-only diagnostics: raw_corr={raw_corr:.4f}, RSPS={diag["rsps"]:.4f}', flush=True)

        print('  Training GRU (no graph)...', flush=True)
        gru = run_model(p, 'gru', None, train_seed, runtime)
        print('  Training GRU+GCN (geographic graph)...', flush=True)
        gnn = run_model(p, 'gcn', geo, train_seed, runtime)
        gain = 100.0 * (gru['mae'] - gnn['mae']) / (gru['mae'] + 1e-12)

        row = {
            'scenario': tag,
            'scenario_type': scenario_type,
            'coupling': coupling,
            'shared_shock_sigma': shared_sigma,
            'data_seed': data_seed,
            'train_seed': train_seed,
            'raw_neighbor_corr': raw_corr,
            **diag,
            'gru_mae': gru['mae'], 'gru_rmse': gru['rmse'], 'gru_r2': gru['r2'],
            'gnn_mae': gnn['mae'], 'gnn_rmse': gnn['rmse'], 'gnn_r2': gnn['r2'],
            'gnn_mae_gain_pct': gain,
            'gnn_graph_gate': gnn.get('graph_gate', np.nan),
            'gru_epochs': gru['epochs'], 'gnn_epochs': gnn['epochs'],
        }
        rows.append(row)
        print(
            f'  RESULT {tag}: GRU_MAE={gru["mae"]:.5f} GNN_MAE={gnn["mae"]:.5f} '
            f'gain={gain:.2f}% gate={gnn.get("graph_gate", float("nan")):.3f} '
            f'scenario_time={(time.perf_counter()-t0)/60:.1f} min', flush=True
        )

        # Crash-safe checkpoint after each scenario.
        pd.DataFrame(rows).to_csv(
            RESULTS_DIR / f'novelty_diagnostic_ds{data_seed}_ts{train_seed}_{mode}.csv', index=False
        )

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / f'novelty_diagnostic_ds{data_seed}_ts{train_seed}_{mode}.csv'
    df.to_csv(out_csv, index=False)

    if len(df) >= 3:
        rsps_assoc = float(df[['rsps', 'gnn_mae_gain_pct']].corr().iloc[0, 1])
        raw_assoc = float(df[['raw_neighbor_corr', 'gnn_mae_gain_pct']].corr().iloc[0, 1])
        pd.DataFrame([{
            'data_seed': data_seed, 'train_seed': train_seed, 'mode': mode,
            'pearson_rsps_vs_gnn_gain': rsps_assoc,
            'pearson_rawcorr_vs_gnn_gain': raw_assoc,
            'n_scenarios': len(df),
        }]).to_csv(RESULTS_DIR / f'novelty_association_ds{data_seed}_ts{train_seed}_{mode}.csv', index=False)
        print(f'\nAssociation: RSPS vs GNN gain={rsps_assoc:.4f}; raw corr vs GNN gain={raw_assoc:.4f}', flush=True)

    make_novelty_plots(out_csv, RESULTS_DIR)
    return df


def run_graph_construction(data_seed=42, train_seed=42, coupling=0.75, shared_sigma=0.0, runtime=None):
    runtime = runtime or default_runtime()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_graphs(seed=train_seed)
    geo = load_geo()
    p, tag = ensure_scenario(coupling, shared_sigma, data_seed)
    ts = np.load(p)

    print('\nBuilding raw-correlation and residual-predictive graphs from TRAINING data only...', flush=True)
    raw_edge, _ = raw_correlation_graph(ts, k=6)
    residual_edge, _ = residual_predictive_graph(ts, k=6)
    raw_graph = graph_dict(raw_edge)
    residual_graph = graph_dict(residual_edge)

    rows = []
    specs = [
        ('GRU_no_graph', 'gru', None),
        ('GNN_geographic', 'gcn', geo),
        ('GNN_raw_correlation', 'gcn', raw_graph),
        ('GNN_residual_predictive', 'gcn', residual_graph),
    ]
    for i, (label, kind, graph) in enumerate(specs, 1):
        print(f'[{i}/{len(specs)}] {label}', flush=True)
        m = run_model(p, kind, graph, train_seed, runtime)
        m.update({
            'label': label, 'scenario': tag, 'coupling': coupling,
            'shared_shock_sigma': shared_sigma, 'data_seed': data_seed,
            'train_seed': train_seed,
        })
        rows.append(m)
        print(f'  MAE={m["mae"]:.6f} R2={m["r2"]:.4f}', flush=True)
        pd.DataFrame(rows).to_csv(
            RESULTS_DIR / f'novelty_graphs_ds{data_seed}_ts{train_seed}_{tag}.csv', index=False
        )
    return pd.DataFrame(rows)


def default_runtime():
    return {'device': resolve_device('auto'), 'batch_size': 4, 'max_epochs': 100, 'patience': 15, 'amp': True, 'verbose': True}


def runtime_from_args(a):
    d = resolve_device(a.device)
    return {
        'device': d,
        'batch_size': a.batch_size,
        'max_epochs': a.epochs,
        'patience': a.patience,
        'amp': not a.no_amp,
        'verbose': not a.quiet,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', choices=['diagnostic', 'graphs', 'all'], default='diagnostic')
    ap.add_argument('--mode', choices=['smoke', 'quick', 'full'], default='quick')
    ap.add_argument('--data-seed', type=int, default=42)
    ap.add_argument('--train-seed', type=int, default=42)
    ap.add_argument('--graph-coupling', type=float, default=0.75)
    ap.add_argument('--graph-shared-shock', type=float, default=0.0)
    ap.add_argument('--device', default='auto', help='auto, cpu, cuda, cuda:0')
    ap.add_argument('--batch-size', type=int, default=4, help='4 is safe for many laptop GPUs; try 8 if VRAM allows.')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--no-amp', action='store_true', help='Disable CUDA mixed precision.')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    runtime = runtime_from_args(a)
    print(device_report(runtime['device']), flush=True)
    print(f"Runtime: batch_size={runtime['batch_size']} epochs={runtime['max_epochs']} patience={runtime['patience']} amp={runtime['amp']}", flush=True)

    if a.suite in ['diagnostic', 'all']:
        run_diagnostic_suite(a.data_seed, a.train_seed, a.mode, runtime)
    if a.suite in ['graphs', 'all']:
        run_graph_construction(
            a.data_seed, a.train_seed, coupling=a.graph_coupling,
            shared_sigma=a.graph_shared_shock, runtime=runtime,
        )


if __name__ == '__main__':
    main()
