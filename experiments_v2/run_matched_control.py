"""Matched-training control for the warm-start residual GNN.

Scientific purpose
------------------
The warm-start experiment showed that a residual GNN can improve after being
initialized from a trained GRU, but it also receives extra optimization after
that baseline has already converged. This script adds a matched temporal
control so the graph-specific effect is isolated.

For each coupling and training seed:
  1) Train one GRU baseline.
  2) Clone that exact GRU and continue fine-tuning it WITHOUT a graph for the
     same joint fine-tuning budget used by the GNN. (The GNN's graph-only stage
     has no temporal-control analogue; the control simply remains at the same
     GRU checkpoint during that stage.)
  3) Initialize a residual GNN from the exact same GRU, train only its graph
     correction branch, then joint fine-tune for the exact same duration/LR as
     the temporal control.

The primary estimand is therefore:
    graph_specific_gain_pct = (GRU_FT_MAE - GNN_MAE) / GRU_FT_MAE * 100

Default grid: coupling={0.0, 0.5, 1.0} x train_seed={42,52,62}, data_seed=42.
"""

import argparse
import copy
import time

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

    # ------------------------------------------------------------------
    # Stage 1: shared temporal baseline. Both downstream branches start
    # from this exact checkpoint.
    # ------------------------------------------------------------------
    print('  Stage 1/4: train shared GRU baseline', flush=True)
    set_seed(train_seed)
    gru = GRUForecast(HIDDEN_DIM, HORIZON)
    gru, gru_best, gru_epochs = train_model(
        gru, tr, va, None, None, seed=train_seed,
        verbose=runtime['verbose'], device=runtime['device'],
        batch_size=runtime['batch_size'], max_epochs=runtime['gru_epochs'],
        patience=runtime['patience'], amp=runtime['amp'], lr=LR,
    )
    baseline_m = evaluate(gru, te, scaler, None, runtime)
    print(
        f'    baseline GRU test MAE={baseline_m["mae"]:.5f}, best_val={gru_best:.6f}',
        flush=True,
    )

    # Keep an immutable CPU copy of the shared baseline so the two branches
    # are guaranteed to start from identical learned weights.
    shared_state = {k: v.detach().cpu().clone() for k, v in gru.state_dict().items()}

    # ------------------------------------------------------------------
    # Stage 2: matched temporal-control branch. It receives the same joint
    # fine-tuning duration, LR, validation selection and seed convention as
    # the final GNN fine-tuning stage, but no graph information.
    # ------------------------------------------------------------------
    print('  Stage 2/4: matched GRU fine-tuning control (no graph)', flush=True)
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
    print(
        f'    matched GRU-FT test MAE={gru_ft_m["mae"]:.5f}, '
        f'best_val={gru_ft_best:.6f}', flush=True,
    )

    # ------------------------------------------------------------------
    # Stage 3: treatment branch. Initialize from the SAME shared GRU and
    # allow only the graph correction to move.
    # ------------------------------------------------------------------
    print('  Stage 3/4: warm-start GNN; train graph correction only', flush=True)
    set_seed(train_seed + 10000)
    gnn = GRUGCNForecast(HIDDEN_DIM, HORIZON, DROPOUT)
    # load through a temporary GRU so initialize_from_gru gets exactly the
    # shared checkpoint regardless of what happened in the GRU-FT branch.
    shared_gru = GRUForecast(HIDDEN_DIM, HORIZON)
    shared_gru.load_state_dict(shared_state)
    gnn.initialize_from_gru(shared_gru)
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

    # ------------------------------------------------------------------
    # Stage 4: joint fine-tune. This uses the SAME epochs/LR/patience and
    # seed convention as the GRU-FT control in Stage 2.
    # ------------------------------------------------------------------
    print('  Stage 4/4: matched joint GNN fine-tune', flush=True)
    gnn.set_all_trainable()
    gnn, gnn_best, gnn_ft_epochs = train_model(
        gnn, tr, va, None, geo, seed=train_seed + 1000,
        verbose=runtime['verbose'], device=runtime['device'],
        batch_size=runtime['batch_size'], max_epochs=runtime['finetune_epochs'],
        patience=runtime['patience'], amp=runtime['amp'], lr=runtime['finetune_lr'],
        include_initial=True,
    )
    gnn_m = evaluate(gnn, te, scaler, geo, runtime)

    baseline_to_ft = 100.0 * (baseline_m['mae'] - gru_ft_m['mae']) / (baseline_m['mae'] + 1e-12)
    baseline_to_gnn = 100.0 * (baseline_m['mae'] - gnn_m['mae']) / (baseline_m['mae'] + 1e-12)
    graph_specific = 100.0 * (gru_ft_m['mae'] - gnn_m['mae']) / (gru_ft_m['mae'] + 1e-12)

    return {
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

        'gnn_mae': gnn_m['mae'],
        'gnn_rmse': gnn_m['rmse'],
        'gnn_r2': gnn_m['r2'],
        'gnn_best_val_mse': gnn_best,
        'graph_stage_best_val_mse': graph_best,
        'graph_stage_epochs': graph_epochs,
        'gnn_finetune_epochs': gnn_ft_epochs,

        'initial_graph_gate': initial_gate,
        'graph_stage_gate': graph_only_gate,
        'final_graph_gate': gnn.graph_gate(),

        'gru_ft_gain_over_baseline_pct': baseline_to_ft,
        'gnn_gain_over_baseline_pct': baseline_to_gnn,
        'graph_specific_gain_pct': graph_specific,
    }


def run(data_seed, runtime):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_graphs(seed=42)
    geo = load_geo()

    rows = []
    out_csv = RESULTS_DIR / f'matched_control_diagnostic_ds{data_seed}.csv'
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
                'scenario': tag,
                'coupling': coupling,
                'data_seed': data_seed,
                'train_seed': seed,
                'raw_neighbor_corr': raw_corr,
                **diag,
                **out,
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_csv, index=False)

            print(
                f'  RESULT baseline={out["gru_baseline_mae"]:.5f} '
                f'GRU_FT={out["gru_ft_mae"]:.5f} GNN={out["gnn_mae"]:.5f} | '
                f'extra_train_gain={out["gru_ft_gain_over_baseline_pct"]:+.2f}% '
                f'graph_specific_gain={out["graph_specific_gain_pct"]:+.2f}% '
                f'gate={out["final_graph_gate"]:.4f} '
                f'time={(time.perf_counter()-t0)/60:.1f} min',
                flush=True,
            )

    df = pd.DataFrame(rows)
    summary = df.groupby('coupling', as_index=False).agg(
        rsps=('rsps', 'first'),
        raw_neighbor_corr=('raw_neighbor_corr', 'first'),
        mean_gru_baseline_mae=('gru_baseline_mae', 'mean'),
        sd_gru_baseline_mae=('gru_baseline_mae', 'std'),
        mean_gru_ft_mae=('gru_ft_mae', 'mean'),
        sd_gru_ft_mae=('gru_ft_mae', 'std'),
        mean_gnn_mae=('gnn_mae', 'mean'),
        sd_gnn_mae=('gnn_mae', 'std'),
        mean_extra_train_gain_pct=('gru_ft_gain_over_baseline_pct', 'mean'),
        sd_extra_train_gain_pct=('gru_ft_gain_over_baseline_pct', 'std'),
        mean_graph_specific_gain_pct=('graph_specific_gain_pct', 'mean'),
        sd_graph_specific_gain_pct=('graph_specific_gain_pct', 'std'),
        min_graph_specific_gain_pct=('graph_specific_gain_pct', 'min'),
        max_graph_specific_gain_pct=('graph_specific_gain_pct', 'max'),
        mean_gnn_gain_over_baseline_pct=('gnn_gain_over_baseline_pct', 'mean'),
        mean_final_gate=('final_graph_gate', 'mean'),
    )

    summary_path = RESULTS_DIR / f'matched_control_summary_ds{data_seed}.csv'
    summary.to_csv(summary_path, index=False)

    # Small association file: only three coupling conditions, so this is a
    # descriptive sanity check, not inferential evidence.
    assoc = pd.DataFrame([{
        'data_seed': data_seed,
        'n_couplings': len(summary),
        'pearson_rsps_vs_mean_graph_specific_gain': summary['rsps'].corr(summary['mean_graph_specific_gain_pct']),
        'pearson_rawcorr_vs_mean_graph_specific_gain': summary['raw_neighbor_corr'].corr(summary['mean_graph_specific_gain_pct']),
    }])
    assoc_path = RESULTS_DIR / f'matched_control_association_ds{data_seed}.csv'
    assoc.to_csv(assoc_path, index=False)

    print('\n=== Matched-control summary ===', flush=True)
    print(summary.to_string(index=False), flush=True)
    print('\n=== Descriptive association (3 coupling points only) ===', flush=True)
    print(assoc.to_string(index=False), flush=True)
    print(f'\nSaved: {out_csv}', flush=True)
    print(f'Saved: {summary_path}', flush=True)
    print(f'Saved: {assoc_path}', flush=True)


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
        'device': resolve_device(a.device),
        'batch_size': a.batch_size,
        'gru_epochs': a.gru_epochs,
        'graph_epochs': a.graph_epochs,
        'finetune_epochs': a.finetune_epochs,
        'patience': a.patience,
        'graph_patience': a.graph_patience,
        'graph_lr': a.graph_lr,
        'finetune_lr': a.finetune_lr,
        'amp': not a.no_amp,
        'verbose': not a.quiet,
    }

    print(device_report(runtime['device']), flush=True)
    print(
        'Matched-control grid: couplings=0,0.5,1.0; train seeds=42,52,62; '
        f'graph_lr={a.graph_lr:g}; finetune_lr={a.finetune_lr:g}; '
        f'joint_finetune_epochs={a.finetune_epochs}',
        flush=True,
    )
    run(a.data_seed, runtime)


if __name__ == '__main__':
    main()
