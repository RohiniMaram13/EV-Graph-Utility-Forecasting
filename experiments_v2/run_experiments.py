import argparse
import time
import numpy as np
import pandas as pd
import torch

from experiments_v2.config import ROOT, DATA_DIR, RESULTS_DIR, COUPLING_LEVELS, ALL_STATIC_FEATURES, SOCIO_FEATURES, HIDDEN_DIM, HORIZON, DROPOUT
from experiments_v2.graphs import save_graphs
from experiments_v2.generator import generate, scenario_tag
from experiments_v2.data import load_dataset, static_features
from experiments_v2.models import GRUForecast, GRUStaticForecast, GRUGCNForecast, STGNNForecastV2
from experiments_v2.train_eval import train_model, predict
from experiments_v2.metrics import metrics, subgroup_metrics
from experiments_v2.diagnostics import mean_edge_correlation, mean_random_pair_correlation
from experiments_v2.device_utils import resolve_device, device_report


def load_graph(kind, seed=42):
    if kind == 'geo':
        return torch.load(DATA_DIR / 'graph_geo.pt', weights_only=False)
    if kind == 'weighted':
        return torch.load(DATA_DIR / 'graph_geo_weighted.pt', weights_only=False)
    if kind == 'random':
        p = DATA_DIR / f'graph_random_seed{seed}.pt'
        if not p.exists(): save_graphs(seed=seed)
        return torch.load(p, weights_only=False)
    return None


def naive_predictions(ts, starts):
    out = {}; yall=[]; last=[]; daily=[]; weekly=[]; havg=[]
    for t in starts:
        x = ts[:, t:t+168]; y = ts[:, t+168:t+192]
        yall.append(y)
        last.append(np.repeat(x[:, -1:], 24, axis=1))
        daily.append(x[:, -24:])
        weekly.append(x[:, :24])
        havg.append(x.reshape(x.shape[0], 7, 24).mean(axis=1))
    y = np.stack(yall)
    out['persistence_last'] = np.stack(last)
    out['daily_naive'] = np.stack(daily)
    out['weekly_naive'] = np.stack(weekly)
    out['historical_avg_7d'] = np.stack(havg)
    return out, y


def model_for(name, n_static=0):
    if name == 'gru': return GRUForecast(HIDDEN_DIM, HORIZON)
    if name == 'gru_static': return GRUStaticForecast(n_static, HIDDEN_DIM, HORIZON)
    if name == 'gru_gcn': return GRUGCNForecast(HIDDEN_DIM, HORIZON, DROPOUT)
    if name == 'stgnn': return STGNNForecastV2(n_static, HIDDEN_DIM, HORIZON, DROPOUT)
    raise ValueError(name)


def run_one(ts_path, train_seed, model_name, runtime, static_cols=None, graph_kind=None, save_preds=False, label=None):
    ts, tsn, scaler, starts, sets = load_dataset(str(ts_path))
    tr, va, te = sets
    xs = static_features(static_cols or [])
    graph = load_graph(graph_kind, seed=train_seed) if graph_kind else None
    model = model_for(model_name, 0 if xs is None else xs.shape[1])
    model, best_val, epochs = train_model(
        model, tr, va, xs, graph, seed=train_seed, verbose=runtime['verbose'],
        device=runtime['device'], batch_size=runtime['batch_size'],
        max_epochs=runtime['max_epochs'], patience=runtime['patience'], amp=runtime['amp']
    )
    pred, target = predict(
        model, te, scaler, xs, graph, device=runtime['device'],
        batch_size=runtime['batch_size'], amp=runtime['amp']
    )
    m = metrics(pred, target)
    m.update({'model': model_name, 'static': '+'.join(static_cols or []) or 'none', 'graph': graph_kind or 'none', 'train_seed': train_seed, 'best_val_mse': best_val, 'epochs': epochs})
    if save_preds:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stem = label or f'{model_name}_{graph_kind or "none"}_s{train_seed}'
        np.savez_compressed(RESULTS_DIR / f'pred_{stem}.npz', pred=pred, target=target)
    return m, pred, target


def ensure_data(coupling, data_seed):
    tag = scenario_tag(coupling, 0.0, data_seed)
    p = DATA_DIR / f'timeseries_{tag}.npy'
    if not p.exists():
        generate(coupling=coupling, seed=data_seed, shared_shock_sigma=0.0)
    return p


def suite_core(data_seed, train_seed, reference_coupling, runtime):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True); save_graphs(seed=train_seed)
    p = ensure_data(reference_coupling, data_seed)
    ts, _, _, starts, _ = load_dataset(str(p))
    rows=[]
    naive, y = naive_predictions(ts, starts[2])
    for name, pred in naive.items():
        m=metrics(pred,y); m.update({'experiment':'E1_baselines','model':name,'static':'none','graph':'none','train_seed':-1,'label':name}); rows.append(m)

    specs = [
        ('gru', [], None, 'GRU'),
        ('gru_static', ALL_STATIC_FEATURES, None, 'GRU+Static'),
        ('gru_gcn', [], 'geo', 'GRU+GCN'),
        ('stgnn', ALL_STATIC_FEATURES, 'geo', 'STGNN'),
    ]
    saved = {}
    for i,(mn, sc, g, label) in enumerate(specs,1):
        print(f'Core model [{i}/{len(specs)}] {label}', flush=True)
        m,pred,target=run_one(p,train_seed,mn,runtime,sc,g,save_preds=(label in ['GRU','STGNN']),label=f'core_{label}_c{reference_coupling:.2f}')
        m.update({'experiment':'E1_E2_core','label':label,'coupling':reference_coupling,'data_seed':data_seed}); rows.append(m); saved[label]=(pred,target)
        pd.DataFrame(rows).to_csv(RESULTS_DIR/'core_results.csv',index=False)

    for cols,label in [(['ev_count'],'EV_count'),(SOCIO_FEATURES,'Socio'),(ALL_STATIC_FEATURES,'All_static')]:
        print(f'Static ablation: {label}', flush=True)
        m,_,_=run_one(p,train_seed,'gru_static',runtime,cols,None)
        m.update({'experiment':'E3_static_ablation','label':label,'coupling':reference_coupling,'data_seed':data_seed}); rows.append(m)
        pd.DataFrame(rows).to_csv(RESULTS_DIR/'core_results.csv',index=False)

    print('Graph topology: no graph', flush=True)
    m,_,_=run_one(p,train_seed,'gru',runtime,[],None); m.update({'experiment':'E5_graph_topology','label':'No_graph_GRU','coupling':reference_coupling,'data_seed':data_seed}); rows.append(m)
    for g in ['random','geo','weighted']:
        print(f'Graph topology: {g}', flush=True)
        m,_,_=run_one(p,train_seed,'gru_gcn',runtime,[],g)
        m.update({'experiment':'E5_graph_topology','label':g,'coupling':reference_coupling,'data_seed':data_seed}); rows.append(m)
        pd.DataFrame(rows).to_csv(RESULTS_DIR/'core_results.csv',index=False)

    ev = pd.read_csv(ROOT/'data'/'processed'/'node_features.csv').ev_count.to_numpy()
    sg = subgroup_metrics(saved['STGNN'][0], saved['STGNN'][1], ev)
    pd.DataFrame(sg).to_csv(RESULTS_DIR/'E6_subgroups.csv',index=False)
    df=pd.DataFrame(rows); df.to_csv(RESULTS_DIR/'core_results.csv',index=False)
    cols=[c for c in ['experiment','label','model','mae','rmse','nmae','r2','peak_mae'] if c in df.columns]
    print(df[cols].fillna('').to_string(index=False))
    return df


def suite_coupling(data_seeds, train_seeds, runtime):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True); save_graphs(seed=train_seeds[0])
    rows=[]; diag=[]
    geo=load_graph('geo',seed=train_seeds[0])['edge_index']
    total=len(data_seeds)*len(COUPLING_LEVELS)*len(train_seeds)*2; done=0
    for ds in data_seeds:
        for c in COUPLING_LEVELS:
            p=ensure_data(c,ds); ts=np.load(p)
            diag.append({'data_seed':ds,'coupling':c,'neighbor_corr':mean_edge_correlation(ts,geo),'random_pair_corr':mean_random_pair_correlation(ts,seed=ds+1000)})
            for tsd in train_seeds:
                for mn, sc, g, label in [('gru',[],None,'GRU'),('stgnn',ALL_STATIC_FEATURES,'geo','STGNN')]:
                    done += 1
                    print(f'Coupling run [{done}/{total}] data_seed={ds} coupling={c:.2f} train_seed={tsd} model={label}', flush=True)
                    m,_,_=run_one(p,tsd,mn,runtime,sc,g)
                    m.update({'experiment':'E4_coupling_sweep','label':label,'coupling':c,'data_seed':ds}); rows.append(m)
                    pd.DataFrame(rows).to_csv(RESULTS_DIR/'E4_coupling_runs.csv',index=False)
    pd.DataFrame(diag).to_csv(RESULTS_DIR/'E0_spatial_diagnostics.csv',index=False)
    df=pd.DataFrame(rows)
    summary=df.groupby(['coupling','label']).agg(mae_mean=('mae','mean'),mae_std=('mae','std'),rmse_mean=('rmse','mean'),r2_mean=('r2','mean'),peak_mae_mean=('peak_mae','mean')).reset_index()
    summary.to_csv(RESULTS_DIR/'E4_coupling_summary.csv',index=False)
    print(summary.to_string(index=False))
    return df


def runtime_from_args(a):
    return {'device':resolve_device(a.device),'batch_size':a.batch_size,'max_epochs':a.epochs,'patience':a.patience,'amp':not a.no_amp,'verbose':not a.quiet}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--suite',choices=['core','coupling','full'],default='core')
    ap.add_argument('--data-seeds',nargs='*',type=int,default=[42])
    ap.add_argument('--train-seeds',nargs='*',type=int,default=[42])
    ap.add_argument('--reference-coupling',type=float,default=.50)
    ap.add_argument('--device',default='auto')
    ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--patience',type=int,default=15)
    ap.add_argument('--no-amp',action='store_true')
    ap.add_argument('--quiet',action='store_true')
    a=ap.parse_args(); runtime=runtime_from_args(a)
    print(device_report(runtime['device']), flush=True)
    print(f"Runtime: batch_size={runtime['batch_size']} epochs={runtime['max_epochs']} patience={runtime['patience']} amp={runtime['amp']}", flush=True)
    if a.suite in ['core','full']:
        suite_core(a.data_seeds[0],a.train_seeds[0],a.reference_coupling,runtime)
    if a.suite in ['coupling','full']:
        suite_coupling(a.data_seeds,a.train_seeds,runtime)

if __name__=='__main__': main()
