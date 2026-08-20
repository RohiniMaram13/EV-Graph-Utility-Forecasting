import argparse
from pathlib import Path
import pandas as pd

from experiments_v2.config import RESULTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph-seed', type=int, default=777)
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(RESULTS_DIR.glob(f'full_graph_control_diagnostic_ds*_gs{args.graph_seed}.csv'))
    if not files:
        raise SystemExit('No full graph-control diagnostic CSV files found in results_v2/.')

    frames = [pd.read_csv(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=['scenario_type','coupling','shared_shock_sigma','data_seed','train_seed','graph_seed'], keep='last')
    df.to_csv(RESULTS_DIR / f'paper_all_runs_gs{args.graph_seed}.csv', index=False)

    scenario = df.groupby(['scenario_type','coupling','shared_shock_sigma'], as_index=False).agg(
        n_runs=('geo_gain_over_perm_pct','size'),
        n_data_seeds=('data_seed','nunique'),
        n_train_seeds=('train_seed','nunique'),
        geo_rsps_mean=('geo_rsps','mean'),
        geo_rsps_sd=('geo_rsps','std'),
        geo_rawcorr_mean=('geo_raw_neighbor_corr','mean'),
        geo_rawcorr_sd=('geo_raw_neighbor_corr','std'),
        perm_rsps_mean=('perm_rsps','mean'),
        perm_rawcorr_mean=('perm_raw_neighbor_corr','mean'),
        gru_ft_mae_mean=('gru_ft_mae','mean'),
        gru_ft_mae_sd=('gru_ft_mae','std'),
        geo_gnn_mae_mean=('geo_gnn_mae','mean'),
        geo_gnn_mae_sd=('geo_gnn_mae','std'),
        perm_gnn_mae_mean=('perm_gnn_mae','mean'),
        perm_gnn_mae_sd=('perm_gnn_mae','std'),
        geo_gain_vs_gru_mean=('geo_gain_over_gru_ft_pct','mean'),
        geo_gain_vs_gru_sd=('geo_gain_over_gru_ft_pct','std'),
        perm_gain_vs_gru_mean=('perm_gain_over_gru_ft_pct','mean'),
        perm_gain_vs_gru_sd=('perm_gain_over_gru_ft_pct','std'),
        geo_vs_perm_gain_mean=('geo_gain_over_perm_pct','mean'),
        geo_vs_perm_gain_sd=('geo_gain_over_perm_pct','std'),
    )
    scenario['rsps_alignment_advantage'] = scenario['geo_rsps_mean'] - scenario['perm_rsps_mean']
    scenario['rawcorr_alignment_advantage'] = scenario['geo_rawcorr_mean'] - scenario['perm_rawcorr_mean']
    scenario.to_csv(RESULTS_DIR / f'paper_scenario_summary_gs{args.graph_seed}.csv', index=False)

    per_ds = df.groupby(['data_seed','scenario_type','coupling','shared_shock_sigma'], as_index=False).agg(
        geo_rsps=('geo_rsps','first'),
        perm_rsps=('perm_rsps','first'),
        geo_rawcorr=('geo_raw_neighbor_corr','first'),
        perm_rawcorr=('perm_raw_neighbor_corr','first'),
        geo_vs_perm_gain=('geo_gain_over_perm_pct','mean'),
        geo_gain_vs_gru=('geo_gain_over_gru_ft_pct','mean'),
    )
    per_ds['rsps_alignment_advantage'] = per_ds['geo_rsps'] - per_ds['perm_rsps']
    per_ds['rawcorr_alignment_advantage'] = per_ds['geo_rawcorr'] - per_ds['perm_rawcorr']

    assoc_rows = []
    for ds, g in per_ds.groupby('data_seed'):
        assoc_rows.append({
            'data_seed': ds,
            'n_conditions': len(g),
            'pearson_geo_rsps_vs_geo_gain': g['geo_rsps'].corr(g['geo_gain_vs_gru']),
            'pearson_geo_rawcorr_vs_geo_gain': g['geo_rawcorr'].corr(g['geo_gain_vs_gru']),
            'pearson_geo_rsps_vs_geo_vs_perm_gain': g['geo_rsps'].corr(g['geo_vs_perm_gain']),
            'pearson_rsps_alignment_advantage_vs_geo_vs_perm_gain': g['rsps_alignment_advantage'].corr(g['geo_vs_perm_gain']),
            'pearson_rawcorr_alignment_advantage_vs_geo_vs_perm_gain': g['rawcorr_alignment_advantage'].corr(g['geo_vs_perm_gain']),
        })
    assoc = pd.DataFrame(assoc_rows)
    assoc.to_csv(RESULTS_DIR / f'paper_association_by_data_seed_gs{args.graph_seed}.csv', index=False)

    pooled = per_ds.groupby(['scenario_type','coupling','shared_shock_sigma'], as_index=False).mean(numeric_only=True)
    pooled_assoc = pd.DataFrame([{
        'n_conditions': len(pooled),
        'n_data_seeds': per_ds['data_seed'].nunique(),
        'pearson_geo_rsps_vs_geo_gain': pooled['geo_rsps'].corr(pooled['geo_gain_vs_gru']),
        'pearson_geo_rawcorr_vs_geo_gain': pooled['geo_rawcorr'].corr(pooled['geo_gain_vs_gru']),
        'pearson_geo_rsps_vs_geo_vs_perm_gain': pooled['geo_rsps'].corr(pooled['geo_vs_perm_gain']),
        'pearson_rsps_alignment_advantage_vs_geo_vs_perm_gain': pooled['rsps_alignment_advantage'].corr(pooled['geo_vs_perm_gain']),
        'pearson_rawcorr_alignment_advantage_vs_geo_vs_perm_gain': pooled['rawcorr_alignment_advantage'].corr(pooled['geo_vs_perm_gain']),
    }])
    pooled_assoc.to_csv(RESULTS_DIR / f'paper_association_pooled_gs{args.graph_seed}.csv', index=False)

    paper_table = scenario[[
        'scenario_type','coupling','shared_shock_sigma','n_runs',
        'geo_rawcorr_mean','geo_rsps_mean','gru_ft_mae_mean','geo_gnn_mae_mean','perm_gnn_mae_mean',
        'geo_gain_vs_gru_mean','geo_gain_vs_gru_sd','geo_vs_perm_gain_mean','geo_vs_perm_gain_sd'
    ]].copy()
    paper_table.to_csv(RESULTS_DIR / f'paper_main_table_gs{args.graph_seed}.csv', index=False)

    print(f'Aggregated {len(df)} runs from {len(files)} diagnostic files.')
    print(scenario.to_string(index=False))
    print('\nPer-data-seed associations:')
    print(assoc.to_string(index=False))
    print('\nPooled association:')
    print(pooled_assoc.to_string(index=False))


if __name__ == '__main__':
    main()
