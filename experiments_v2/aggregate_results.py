from pathlib import Path
import pandas as pd
from experiments_v2.config import RESULTS_DIR


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    diag_files = sorted(RESULTS_DIR.glob('novelty_diagnostic_ds*_ts*_full.csv'))
    if diag_files:
        d = pd.concat([pd.read_csv(p) for p in diag_files], ignore_index=True).drop_duplicates(
            subset=['scenario','data_seed','train_seed'], keep='last'
        )
        d.to_csv(RESULTS_DIR/'novelty_diagnostic_ALL.csv', index=False)
        group = d.groupby(['scenario_type','coupling','shared_shock_sigma']).agg(
            n=('gnn_mae_gain_pct','size'),
            rsps_mean=('rsps','mean'), rsps_std=('rsps','std'),
            raw_corr_mean=('raw_neighbor_corr','mean'), raw_corr_std=('raw_neighbor_corr','std'),
            gru_mae_mean=('gru_mae','mean'), gru_mae_std=('gru_mae','std'),
            gnn_mae_mean=('gnn_mae','mean'), gnn_mae_std=('gnn_mae','std'),
            gnn_gain_mean=('gnn_mae_gain_pct','mean'), gnn_gain_std=('gnn_mae_gain_pct','std'),
        ).reset_index()
        group.to_csv(RESULTS_DIR/'novelty_diagnostic_SUMMARY.csv', index=False)
        assoc = pd.DataFrame([{
            'n_rows': len(d),
            'pearson_rsps_vs_gnn_gain': d[['rsps','gnn_mae_gain_pct']].corr().iloc[0,1],
            'pearson_rawcorr_vs_gnn_gain': d[['raw_neighbor_corr','gnn_mae_gain_pct']].corr().iloc[0,1],
        }])
        assoc.to_csv(RESULTS_DIR/'novelty_association_ALL.csv', index=False)
        print('\nNovelty diagnostic summary:')
        print(group.to_string(index=False))
        print('\nAssociation summary:')
        print(assoc.to_string(index=False))

    graph_files = sorted(RESULTS_DIR.glob('novelty_graphs_ds*_ts*_*.csv'))
    if graph_files:
        g = pd.concat([pd.read_csv(p) for p in graph_files], ignore_index=True).drop_duplicates(
            subset=['label','scenario','data_seed','train_seed'], keep='last'
        )
        g.to_csv(RESULTS_DIR/'novelty_graphs_ALL.csv', index=False)
        gs = g.groupby('label').agg(
            n=('mae','size'), mae_mean=('mae','mean'), mae_std=('mae','std'),
            rmse_mean=('rmse','mean'), r2_mean=('r2','mean'), peak_mae_mean=('peak_mae','mean')
        ).reset_index()
        gs.to_csv(RESULTS_DIR/'novelty_graphs_SUMMARY.csv', index=False)
        print('\nGraph construction summary:')
        print(gs.to_string(index=False))


if __name__ == '__main__':
    main()
