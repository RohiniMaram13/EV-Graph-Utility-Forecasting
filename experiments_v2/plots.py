from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def make_novelty_plots(csv_path, out_dir):
    df = pd.read_csv(csv_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df['rsps'], df['gnn_mae_gain_pct'])
    ax.set_xlabel('Residual Spatial Predictability Score (RSPS)')
    ax.set_ylabel('GNN MAE improvement over GRU (%)')
    ax.set_title('Does residual spatial predictability anticipate GNN utility?')
    fig.tight_layout()
    fig.savefig(out / 'rsps_vs_gnn_gain.png', dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df['raw_neighbor_corr'], df['gnn_mae_gain_pct'])
    ax.set_xlabel('Raw neighbor correlation')
    ax.set_ylabel('GNN MAE improvement over GRU (%)')
    ax.set_title('Raw correlation vs GNN utility')
    fig.tight_layout()
    fig.savefig(out / 'rawcorr_vs_gnn_gain.png', dpi=220)
    plt.close(fig)
