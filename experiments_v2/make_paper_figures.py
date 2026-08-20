import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from experiments_v2.config import RESULTS_DIR


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph-seed', type=int, default=777)
    args = ap.parse_args()
    p = RESULTS_DIR / f'paper_scenario_summary_gs{args.graph_seed}.csv'
    if not p.exists():
        raise SystemExit(f'Missing {p}; run aggregate_paper_results first.')
    df = pd.read_csv(p)
    out = RESULTS_DIR / 'figures'
    out.mkdir(parents=True, exist_ok=True)

    true = df[df['scenario_type'] == 'true_coupling'].sort_values('coupling')
    false = df[df['scenario_type'] == 'false_signal'].sort_values('shared_shock_sigma')

    fig, ax = plt.subplots(figsize=(6.4,4.2))
    ax.errorbar(true['coupling'], true['geo_vs_perm_gain_mean'], yerr=true['geo_vs_perm_gain_sd'], marker='o', capsize=3)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel('Injected spatial coupling')
    ax.set_ylabel('Geographic vs permuted GNN gain (%)')
    ax.set_title('Graph-alignment advantage under true spatial coupling')
    save(fig, out / 'coupling_vs_graph_alignment_advantage.pdf')
    save(plt.figure(), out / '.placeholder.pdf') if False else None

    fig, ax = plt.subplots(figsize=(6.4,4.2))
    ax.errorbar(false['geo_rawcorr_mean'], false['geo_vs_perm_gain_mean'], yerr=false['geo_vs_perm_gain_sd'], marker='o', capsize=3)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel('Raw geographic-neighbor correlation')
    ax.set_ylabel('Geographic vs permuted GNN gain (%)')
    ax.set_title('False-signal conditions: correlation without graph value')
    save(fig, out / 'false_signal_rawcorr_vs_graph_advantage.pdf')

    fig, ax = plt.subplots(figsize=(6.4,4.2))
    ax.scatter(df['rsps_alignment_advantage'], df['geo_vs_perm_gain_mean'])
    for _, r in df.iterrows():
        label = f"c={r['coupling']:.2f}" if r['scenario_type']=='true_coupling' else f"shock={r['shared_shock_sigma']:.2f}"
        ax.annotate(label, (r['rsps_alignment_advantage'], r['geo_vs_perm_gain_mean']), fontsize=7, xytext=(3,3), textcoords='offset points')
    ax.axhline(0, linewidth=1)
    ax.set_xlabel('RSPS alignment advantage (geographic - permuted)')
    ax.set_ylabel('Geographic vs permuted GNN gain (%)')
    ax.set_title('Residual predictive alignment and graph-specific value')
    save(fig, out / 'rsps_alignment_vs_graph_advantage.pdf')

    fig, ax = plt.subplots(figsize=(6.4,4.2))
    ax.plot(true['coupling'], true['geo_gain_vs_gru_mean'], marker='o', label='Geographic GNN vs GRU-FT')
    ax.plot(true['coupling'], true['perm_gain_vs_gru_mean'], marker='s', label='Permuted GNN vs GRU-FT')
    ax.set_xlabel('Injected spatial coupling')
    ax.set_ylabel('Gain over matched GRU-FT (%)')
    ax.set_title('Aligned vs misaligned graph treatment')
    ax.legend()
    save(fig, out / 'geo_vs_permuted_gain.pdf')

    print(f'Figures saved to {out}')


if __name__ == '__main__':
    main()
