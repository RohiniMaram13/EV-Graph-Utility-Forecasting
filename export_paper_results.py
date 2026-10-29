"""Run on Ubuntu from the forecasting project: python -u export_paper_results.py
Creates paper figures and a ZIP of small final-result artifacts. No training.
This is a paper handoff package, NOT a complete raw-data/checkpoint backup.
"""
from pathlib import Path
import hashlib
import json
import shutil
import tempfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def select(root, prefix):
    candidates = sorted(p for p in (root/'revision_runs').glob(prefix+'*') if p.is_dir())
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one completed {prefix} run; found {len(candidates)}. '
                           'Do not delete any runs; send this message so we can select the intended run.')
    return candidates[0]


def main():
    root = Path.cwd().resolve()
    runs = {name: select(root, prefix) for name, prefix in {
        'seasonal_naive':'seasonal_naive_', 'dlinear':'dlinear_',
        'caltech_five_seeds':'caltech_extra_seeds_', 'stgcn':'stgcn_full_',
        'urbanev':'urbanev_full_', 'urbanev_rsps':'urbanev_rsps_'}.items()}
    seasonal = pd.read_csv(runs['seasonal_naive']/'metrics.csv')
    dlinear = pd.read_csv(runs['dlinear']/'seed_metrics.csv')
    caltech = pd.read_csv(runs['caltech_five_seeds']/'combined_five_seed_metrics.csv')
    stgcn = pd.read_csv(runs['stgcn']/'all_runs.csv')
    urbanev = pd.read_csv(runs['urbanev']/'all_runs.csv')
    urban_baseline = pd.read_csv(runs['urbanev']/'seasonal_naive_metrics.csv')
    st_seeds = pd.read_csv(runs['stgcn']/'seed_summary.csv')
    ur_seeds = pd.read_csv(runs['urbanev']/'seed_summary.csv')
    rsps = pd.read_csv(runs['urbanev_rsps']/'rsps_neural_comparison.csv')
    assert len(caltech) == 5 and set(caltech.train_seed) == {42,52,62,72,82}
    assert len(dlinear) == 3 and set(dlinear.seed) == {42,52,62}
    assert len(stgcn) == 63 and len(rsps) == 21
    assert len(ur_seeds) == len(st_seeds) == 3
    for name in ['stgcn','urbanev']:
        frame = stgcn if name == 'stgcn' else urbanev
        graph = frame[frame.treatment.isin(['aligned','geographic','permuted'])]
        assert len(graph) == 63 and not graph.duplicated(['seed','permutation_id']).any()
        for seed in [42,52,62]:
            assert set(graph.loc[graph.seed==seed,'permutation_id']) == set(range(-1,20))
    parent = root/'paper_exports'
    parent.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='paper_revision_', dir=parent))
    figures = output/'figures'
    figures.mkdir()
    rows = []

    def add(dataset, model, frame, mae='mae', rmse='rmse', seeded=True):
        assert np.isfinite(frame[[mae,rmse]].to_numpy()).all()
        rows.append(dict(dataset=dataset, model=model, n_seeds=len(frame) if seeded else 0,
                         mae_mean=frame[mae].mean(), mae_std=frame[mae].std(ddof=1) if seeded else np.nan,
                         rmse_mean=frame[rmse].mean(), rmse_std=frame[rmse].std(ddof=1) if seeded else np.nan))

    add('Caltech Pod-16','Weekly naive',seasonal,seeded=False)
    add('Caltech Pod-16','DLinear',dlinear)
    add('Caltech Pod-16','GRU',caltech,'gru_baseline_mae','gru_baseline_rmse')
    add('Caltech Pod-16','GRU-FT',caltech,'gru_ft_mae','gru_ft_rmse')
    add('Caltech Pod-16','Pod GNN',caltech,'real_gnn_mae','real_gnn_rmse')
    add('Caltech Pod-16','Adapted STGCN',stgcn[stgcn.treatment=='aligned'])
    add('UrbanEV','Weekly naive',urban_baseline,seeded=False)
    for code,label in [('gru','GRU'),('gru_ft','GRU-FT'),('geographic','Geographic GNN')]:
        add('UrbanEV',label,urbanev[urbanev.treatment==code])
    table = pd.DataFrame(rows)
    table.to_csv(output/'forecast_comparison.csv',index=False)
    plt.rcParams.update({'font.size':10, 'pdf.fonttype':42, 'ps.fonttype':42})

    def save(fig, name):
        for extension in ['pdf','png','svg']:
            fig.savefig(figures/f'{name}.{extension}',dpi=220,bbox_inches='tight')
        plt.close(fig)

    for dataset, filename in [('Caltech Pod-16','caltech_forecasting'),('UrbanEV','urbanev_forecasting')]:
        frame = table[table.dataset==dataset]
        fig, axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
        labels = [f"{r.model}\n" + (f"{r.n_seeds} seeds" if r.n_seeds else 'deterministic')
                  for r in frame.itertuples()]
        for ax,metric in zip(axes,['mae','rmse']):
            ax.bar(np.arange(len(frame)),frame[metric+'_mean'],
                   yerr=frame[metric+'_std'].fillna(0),capsize=3,color='#4477AA')
            ax.set_xticks(np.arange(len(frame)),labels,rotation=35,ha='right')
            ax.set_ylabel(metric.upper()+' (kWh)')
            ax.set_ylim(bottom=0)
            ax.grid(axis='y',alpha=.2)
            ax.set_axisbelow(True)
        save(fig,filename)

    alignment = []
    for label,frame,seedcol,gaincol in [
        ('Caltech: original GNN',caltech,'train_seed','real_vs_mean_perm_gain_pct'),
        ('Caltech: adapted STGCN',st_seeds,'seed','alignment_gain_pct'),
        ('UrbanEV: geographic GNN',ur_seeds,'seed','alignment_gain_pct')]:
        for _,row in frame.iterrows():
            alignment.append(dict(experiment=label,seed=int(row[seedcol]),gain_pct=row[gaincol]))
    align = pd.DataFrame(alignment)
    align.to_csv(output/'alignment_by_seed.csv',index=False)
    fig,axes = plt.subplots(1,3,figsize=(10,3.5),layout='constrained')
    for ax,(label,frame) in zip(axes,align.groupby('experiment',sort=False)):
        values = frame.gain_pct.to_numpy()
        ax.scatter(frame.seed,values,color='#4477AA',s=35)
        ax.axhline(0,color='black',linestyle='--',linewidth=.8)
        ax.axhline(values.mean(),color='#CC6677',linewidth=1,label='Seed mean')
        ax.set_xticks(frame.seed)
        ax.set_xlabel('Training seed')
        ax.set_title(label,fontsize=10)
        ax.set_ylabel('Alignment gain (%)')
        ax.legend(fontsize=8)
    save(fig,'alignment_by_seed')

    fig,ax = plt.subplots(figsize=(5,3.8),layout='constrained')
    for mask,label,color,marker in [(rsps.permutation_id>=0,'Shuffled','#4477AA','o'),
                                   (rsps.permutation_id==-1,'Geographic','#CC6677','D')]:
        ax.scatter(rsps.loc[mask,'rsps'],rsps.loc[mask,'neural_test_mae_mean'],
                   label=label,color=color,marker=marker,s=38)
    ax.set_xlabel('Validation RSPS')
    ax.set_ylabel('Mean neural test MAE (estimated kWh)')
    ax.legend()
    save(fig,'urbanev_rsps_neural')

    # Bundle small exact evidence; preserve original run directory names inside
    # named groups. Raw data, prediction tensors and checkpoints remain on server.
    for label,folder in runs.items():
        for path in folder.rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            if path.suffix.lower() not in {'.csv','.json','.txt','.py','.md'} and path.name != 'graph_and_scaler.npz':
                continue
            destination = output/'evidence'/label/path.relative_to(folder)
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,destination)
    original = root/'results_real/real_gnn_seed_summary.csv'
    shutil.copy2(original,output/'original_caltech_three_seeds.csv')
    shutil.copy2(Path(__file__).resolve(),output/'export_paper_results.py')
    notes = '''PAPER HANDOFF PACKAGE — new revision experiments only
Figures are generated from full-precision CSVs, not rounded console output.
FORECAST FIGURES: bars are means; error bars are sample SD across training seeds,
not confidence intervals. Weekly naive is deterministic. Seed counts are labeled.
Caltech original-model five-seed results combine original and added runs; retain
the original three-seed table separately and disclose execution environments.
UrbanEV volume is estimated energy. Caltech hourly energy and UrbanEV regional
volume have different scales and should not be pooled into one error metric.
ALIGNMENT FIGURE: positive = lower MAE for aligned graph versus mean shuffled
graph within each seed. Panels have different y-axis scales to expose small
UrbanEV effects; inspect axis labels and disclose this in the paper caption.
RSPS SCATTER: descriptive only; points share observations and topology. RSPS
uses validation residual MSE; neural MAE uses test hourly forecasts. No fit line
or significance claim is included. This is a retrospective extension.
Captions belong in LaTeX; full captions have not been embedded in graphics.

This package contains result tables, settings, small evidence files, source
copies, and new figures. It excludes raw/input data, large prediction arrays,
model checkpoints, and the original Washington figures. It is NOT a full backup.
Keep a separate copy of the complete repository and revision_runs/revision_inputs.
The manuscript has not yet been edited by this export script.
'''
    (output/'READ_ME_FIRST.txt').write_text(notes)
    manifest = {str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in output.rglob('*') if p.is_file()}
    (output/'sha256_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive = shutil.make_archive(str(output),'zip',root_dir=output.parent,base_dir=output.name)
    print('\nEXPORT COMPLETE\n'+table.to_string(index=False),flush=True)
    print(f'\nFigures and tables: {output}\nTRANSFER THIS ZIP: {archive}',flush=True)


if __name__ == '__main__':
    main()
