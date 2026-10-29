"""Geographic graph validation on published UrbanEV estimated charging volume.
First: python -u run_urbanev.py --smoke
Full:  python -u run_urbanev.py
Optional --snapshot PATH selects a downloaded snapshot if several exist.

Fixed protocol: all 275 published zones, 181 days; first 126 days train,
next 27 validation, last 28 test. History 168 hours; forecast 24; daily stride.
Graph: binary symmetric union of each zone's five nearest other zones in the
published distance matrix. No load-derived edges or test-based node selection.
Three seeds and 20 topology-preserving permutations (fixed RNG seed 20261029).
Published preprocessing (including forward/back filling) cannot be undone here;
this is not an independently leakage-audited raw dataset. Volume is estimated
energy, not metered energy. Row order in the unlabeled-row distance matrix is
interpreted as its published column order, checked for symmetry/zero diagonal.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import torch


def prepare(snapshot):
    manifest = json.loads((snapshot / 'download_manifest.json').read_text())
    for relative, item in manifest['files'].items():
        actual = hashlib.sha256((snapshot / relative).read_bytes()).hexdigest()
        assert actual == item['sha256'], f'Downloaded file changed: {relative}'
    frame = pd.read_csv(snapshot / 'data/volume.csv')
    times = pd.to_datetime(frame.pop('time'), errors='raise')
    ids = list(frame.columns)
    assert len(ids) == len(set(ids)) == 275 and len(times) == 4344
    assert times.iloc[0] == pd.Timestamp('2022-09-01')
    assert times.iloc[-1] == pd.Timestamp('2023-02-28 23:00:00')
    assert (times.diff().dropna() == pd.Timedelta(hours=1)).all()
    energy = frame.to_numpy(dtype=np.float32)
    assert np.isfinite(energy).all() and (energy >= 0).all()
    distance_frame = pd.read_csv(snapshot / 'data/distance.csv')
    # These source files have column labels but no row-ID column. Refuse to
    # silently reorder either file: the source order must already match.
    assert list(distance_frame.columns) == ids, 'Distance/target node orders differ'
    distances = distance_frame.to_numpy(dtype=np.float64)
    assert distances.shape == (275, 275) and np.isfinite(distances).all()
    assert (distances >= 0).all() and np.allclose(distances, distances.T, atol=1e-6)
    assert np.allclose(np.diag(distances), 0, atol=1e-6)
    info = pd.read_csv(snapshot / 'data/inf.csv')
    assert set(info.TAZID.astype(int).astype(str)) == set(ids), 'Metadata zone IDs differ'
    nearest_distances = distances.copy()
    np.fill_diagonal(nearest_distances, np.inf)
    neighbors = np.argsort(nearest_distances, axis=1, kind='stable')[:, :5]
    adjacency = np.zeros((275, 275), dtype=np.float32)
    adjacency[np.arange(275)[:, None], neighbors] = 1
    adjacency = np.maximum(adjacency, adjacency.T)
    assert np.array_equal(adjacency, adjacency.T) and not np.diag(adjacency).any()
    assert (adjacency.sum(1) >= 5).all()
    mean, std = energy[:126*24].mean(0), energy[:126*24].std(0)
    std[std < 1e-8] = 1.0
    normalized = (energy - mean) / std

    def windows(days):
        x = np.stack([normalized[d*24-168:d*24].T for d in days])
        y = np.stack([normalized[d*24:(d+1)*24].T for d in days])
        return torch.from_numpy(x), torch.from_numpy(y)

    data = {'train': windows(range(7, 126)), 'validation': windows(range(126, 153)),
            'test': windows(range(153, 181)), 'scaler': {'mean': mean[:, None], 'std': std[:, None]},
            'adjacency': adjacency, 'ids': ids, 'energy': energy}
    for split, size in [('train', 119), ('validation', 27), ('test', 28)]:
        assert data[split][0].shape == (size, 275, 168)
        assert data[split][1].shape == (size, 275, 24)
    details = {'source_commit': manifest['commit'], 'target': 'volume.csv: estimated energy (kWh)',
               'train_dates': [str(times.iloc[0]), str(times.iloc[126*24-1])],
               'validation_dates': [str(times.iloc[126*24]), str(times.iloc[153*24-1])],
               'test_dates': [str(times.iloc[153*24]), str(times.iloc[-1])],
               'time_zone': 'timestamps kept as published; not converted to UTC',
               'lookback': 168, 'horizon': 24, 'stride': 24, 'nodes': 275,
               'graph': 'binary symmetric union of 5 nearest neighbors in published distance matrix',
               'tie_rule': 'stable published node order', 'edges': int(adjacency.sum()/2),
               'degree_min': float(adjacency.sum(1).min()), 'degree_max': float(adjacency.sum(1).max()),
               'normalization': 'per-node mean/std fitted on first 126 days only',
               'limitations': ['Published volume is estimated, not metered.',
                              'Upstream cleaning and bidirectional imputation predate our split.',
                              'Distance rows have no IDs; their order is assumed to match the published column order.',
                              'Only 28 daily test windows; no causal or broad geographic generalization.']}
    return data, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    root, source = Path.cwd().resolve(), Path(__file__).resolve()
    assert (root / 'real_validation/run_real_graph_validation.py').is_file(), 'Run from project root'
    choices = sorted((root / 'revision_inputs').glob('urbanev_*'))
    if args.snapshot:
        snapshot = args.snapshot.resolve()
    elif len(choices) == 1:
        snapshot = choices[0]
    else:
        raise RuntimeError('Use --snapshot PATH to identify the downloaded UrbanEV folder')
    data, details = prepare(snapshot)
    sys.path.insert(0, str(root))
    from real_validation.run_real_graph_validation import train_temporal_controls, adjacency_to_graph
    from experiments_v2.run_full_graph_control import train_gnn_branch
    from experiments_v2.metrics import metrics
    from experiments_v2.config import LR, WEIGHT_DECAY, HIDDEN_DIM, DROPOUT, PATIENCE
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; activate the tested environment')
    seeds = [42] if args.smoke else [42, 52, 62]
    nperm = 1 if args.smoke else 20
    rng = np.random.default_rng(20261029)
    permutations = [rng.permutation(275) for _ in range(20)]
    runtime = {'device': torch.device('cuda'), 'batch_size': 4, 'amp': True, 'verbose': True,
               'gru_epochs': 2 if args.smoke else 100, 'graph_epochs': 2 if args.smoke else 30,
               'finetune_epochs': 2 if args.smoke else 70, 'patience': PATIENCE,
               'graph_patience': PATIENCE, 'graph_lr': 0.001, 'finetune_lr': 0.0005}
    parent = root / 'revision_runs'
    parent.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='urbanev_SMOKE_' if args.smoke else 'urbanev_full_', dir=parent))
    shutil.copy2(source, output / 'run_urbanev.py')
    shutil.copy2(snapshot / 'download_manifest.json', output / 'download_manifest.json')
    code_paths = ['real_validation/run_real_graph_validation.py', 'experiments_v2/run_full_graph_control.py',
                  'experiments_v2/models.py', 'experiments_v2/train_eval.py', 'experiments_v2/config.py',
                  'experiments_v2/metrics.py', 'experiments_v2/device_utils.py']
    hashes = {}
    for name in code_paths:
        destination = output / 'source_snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, destination)
        hashes[name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    details.update(smoke_only=args.smoke, train_seeds=seeds, n_permutations=nperm,
                   permutation_seed=20261029, runtime={**runtime, 'device': 'cuda'},
                   baseline_lr=LR, weight_decay=WEIGHT_DECAY, hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
                   torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                   source_sha256=hashes,
                   metric_note='MAE/RMSE on unscaled estimated kWh. Peak-hours labels use existing code hour indices; not a local tariff definition.')
    # Record choices before any training or test metrics are evaluated.
    (output / 'protocol.json').write_text(json.dumps(details, indent=2) + '\n')
    np.savez_compressed(output / 'graph_and_scaler.npz', adjacency=data['adjacency'],
                        node_ids=data['ids'], permutations=np.stack(permutations), **data['scaler'])
    print(json.dumps(details, indent=2), flush=True)
    print(f'\nOUTPUT: {output}\nSMOKE ONLY: {args.smoke}', flush=True)
    x, y = (t.numpy() for t in data['test'])
    mean, std = data['scaler']['mean'], data['scaler']['std']
    prediction, target = x[:, :, :24]*std+mean, y*std+mean
    raw = np.stack([data['energy'][d*24-168:d*24-144].T for d in range(153, 181)])
    np.testing.assert_allclose(prediction, raw, rtol=1e-5, atol=1e-3)
    baseline = metrics(prediction, target)
    pd.DataFrame([baseline]).to_csv(output / 'seasonal_naive_metrics.csv', index=False)
    np.savez_compressed(output / 'seasonal_naive_predictions.npz', prediction=prediction, target=target)
    print('\nWEEKLY BASELINE:', baseline, flush=True)
    rows, summaries = [], []
    start = time.perf_counter()
    for seed in seeds:
        print(f'\nTRAINING SEED {seed}', flush=True)
        temporal = train_temporal_controls(data['train'], data['validation'], data['test'],
                                            data['scaler'], seed, runtime)
        torch.save(temporal['shared_state'], output / f'gru_warmstart_seed_{seed}.pt')
        for label, key in [('gru', 'baseline'), ('gru_ft', 'gru_ft')]:
            rows.append({'seed': seed, 'treatment': label, 'permutation_id': -1, **temporal[key]})
        outcomes = []
        treatments = [(-1, data['adjacency'])] + [(i, data['adjacency'][np.ix_(p, p)])
                                                for i, p in enumerate(permutations[:nperm])]
        for index, adjacency in treatments:
            assert np.array_equal(np.sort(adjacency.sum(1)), np.sort(data['adjacency'].sum(1)))
            assert adjacency.sum() == data['adjacency'].sum()
            label = 'geographic' if index == -1 else 'permuted'
            print(f'\nSEED {seed}: {label} {index}', flush=True)
            result = train_gnn_branch(temporal['shared_state'], data['train'], data['validation'],
                                     data['test'], data['scaler'], adjacency_to_graph(adjacency),
                                     seed, runtime, f'{label}-{index}')
            outcomes.append(result)
            rows.append({'seed': seed, 'treatment': label, 'permutation_id': index, **result})
            pd.DataFrame(rows).to_csv(output / 'all_runs.csv', index=False)
        real, perm = outcomes[0], np.array([v['mae'] for v in outcomes[1:]])
        ft = temporal['gru_ft']['mae']
        summaries.append(dict(seed=seed, gru_ft_mae=ft, geographic_mae=real['mae'],
                              geographic_rmse=real['rmse'], perm_mae_mean=perm.mean(),
                              gain_over_gru_ft_pct=100*(ft-real['mae'])/ft,
                              alignment_gain_pct=100*(perm.mean()-real['mae'])/perm.mean(),
                              permutations_beaten_pct=100*(perm>real['mae']).mean()))
        pd.DataFrame(summaries).to_csv(output / 'seed_summary.csv', index=False)
        print(pd.DataFrame([summaries[-1]]).to_string(index=False), flush=True)
    summary = pd.DataFrame(summaries).drop(columns='seed').agg(['mean', 'std'])
    summary.to_csv(output / 'summary_metrics.csv', index_label='statistic')
    text = ('SMOKE ONLY - NOT PAPER RESULTS\n' if args.smoke else 'FULL URBANEV GEOGRAPHIC RESULTS\n')
    text += summary.to_string() + f'\nTotal minutes: {(time.perf_counter()-start)/60:.2f}\n'
    (output / 'summary.txt').write_text(text)
    print('\n' + text + f'Saved to: {output}', flush=True)


if __name__ == '__main__':
    main()
