"""UrbanEV RSPS extension, using saved geographic neural-run artifacts.
Run from project root: python -u run_urbanev_rsps.py
Optional: --neural-run PATH --snapshot PATH
No neural retraining. Reuses the frozen 20 graph permutations. Mirrors the
original Caltech diagnostic: daily sums, training-only daily normalization,
weekly differences, pooled one-day-lag ridge with unpenalized intercept and
alpha=1. Fits training days 8..125; evaluates validation days 126..152.
This is a retrospective diagnostic extension, not a prospective screening test.
One new dataset cannot establish general cross-dataset predictive validity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd


def ridge(x, y):
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def calculate_rsps(residuals, adjacency, train_days=126, val_days=27):
    degree = adjacency.sum(axis=1)
    w = np.divide(adjacency, degree[:, None], out=np.zeros_like(adjacency, dtype=float),
                  where=degree[:, None] > 0)

    def samples(days):
        previous, target = residuals[days-8], residuals[days-7]
        own = previous.reshape(-1, 1)
        neighbor = (previous @ w.T).reshape(-1)
        return own, np.column_stack([own[:, 0], neighbor]), target.reshape(-1)

    self_tr, spatial_tr, ytr = samples(np.arange(8, train_days))
    self_va, spatial_va, yva = samples(np.arange(train_days, train_days+val_days))
    bself, bspatial = ridge(self_tr, ytr), ridge(spatial_tr, ytr)
    pself = np.column_stack([np.ones(len(self_va)), self_va]) @ bself
    pspatial = np.column_stack([np.ones(len(spatial_va)), spatial_va]) @ bspatial
    mse_self = float(np.mean((yva-pself)**2))
    mse_spatial = float(np.mean((yva-pspatial)**2))
    result = dict(rsps=(mse_self-mse_spatial)/(mse_self+1e-12), mse_self=mse_self,
                  mse_spatial=mse_spatial, neighbor_coef=float(bspatial[-1]),
                  n_train_samples=len(ytr), n_validation_samples=len(yva))
    return result, pself, pspatial, yva


def select_one(paths, description):
    paths = list(paths)
    if len(paths) != 1:
        raise RuntimeError(f'Expected one {description}; found {len(paths)}. Specify its path explicitly.')
    return paths[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--neural-run', type=Path)
    parser.add_argument('--snapshot', type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    run = args.neural_run or select_one((root/'revision_runs').glob('urbanev_full_*'), 'full UrbanEV neural run (--neural-run)')
    protocol = json.loads((run/'protocol.json').read_text())
    assert not protocol['smoke_only'] and protocol['n_permutations'] == 20
    assert protocol['train_seeds'] == [42, 52, 62]
    assert protocol['train_dates'] == ['2022-09-01 00:00:00', '2023-01-04 23:00:00']
    assert protocol['validation_dates'] == ['2023-01-05 00:00:00', '2023-01-31 23:00:00']
    manifest = json.loads((run/'download_manifest.json').read_text())
    assert protocol['source_commit'] == manifest['commit']
    if args.snapshot:
        snapshot = args.snapshot
    else:
        candidates = [p for p in (root/'revision_inputs').glob('urbanev_*')
                      if (p/'download_manifest.json').is_file()
                      and json.loads((p/'download_manifest.json').read_text())['commit'] == manifest['commit']]
        snapshot = select_one(candidates, 'matching snapshot (--snapshot)')
    volume = snapshot/'data/volume.csv'
    assert hashlib.sha256(volume.read_bytes()).hexdigest() == manifest['files']['data/volume.csv']['sha256']
    saved = np.load(run/'graph_and_scaler.npz', allow_pickle=False)
    adjacency = saved['adjacency'].astype(float)
    permutations = saved['permutations']
    frame = pd.read_csv(volume)
    times = pd.to_datetime(frame.pop('time'))
    assert list(frame.columns) == list(saved['node_ids'].astype(str))
    assert adjacency.shape == (275, 275) and permutations.shape == (20, 275)
    assert np.array_equal(adjacency, adjacency.T) and not np.diag(adjacency).any()
    assert int(adjacency.sum()/2) == protocol['edges']
    assert len(times) == 4344 and (times.diff().dropna() == pd.Timedelta(hours=1)).all()
    # Match the neural loader's float32 input quantization, then compute the
    # diagnostic in float64 as in the original Caltech diagnostic script.
    hourly = frame.to_numpy(dtype=np.float32).astype(np.float64)
    assert np.isfinite(hourly).all() and (hourly >= 0).all()
    # Test-period target values are excluded from diagnostic calculations.
    daily = hourly[:153*24].reshape(153, 24, 275).sum(axis=1)
    mean, std = daily[:126].mean(0), daily[:126].std(0)
    std[std < 1e-8] = 1
    z = (daily-mean)/std
    residuals = z[7:]-z[:-7]
    neural = pd.read_csv(run/'all_runs.csv')
    graph_neural = neural[neural.treatment.isin(['geographic', 'permuted'])].copy()
    assert len(graph_neural) == 63 and not graph_neural.duplicated(['seed','permutation_id']).any()
    for seed in [42, 52, 62]:
        group = graph_neural[graph_neural.seed == seed]
        assert set(group.permutation_id) == set(range(-1,20))
    parent = root/'revision_runs'
    output = Path(tempfile.mkdtemp(prefix='urbanev_rsps_', dir=parent))
    shutil.copy2(Path(__file__).resolve(), output/'run_urbanev_rsps.py')
    details = dict(neural_run=str(run.resolve()), snapshot=str(snapshot.resolve()),
                   source_commit=protocol['source_commit'], ridge_alpha=1.0,
                   intercept='unpenalized', aggregation='daily estimated-energy totals',
                   normalization='training-only daily per-node mean and standard deviation',
                   residual='weekly difference of standardized daily totals',
                   predictors='own residual at lag 1; spatial model adds row-normalized neighbor residual at lag 1',
                   fitting='pooled across nodes and training target days 8..125',
                   evaluation='validation target days 126..152 (27 days); no test-target fitting or scoring',
                   extension='retrospective; fixed original diagnostic, no tuning to observed neural results',
                   caveats=protocol['limitations'],
                   evidence_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in [volume, run/'all_runs.csv', run/'protocol.json', run/'graph_and_scaler.npz']})
    (output/'protocol.json').write_text(json.dumps(details, indent=2)+'\n')
    rows, predictions = [], []
    for index in range(-1,20):
        if index == -1:
            graph = adjacency
        else:
            permutation = permutations[index]
            assert sorted(permutation.tolist()) == list(range(275))
            graph = adjacency[np.ix_(permutation, permutation)]
            assert np.array_equal(np.sort(graph.sum(1)), np.sort(adjacency.sum(1)))
        score, self_pred, spatial_pred, target = calculate_rsps(residuals, graph)
        assert np.isfinite(list(score.values())).all()
        rows.append(dict(permutation_id=index, treatment='geographic' if index == -1 else 'permuted', **score))
        predictions.append(spatial_pred)
    scores = pd.DataFrame(rows)
    scores.to_csv(output/'rsps_by_graph.csv', index=False)
    np.savez_compressed(output/'validation_predictions.npz', spatial_predictions=np.stack(predictions),
                        self_prediction=self_pred, target=target, node_ids=saved['node_ids'],
                        permutation_ids=np.arange(-1,20), daily_mean=mean, daily_std=std)
    comparison = scores.merge(graph_neural.groupby('permutation_id', as_index=False).mae.mean()
                              .rename(columns={'mae':'neural_test_mae_mean'}), on='permutation_id', validate='one_to_one')
    comparison.to_csv(output/'rsps_neural_comparison.csv', index=False)
    actual = float(scores.loc[scores.permutation_id == -1, 'rsps'].iloc[0])
    controls = scores.loc[scores.permutation_id >= 0, 'rsps']
    neural_seeds = pd.read_csv(run/'seed_summary.csv')
    association = comparison.rsps.corr(comparison.neural_test_mae_mean)
    report = (f'URBANEV RSPS (validation, 27 days; 275 zones)\n'
              f'Aligned RSPS: {actual:.8f}\n'
              f'Mean shuffled RSPS: {controls.mean():.8f}\n'
              f'RSPS alignment advantage: {actual-controls.mean():+.8f} (raw score difference)\n'
              f'Aligned RSPS exceeds {int((actual>controls).sum())}/20 shuffled scores\n'
              f'Neural test MAE alignment gain: {neural_seeds.alignment_gain_pct.mean():+.6f}% '
              f'+/- {neural_seeds.alignment_gain_pct.std(ddof=1):.6f}% across 3 seeds\n'
              f'Exploratory Pearson r, RSPS versus mean neural test MAE across 21 graphs: {association:.6f}\n'
              'For that correlation, negative means higher RSPS accompanies lower neural error.\n'
              'Graphs share data and topology: no independent-sample significance claim.\n'
              'RSPS is dimensionless validation MSE reduction; neural gain is test MAE percent.\n'
              'A single geographic case cannot establish general tracking across datasets.\n')
    (output/'summary.txt').write_text(report)
    print(report + f'Saved to: {output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
