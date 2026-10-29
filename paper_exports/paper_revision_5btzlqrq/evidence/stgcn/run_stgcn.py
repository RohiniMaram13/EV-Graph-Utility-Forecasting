"""STGCN (Chebyshev) adapted for Caltech 168-hour -> direct 24-hour forecasts.

From the repository root, first run: python -u run_stgcn.py --smoke
Then for reportable runs: python -u run_stgcn.py

Architecture reference: https://github.com/VeritasYin/STGCN_IJCAI-18
Two temporal/spatial/temporal blocks, channels [1,32,64], [64,32,128],
Kt=3, Ks=3, node/channel layer normalization, dropout 0.3. Output temporal
convolution spans all remaining history. Unlike the original one-step head,
the final projection produces 24 horizons directly. PyTorch implementation,
Xavier initialization, and the existing project's Adam/MSE training protocol;
this is an explicitly adapted baseline, not an exact original-paper replication.
Every aligned/shuffled treatment within a seed has identical initial parameters
and the same training budget. There is no GRU warm start in this architecture.
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
from torch import nn
from torch.nn import functional as F


def graph_supports(adjacency):
    a = np.asarray(adjacency, dtype=np.float64)
    assert a.shape == (16, 16) and np.allclose(a, a.T)
    assert np.all(a >= 0) and np.allclose(np.diag(a), 0)
    degree = a.sum(1)
    assert np.all(degree > 0)
    laplacian = np.eye(16) - a / np.sqrt(degree[:, None] * degree[None, :])
    largest = np.linalg.eigvalsh(laplacian)[-1]
    scaled = 2 * laplacian / largest - np.eye(16)
    return torch.tensor(np.stack([np.eye(16), scaled, 2 * scaled @ scaled - np.eye(16)]),
                        dtype=torch.float32)


class Align(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.cout = cout
        self.projection = nn.Conv2d(cin, cout, 1, bias=False) if cin > cout else nn.Identity()

    def forward(self, x):
        x = self.projection(x)
        if x.shape[1] < self.cout:
            x = torch.cat([x, x.new_zeros(x.shape[0], self.cout-x.shape[1], *x.shape[2:])], dim=1)
        return x


class Temporal(nn.Module):
    def __init__(self, cin, cout, width, gated):
        super().__init__()
        self.width, self.gated = width, gated
        self.align = Align(cin, cout)
        self.conv = nn.Conv2d(cin, cout * (2 if gated else 1), (1, width))

    def forward(self, x):
        residual = self.align(x)[..., self.width-1:]
        value = self.conv(x)
        if self.gated:
            value, gate = value.chunk(2, dim=1)
            return (value + residual) * torch.sigmoid(gate)
        return torch.relu(value + residual)


class Spatial(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mix = nn.Conv2d(3 * channels, channels, 1)

    def forward(self, x, supports):
        # All three polynomial orders are mixed with independent channel weights.
        filtered = torch.einsum('knm,bcmt->bkcnt', supports, x)
        b, k, c, n, t = filtered.shape
        return torch.relu(self.mix(filtered.reshape(b, k*c, n, t)) + x)


class NodeChannelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm((16, channels), eps=1e-6)

    def forward(self, x):
        return self.norm(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1).contiguous()


class Block(nn.Module):
    def __init__(self, cin, middle, cout):
        super().__init__()
        self.first = Temporal(cin, middle, 3, True)
        self.spatial = Spatial(middle)
        self.last = Temporal(middle, cout, 3, False)
        self.norm = NodeChannelNorm(cout)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, supports):
        return self.dropout(self.norm(self.last(self.spatial(self.first(x), supports))))


class STGCN(nn.Module):
    def __init__(self, adjacency):
        super().__init__()
        # Graph supports are not learned or copied with initialization weights.
        self.register_buffer('supports', graph_supports(adjacency), persistent=False)
        self.first = Block(1, 32, 64)
        self.second = Block(64, 32, 128)
        self.output_temporal = Temporal(128, 128, 160, True)
        self.output_norm = NodeChannelNorm(128)
        self.output_hidden = nn.Conv2d(128, 128, 1)
        self.head = nn.Conv2d(128, 24, 1, bias=False)
        self.node_bias = nn.Parameter(torch.zeros(1, 24, 16, 1))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, x_static=None, edge_index=None, edge_weight=None):
        x = self.first(x.unsqueeze(1), self.supports)
        x = self.second(x, self.supports)
        x = self.output_norm(self.output_temporal(x))
        x = self.head(torch.sigmoid(self.output_hidden(x))) + self.node_bias
        return x.squeeze(-1).transpose(1, 2)


def self_check(adjacency, permutation, device):
    # Test the mathematical permutation identity in double precision. Float32
    # CUDA convolution/reduction rounding can depend on node order, especially
    # through the long output convolution. Actual smoke/full training below
    # still uses the original float32/AMP protocol.
    dtype = torch.float64
    model = STGCN(adjacency).to(device=device, dtype=dtype).eval()
    shuffled = STGCN(adjacency[np.ix_(permutation, permutation)]).to(device=device, dtype=dtype).eval()
    shuffled.load_state_dict(model.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), shuffled.parameters()))
    p = torch.as_tensor(permutation, device=device)
    torch.testing.assert_close(shuffled.supports, model.supports[:, p][:, :, p])
    # Independent polynomial-filter multiplication check.
    x = torch.randn(2, 3, 16, 5, device=device, dtype=dtype)
    fast = torch.einsum('knm,bcmt->bkcnt', model.supports, x)
    slow = torch.stack([torch.stack([s @ x[b, c] for c in range(3)])
                        for b in range(2) for s in model.supports]).reshape(2, 3, 3, 16, 5)
    torch.testing.assert_close(fast, slow, rtol=1e-9, atol=1e-10)
    history = torch.randn(2, 16, 168, device=device, dtype=dtype)
    prediction = model(history)
    assert prediction.shape == (2, 16, 24) and torch.isfinite(prediction).all()
    # At initialization the node-affine parameters are uniform: simultaneous
    # permutation of input nodes and graph must only permute the output nodes.
    with torch.no_grad():
        permuted_prediction = shuffled(history[:, p])
        difference = (permuted_prediction - prediction[:, p]).abs().max().item()
        print(f'Double-precision permutation check: maximum difference {difference:.3e}', flush=True)
        torch.testing.assert_close(permuted_prediction, prediction[:, p], rtol=1e-9, atol=1e-10)
        changed = shuffled(history)
        assert (changed - prediction).abs().max().item() > 1e-6, 'Graph change has no effect'
    prediction.square().mean().backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
    print('STGCN double-precision graph, shape, shared-initialization and backward checks PASSED', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--smoke', action='store_true', help='Two epochs, one seed, one permutation; NOT paper results')
    args = parser.parse_args()
    root, source = args.repo.resolve(), Path(__file__).resolve()
    files = ['real_validation/run_real_graph_validation.py', 'experiments_v2/train_eval.py',
             'experiments_v2/config.py', 'experiments_v2/metrics.py', 'experiments_v2/device_utils.py',
             'data/real_acn/caltech_hourly_energy.npz', 'data/real_acn/caltech_pod_adjacency.npy',
             'results_real/real_graph_permutations.json']
    for name in files:
        if not (root / name).is_file():
            raise FileNotFoundError(f'Missing {root / name}; run from the repository root')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; activate the tested environment')
    os.chdir(root)
    sys.path.insert(0, str(root))
    from real_validation.run_real_graph_validation import load_real_pod_dataset
    from experiments_v2.train_eval import train_model, predict, set_seed
    from experiments_v2.metrics import metrics
    from experiments_v2.config import WEIGHT_DECAY

    data = load_real_pod_dataset()
    assert tuple(data['train'][0].shape) == (462, 16, 168)
    assert tuple(data['validation'][0].shape) == (100, 16, 168)
    assert tuple(data['test'][0].shape) == (101, 16, 168)
    adjacency = data['adjacency']
    controls = json.loads((root / files[-1]).read_text())
    assert len(controls) == 20
    for control in controls:
        assert sorted(control['pod16']) == list(range(16))
    device = torch.device('cuda')
    set_seed(0)
    self_check(adjacency, np.array(controls[0]['pod16']), device)
    seeds, epochs = ([42], 2) if args.smoke else ([42, 52, 62], 100)
    controls = controls[:1] if args.smoke else controls
    parent = root / 'revision_runs'
    parent.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='stgcn_SMOKE_' if args.smoke else 'stgcn_full_', dir=parent))
    shutil.copy2(source, output / 'run_stgcn.py')
    settings = dict(model='STGCN-Cheb, direct 24-hour adaptation', smoke_only=args.smoke,
                    seeds=seeds, max_epochs=epochs, patience=15, batch_size=4, lr=0.001,
                    weight_decay=WEIGHT_DECAY, amp=True, dropout=0.3, temporal_kernel=3,
                    chebyshev_terms=3, channels=[[1,32,64],[64,32,128]],
                    initialization='Xavier uniform, equal learned parameters within each seed',
                    selection='minimum validation normalized MSE', loss='normalized MSE',
                    controls=controls, torch=str(torch.__version__), cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0),
                    source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files})
    (output / 'run_details.json').write_text(json.dumps(settings, indent=2) + '\n')
    print(f'\nOUTPUT: {output}\nSMOKE ONLY: {args.smoke}', flush=True)
    rows, summaries = [], []
    start = time.perf_counter()
    for seed in seeds:
        set_seed(seed)
        reference = STGCN(adjacency)
        initial = {key: value.detach().clone() for key, value in reference.state_dict().items()}
        print(f'Parameters: {sum(v.numel() for v in reference.parameters()):,}', flush=True)
        del reference
        treatments = [('aligned', -1, adjacency)] + [
            ('permuted', c['permutation_id'], adjacency[np.ix_(c['pod16'], c['pod16'])]) for c in controls]
        for treatment, permutation_id, graph in treatments:
            set_seed(seed)
            model = STGCN(graph)
            model.load_state_dict(initial)
            assert all(torch.equal(value, initial[key]) for key, value in model.state_dict().items())
            print(f'\nSEED {seed}: {treatment} {permutation_id}', flush=True)
            tick = time.perf_counter()
            model, best, trained_epochs = train_model(
                model, data['train'], data['validation'], seed=seed, verbose=True,
                device=device, batch_size=4, max_epochs=epochs, patience=15, amp=True, lr=0.001)
            prediction, target = predict(model, data['test'], data['scaler'], device=device, batch_size=4, amp=True)
            assert np.isfinite(prediction).all() and np.isfinite(target).all()
            row = dict(seed=seed, treatment=treatment, permutation_id=permutation_id,
                       best_validation_mse=best, epochs_run=trained_epochs,
                       runtime_minutes=(time.perf_counter()-tick)/60, **metrics(prediction, target))
            rows.append(row)
            pd.DataFrame(rows).to_csv(output / 'all_runs.csv', index=False)
            name = f'seed_{seed}_{treatment}_{permutation_id}'
            np.savez_compressed(output / f'{name}.npz', prediction=prediction, target=target,
                                station_ids=data['station_ids'], adjacency=graph)
            torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, output / f'{name}.pt')
            print(pd.DataFrame([row]).to_string(index=False), flush=True)
            del model
        seed_rows = pd.DataFrame([row for row in rows if row['seed'] == seed])
        real = seed_rows[seed_rows.treatment == 'aligned'].iloc[0]
        perm = seed_rows[seed_rows.treatment == 'permuted']
        summaries.append(dict(seed=seed, aligned_mae=real.mae, aligned_rmse=real.rmse,
                              perm_mae_mean=perm.mae.mean(), perm_mae_std=perm.mae.std(ddof=1),
                              alignment_gain_pct=100 * (perm.mae.mean()-real.mae)/perm.mae.mean(),
                              permutations_beaten_pct=100 * (perm.mae > real.mae).mean()))
        pd.DataFrame(summaries).to_csv(output / 'seed_summary.csv', index=False)
    summary = pd.DataFrame(summaries).drop(columns='seed').agg(['mean', 'std'])
    summary.to_csv(output / 'summary_metrics.csv', index_label='statistic')
    report = ('SMOKE TEST ONLY - DO NOT REPORT AS EXPERIMENT RESULTS\n' if args.smoke else 'FULL STGCN RESULTS\n')
    report += summary.to_string() + f'\nTotal minutes: {(time.perf_counter()-start)/60:.2f}\n'
    (output / 'summary.txt').write_text(report)
    print('\n' + report + f'Saved to: {output}', flush=True)


if __name__ == '__main__':
    main()
