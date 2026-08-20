import copy
import random
import time
import numpy as np
import torch
import torch.nn.functional as F

from experiments_v2.config import MAX_EPOCHS, PATIENCE, LR, WEIGHT_DECAY, BATCH_SIZE
from experiments_v2.device_utils import resolve_device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_graph(graph, device):
    if graph is None:
        return None
    return {
        'edge_index': graph['edge_index'].to(device, non_blocking=True),
        'edge_weight': None if graph.get('edge_weight') is None else graph['edge_weight'].to(device, non_blocking=True),
    }


def _iter_batches(X, Y, batch_size, shuffle=False, device=None):
    n = len(X)
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for s in range(0, n, batch_size):
        ii = idx[s:s + batch_size]
        xb = X[ii]
        yb = None if Y is None else Y[ii]
        if device is not None:
            xb = xb.to(device, non_blocking=True)
            if yb is not None:
                yb = yb.to(device, non_blocking=True)
        yield xb, yb


def _mse_batched(model, dataset, x_static, graph, device, batch_size, amp):
    X, Y = dataset
    total_sse = 0.0
    total_n = 0
    model.eval()
    with torch.inference_mode():
        for xb, yb in _iter_batches(X, Y, batch_size, shuffle=False, device=device):
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(amp and device.type == 'cuda')):
                pred = model(xb, x_static, graph['edge_index'] if graph else None, graph.get('edge_weight') if graph else None)
                diff = pred.float() - yb.float()
            total_sse += float((diff * diff).sum().item())
            total_n += diff.numel()
    return total_sse / max(total_n, 1)


def train_model(model, train_set, val_set, x_static=None, graph=None, seed=42,
                verbose=False, device='auto', batch_size=None, max_epochs=None,
                patience=None, amp=True, lr=None, include_initial=False):
    set_seed(seed)
    device = resolve_device(device) if not isinstance(device, torch.device) else device
    batch_size = int(batch_size or BATCH_SIZE)
    max_epochs = int(max_epochs or MAX_EPOCHS)
    patience = int(patience or PATIENCE)
    lr = float(LR if lr is None else lr)

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    model = model.to(device)
    xs = None if x_static is None else x_static.to(device, non_blocking=True)
    g = _move_graph(graph, device)
    Xtr, Ytr = train_set

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError('No trainable parameters are enabled for this training stage.')
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=max(2, min(7, patience // 2)), factor=.5, min_lr=1e-5
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(amp and device.type == 'cuda'))

    best = float('inf')
    best_state = None
    bad = 0
    t0 = time.perf_counter()

    # For warm-start stages, treat the incoming model as an eligible checkpoint.
    # This prevents graph-only or fine-tuning stages from being forced to accept
    # a worse solution than the already-trained temporal baseline.
    if include_initial:
        best = _mse_batched(model, val_set, xs, g, device, batch_size, amp)
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f'    initial_val={best:.6f} (eligible checkpoint)', flush=True)

    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        nb = 0
        ep0 = time.perf_counter()
        for xb, yb in _iter_batches(Xtr, Ytr, batch_size, shuffle=True, device=device):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(amp and device.type == 'cuda')):
                pred = model(xb, xs, g['edge_index'] if g else None, g.get('edge_weight') if g else None)
                loss = F.mse_loss(pred, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach().item())
            nb += 1

        vloss = _mse_batched(model, val_set, xs, g, device, batch_size, amp)
        scheduler.step(vloss)
        if vloss < best - 1e-7:
            best = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == max_epochs or bad >= patience):
            sec = time.perf_counter() - ep0
            lr = optimizer.param_groups[0]['lr']
            print(f'    epoch={epoch:03d} train={total/max(nb,1):.6f} val={vloss:.6f} lr={lr:.2e} time={sec:.1f}s', flush=True)
        if bad >= patience:
            break

    if best_state is None:
        raise RuntimeError('Training completed without a valid checkpoint.')
    model.load_state_dict(best_state)
    model.to(device)
    if verbose:
        print(f'    best_val={best:.6f}, epochs={epoch}, total_time={(time.perf_counter()-t0)/60:.1f} min', flush=True)
    return model, best, epoch


def predict(model, test_set, scaler, x_static=None, graph=None, device='auto', batch_size=None, amp=True):
    device = resolve_device(device) if not isinstance(device, torch.device) else device
    batch_size = int(batch_size or BATCH_SIZE)
    xs = None if x_static is None else x_static.to(device, non_blocking=True)
    g = _move_graph(graph, device)
    Xt, Yt = test_set
    model = model.to(device)
    model.eval()
    preds = []
    targets = []
    with torch.inference_mode():
        for xb, yb in _iter_batches(Xt, Yt, batch_size, shuffle=False, device=device):
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(amp and device.type == 'cuda')):
                p = model(xb, xs, g['edge_index'] if g else None, g.get('edge_weight') if g else None)
            preds.append(p.float().cpu())
            targets.append(yb.float().cpu())
    p = torch.cat(preds, dim=0).numpy()
    y = torch.cat(targets, dim=0).numpy()
    # scaler mean/std may be scalars or [N,1]; NumPy broadcasts both
    # correctly across [windows, nodes, horizon].
    p = p * scaler['std'] + scaler['mean']
    y = y * scaler['std'] + scaler['mean']
    return p, y
