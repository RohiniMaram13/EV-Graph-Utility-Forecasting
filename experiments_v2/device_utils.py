import platform
import torch


def resolve_device(requested='auto'):
    requested = (requested or 'auto').lower()
    if requested == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    if requested.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but torch.cuda.is_available() is False. Install a CUDA-enabled PyTorch build.')
    return torch.device(requested)


def device_report(device):
    lines = [
        f'Python platform: {platform.platform()}',
        f'PyTorch: {torch.__version__}',
        f'Device: {device}',
    ]
    if device.type == 'cuda':
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        lines += [
            f'GPU: {props.name}',
            f'GPU memory: {props.total_memory / 1024**3:.2f} GB',
            f'CUDA runtime reported by PyTorch: {torch.version.cuda}',
            f'cuDNN: {torch.backends.cudnn.version()}',
        ]
    return '\n'.join(lines)
