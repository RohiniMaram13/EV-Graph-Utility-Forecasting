import torch
from experiments_v2.device_utils import resolve_device, device_report

d = resolve_device('auto')
print(device_report(d))
if d.type != 'cuda':
    raise SystemExit('\nERROR: CUDA GPU is not available to this PyTorch installation.')

# Small tensor smoke test.
x = torch.randn(2048, 2048, device=d)
y = x @ x.T
print(f'CUDA tensor test OK. checksum={float(y[0,0]):.4f}')
