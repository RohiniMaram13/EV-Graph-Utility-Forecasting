#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# Known-good combination for the user's RTX 5090 / CUDA-12.8-capable driver.
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt

python check_gpu.py
