#!/usr/bin/env bash
set -euo pipefail

module load python3/3.12.11

if [[ ! -f requirements.txt ]]; then
    echo "Run this script from the dtu-cyp-project directory." >&2
    exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
    python3.12 -m venv .venv
fi

source .venv/bin/activate

python -c 'import sys; assert sys.version_info[:3] == (3, 12, 11), f"Expected Python 3.12.11, got {sys.version}"'

python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt

python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA runtime:", torch.version.cuda)'
