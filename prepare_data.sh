#!/bin/bash
#BSUB -q hpc
#BSUB -J cyp_prepare
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 04:00
#BSUB -o outputs/cyp_prepare_%J.out
#BSUB -e outputs/cyp_prepare_%J.err
set -euo pipefail

# Submit from the starter directory. Set up this venv BEFORE submitting.
cd "${LS_SUBCWD:?Submit using bsub from the starter directory}"
source "${CYP_VENV:-$PWD/.venv}/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Download labelled training tables, generate conformers, and split train/val/test.
# Rerunning reuses existing conformer files.
# OUT selects the output directory and CONFORMERS the number of conformers per molecule:
#   bsub -env "OUT=data_mc, CONFORMERS=10" < prepare_data.sh
# Changing CONFORMERS requires a NEW OUT directory; the manifest guard enforces this.
python -u prepare_data.py --out "${OUT:-data}" --workers 8 --seed 42 \
    --conformers "${CONFORMERS:-1}"
