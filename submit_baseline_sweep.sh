#!/usr/bin/env bash
set -euo pipefail

# Submit a small Morgan-MLP learning-rate comparison:
#   regression:      1e-3 (starter default), 3e-4
#   classification:  1e-3 (starter default), 3e-4
# Select configurations using validation metrics only. The starter finetune.py
# also writes test metrics on every run; do not inspect those during tuning.
# Run this from anywhere; submissions are made from the project directory.
cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v bsub >/dev/null 2>&1; then
    echo "bsub is unavailable. Run this script on the DTU HPC." >&2
    exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
    echo "Missing .venv. Create the project environment before submitting jobs." >&2
    exit 1
fi

if ! grep -q 'module load python3/3.12.11' finetune.sh; then
    echo "Add 'module load python3/3.12.11' to finetune.sh before its venv activation." >&2
    exit 1
fi

for task in regression classification; do
    for split in train val test; do
        if [[ ! -f "data/${task}/${split}.csv" ]]; then
            echo "Missing data/${task}/${split}.csv; prepare the data first." >&2
            exit 1
        fi
    done
done

submit_job() {
    local task="$1"
    local setting="$2"
    local learning_rate="$3"
    local job_name="morgan_${task}_${setting}"
    local run_dir="runs/${task}/morgan_mlp/${job_name}"

    if [[ -e "$run_dir" ]]; then
        echo "Run directory already exists: $run_dir" >&2
        echo "Choose a new job name before resubmitting this configuration." >&2
        exit 1
    fi

    echo "Submitting ${job_name} (readout learning rate ${learning_rate})"
    bsub -J "$job_name" \
        -env "MODEL=morgan_mlp, TASK=${task}, OVERRIDES=model.readout_lr=${learning_rate} hydra.run.dir=${run_dir}" \
        < finetune.sh
}

submit_job regression default 1e-3
submit_job regression low_lr 3e-4
submit_job classification default 1e-3
submit_job classification low_lr 3e-4
