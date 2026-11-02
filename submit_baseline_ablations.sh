#!/usr/bin/env bash
set -euo pipefail

# Submit three one-factor-at-a-time Morgan-MLP ablations on both tasks.
# Keep the learning rate at the current validation-sweep choice, 1e-3.
# The starter runner also writes test metrics; use validation metrics only.
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
    local overrides="$3"
    local job_name="morgan_${task}_${setting}"
    local run_dir="runs/${task}/morgan_mlp/${job_name}"

    if [[ -e "$run_dir" ]]; then
        echo "Run directory already exists: $run_dir" >&2
        echo "Choose a new job name before resubmitting this ablation." >&2
        exit 1
    fi

    echo "Submitting ${job_name} (${overrides})"
    bsub -J "$job_name" \
        -env "MODEL=morgan_mlp, TASK=${task}, OVERRIDES=model.readout_lr=1e-3 ${overrides} hydra.run.dir=${run_dir}" \
        < finetune.sh
}

for task in regression classification; do
    submit_job "$task" arch_small "readout.hidden_dim=256 readout.num_layers=2"
    submit_job "$task" batch128 "train.batch_size=128"
    submit_job "$task" weight_decay_1e3 "train.weight_decay=1e-3"
done
