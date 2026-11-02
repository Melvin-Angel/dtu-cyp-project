#!/usr/bin/env bash
set -euo pipefail

# Targeted Task 2 refinements. All jobs keep the selected LR (1e-3) and MLP
# architecture, and each run directory has a unique name.
# This one-shot guard prevents accidental duplicate submission if this helper is
# run twice. Check the jobs.txt file and bjobs before removing the guard.
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

for baseline in \
    runs/regression/morgan_mlp/morgan_regression_batch128/validation_metrics.json \
    runs/classification/morgan_mlp/morgan_classification_batch128/validation_metrics.json; do
    if [[ ! -f "$baseline" ]]; then
        echo "Missing selected batch-128 control: $baseline" >&2
        exit 1
    fi
done

jobs=(
  "regression regression_batch256 train.batch_size=256"
  "regression regression_weight_decay_3e2 train.batch_size=128 train.weight_decay=3e-2"
  "classification classification_batch256 train.batch_size=256"
  "classification classification_weight_decay_1e3_batch128 train.batch_size=128 train.weight_decay=1e-3"
  "classification classification_weight_decay_1e4_batch128 train.batch_size=128 train.weight_decay=1e-4"
)

for item in "${jobs[@]}"; do
    read -r task setting overrides <<< "$item"
    job_name="morgan_${setting}"
    run_dir="runs/${task}/morgan_mlp/${job_name}"
    if [[ -e "$run_dir" ]]; then
        echo "Run directory already exists: $run_dir" >&2
        echo "Refusing to submit a duplicate configuration." >&2
        exit 1
    fi
done

guard_dir="runs/.baseline_refinement_2026-11-02_submitted"
if ! mkdir "$guard_dir" 2>/dev/null; then
    echo "Submission guard exists: $guard_dir" >&2
    echo "Check bjobs and ${guard_dir}/jobs.txt before considering another submission." >&2
    exit 1
fi

for item in "${jobs[@]}"; do
    read -r task setting overrides <<< "$item"
    job_name="morgan_${setting}"
    run_dir="runs/${task}/morgan_mlp/${job_name}"
    echo "Submitting ${job_name}: ${overrides}"
    response=$(bsub -J "$job_name" \
        -env "MODEL=morgan_mlp, TASK=${task}, OVERRIDES=model.readout_lr=1e-3 ${overrides} hydra.run.dir=${run_dir}" \
        < finetune.sh)
    printf '%s: %s\n' "$job_name" "$response" | tee -a "${guard_dir}/jobs.txt"
done
