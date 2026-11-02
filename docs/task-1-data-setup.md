# Task 1: Data setup and starter pipeline

**Status:** Data preparation and initial dataset inspection completed.  
**Recorded:** 2026-11-02  
**Evidence source:** Terminal output shared from the DTU HPC run. Values below were not re-read from the remote filesystem by this assistant.

## Objective

Prepare the provided CYP regression and classification datasets, generate molecular conformers, inspect labels and evaluation metrics, and retain the default train/validation/test split seed of 42.

## Data preparation configuration

Recorded in `data/manifest.json`:

- Data revision: `146282d2ce1532f1803679a834a9bec9dac1e482`
- Split seed: `42`
- Validation fraction: `0.15`
- Test fraction: `0.15`
- Row limit: `null` (no smoke-test row limit)
- RDKit version: `2026.09.1`
- Conformer protocol: `atom-jepa-0.3.0`

### Split counts

| Task | Train | Validation | Test | Total rows |
|---|---:|---:|---:|---:|
| Regression | 3,415 | 728 | 762 | 4,905 |
| Classification | 3,382 | 715 | 723 | 4,820 |

The task tables have different row counts because each contains molecules with labels for that task. The preparation code assigns molecule-grouped splits across the combined data so canonical-SMILES duplicates remain in the same split.

## Conformers and exclusions

- Conformer files found: `6,143` `.npz` files under `data/conformers/`.
- Failed conformers recorded: `2`.
- The names shown in `data/excluded_conformer_failures.csv` were `OCNT-0453746` and `OCNT-0453782`. These match the two exclusions noted in the project README.

## Dataset labels inspected

The regression CSV header contains four pIC50 targets:

- `CYP1A2_pIC50_direct_inhibition`
- `CYP2C9_pIC50_direct_inhibition`
- `CYP2D6_pIC50_direct_inhibition`
- `CYP3A4_pIC50_direct_inhibition`

It also contains the corresponding `_conf_low`, `_conf_high`, and `_std` columns, plus `canonical_smiles`, `key`, and `split`. Higher pIC50 indicates stronger inhibition. The starter metric uses the confidence bounds when calculating ST-RAE.

The classification CSV header contains two binary time-dependent inhibition targets:

- `CYP2D6_is_TDI`
- `CYP3A4_is_TDI`

The TDI labels are represented as numeric 0/1 values in the prepared data.

## Evaluation metrics in the starter code

As implemented in [`finetune.py`](../finetune.py):

- **Regression:** per-target MAE, R-squared, and ST-RAE; the aggregate is the macro-average of ST-RAE across the four targets. Lower is better.
- **Classification:** per-target MCC, AUROC, and average precision; MCC uses a fixed probability threshold of 0.5. The aggregate is the macro-average of MCC across the two targets. Higher is better.
- Missing target values are masked from the loss and the relevant per-target metric.

## Environment setup notes

- The environment was set up on DTU HPC using the available `python3/3.12.11` module and the project dependency list.
- The first PyTorch/CUDA-wheel installation attempt stopped with `No space left on device`.
- At diagnosis, the shared filesystem reported 48 TB available and 91% inode use; `quota -s` printed no quota details. The visible user-space usage included a 5.8 GB pip cache and a 7.0 GB `.venv`.
- The user reported that installation completed after clearing the pip cache and retrying with `--no-cache-dir`.
- A `pip check` result and the final printed PyTorch/CUDA runtime versions have not been recorded here yet.

## Current boundary and follow-up

- Keep these prepared splits and seed 42 unchanged for subsequent experiments.
- Data preparation and initial label/metric inspection are complete based on the outputs shared.
- No model-training results are recorded in this note yet.

## Project files

- [`README.md`](../README.md)
- [`prepare_data.py`](../prepare_data.py)
- [`prepare_data.sh`](../prepare_data.sh)
- [`finetune.py`](../finetune.py)
