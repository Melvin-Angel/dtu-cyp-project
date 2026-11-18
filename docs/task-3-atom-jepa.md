# Task 3: Fine-tuning Atom-JEPA

**Status:** Environment set up independently, data regenerated, and reference Atom-JEPA runs completed on both tasks at starter defaults. Investigation of fine-tuning directions not yet started.
**Recorded:** 2026-11-18
**Evidence source:** Run artifacts downloaded from the DTU HPC to the local repository (`config.json`, `history.jsonl`, `validation_metrics.json`, `baseline_metrics.json`). Test metric files were not opened.

## Objective

Fine-tune the pretrained Atom-JEPA encoder on both prediction tasks, establish a reference result at the starter defaults, and use it to choose which fine-tuning directions to investigate. Final configurations are selected on validation only.

## Environment

A second working copy was set up on the HPC, separate from the one used for Tasks 1 and 2, so that both team members can run jobs independently.

- Location: `$BLACKHOLE/dtu-cyp-project` (`/dtu/blackhole/02/223906/dtu-cyp-project`).
- Clone of the team fork `Melvin-Angel/dtu-cyp-project` at commit `ffe280a`.
- Virtual environment built with the `python3/3.12.11` module, following [`setup_hpc_env.sh`](../setup_hpc_env.sh) by hand.
- `pip install --no-cache-dir` was used throughout. The uncached install was the failure mode recorded in [`task-1-data-setup.md`](task-1-data-setup.md); using `--no-cache-dir` from the start avoided it.
- Installed versions recorded in each run's `config.json`: torch `2.6.0+cu124`, atom-jepa `0.3.0`, RDKit `2026.9.2`, numpy `2.5.5`, pandas `2.3.3`, scikit-learn `1.9.1`, hydra-core `1.3.2`.

`$BLACKHOLE` is scratch storage that is cleared at service windows and deleted at the end of January 2027. Code is kept on GitHub and `data/` is reproducible, but `runs/` is not, so run artifacts are copied to a local machine after each batch of jobs.

## Data regeneration

`prepare_data.py` was rerun in this working copy with the default seed (job `30099194`, 167 s wall time, completed without error).

The resulting `data/manifest.json` matches the counts recorded in [`task-1-data-setup.md`](task-1-data-setup.md) exactly:

| Task | Train | Validation | Test |
|---|---:|---:|---:|
| Regression | 3,415 | 728 | 762 |
| Classification | 3,382 | 715 | 723 |

6,143 conformer files were written and 2 conformers failed, as before. RDKit differs between the two working copies (`2026.09.2` here, `2026.09.1` for Tasks 1 and 2). The split counts are identical, so the splits are very likely identical as well, but this has not been proven. `finetune.py` records a `split_sha256` per CSV in every run's `config.json`; comparing those hashes across the two working copies would confirm it. The hashes for this copy are:

| Split | sha256 (regression) |
|---|---|
| train | `c9c72ba7ffef582ed884f9de12d8d36e1cfe007f3247782cdd330d7ec0f01779` |
| val | `f2bfab24a433c80ae727d3650e2de392b62621071469d7b9d8008debefa2cebc` |
| test | `5d0d4257ac7af49b345d1f5cc87f43f40081bf71c73bdf33cda41d76b4c69a08` |

## Reference runs

Three jobs were submitted on the `gpuv100` queue.

| Job ID | Task | Epochs | Run folder | Purpose |
|---:|---|---:|---|---|
| 30099246 | Regression | 1 | `runs/regression/atom_jepa/2026-11-17_18-12-09` | Pipeline smoke test |
| 30099574 | Regression | 20 | `runs/regression/atom_jepa/2026-11-17_19-40-38` | Reference run |
| 30099575 | Classification | 20 | `runs/classification/atom_jepa/2026-11-17_19-40-49` | Reference run |

The single-epoch job was run first to verify the GPU path, the pretrained checkpoint download, conformer loading and the evaluation code before committing GPU time to full runs. It is not treated as an experimental result.

All three used the starter configuration in [`conf/model/atom_jepa.yaml`](../conf/model/atom_jepa.yaml) unchanged:

- Pretrained `molecules` checkpoint, fp32, encoder not frozen.
- Encoder learning rate `1e-5`, readout learning rate `1e-3`.
- MLP readout: 3 linear layers, hidden size 512, dropout `0.1`.
- 20 epochs, batch size 8, weight decay `1e-4`, gradient clipping `1.0`, seed 42.

fp32 was kept because `gpuv100` does not support bf16 natively. Both 20-epoch jobs finished well inside the 4-hour wall clock, so no queue or precision change was needed.

## Results

Selection metrics are macro ST-RAE for regression (lower is better) and macro MCC at the fixed 0.5 threshold for classification (higher is better). Both are validation values at the best epoch, as chosen by the starter training loop.

| Task | Atom-JEPA | Best epoch | Task 2 baseline | Predict-the-mean floor |
|---|---:|---:|---:|---:|
| Regression (macro ST-RAE) | **0.692313** | 10 | 0.842870 | 0.983028 |
| Classification (macro MCC) | **0.286998** | 14 | 0.276712 | 0.000000 |

The baseline column is the validation-selected Morgan MLP from [`task-2-baseline.md`](task-2-baseline.md). The floor is `baseline_metrics.json`, written by [`finetune.py`](../finetune.py) from the training-set mean; for classification a constant prediction gives MCC exactly 0, which is what that metric is designed to do.

### Per-target validation scores

Regression, at the selected epoch:

| Target | ST-RAE | MAE | R² |
|---|---:|---:|---:|
| CYP1A2 | 0.874383 | 0.730433 | 0.215335 |
| CYP2C9 | 0.538147 | 0.443344 | 0.543378 |
| CYP2D6 | 0.852745 | 0.601270 | 0.186607 |
| CYP3A4 | 0.503975 | 0.530156 | 0.611706 |

Classification, at the selected epoch:

| Target | MCC | AUROC | AP | Positive fraction |
|---|---:|---:|---:|---:|
| CYP2D6 TDI | 0.204734 | 0.616291 | 0.406570 | 0.273913 |
| CYP3A4 TDI | 0.369262 | 0.781354 | 0.433047 | 0.203774 |

## Observations

**Atom-JEPA beats the fingerprint baseline on both tasks, by very different margins.** Regression improves by 0.150557 ST-RAE, a substantial gain. Classification improves by 0.010286 MCC, which is within the run-to-run variation the Task 2 notes warn about and should not be presented as a reliable gain from a single run.

**Both runs overfit.** Training loss falls monotonically from 0.8729 to 0.1583 over the 20 regression epochs, while validation ST-RAE stops improving after epoch 10 and then fluctuates between 0.69 and 0.78 with no trend. Classification shows the same shape: loss 0.5342 to 0.1259, best validation MCC at epoch 14, no improvement afterwards. Roughly half the training budget produces no generalisation benefit in either task.

This is expected given the setup. The pretrained encoder is being fine-tuned end to end on about 3,400 training molecules, which is a small dataset relative to the number of parameters being updated.

**The same enzymes are hard in both tasks.** CYP2D6 is the weakest regression target (ST-RAE 0.852745, R² 0.186607) and the weaker classification target (MCC 0.204734). CYP3A4 is the strongest in both (ST-RAE 0.503975, MCC 0.369262). CYP2D6 was also the weakest target for the Task 2 baseline, so this is a property of the data rather than of Atom-JEPA.

**Useful reference point from the smoke test.** The single-epoch run already reached macro ST-RAE 0.761364, better than the fully tuned Task 2 baseline. Most of Atom-JEPA's regression advantage is present after one pass over the data, which supports the interpretation that the gain comes from the pretrained representation rather than from the fine-tuning schedule.

## Planned directions

The project requires at least two distinct directions, and states that combining two routine configuration changes is not sufficient on its own. The two chosen below address different parts of the pipeline and both require code changes.

**1. Freezing strategy.** Motivated directly by the overfitting observed above. Planned comparison on both tasks, holding everything else at the reference settings:

- Frozen encoder, readout head only (`model.freeze_encoder=true`, a config change).
- Full fine-tuning (the reference runs above).
- Freeze the encoder for the first N epochs, then unfreeze and train both together. This does not exist in the starter code and requires a change to the epoch loop in [`finetune.py`](../finetune.py), since `freeze_encoder` is currently fixed for the whole run.

The question is whether the overfitting comes from updating the pretrained weights or from the readout head, and whether a warm-up period for the head changes the answer.

**2. Multiple conformers.** Each molecule currently has a single 3D conformer, generated once in `prepare_data.py` and read by the `Conformers` dataset in [`models/atom_jepa.py`](../models/atom_jepa.py). Real molecules adopt many shapes, so a single conformer is an arbitrary sample. Planned changes:

- Generate several conformers per molecule in data preparation.
- Sample a random conformer per molecule per epoch during training, as a form of data augmentation.
- Average predictions over multiple conformers at evaluation time.

Training-time sampling and evaluation-time averaging are separable and will be compared separately, since they test different claims: one that conformer variation regularises training, the other that averaging reduces prediction variance.

## Test-set boundary

No test metrics or test predictions have been opened. `finetune.py` writes them for every run, including the three above. They will be used only once, for the final comparison in Part 5, after configurations are selected on validation.

## Analysis scripts

- [`plot_history_task3.py`](../scripts/plot_history_task3.py) plots training loss and the validation metric as two stacked panels sharing the epoch axis, and marks the selected epoch. It reads `history.jsonl` only. Two panels were used rather than twin y-axes so that the two curves, which have unrelated scales, are not visually related to each other.

## Project files

- [`conf/model/atom_jepa.yaml`](../conf/model/atom_jepa.yaml)
- [`models/atom_jepa.py`](../models/atom_jepa.py)
- [`models/readouts.py`](../models/readouts.py)
- [`finetune.py`](../finetune.py)
