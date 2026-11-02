# Task 2: Molecular-feature baseline

**Status:** Main validation sweep and five targeted refinement jobs analyzed; validation plots regenerated.  
**Recorded:** 2026-11-02  
**Evidence source:** HPC job logs and downloaded validation artifacts. Test metric files and held-out scores were not opened during this analysis.

## Baseline and experiment plan

- Baseline: `morgan_mlp` (Morgan count fingerprints followed by the starter MLP).
- Initial sweep: `model.readout_lr=1e-3` (starter default) and `3e-4` on both tasks.
- Follow-up ablations: smaller MLP architecture, batch size 128, and weight decay `1e-3`, each compared with the starter setting on both tasks.
- Five configurations were compared per task. The follow-up submitter was accidentally run twice, so the six ablation configurations were each submitted twice with the same seed and settings.
- Apart from the specific factor under comparison, settings were held at their starter defaults. The prepared data and split seed 42 were unchanged.

The starter Morgan configuration uses radius 2, 1,024-bit count fingerprints, quantile scaling fitted on training data, a 3-layer MLP with hidden dimension 512 and dropout 0.1, 100 epochs, batch size 64, and weight decay `1e-2`.

## Initial learning-rate jobs

| Job ID | Job name | Task | Learning rate | Status at analysis |
|---:|---|---|---:|---|
| 29872534 | `morgan_regression_default` | Regression | `1e-3` | Completed |
| 29872535 | `morgan_regression_low_lr` | Regression | `3e-4` | Completed |
| 29872536 | `morgan_classification_default` | Classification | `1e-3` | Completed |
| 29872537 | `morgan_classification_low_lr` | Classification | `3e-4` | Completed |

All four jobs were accepted by the `gpuv100` queue. The HPC copy of `finetune.sh` was confirmed to load `python3/3.12.11` before activating the virtual environment.

## Selection and test-set handling

- Select a setting independently for each task using validation metrics only.
- Regression selection metric: macro-average ST-RAE; lower is better.
- Classification selection metric: macro-average MCC at the fixed 0.5 threshold; higher is better.
- The starter runner writes test metrics for every run. Do not inspect or use those test outputs while selecting the baseline configuration.

## Results

### Initial learning-rate sweep

| Task | Selection metric | LR `1e-3` | Best epoch | LR `3e-4` | Best epoch | Best in LR sweep |
|---|---|---:|---:|---:|---:|---|
| Regression | Macro ST-RAE (lower is better) | 0.84594 | 1 | 0.84929 | 2 | `1e-3` |
| Classification | Macro MCC (higher is better) | 0.21664 | 7 | 0.19938 | 9 | `1e-3` |

The default learning rate had a 0.00335 lower macro ST-RAE for regression and a 0.01726 higher macro MCC for classification on this validation split.

### Per-target validation scores

Regression ST-RAE:

| Target | LR `1e-3` | LR `3e-4` | Better on this target |
|---|---:|---:|---|
| CYP1A2 | 0.94697 | 0.97325 | `1e-3` |
| CYP2C9 | 0.75011 | 0.73139 | `3e-4` |
| CYP2D6 | 0.97539 | 1.00392 | `1e-3` |
| CYP3A4 | 0.71131 | 0.68863 | `3e-4` |

Classification MCC:

| Target | LR `1e-3` | LR `3e-4` | Better on this target |
|---|---:|---:|---|
| CYP2D6 TDI | 0.15893 | 0.16185 | `3e-4` |
| CYP3A4 TDI | 0.27434 | 0.23690 | `1e-3` |

### Observations

- The macro metric favors the default rate for both tasks, but the improvement is small for regression and the preferred rate differs for some individual targets.
- Regression selected its best checkpoint very early (epoch 1 or 2); later validation ST-RAE values were generally worse and fluctuated. This makes validation-based checkpoint selection important for this run.
- Classification reached its best MCC at epochs 7 and 9. Its later validation curves fluctuate around a lower plateau; the default rate gives the stronger macro MCC, mainly due to CYP3A4.
- This is one run per learning rate with the fixed seed and split. Treat these as findings for this validation split, not evidence that the default rate is universally better.

### Plots and reproducibility

- Initial learning-rate plots: [curves PDF](../../02456_report_template/02456_report_template/figures/baseline/task2_validation_learning_curves.pdf) | [score comparison PDF](../../02456_report_template/02456_report_template/figures/baseline/task2_validation_score_comparison.pdf)
- Full validation plots: [curves PDF](../../02456_report_template/02456_report_template/figures/baseline/task2_all_validation_learning_curves.pdf) | [score comparison PDF](../../02456_report_template/02456_report_template/figures/baseline/task2_all_validation_score_comparison.pdf)
- Analysis scripts: [`analyze_task2_baseline.py`](../scripts/analyze_task2_baseline.py), [`analyze_task2_all_configs.py`](../scripts/analyze_task2_all_configs.py), and [`audit_task2_duplicate_import.py`](../scripts/audit_task2_duplicate_import.py). They read validation metrics and histories only; test artifacts were not opened.

### Initial five-configuration validation comparison

| Configuration | Regression macro ST-RAE (lower is better) | Best epoch | Classification macro MCC (higher is better) | Best epoch |
|---|---:|---:|---:|---:|
| Default MLP, LR `1e-3`, batch 64, weight decay `1e-2` | 0.84594 | 1 | 0.21664 | 7 |
| LR `3e-4` | 0.84929 | 2 | 0.19938 | 9 |
| Smaller MLP: 2 layers, hidden size 256 | 0.85539 | 1 | 0.16874 | 6 |
| Batch size 128 | **0.84301** | 1 | **0.25581** | 4 |
| Weight decay `1e-3` | 0.84635 | 1 | 0.23496 | 32 |

**Provisional selection after the first five configurations:** Morgan count fingerprints with the default 3-layer, hidden-size-512 MLP, dropout `0.1`, learning rate `1e-3`, batch size 128, and weight decay `1e-2`. The targeted refinement below updates this selection independently for each task.

Batch size 128 improved macro ST-RAE by 0.00293 for regression and macro MCC by 0.03917 for classification relative to the default run. The smaller MLP performed worse on both tasks. Reducing weight decay slightly worsened regression but improved classification over the default; it did not outperform batch size 128.

Per-target, batch size 128 improved three of four regression ST-RAE values; CYP2C9 was worse. It improved MCC for both classification targets. The weight-decay variant's classification improvement over default was concentrated on CYP2D6, while CYP3A4 MCC was lower.

### Duplicate-submission audit and local cleanup

- The six ablation configurations were each submitted twice. Their shared run folders contained 200 history rows: two 100-epoch blocks that matched exactly across all fields.
- Separate per-job stdout logs for both submissions each contained 100 validation epochs and matched the corresponding configuration history. Duplicate jobs are treated as accidental repeats, not independent experimental replicates.
- Unique and duplicate job IDs were: regression architecture 29872843/29872851, regression batch 128 29872844/29872852, regression weight decay 29872845/29872853, classification architecture 29872846/29872854, classification batch 128 29872847/29872855, and classification weight decay 29872849/29872856.
- The follow-up jobs were submitted with [`submit_baseline_ablations.sh`](../submit_baseline_ablations.sh).
- The canonical local run folders now contain one verified 100-epoch history per configuration. One stdout/stderr pair per unique ablation was retained in `outputs/`.
- The imported staging snapshot `hpc_ablation_import_2026-11-02/` still contains redundant copies. Its recursive removal was blocked by the environment's file-operation policy; no staged files were deleted.

### Test-set boundary

No test metrics or test predictions were used to choose the baseline configuration. The runner generated test artifacts for each job; only the validation-selected configuration should be used for the final test comparison.

### Targeted refinement batch

The five additional configurations were run against the provisional batch-128 setting. The selected Morgan MLP architecture, learning rate `1e-3`, data preparation, and seed 42 were retained; only the listed training parameter(s) changed.

The submitter [`submit_baseline_refinement.sh`](../submit_baseline_refinement.sh) used a one-shot guard against accidental duplicate submission.

| Job ID | Task | Configuration | Status |
|---:|---|---|---|
| 29873126 | Regression | Batch size 256 | Completed and analyzed |
| 29873127 | Regression | Weight decay `3e-2`, batch size 128 | Completed and analyzed |
| 29873128 | Classification | Batch size 256 | Completed and analyzed |
| 29873129 | Classification | Weight decay `1e-3`, batch size 128 | Completed and analyzed |
| 29873130 | Classification | Weight decay `1e-4`, batch size 128 | Completed and analyzed |

Job IDs are matched to configurations by the submitter's submission order; `bjobs` truncated some job names. The imported configs, validation metrics, and epoch histories were checked. The analysis script did not open test metrics or test predictions.

#### Refinement results

| Task | Configuration | Macro validation metric | Best epoch |
|---|---|---:|---:|
| Regression | Batch 256, weight decay `1e-2` | 0.857961 ST-RAE | 2 |
| Regression | Batch 128, weight decay `3e-2` | **0.842870 ST-RAE** | 1 |
| Classification | Batch 256, weight decay `1e-2` | 0.253390 MCC | 9 |
| Classification | Batch 128, weight decay `1e-3` | **0.276712 MCC** | 68 |
| Classification | Batch 128, weight decay `1e-4` | 0.271218 MCC | 39 |

#### Cumulative validation comparison

The table combines the initial sweep, first ablations, and refinement jobs. A dash means that configuration was not run for that task.

| Configuration | Regression macro ST-RAE (best epoch) | Classification macro MCC (best epoch) |
|---|---:|---:|
| Default: LR `1e-3`, batch 64, weight decay `1e-2` | 0.845944 (1) | 0.216635 (7) |
| LR `3e-4`, batch 64, weight decay `1e-2` | 0.849294 (2) | 0.199377 (9) |
| Smaller MLP: 2 linear layers, hidden size 256 | 0.855390 (1) | 0.168736 (6) |
| LR `1e-3`, batch 128, weight decay `1e-2` | 0.843011 (1) | 0.255809 (4) |
| LR `1e-3`, batch 64, weight decay `1e-3` | 0.846351 (1) | 0.234965 (32) |
| LR `1e-3`, batch 256, weight decay `1e-2` | 0.857961 (2) | 0.253390 (9) |
| LR `1e-3`, batch 128, weight decay `3e-2` | **0.842870 (1)** | — |
| LR `1e-3`, batch 128, weight decay `1e-3` | — | **0.276712 (68)** |
| LR `1e-3`, batch 128, weight decay `1e-4` | — | 0.271218 (39) |

The final validation selections are task-specific:

- **Regression:** Morgan count fingerprints; 3-layer MLP, hidden size 512, dropout `0.1`; learning rate `1e-3`, batch size 128, weight decay `3e-2`. Macro validation ST-RAE is 0.842870 at epoch 1.
- **Classification:** the same fingerprints and MLP; learning rate `1e-3`, batch size 128, weight decay `1e-3`. Macro validation MCC is 0.276712 at epoch 68.

The regression refinement is a near tie with the provisional setting (batch 128, weight decay `1e-2`, macro ST-RAE 0.843011): the difference is only 0.000141. The stronger weight decay therefore wins numerically on this validation split, but the result is too small to interpret as a stable general improvement from a single run. Batch size 256 was worse for regression (0.857961). For classification, combining batch size 128 with weight decay `1e-3` improved macro MCC by 0.020903 over the provisional batch-128 result (0.255809); weight decay `1e-4` also scored well (0.271218), but below `1e-3`. Batch size 256 did not improve on batch 128 with the original weight decay.

At the selected classification setting, CYP2D6 MCC was 0.314465 and CYP3A4 MCC was 0.238960. Relative to batch 128 with weight decay `1e-2`, this raises CYP2D6 MCC by 0.110767 while lowering CYP3A4 MCC by 0.068960. The macro gain reflects this trade-off rather than a gain on both endpoints. Regression ST-RAE at the selected setting was 0.929340 (CYP1A2), 0.783441 (CYP2C9), 0.958560 (CYP2D6), and 0.700137 (CYP3A4).

The smaller architecture remained worse than the default MLP on both tasks. Lowering the learning rate to `3e-4` also did not beat `1e-3` on the macro metrics. Learning curves vary substantially across epochs, especially for classification; the best classification checkpoint occurred at epoch 68. These are single-run, seed-42 validation comparisons, so small differences and isolated peaks should be treated cautiously. Use the two selected configurations for the final comparison and consult the test set only at that stage.

#### Refreshed plots

- **Main report figure:** [validation score comparison (PDF)](../../02456_report_template/02456_report_template/figures/baseline/task2_all_validation_score_comparison.pdf) compares the best validation macro score for every tested setting. Place it immediately after the paragraph introducing validation-based configuration selection.
- **Training-dynamics figure:** [validation learning curves (PDF)](../../02456_report_template/02456_report_template/figures/baseline/task2_all_validation_learning_curves.pdf) shows the metric across epochs and marks each run's selected epoch. Given the four-page body limit, this is a good appendix figure; cite it in the baseline discussion when noting noisy curves and early or late best epochs.
- PNG previews are available beside each PDF. Both plots were regenerated from the saved validation histories and metrics using [`analyze_task2_all_configs.py`](../scripts/analyze_task2_all_configs.py).

### Draft report section for inspiration

The draft below is a starting point for rewriting in your own voice. Keep the task-specific selection and the validation-only boundary clear.

> **Molecular-feature baseline.** We used an MLP operating on radius-2 Morgan count fingerprints of length 1,024 as the molecular-feature baseline. The starter readout has three linear layers in total, including the output layer, with two hidden layers of width 512 and dropout 0.1; quantile scaling was fitted on the training data. We retained the supplied train, validation, and test splits with seed 42 and varied the learning rate, network size, batch size, and weight decay. Configurations were selected independently for inhibition regression and TDI classification using validation data only. We compared macro ST-RAE for regression (lower is better) and macro MCC at a 0.5 threshold for classification (higher is better).
>
> The best regression result used learning rate `1e-3`, batch size 128, and weight decay `3e-2`, giving validation macro ST-RAE 0.8429. This was only 0.0001 below the batch-128, weight-decay-`1e-2` setting, so the numerical advantage is marginal. For classification, batch size 128 with weight decay `1e-3` gave the highest validation macro MCC, 0.2767, compared with 0.2558 for batch size 128 with weight decay `1e-2`, and 0.2166 for the default batch-size-64 run. This macro improvement came from a higher CYP2D6 MCC and a lower CYP3A4 MCC. The smaller network and batch size 256 did not improve the macro scores. Each setting is represented by one unique seed-42 training trajectory; the accidental duplicate ablation submissions produced identical histories. Since validation curves fluctuate, these choices are empirical selections for the fixed split rather than evidence of seed-independent gains.

**Figure placement for a short report:** Put the [score-comparison figure](../../02456_report_template/02456_report_template/figures/baseline/task2_all_validation_score_comparison.pdf) after the sentence describing the two validation metrics and before discussing the selected settings. After discussing the best epochs and curve variability, cite the [learning-curve figure](../../02456_report_template/02456_report_template/figures/baseline/task2_all_validation_learning_curves.pdf); place it in the appendix if the body is tight on space.
