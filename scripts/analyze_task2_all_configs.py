"""Compare all Task 2 Morgan-MLP configurations using validation data only.

Reads run config.json, validation_metrics.json, and history.jsonl files. It never
opens test_metrics.json or test_predictions.csv. Duplicate 200-row histories are
checked as two 100-epoch blocks and reduced in-memory for plotting only.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "runs"
FIGURE_DIR = (
    ROOT.parent
    / "02456_report_template"
    / "02456_report_template"
    / "figures"
    / "baseline"
)
CONFIGS = {
    "regression": [
        ("lr_1e3", "LR 1e-3 (default)", "morgan_regression_default"),
        ("lr_3e4", "LR 3e-4", "morgan_regression_low_lr"),
        ("arch_small", "Smaller MLP", "morgan_regression_arch_small"),
        ("batch128", "Batch 128", "morgan_regression_batch128"),
        ("weight_decay_1e3", "Weight decay 1e-3", "morgan_regression_weight_decay_1e3"),
        ("batch256", "Batch 256", "morgan_regression_batch256"),
        ("weight_decay_3e2", "Weight decay 3e-2", "morgan_regression_weight_decay_3e2"),
    ],
    "classification": [
        ("lr_1e3", "LR 1e-3 (default)", "morgan_classification_default"),
        ("lr_3e4", "LR 3e-4", "morgan_classification_low_lr"),
        ("arch_small", "Smaller MLP", "morgan_classification_arch_small"),
        ("batch128", "Batch 128", "morgan_classification_batch128"),
        ("weight_decay_1e3", "Weight decay 1e-3", "morgan_classification_weight_decay_1e3"),
        ("batch256", "Batch 256", "morgan_classification_batch256"),
        ("weight_decay_1e3_batch128", "WD 1e-3, batch 128", "morgan_classification_weight_decay_1e3_batch128"),
        ("weight_decay_1e4_batch128", "WD 1e-4, batch 128", "morgan_classification_weight_decay_1e4_batch128"),
    ],
}
METRIC = {"regression": "macro_ST-RAE", "classification": "macro_MCC"}
COLOR_BY_ID = {
    "lr_1e3": "#0072B2",
    "lr_3e4": "#D55E00",
    "arch_small": "#009E73",
    "batch128": "#CC79A7",
    "weight_decay_1e3": "#E69F00",
    "batch256": "#56B4E9",
    "weight_decay_3e2": "#F0E442",
    "weight_decay_1e3_batch128": "#332288",
    "weight_decay_1e4_batch128": "#999999",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_history(path: Path) -> tuple[list[dict], bool]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) == 100:
        return rows, False
    if len(rows) != 200:
        raise ValueError(f"Expected 100 epochs or two duplicate blocks, got {len(rows)} in {path}")
    first, second = rows[:100], rows[100:]
    expected_epochs = list(range(1, 101))
    if [row["epoch"] for row in first] != expected_epochs or [row["epoch"] for row in second] != expected_epochs:
        raise ValueError(f"History is not two complete 1..100 epoch blocks: {path}")
    return first, first == second


def read_run(task: str, run_name: str) -> dict:
    path = RUNS_ROOT / task / "morgan_mlp" / run_name
    return {
        "path": path,
        "config": load_json(path / "config.json"),
        "validation": load_json(path / "validation_metrics.json"),
        "history": load_history(path / "history.jsonl")[0],
        "history_duplicate_identical": load_history(path / "history.jsonl")[1],
    }


def best_row(task: str, history: list[dict]) -> dict:
    metric = METRIC[task]
    return (min if task == "regression" else max)(history, key=lambda row: row[metric])


def flattened_differences(a: dict, b: dict) -> set[str]:
    differences = set()
    for section in ("model", "readout", "train"):
        keys = set(a[section]) | set(b[section])
        differences.update(
            f"{section}.{key}" for key in keys if a[section].get(key) != b[section].get(key)
        )
    for key in ("task", "seed", "split_sha256"):
        if a.get(key) != b.get(key):
            differences.add(key)
    return differences


def validate_experiment_design(data: dict) -> None:
    expected_diffs = {
        "regression": {
            "lr_3e4": {"model.readout_lr"},
            "arch_small": {"readout.hidden_dim", "readout.num_layers"},
            "batch128": {"train.batch_size"},
            "weight_decay_1e3": {"train.weight_decay"},
            "batch256": {"train.batch_size"},
            "weight_decay_3e2": {"train.batch_size", "train.weight_decay"},
        },
        "classification": {
            "lr_3e4": {"model.readout_lr"},
            "arch_small": {"readout.hidden_dim", "readout.num_layers"},
            "batch128": {"train.batch_size"},
            "weight_decay_1e3": {"train.weight_decay"},
            "batch256": {"train.batch_size"},
            "weight_decay_1e3_batch128": {"train.batch_size", "train.weight_decay"},
            "weight_decay_1e4_batch128": {"train.batch_size", "train.weight_decay"},
        },
    }
    for task, runs in data.items():
        control = runs["lr_1e3"]["config"]
        if control["task"] != task or control["seed"] != 42:
            raise ValueError(f"Unexpected task or seed in default {task} run")
        for config_id, expected in expected_diffs[task].items():
            cfg = runs[config_id]["config"]
            if flattened_differences(control, cfg) != expected:
                raise ValueError(
                    f"Unexpected setting changes for {task}/{config_id}: "
                    f"{sorted(flattened_differences(control, cfg))}"
                )
        for config_id, run in runs.items():
            key = METRIC[task]
            selected = best_row(task, run["history"])
            if not np.isclose(run["validation"][key], selected[key], rtol=1e-8, atol=1e-10):
                raise ValueError(f"Saved validation metric does not match history: {task}/{config_id}")


def make_learning_curves(data: dict) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.3))
    panels = [
        ("regression", "macro_ST-RAE", "Regression: validation macro ST-RAE", "Lower is better"),
        ("classification", "macro_MCC", "Classification: validation macro MCC", "Higher is better"),
    ]
    for ax, (task, metric, title, note) in zip(axes, panels):
        for config_id, label, _ in CONFIGS[task]:
            history = data[task][config_id]["history"]
            epochs = [row["epoch"] for row in history]
            values = [row[metric] for row in history]
            selected = best_row(task, history)
            color = COLOR_BY_ID[config_id]
            ax.plot(epochs, values, color=color, linewidth=1.35, label=label)
            ax.scatter([selected["epoch"]], [selected[metric]], color=color, s=18, zorder=3)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.text(0.98, 0.03, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
        ax.grid(True, alpha=0.25, linewidth=0.6)
    legend_items = {}
    for task_configs in CONFIGS.values():
        for config_id, label, _ in task_configs:
            legend_items.setdefault(config_id, (label, COLOR_BY_ID[config_id]))
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=color, linewidth=1.8, label=label)
        for label, color in legend_items.values()
    ]
    labels = [label for label, _ in legend_items.values()]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=3, frameon=False, fontsize=7.5)
    fig.suptitle("Morgan MLP validation curves across training choices", fontsize=11)
    fig.tight_layout(rect=(0, 0.2, 1, 0.93))
    png = FIGURE_DIR / "task2_all_validation_learning_curves.png"
    pdf = png.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def make_score_comparison(data: dict) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), constrained_layout=True)
    for ax, task, title, note in [
        (axes[0], "regression", "Regression: macro ST-RAE", "Lower is better"),
        (axes[1], "classification", "Classification: macro MCC", "Higher is better"),
    ]:
        task_configs = CONFIGS[task]
        labels = [label for _, label, _ in task_configs]
        scores = [data[task][config_id]["validation"][METRIC[task]] for config_id, _, _ in task_configs]
        colors = [COLOR_BY_ID[config_id] for config_id, _, _ in task_configs]
        bars = ax.bar(range(len(scores)), scores, color=colors, width=0.68)
        offset = max(abs(value) for value in scores) * 0.018
        for bar, value in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(METRIC[task])
        ax.text(0.98, 0.03, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Best validation score by baseline configuration", fontsize=11)
    png = FIGURE_DIR / "task2_all_validation_score_comparison.png"
    pdf = png.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        task: {config_id: read_run(task, run_name) for config_id, _, run_name in configs}
        for task, configs in CONFIGS.items()
    }
    validate_experiment_design(data)

    print("Validation-only Task 2 results (test files were not opened):")
    for task, configs in CONFIGS.items():
        metric = METRIC[task]
        print(f"\n{task.title()} | {metric} ({'lower' if task == 'regression' else 'higher'} is better)")
        for config_id, label, _ in configs:
            run = data[task][config_id]
            selected = best_row(task, run["history"])
            print(
                f"  {label:22s} score={run['validation'][metric]:.9f} "
                f"best_epoch={selected['epoch']:>3} "
                f"duplicate_history_identical={run['history_duplicate_identical']}"
            )
        best_config = (min if task == "regression" else max)(
            configs, key=lambda item: data[task][item[0]]["validation"][metric]
        )
        print(f"  Selected by validation: {best_config[1]}")
        print("  Per-target primary scores (in configuration order above):")
        targets = [
            target for target, metrics in data[task]["lr_1e3"]["validation"].items()
            if isinstance(metrics, dict)
        ]
        endpoint_metric = "ST-RAE" if task == "regression" else "MCC"
        for target in targets:
            values = [
                data[task][config_id]["validation"][target][endpoint_metric]
                for config_id, _, _ in configs
            ]
            print(f"    {target}: " + ", ".join(f"{value:.6f}" for value in values))

    print("\nPlots:")
    for path in (*make_learning_curves(data), *make_score_comparison(data)):
        print(path)


if __name__ == "__main__":
    main()
