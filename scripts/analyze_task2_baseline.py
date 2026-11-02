"""Summarize Task 2 validation runs and create report-ready validation plots.

This script reads only config.json, validation_metrics.json, and history.jsonl.
It deliberately does not open test_metrics.json or test_predictions.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
FIGURE_DIR = (
    PROJECT_ROOT.parent
    / "02456_report_template"
    / "02456_report_template"
    / "figures"
    / "baseline"
)

RUNS = {
    "regression": {
        "1e-3": "morgan_regression_default",
        "3e-4": "morgan_regression_low_lr",
    },
    "classification": {
        "1e-3": "morgan_classification_default",
        "3e-4": "morgan_classification_low_lr",
    },
}
PRIMARY = {"regression": ("macro_ST-RAE", "min"), "classification": ("macro_MCC", "max")}
COLORS = {"1e-3": "#0072B2", "3e-4": "#D55E00"}


def read_run(task: str, run_name: str) -> dict:
    path = RUNS_ROOT / task / "morgan_mlp" / run_name
    with (path / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    with (path / "validation_metrics.json").open(encoding="utf-8") as stream:
        validation = json.load(stream)
    with (path / "history.jsonl").open(encoding="utf-8") as stream:
        history = [json.loads(line) for line in stream if line.strip()]
    return {"path": path, "config": config, "validation": validation, "history": history}


def best_row(task: str, history: list[dict]) -> dict:
    metric, direction = PRIMARY[task]
    key = lambda row: row[metric]
    return (min if direction == "min" else max)(history, key=key)


def compare_configurations(task: str, default: dict, lower: dict) -> None:
    a, b = default["config"], lower["config"]
    a_model, b_model = dict(a["model"]), dict(b["model"])
    a_lr, b_lr = a_model.pop("readout_lr"), b_model.pop("readout_lr")
    comparable = (a_model, a["train"], a["readout"], a["seed"], a["split_sha256"])
    other = (b_model, b["train"], b["readout"], b["seed"], b["split_sha256"])
    if comparable != other:
        raise ValueError(f"Unexpected configuration differences in {task} besides learning rate")
    if not np.isclose(a_lr, 1e-3) or not np.isclose(b_lr, 3e-4):
        raise ValueError(f"Unexpected learning-rate values in {task}: {a_lr}, {b_lr}")


def build_summary(data: dict) -> list[dict]:
    summary = []
    for task, settings in data.items():
        metric, direction = PRIMARY[task]
        for lr, run in settings.items():
            selected = best_row(task, run["history"])
            saved_score = run["validation"][metric]
            if not np.isclose(saved_score, selected[metric], rtol=1e-8, atol=1e-10):
                raise ValueError(f"Saved validation score does not match history for {task}, lr={lr}")
            summary.append({
                "task": task,
                "learning_rate": lr,
                "metric": metric,
                "score": float(saved_score),
                "best_epoch": int(selected["epoch"]),
                "epochs": len(run["history"]),
                "direction": direction,
            })
    return summary


def make_learning_curves(data: dict) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)
    panels = [
        ("regression", "macro_ST-RAE", "Regression: validation macro ST-RAE", "Lower is better"),
        ("classification", "macro_MCC", "Classification: validation macro MCC", "Higher is better"),
    ]
    for ax, (task, metric, title, note) in zip(axes, panels):
        for lr, run in data[task].items():
            epochs = [row["epoch"] for row in run["history"]]
            values = [row[metric] for row in run["history"]]
            selected = best_row(task, run["history"])
            ax.plot(epochs, values, color=COLORS[lr], linewidth=1.6, label=f"LR {lr}")
            ax.scatter([selected["epoch"]], [selected[metric]], color=COLORS[lr], s=24, zorder=3)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.text(0.98, 0.03, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Morgan MLP validation learning curves", fontsize=11)
    out = FIGURE_DIR / "task2_validation_learning_curves.png"
    pdf_out = out.with_suffix(".pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)
    return out, pdf_out


def make_score_comparison(data: dict) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.15), constrained_layout=True)
    panels = [
        ("regression", "macro_ST-RAE", "Regression", "Lower is better"),
        ("classification", "macro_MCC", "Classification", "Higher is better"),
    ]
    labels = ["1e-3\n(default)", "3e-4\n(lower LR)"]
    lrs = ["1e-3", "3e-4"]
    for ax, (task, metric, title, note) in zip(axes, panels):
        scores = [data[task][lr]["validation"][metric] for lr in lrs]
        bars = ax.bar(labels, scores, color=[COLORS[lr] for lr in lrs], width=0.62)
        span = max(scores) - min(scores)
        offset = max(span * 0.04, max(abs(value) for value in scores) * 0.025)
        for bar, value in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{title}: {metric}", fontsize=9)
        ax.set_ylabel(metric)
        ax.text(0.98, 0.03, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Best validation score by learning rate", fontsize=11)
    out = FIGURE_DIR / "task2_validation_score_comparison.png"
    pdf_out = out.with_suffix(".pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)
    return out, pdf_out


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        task: {lr: read_run(task, run_name) for lr, run_name in settings.items()}
        for task, settings in RUNS.items()
    }
    for task, settings in data.items():
        compare_configurations(task, settings["1e-3"], settings["3e-4"])
    summary = build_summary(data)

    print("Validation-only comparison (test files were not opened):")
    for row in summary:
        print(
            f"{row['task']:14s} LR={row['learning_rate']:5s} "
            f"{row['metric']}={row['score']:.9f} "
            f"best_epoch={row['best_epoch']} epochs={row['epochs']}"
        )

    for task, settings in data.items():
        print(f"\n{task} per-target validation metrics:")
        for lr, run in settings.items():
            print(f"  LR {lr}:")
            for target, metrics in run["validation"].items():
                if isinstance(metrics, dict):
                    metric_name = "ST-RAE" if task == "regression" else "MCC"
                    print(f"    {target}: {metric_name}={metrics[metric_name]:.6f}")

    print("\nPlots:")
    for output in (*make_learning_curves(data), *make_score_comparison(data)):
        print(output)


if __name__ == "__main__":
    main()
