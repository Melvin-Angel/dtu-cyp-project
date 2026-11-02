"""Audit duplicated Task 2 HPC downloads without reading any test artifacts.

Reads staged config/validation/history files and per-job stdout only. For stdout,
only standalone JSON lines containing an epoch field are parsed; held-out test
summary lines are ignored.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "hpc_ablation_import_2026-11-02"
RUNS = STAGE / "runs"
OUTPUTS = STAGE / "outputs"
ABLATIONS = {
    "regression": [
        "morgan_regression_arch_small",
        "morgan_regression_batch128",
        "morgan_regression_weight_decay_1e3",
    ],
    "classification": [
        "morgan_classification_arch_small",
        "morgan_classification_batch128",
        "morgan_classification_weight_decay_1e3",
    ],
}
METRIC = {"regression": "macro_ST-RAE", "classification": "macro_MCC"}
REG_TARGETS = {
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
}
CLS_TARGETS = {"CYP2D6_is_TDI", "CYP3A4_is_TDI"}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def best(history: list[dict], task: str) -> dict:
    metric = METRIC[task]
    return (min if task == "regression" else max)(history, key=lambda row: row[metric])


def validate_config(task: str, name: str, cfg: dict) -> None:
    model, readout, train = cfg["model"], cfg["readout"], cfg["train"]
    if cfg["task"] != task or model["name"] != "morgan_mlp":
        raise ValueError(f"Unexpected model/task in {name}")
    if not np.isclose(model["readout_lr"], 1e-3):
        raise ValueError(f"Unexpected learning rate in {name}: {model['readout_lr']}")
    if name.endswith("arch_small"):
        valid = readout["hidden_dim"] == 256 and readout["num_layers"] == 2
    elif name.endswith("batch128"):
        valid = train["batch_size"] == 128
    elif name.endswith("weight_decay_1e3"):
        valid = np.isclose(train["weight_decay"], 1e-3)
    else:
        raise ValueError(f"Unknown ablation run: {name}")
    if not valid:
        raise ValueError(f"Saved config does not match expected ablation: {name}")
    if cfg["seed"] != 42:
        raise ValueError(f"Unexpected seed in {name}: {cfg['seed']}")


def near_equal(a: list[dict], b: list[dict], task: str) -> bool:
    if len(a) != len(b):
        return False
    metric = METRIC[task]
    return all(
        x["epoch"] == y["epoch"] and np.isclose(x[metric], y[metric], rtol=0, atol=1e-10)
        for x, y in zip(a, b)
    )


def read_epoch_log(path: Path) -> tuple[str | None, list[dict]]:
    rows = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            stripped = line.strip()
            # Do not parse non-epoch lines (including the held-out test summary).
            if not stripped.startswith("{") or '"epoch"' not in stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and "epoch" in item:
                rows.append(item)
    if not rows:
        return None, rows
    keys = set(rows[0])
    task = "regression" if keys & REG_TARGETS else "classification" if keys & CLS_TARGETS else None
    return task, rows


def main() -> None:
    if not RUNS.is_dir() or not OUTPUTS.is_dir():
        raise FileNotFoundError(f"Expected staged runs and outputs under {STAGE}")

    chunks_by_run = {}
    print("Staged run-directory audit (test artifacts not opened):")
    for task, names in ABLATIONS.items():
        for name in names:
            folder = RUNS / task / "morgan_mlp" / name
            cfg = read_json(folder / "config.json")
            validate_config(task, name, cfg)
            history = read_jsonl(folder / "history.jsonl")
            if len(history) != 200:
                raise ValueError(f"Expected two 100-epoch histories in {name}; found {len(history)} rows")
            first, second = history[:100], history[100:]
            expected_epochs = list(range(1, 101))
            if [row["epoch"] for row in first] != expected_epochs or [row["epoch"] for row in second] != expected_epochs:
                raise ValueError(f"History blocks are not two clean 1..100 epoch sequences: {name}")
            duplicate = near_equal(first, second, task)
            validation = read_json(folder / "validation_metrics.json")
            metric = METRIC[task]
            best_first, best_second = best(first, task), best(second, task)
            saved = validation[metric]
            saved_matches = [
                label for label, item in (("first", best_first), ("second", best_second))
                if np.isclose(saved, item[metric], rtol=1e-8, atol=1e-10)
            ]
            if not saved_matches:
                raise ValueError(f"Saved validation score does not match either history block: {name}")
            chunks_by_run[(task, name)] = (first, second)
            print(
                f"{name}: duplicate_histories_identical={duplicate}; "
                f"best_first=epoch{best_first['epoch']} {metric}={best_first[metric]:.9f}; "
                f"best_second=epoch{best_second['epoch']} {metric}={best_second[metric]:.9f}; "
                f"saved_validation_matches={','.join(saved_matches)}"
            )

    print("\nPer-job stdout validation audit (only epoch JSON lines parsed):")
    log_count = 0
    for path in sorted(OUTPUTS.glob("cyp_298728*.out")):
        task, rows = read_epoch_log(path)
        if not rows:
            continue
        log_count += 1
        metric = METRIC[task]
        winner = best(rows, task)
        signatures = []
        for (run_task, name), (first, second) in chunks_by_run.items():
            if run_task != task:
                continue
            if near_equal(rows, first, task):
                signatures.append(name)
            elif near_equal(rows, second, task):
                signatures.append(name)
        print(
            f"{path.stem}: task={task}, epochs={len(rows)}, "
            f"best=epoch{winner['epoch']} {metric}={winner[metric]:.9f}, "
            f"matched_run={','.join(signatures) if signatures else 'no exact history match'}"
        )
    print(f"Parsed {log_count} ablation stdout logs. Test metrics were not parsed.")


if __name__ == "__main__":
    main()
