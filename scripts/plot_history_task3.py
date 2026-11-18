"""Plot training loss and the validation metric across epochs for one run.

Usage:
    python scripts/plot_history_task3.py runs/regression/atom_jepa/<timestamp> [name]

Reads history.jsonl only; never opens test metrics or test predictions.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

run = Path(sys.argv[1])
file_name = sys.argv[2] if len(sys.argv) > 2 else "history"

task = "classification" if "classification" in str(run) else "regression"
metric = "macro_MCC" if task == "classification" else "macro_ST-RAE"
best = max if task == "classification" else min

history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines() if line.strip()]
epochs = [row["epoch"] for row in history]
losses = [row["train_loss"] for row in history]
scores = [row[metric] for row in history]
peak = best(history, key=lambda row: row[metric])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5.5), sharex=True)

ax1.plot(epochs, losses, marker="o", ms=4, color="#b45309")
ax1.set_ylabel("training loss")
ax1.set_title(f"{task} / {run.parent.name} / {run.name}", fontsize=11)

ax2.plot(epochs, scores, marker="o", ms=4, color="#0f766e")
ax2.set_ylabel(f"validation {metric}")
ax2.set_xlabel("epoch")
ax2.annotate(
    f"best: epoch {peak['epoch']} = {peak[metric]:.4f}",
    xy=(peak["epoch"], peak[metric]),
    xytext=(6, 6),
    textcoords="offset points",
    fontsize=9,
    color="#333333",
)

for ax in (ax1, ax2):
    ax.axvline(peak["epoch"], color="#555555", linestyle="--", lw=1)
    ax.grid(alpha=0.25, lw=0.6)

ax2.set_xticks(epochs)
fig.tight_layout()

out = run / f"{file_name}.png"
fig.savefig(out, dpi=150)
print("wrote", out)
