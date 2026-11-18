"""Plot the validation metric across epochs for one run."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

run = Path(sys.argv[1])
task = "classification" if "classification" in str(run) else "regression"
metric = "macro_MCC" if task == "classification" else "macro_ST-RAE"
best = max if task == "classification" else min

history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines() if line.strip()]
epochs = [row["epoch"] for row in history]
scores = [row[metric] for row in history]
peak = best(history, key=lambda row: row[metric])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5.5), sharex=True)

ax1.plot(epochs, losses, marker="o", ms=4, color="#b45309")
ax1.set_ylabel("training loss")

ax2.plot(epochs, scores, marker="o", ms=4, color="#0f766e")
ax2.set_ylabel(f"validation {metric}")
ax2.set_xlabel("epoch")

for ax in (ax1, ax2):
    ax.axvline(peak["epoch"], color="#555", linestyle="--", lw=1)

