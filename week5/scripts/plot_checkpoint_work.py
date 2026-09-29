"""Run with: uv run --no-project --with matplotlib python week5/scripts/plot_checkpoint_work.py"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


week = Path(__file__).resolve().parents[1]
artifacts = week / "artifacts"
records = []
for path in artifacts.glob("checkpoint-*/result.json"):
    statistics = json.loads(path.read_text())["statistics"]
    assert statistics["storage"] == "treeverse"
    calls = {shot["scheduler_forward_calls"] for shot in statistics["per_shot"]}
    assert len(calls) == 1
    records.append((statistics["checkpoints"], calls.pop(), statistics["peak_saved_bytes"]))
records.sort()
budgets, forward_steps, storage_bytes = zip(*records)

full = json.loads((artifacts / "adjoint/result.json").read_text())["statistics"]
assert full["storage"] == "full"
full_calls = {shot["scheduler_forward_calls"] for shot in full["per_shot"]}
assert len(full_calls) == 1
full_steps = full_calls.pop()
full_bytes = full["peak_saved_bytes"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
axes[0].plot(budgets, forward_steps, "o-", color="#2563eb", label="Treeverse")
axes[0].axhline(full_steps, color="#dc2626", linestyle="--", linewidth=1.5,
                label=f"Full history ({full_steps:,})")
axes[0].set(title="Forward work per shot", xlabel="Checkpoint budget (additional slots)",
            ylabel="Forward steps per shot", yscale="log", xticks=budgets)
axes[0].grid(alpha=0.25, which="both")
axes[0].legend()

axes[1].plot(budgets, storage_bytes, "o-", color="#16a34a", label="Treeverse")
axes[1].axhline(full_bytes, color="#dc2626", linestyle="--", linewidth=1.5,
                label=f"Full history ({full_bytes:,} B)")
axes[1].set(title="Peak saved-state storage", xlabel="Checkpoint budget (additional slots)",
            ylabel="Peak saved-state storage (bytes)", xticks=budgets)
axes[1].set_ylim(0, full_bytes * 1.08)
axes[1].grid(alpha=0.25)
axes[1].legend(loc="upper right")

detail = axes[1].inset_axes([0.27, 0.33, 0.63, 0.42])
detail.plot(budgets, storage_bytes, "o-", color="#16a34a", markersize=4)
detail.set(title="Checkpoint detail", xticks=budgets)
detail.tick_params(labelsize=8)
detail.grid(alpha=0.2)

fig.savefig(artifacts / "checkpoint-work.png", dpi=180)
plt.close(fig)
