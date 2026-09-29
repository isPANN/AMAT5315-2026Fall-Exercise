"""Run with: uv run --no-project --with numpy --with matplotlib python week5/scripts/compare_checkpoints.py"""

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


week = Path(__file__).resolve().parents[1]
artifacts = week / "artifacts"
reference = np.load(artifacts / "adjoint/image.npy")
reference_norm = np.linalg.norm(reference)
for budget in (1, 3, 5, 10):
    run = artifacts / f"checkpoint-{budget}"
    image = np.load(run / "image.npy")
    assert image.shape == reference.shape
    error = np.linalg.norm(image - reference) / reference_norm
    statistics = json.loads((run / "result.json").read_text())["statistics"]
    assert statistics["checkpoints"] == budget
    print(f"Budget {budget}: relative L2 error {error:.12e}; "
          f"peak saved states {statistics['peak_saved_states']}")

missing = duplicated = invalid_restores = budget_overruns = 0
order_violations = 0
files_audited = 0
for run in sorted({path.parent for path in artifacts.glob("*/actions-*.json")}):
    result = json.loads((run / "result.json").read_text())
    statistics = result["statistics"]
    budget = statistics["checkpoints"]
    files = sorted(run.glob("actions-*.json"))
    assert {path.name for path in files} == {
        shot["actions_file"] for shot in statistics["per_shot"]
    }
    run_calls = run_grads = 0
    run_peak = 0
    for path in files:
        shot = int(path.stem.split("-")[1])
        shot_stats = statistics["per_shot"][shot]
        actions = json.loads(path.read_text())
        saved = {0}
        working = 0
        grads = []
        calls = 0
        peak = 1
        for entry in actions:
            action, step = entry["action"], entry["step"]
            if action == "restore":
                if step not in saved:
                    invalid_restores += 1
                working = step
            elif action == "call":
                assert step == working
                working += 1
                calls += 1
            elif action == "store":
                assert step == working and step not in saved
                saved.add(step)
            elif action == "grad":
                assert step in saved
                grads.append(step)
            elif action == "fetch":
                assert step in saved and step != 0
                saved.remove(step)
            else:
                raise ValueError(f"unknown action {action}")
            assert entry["saved_states"] == len(saved)
            budget_overruns += len(saved) > budget + 1
            peak = max(peak, len(saved))
        counts = Counter(grads)
        missing += len(set(range(result["steps"])) - counts.keys())
        duplicated += sum(count - 1 for count in counts.values() if count > 1)
        order_violations += sum(next_step != step - 1
                                for step, next_step in zip(grads, grads[1:]))
        assert saved == {0}
        assert calls == shot_stats["scheduler_forward_calls"]
        assert len(grads) == shot_stats["reverse_calls"]
        assert peak == shot_stats["peak_saved_states"]
        run_calls += calls
        run_grads += len(grads)
        run_peak = max(run_peak, peak)
        files_audited += 1
    assert run_calls == statistics["scheduler_forward_calls"]
    assert run_grads == statistics["reverse_calls"]
    assert run_peak == statistics["peak_saved_states"]

print(f"Audited action files: {files_audited}")
print(f"Missing reverse steps: {missing}")
print(f"Duplicated reverse steps: {duplicated}")
print(f"Reverse steps out of descending order: {order_violations}")
print(f"Invalid restores: {invalid_restores}")
print(f"Budget overruns: {budget_overruns}")

actions = json.loads((artifacts / "checkpoint-5/actions-0.json").read_text())
fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
for action, color, size in (
    ("call", "#64748b", 6),
    ("restore", "#7c3aed", 18),
    ("store", "#16a34a", 22),
    ("grad", "#ea580c", 14),
    ("fetch", "#dc2626", 20),
):
    indices = [i for i, entry in enumerate(actions) if entry["action"] == action]
    steps = [actions[i]["step"] for i in indices]
    ax.scatter(indices, steps, s=size, c=color, label=action, alpha=0.85,
               linewidths=0)
ax.set(title="Treeverse actions · shot 0 · budget 5", xlabel="Operation index",
       ylabel="Timestep")
ax.legend(ncol=5, loc="upper center")
ax.grid(alpha=0.2)
fig.savefig(artifacts / "checkpoint-actions.png", dpi=180)
plt.close(fig)
