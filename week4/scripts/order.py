# /// script
# dependencies = ["matplotlib", "numpy"]
# ///

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ORDER = ROOT / "artifacts" / "order"
exact = json.loads((ORDER / "exact-t2.json").read_text())
exact_velocity = np.r_[exact["u"], exact["v"]]

steps = np.array([0.4, 0.25, 0.2])
errors = []
for dt in steps:
    frames = [
        json.loads(line)
        for line in (ORDER / f"rk4-dt{dt:g}" / "fields.jsonl").read_text().splitlines()
    ]
    assert frames[-1]["t"] == 2.0
    computed_velocity = np.r_[frames[-1]["u"], frames[-1]["v"]]
    error = np.linalg.norm(computed_velocity - exact_velocity) / np.linalg.norm(exact_velocity)
    errors.append(error)
    print(f"RK4 dt={dt:g}: relative velocity error = {error:.17e}")
errors = np.asarray(errors)
slope, intercept = np.polyfit(np.log(steps), np.log(errors), 1)
assert 3.5 < slope < 4.5

fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
fit_steps = np.geomspace(steps.min(), steps.max(), 100)
ax.loglog(steps, errors, "o", label="RK4 errors")
ax.loglog(
    fit_steps,
    np.exp(intercept) * fit_steps**slope,
    "--",
    label=rf"log-log fit, slope {slope:.2f}",
)
ax.set(xlabel=r"time step $\Delta t$", ylabel="relative velocity error")
ax.grid(which="both", alpha=0.25)
ax.legend(frameon=False)

output = ROOT / "evidence" / "order.png"
output.parent.mkdir(exist_ok=True)
fig.savefig(output, dpi=220)
print(output)
