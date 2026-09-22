# /// script
# dependencies = ["matplotlib", "numpy"]
# ///

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "convergence"


def final_vorticity(dt):
    frames = [
        json.loads(line)
        for line in (ARTIFACTS / f"rk4-dt{dt}" / "fields.jsonl").read_text().splitlines()
    ]
    assert frames[-1]["t"] == 2.0
    omega = np.asarray(frames[-1]["omega"])
    assert omega.size == 128 * 128 and np.all(np.isfinite(omega))
    return omega


reference_dt = "0.0025"
reference = final_vorticity(reference_dt)
steps = np.array([0.02, 0.0125, 0.01])
errors = []
for dt in steps:
    omega = final_vorticity(f"{dt:g}")
    error = np.linalg.norm(omega - reference) / np.linalg.norm(reference)
    errors.append(error)
    print(f"RK4 dt={dt:g}: relative omega error = {error:.17e}")
errors = np.asarray(errors)
slope, intercept = np.polyfit(np.log(steps), np.log(errors), 1)
assert 3.5 < slope < 4.5
print(f"fitted log-log slope = {slope:.17e}")

result = {
    "quantity": "omega",
    "time": 2.0,
    "reference": {"dt": float(reference_dt), "relative_error": 0.0},
    "runs": [
        {"dt": float(dt), "relative_error": float(error)}
        for dt, error in zip(steps, errors)
    ],
    "fitted_log_log_slope": float(slope),
}
output = ROOT / "evidence" / "convergence.json"
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(result, indent=2) + "\n")
print(output)

richardson_at_001 = (
    np.linalg.norm(final_vorticity("0.01") - final_vorticity("0.02"))
    / 15
    / np.linalg.norm(reference)
)
predicted = richardson_at_001 * (steps / 0.01) ** 4
eligible = np.flatnonzero(predicted < 5e-6)
chosen = eligible[np.argmax(steps[eligible])]
assert steps[chosen] == 0.0125
print(
    f"chosen dt={steps[chosen]:g}: predicted error = {predicted[chosen]:.17e}, "
    f"measured error = {errors[chosen]:.17e}"
)

fit_steps = np.geomspace(steps.min(), steps.max(), 100)
fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
ax.loglog(steps, errors, "o", label="measured errors")
ax.loglog(
    fit_steps,
    np.exp(intercept) * fit_steps**slope,
    "--",
    label=rf"log-log fit, slope {slope:.2f}",
)
ax.scatter(
    steps[chosen],
    errors[chosen],
    s=150,
    facecolors="none",
    edgecolors="tab:red",
    linewidths=2,
    label=rf"chosen $\Delta t={steps[chosen]:g}$",
)
ax.set(xlabel=r"time step $\Delta t$", ylabel=r"relative vorticity error")
ax.grid(which="both", alpha=0.25)
ax.legend(frameon=False)

figure = ROOT / "evidence" / "convergence.png"
fig.savefig(figure, dpi=220)
print(figure)
