# /// script
# dependencies = ["numpy"]
# ///

import json
from pathlib import Path

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
slope = np.polyfit(np.log(steps), np.log(errors), 1)[0]
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
