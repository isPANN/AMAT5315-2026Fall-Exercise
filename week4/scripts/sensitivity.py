# /// script
# dependencies = ["matplotlib", "numpy"]
# ///

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "sensitivity"


def relative_distances(case):
    original_lines = (ARTIFACTS / case / "original.jsonl").read_text().splitlines()
    perturbed_lines = (ARTIFACTS / case / "perturbed.jsonl").read_text().splitlines()
    assert len(original_lines) == len(perturbed_lines) == 41
    times = []
    distances = []
    for original_line, perturbed_line in zip(original_lines, perturbed_lines):
        original = json.loads(original_line)
        perturbed = json.loads(perturbed_line)
        assert original["t"] == perturbed["t"]
        original_omega = np.asarray(original["omega"])
        perturbed_omega = np.asarray(perturbed["omega"])
        times.append(original["t"])
        distances.append(
            np.linalg.norm(perturbed_omega - original_omega)
            / np.linalg.norm(original_omega)
        )
    return np.asarray(times), np.asarray(distances)


fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
for case, label in [
    ("taylor-green", r"Taylor–Green, $n=64$, $\nu=0.1$"),
    ("random", r"Random, $n=128$, $\nu=0.004$"),
]:
    times, distances = relative_distances(case)
    assert np.all(np.isfinite(distances)) and np.all(distances > 0)
    ax.semilogy(times, distances, marker="o", markersize=3, label=label)

ax.set(
    xlabel=r"$t$",
    ylabel=r"$\|\omega_\delta-\omega\|_2/\|\omega\|_2$",
    title=(
        r"Sensitivity to $\delta\omega=-7\times10^{-5}M\cos(3x)\cos(4y)$"
        "\n"
        r"RK4, $\Delta t=0.01$"
    ),
)
ax.grid(which="both", alpha=0.2)
ax.legend(frameon=False)

output = ROOT / "evidence" / "sensitivity.png"
output.parent.mkdir(exist_ok=True)
fig.savefig(output, dpi=220)
print(output)
