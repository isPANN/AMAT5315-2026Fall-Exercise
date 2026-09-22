# /// script
# dependencies = ["matplotlib", "numpy"]
# ///

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "artifacts" / "scan"
initial = json.loads((SCAN / "random-initial.json").read_text())
largest_speed = np.hypot(initial["u"], initial["v"]).max()
print(f"largest speed of random initial field: {largest_speed:.17e}")

cases = {
    "Taylor–Green, $n=64$, $\\nu=0.1$": [
        ("taylor-rk4-0.032", "RK4, $\\Delta t=0.032$"),
        ("taylor-rk4-0.033", "RK4, $\\Delta t=0.033$"),
    ],
    "Random, $n=128$, $\\nu=0.004$": [
        ("random-rk4-0.038", "RK4, $\\Delta t=0.038$"),
        ("random-rk4-0.040", "RK4, $\\Delta t=0.040$"),
        ("random-euler-0.01", "Euler, $\\Delta t=0.01$"),
    ],
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
for ax, (title, runs) in zip(axes, cases.items()):
    for name, label in runs:
        data = np.genfromtxt(SCAN / f"{name}.tsv", names=True, delimiter="\t", ndmin=1)
        finite = np.isfinite(data["E"]) & (data["E"] > 0)
        line = ax.semilogy(data["t"][finite], data["E"][finite], marker="o", label=label)[0]
        if not np.isfinite(data["E"][-1]):
            stopped = data["t"][-1]
            last_energy = data["E"][finite][-1]
            ax.axvline(stopped, color=line.get_color(), linestyle=":", alpha=0.7)
            ax.scatter(stopped, last_energy, color=line.get_color(), marker="x", s=55, zorder=3)
            above = name == "random-euler-0.01"
            ax.annotate(
                f"stopped $t={stopped:g}$",
                (stopped, last_energy),
                xytext=(-6, 8 if above else -8),
                textcoords="offset points",
                ha="right",
                va="bottom" if above else "top",
                color=line.get_color(),
                fontsize=9,
            )
    ax.set(xlabel=r"$t$", ylabel=r"energy $E$", title=title)
    ax.margins(y=0.12)
    ax.grid(which="both", alpha=0.2)
    ax.legend(frameon=False)

output = ROOT / "evidence" / "blowup.png"
output.parent.mkdir(exist_ok=True)
fig.savefig(output, dpi=220)
print(output)
