"""Draw with: uv run --with matplotlib --with numpy week3/plot_boltzmann.py"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).parent
temperatures = (3.0, 3.1)
energies = []
for temperature in temperatures:
    path = root / "runs" / f"T{temperature:.1f}" / "series.jsonl"
    with path.open() as rows:
        energies.append(np.array([round(json.loads(row)["E"] * 4096) for row in rows]))

width = 40
low = np.floor(min(values.min() for values in energies) / width) * width
high = np.ceil(max(values.max() for values in energies) / width) * width
edges = np.arange(low, high + width, width)
centers = (edges[:-1] + edges[1:]) / 2
counts = [np.histogram(values, bins=edges)[0] for values in energies]
eligible = (counts[0] >= 5) & (counts[1] >= 5)
log_ratio = np.log(counts[1][eligible] / counts[0][eligible])
slope = 1 / temperatures[0] - 1 / temperatures[1]
intercept = np.mean(log_ratio - slope * centers[eligible])

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, (histogram, ratio) = plt.subplots(1, 2, figsize=(9.2, 4.2), layout="constrained")
colors = ("#176ca4", "#b64b32")
for temperature, count, color in zip(temperatures, counts, colors):
    histogram.stairs(count, edges, label=f"T = {temperature:.1f}", color=color, linewidth=2)
histogram.set(xlabel="Total energy E", ylabel="Sweeps per 40-unit bin")
histogram.legend(frameon=False)

x = centers[eligible]
ratio.scatter(x, log_ratio, color="#36588c", s=24, label="At least 5 sweeps in each bin")
ratio.plot(x, intercept + slope * x, color="#555555", linewidth=1.5, linestyle="--",
           label=rf"Slope $1/3.0 - 1/3.1 = {slope:.5f}$")
ratio.axhline(0, color="#aaaaaa", linewidth=0.8)
ratio.set(xlabel="Total energy E", ylabel=r"$\ln[P_{3.1}(E)/P_{3.0}(E)]$", ylim=(-3.2, 3.2))
ratio.legend(frameon=False)
view = (np.floor((x.min() - width) / 100) * 100, np.ceil((x.max() + width) / 100) * 100)
for axis in (histogram, ratio):
    axis.set_xlim(view)
    axis.set_xticks(np.arange(view[0], view[1] + 1, 200))
    axis.grid(color="#dedede", linewidth=0.7)

(root / "evidence").mkdir(exist_ok=True)
fig.savefig(root / "evidence" / "boltzmann.png", dpi=200)
