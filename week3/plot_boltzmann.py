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
fig, (histogram, ratio) = plt.subplots(2, 1, figsize=(7.4, 7.2), sharex=True, layout="constrained")
colors = ("#176ca4", "#b64b32")
for temperature, count, color in zip(temperatures, counts, colors):
    histogram.stairs(count, edges, label=f"T = {temperature:.1f}", color=color, linewidth=2)
histogram.set(ylabel="Rows per 40-unit bin", title="Ising total-energy distributions")
histogram.legend(frameon=False)

x = centers[eligible]
ratio.scatter(x, log_ratio, color="#333333", s=24, label="Bins with at least 5 rows in each run")
ratio.plot(x, intercept + slope * x, color="#6a3d9a", linewidth=2,
           label=rf"Slope $1/3.0 - 1/3.1 = {slope:.5f}$")
ratio.axhline(0, color="#aaaaaa", linewidth=0.8)
ratio.set(xlabel="Total energy", ylabel=r"$\log[N_{3.1}(E)/N_{3.0}(E)]$")
ratio.legend(frameon=False)
for axis in (histogram, ratio):
    axis.grid(color="#dedede", linewidth=0.7)

(root / "evidence").mkdir(exist_ok=True)
fig.savefig(root / "evidence" / "boltzmann.png", dpi=200)
