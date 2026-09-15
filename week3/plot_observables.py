"""Draw with: uv run --with matplotlib week3/plot_observables.py"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).parent
samples = defaultdict(lambda: [0, 0.0, 0.0])

for metadata_path in sorted((root / "artifacts").glob("*/run.json")):
    metadata = json.loads(metadata_path.read_text())
    if metadata["update"] != "metropolis":
        continue
    run = defaultdict(int)
    with (metadata_path.parent / "series.jsonl").open() as rows:
        for line in rows:
            row = json.loads(line)
            key = (row["L"], row["T"])
            run[row["T"]] += 1
            samples[key][0] += 1
            samples[key][1] += abs(row["M"])
            samples[key][2] += row["M"] ** 2
    expected = {round(temperature, 6): metadata["measure"] for temperature in metadata["t_grid"]}
    if run != expected:
        raise RuntimeError(f"ramp is incomplete: {metadata_path.parent}")

series = defaultdict(list)
for (size, temperature), (count, sum_abs_m, sum_m2) in samples.items():
    mean_abs_m = sum_abs_m / count
    susceptibility = size**2 * (sum_m2 / count - mean_abs_m**2) / temperature
    series[size].append((temperature, mean_abs_m, susceptibility))

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

def draw(column, ylabel, output):
    fig, axis = plt.subplots(figsize=(7.4, 4.8), layout="constrained")
    for size, values in sorted(series.items()):
        values.sort()
        axis.plot(
            [value[0] for value in values],
            [value[column] for value in values],
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=f"L = {size}",
        )
    axis.set(xlabel="Temperature T", ylabel=ylabel)
    axis.grid(color="#dedede", linewidth=0.7)
    axis.legend(frameon=False)
    fig.savefig(root / "evidence" / output, dpi=200)


draw(1, r"Mean absolute magnetization $\langle |M| \rangle$", "magnetization.png")
draw(2, r"Absolute-magnetization susceptibility $\chi_{|M|}$", "susceptibility.png")
