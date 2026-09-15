"""Draw with: uv run --with matplotlib week3/scripts/magnetization_compare.py"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

root = Path(__file__).parents[1]


def load(name):
    directory = root / "artifacts" / name
    metadata = json.loads((directory / "run.json").read_text())
    samples = defaultdict(list)
    with (directory / "series.jsonl").open() as rows:
        for line in rows:
            row = json.loads(line)
            samples[row["T"]].append(abs(row["M"]))
    expected = {round(temperature, 6): metadata["measure"] for temperature in metadata["t_grid"]}
    if {temperature: len(values) for temperature, values in samples.items()} != expected:
        raise RuntimeError(f"ramp is incomplete: {directory}")
    return metadata, {temperature: np.asarray(values) for temperature, values in samples.items()}


metropolis_metadata, metropolis = load("window-l64")
wolff_runs = {size: load(f"wolff-l{size}") for size in (32, 64)}
if metropolis_metadata["update"] != "metropolis" or any(
    metadata["update"] != "wolff" for metadata, _ in wolff_runs.values()
):
    raise RuntimeError("unexpected update method")

replicates = 500
rng = np.random.default_rng(2026)


def bootstrap_error(values, block_length):
    count = len(values)
    full, remainder = divmod(count, block_length)
    starts = rng.integers(0, count, size=(replicates, full + bool(remainder)))
    wrapped = np.concatenate((values, values[:block_length]))
    prefix = np.concatenate(([0.0], np.cumsum(wrapped)))
    block_sums = prefix[np.arange(count) + block_length] - prefix[np.arange(count)]
    totals = block_sums[starts[:, :full]].sum(axis=1)
    if remainder:
        remainder_sums = prefix[np.arange(count) + remainder] - prefix[np.arange(count)]
        totals += remainder_sums[starts[:, -1]]
    return (totals / count).std(ddof=1)


plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, (magnetization, susceptibility) = plt.subplots(1, 2, figsize=(12.0, 4.8), layout="constrained")
colors = {32: "#176ca4", 64: "#b64b32"}
algorithm_handles = []
for label, samples, color in (
    ("Metropolis", metropolis, "#6a3d9a"),
    ("Wolff", wolff_runs[64][1], colors[64]),
):
    temperatures = np.array(sorted(samples))
    means = np.array([samples[temperature].mean() for temperature in temperatures])
    algorithm_handles += magnetization.plot(temperatures, means, color=color, linewidth=1.5,
                                              label=label)
    for offset, block_length, marker in zip((-0.006, 0, 0.006), (2000, 4000, 8000), ("o", "s", "^")):
        errors = [bootstrap_error(samples[temperature], block_length) for temperature in temperatures]
        magnetization.errorbar(temperatures + offset, means, yerr=errors, color=color, marker=marker,
                               linestyle="none", markersize=3.5, capsize=2, alpha=0.85)
magnetization.set(xlabel="Temperature T", ylabel=r"Mean absolute magnetization $\langle |M| \rangle$",
                  title="L = 64")
block_handles = [
    Line2D([], [], color="#333333", marker=marker, linestyle="none", label=f"Block {length}")
    for length, marker in zip((2000, 4000, 8000), ("o", "s", "^"))
]
magnetization.legend(handles=algorithm_handles + block_handles, frameon=False, ncol=2)

peaks = {}
for size, (_, samples) in wolff_runs.items():
    temperatures = np.array(sorted(samples))
    chi = np.array([
        size**2 * (np.mean(values**2) - np.mean(values) ** 2) / temperature
        for temperature, values in sorted(samples.items())
    ])
    susceptibility.plot(temperatures, chi, marker="o", markersize=4, linewidth=1.4,
                        color=colors[size], label=f"L = {size}")
    largest = np.argmax(chi)
    if largest < 2 or largest + 2 >= len(chi):
        raise RuntimeError(f"L={size} chi maximum lacks two temperatures on each side")
    selected = slice(largest - 2, largest + 3)
    center = temperatures[largest]
    coefficients = np.polyfit(temperatures[selected] - center, chi[selected], 2)
    peak = center - coefficients[1] / (2 * coefficients[0])
    if coefficients[0] >= 0 or not temperatures[selected][0] <= peak <= temperatures[selected][-1]:
        raise RuntimeError(f"L={size} quadratic has no maximum within its five points")
    peaks[size] = peak
    grid = np.linspace(temperatures[selected][0], temperatures[selected][-1], 200)
    fit = np.polyval(coefficients, grid - center)
    susceptibility.plot(grid, fit, color=colors[size], linestyle="--", linewidth=1.8)
    susceptibility.scatter(peak, np.polyval(coefficients, peak - center), color=colors[size],
                           marker="x", s=45, zorder=3)

extrapolated = 2 * peaks[64] - peaks[32]
susceptibility.axvline(2.26919, color="#333333", linestyle=":", linewidth=1.5,
                       label="Exact 2.26919")
susceptibility.axvline(extrapolated, color="#2f8f46", linestyle="-.", linewidth=1.5,
                       label=f"Extrapolated {extrapolated:.5f}")
susceptibility.set(xlabel="Temperature T", ylabel=r"Susceptibility $\chi(T)$",
                   title="Wolff cluster updates")
susceptibility.legend(frameon=False)
for axis in (magnetization, susceptibility):
    axis.grid(color="#dedede", linewidth=0.7)

fig.savefig(root / "evidence" / "magnetization-compare.png", dpi=200)
