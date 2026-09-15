"""Draw with: uv run --with matplotlib week3/scripts/bootstrap.py"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).parents[1]
runs = {}
for metadata_path in sorted((root / "artifacts").glob("window-*/run.json")):
    metadata = json.loads(metadata_path.read_text())
    if metadata["update"] != "metropolis":
        continue
    samples = defaultdict(list)
    with (metadata_path.parent / "series.jsonl").open() as rows:
        for line in rows:
            row = json.loads(line)
            samples[row["T"]].append(abs(row["M"]))
    expected = {round(temperature, 6): metadata["measure"] for temperature in metadata["t_grid"]}
    if {temperature: len(values) for temperature, values in samples.items()} != expected:
        raise RuntimeError(f"ramp is incomplete: {metadata_path.parent}")
    runs[metadata["L"]] = {temperature: np.asarray(values) for temperature, values in samples.items()}


def susceptibility(size, temperature, values):
    return size**2 * (np.mean(values**2) - np.mean(values) ** 2) / temperature


def bootstrap_susceptibility(size, temperature, values, block, rng):
    count = len(values)
    full, remainder = divmod(count, block)
    starts = rng.integers(0, count, size=(500, full + bool(remainder)))

    def resampled_mean(quantity):
        wrapped = np.concatenate((quantity, quantity[:block]))
        prefix = np.concatenate(([0.0], np.cumsum(wrapped)))
        block_sums = prefix[np.arange(count) + block] - prefix[np.arange(count)]
        totals = block_sums[starts[:, :full]].sum(axis=1)
        if remainder:
            remainder_sums = prefix[np.arange(count) + remainder] - prefix[np.arange(count)]
            totals += remainder_sums[starts[:, -1]]
        return totals / count

    mean_abs_m = resampled_mean(values)
    mean_m2 = resampled_mean(values**2)
    return size**2 * (mean_m2 - mean_abs_m**2) / temperature


temperatures = {size: np.array(sorted(samples)) for size, samples in runs.items()}
chi = {
    size: np.array([susceptibility(size, temperature, samples[temperature])
                    for temperature in temperatures[size]])
    for size, samples in runs.items()
}
fit_points = {}
for size, values in chi.items():
    peak = np.argmax(values)
    if peak < 2 or peak + 2 >= len(values):
        raise RuntimeError(f"L={size} chi maximum lacks two temperatures on each side")
    fit_points[size] = slice(peak - 2, peak + 3)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
colors = {32: "#176ca4", 64: "#b64b32"}
rng = np.random.default_rng(2026)
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharex=True, sharey=True, layout="constrained")
for axis, block in zip(axes, (2000, 4000, 8000)):
    for size, samples in sorted(runs.items()):
        axis.plot(temperatures[size], chi[size], color=colors[size], marker="o", markersize=3,
                  linewidth=1.3, label=f"L = {size}")
        selected = fit_points[size]
        x = temperatures[size][selected]
        center = x[2]
        grid = np.linspace(x[0], x[-1], 200)
        offset = grid - center
        coefficients = np.polyfit(x - center, chi[size][selected], 2)
        axis.plot(grid, np.polyval(coefficients, offset), color=colors[size], linestyle="--",
                  linewidth=1.8)
        boot = np.column_stack([
            bootstrap_susceptibility(size, temperature, samples[temperature], block, rng)
            for temperature in x
        ])
        coefficients = np.polyfit(x - center, boot.T, 2)
        curves = (coefficients[0, :, None] * offset**2
                  + coefficients[1, :, None] * offset + coefficients[2, :, None])
        axis.fill_between(grid, curves.min(axis=0), curves.max(axis=0), color=colors[size], alpha=0.18)
    axis.axvline(2.26919, color="#333333", linestyle=":", linewidth=1.5,
                 label="T = 2.26919")
    axis.set(title=f"Block length {block}", xlabel="Temperature T")
    axis.grid(color="#dedede", linewidth=0.7)

axes[0].set_ylabel(r"Susceptibility $\chi(T)$")
axes[0].legend(frameon=False)
fig.suptitle("Five-point susceptibility fits and bootstrap envelopes")
fig.savefig(root / "evidence" / "chi-bootstrap.png", dpi=200)
