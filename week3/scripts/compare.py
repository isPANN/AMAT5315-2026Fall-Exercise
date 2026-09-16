"""Draw with: uv run --with numpy --with matplotlib week3/scripts/compare.py"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).parents[1]


def load(name, update):
    directory = root / "artifacts" / name
    run = json.loads((directory / "run.json").read_text())
    if run["L"] != 64 or run["update"] != update:
        raise RuntimeError(f"unexpected run metadata: {directory}")
    samples = defaultdict(list)
    clusters = defaultdict(list)
    with (directory / "series.jsonl").open() as rows:
        for line in rows:
            row = json.loads(line)
            samples[row["T"]].append(abs(row["M"]))
            if update == "wolff":
                clusters[row["T"]].append(row["cluster_size"])
    expected = {round(temperature, 6): run["measure"] for temperature in run["t_grid"]}
    if {temperature: len(values) for temperature, values in samples.items()} != expected:
        raise RuntimeError(f"ramp is incomplete: {directory}")
    return samples, clusters


def autocorrelation_time(values):
    values = np.asarray(values) - np.mean(values)
    size = 1 << (2 * len(values) - 1).bit_length()
    transform = np.fft.rfft(values, size)
    correlation = np.fft.irfft(transform * transform.conjugate(), size)[: len(values)]
    correlation /= correlation[0]
    tau = 0.5
    for lag, value in enumerate(correlation[1:], 1):
        tau += value
        if lag > 6 * tau:
            return tau
    raise RuntimeError("autocorrelation window did not close")


metropolis, _ = load("window-l64", "metropolis")
wolff, clusters = load("wolff-l64", "wolff")

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, axis = plt.subplots(figsize=(7.4, 4.8), layout="constrained")
for label, samples, factors, color in (
    ("Metropolis", metropolis, {temperature: 1 for temperature in metropolis}, "#6a3d9a"),
    ("Wolff", wolff, {temperature: np.mean(values) / 64**2 for temperature, values in clusters.items()}, "#b64b32"),
):
    temperatures = sorted(samples)
    tau = [autocorrelation_time(samples[temperature]) * factors[temperature] for temperature in temperatures]
    axis.plot(temperatures, tau, marker="o", markersize=4, linewidth=1.8, label=label, color=color)

axis.set(yscale="log", xlabel="Temperature T", ylabel=r"Work-normalized $\tau_{\rm int}$ (sweeps)")
axis.grid(color="#dedede", linewidth=0.7, which="both")
axis.legend(frameon=False)
fig.savefig(root / "evidence" / "tau-compare.png", dpi=200)
