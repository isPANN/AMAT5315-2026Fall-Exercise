"""Draw with: uv run --with matplotlib week3/scripts/acf_binning.py"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).parents[1]
directory = root / "artifacts" / "window-l64"
metadata = json.loads((directory / "run.json").read_text())
values = []
with (directory / "series.jsonl").open() as rows:
    for line in rows:
        row = json.loads(line)
        if row["T"] == 2.3:
            values.append(abs(row["M"]))
if metadata["update"] != "metropolis" or metadata["L"] != 64 or len(values) != metadata["measure"]:
    raise RuntimeError("window L=64 ramp is incomplete or has unexpected metadata")
values = np.asarray(values)

centered = values - values.mean()
size = 1 << (2 * len(values) - 1).bit_length()
transform = np.fft.rfft(centered, size)
acf = np.fft.irfft(transform * transform.conjugate(), size)[: len(values)]
acf /= acf[0]

block_lengths = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
errors = []
for length in block_lengths:
    blocks = values.reshape(-1, length).mean(axis=1)
    errors.append(blocks.std(ddof=1) / np.sqrt(len(blocks)))

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, (correlation, binning) = plt.subplots(1, 2, figsize=(10.0, 4.3), layout="constrained")
lags = np.arange(6001)
correlation.plot(lags, acf[: len(lags)], color="#176ca4", linewidth=1.2)
correlation.axhline(0, color="#aaaaaa", linewidth=0.8)
correlation.set(xlabel="Lag (sweeps)", ylabel=r"Autocorrelation $\rho_{|M|}$",
                title="L = 64, T = 2.3")

binning.plot(block_lengths, errors, color="#b64b32", marker="o", markersize=5,
             linewidth=1.5)
binning.set(xscale="log", xlabel="Block length (sweeps)",
            ylabel=r"Standard error of mean $|M|$", title="Blocking estimate")
for axis in (correlation, binning):
    axis.grid(color="#dedede", linewidth=0.7)

fig.savefig(root / "evidence" / "acf-binning.png", dpi=200)
