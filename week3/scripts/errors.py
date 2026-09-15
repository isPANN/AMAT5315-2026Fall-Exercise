"""Run with: uv run --with numpy week3/scripts/errors.py > week3/evidence/errors.txt"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path(__file__).parents[1]
metadata = []
source = {}
for path in sorted((root / "artifacts").glob("*/run.json")):
    run = json.loads(path.read_text())
    if run["update"] != "metropolis":
        continue
    metadata.append((path.parent, run))
    for temperature in run["t_grid"]:
        key = (run["L"], round(temperature, 6))
        if key not in source or run["measure"] > source[key][0]:
            source[key] = (run["measure"], path.parent)

samples = defaultdict(list)
for directory, run in metadata:
    counts = defaultdict(int)
    with (directory / "series.jsonl").open() as rows:
        for line in rows:
            row = json.loads(line)
            key = (row["L"], row["T"])
            counts[row["T"]] += 1
            if source[key][1] == directory:
                samples[key].append(abs(row["M"]))
    expected = {round(temperature, 6): run["measure"] for temperature in run["t_grid"]}
    if counts != expected:
        raise RuntimeError(f"ramp is incomplete: {directory}")


def integrated_autocorrelation_time(values):
    centered = values - values.mean()
    size = 1 << (2 * len(values) - 1).bit_length()
    transform = np.fft.rfft(centered, size)
    correlation = np.fft.irfft(transform * transform.conjugate(), size)[: len(values)]
    correlation /= correlation[0]
    tau = 0.5
    for lag, value in enumerate(correlation[1:], 1):
        tau += value
        if lag > 6 * tau:
            return tau
    raise RuntimeError("autocorrelation window did not close")


print("L\tT\tmean_abs_M\tnaive_se\tblock_se_50\tratio\ttau_int")
for (size, temperature), rows in sorted(samples.items()):
    values = np.asarray(rows)
    naive = values.std(ddof=1) / np.sqrt(len(values))
    blocks = values.reshape(50, -1).mean(axis=1)
    blocked = blocks.std(ddof=1) / np.sqrt(50)
    tau = integrated_autocorrelation_time(values)
    print(
        f"{size}\t{temperature:.6f}\t{values.mean():.8f}\t{naive:.8g}\t"
        f"{blocked:.8g}\t{blocked / naive:.6f}\t{tau:.6f}"
    )
