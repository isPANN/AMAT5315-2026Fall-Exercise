"""Run with: uv run --with numpy week3/scripts/peaks.py"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path(__file__).parents[1]
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
    chi = size**2 * (sum_m2 / count - mean_abs_m**2) / temperature
    series[size].append((temperature, mean_abs_m, chi))

peaks = {}
lowest = {}
for size, values in series.items():
    values.sort()
    largest = max(range(len(values)), key=lambda i: values[i][2])
    if largest < 2 or largest + 2 >= len(values):
        raise RuntimeError(f"L={size} chi maximum lacks two temperatures on each side")
    points = values[largest - 2 : largest + 3]
    center = points[2][0]
    a, b, _ = np.polyfit([point[0] - center for point in points], [point[2] for point in points], 2)
    peak = center - b / (2 * a)
    if a >= 0 or not points[0][0] <= peak <= points[-1][0]:
        raise RuntimeError(f"L={size} quadratic has no maximum within its five points")
    peaks[size] = peak
    lowest[size] = values[0][:2]

critical = 2 * peaks[64] - peaks[32]
lines = [*(f"L={size} T_peak={peaks[size]:.8f}" for size in sorted(peaks)), f"T_c={critical:.8f}"]
lines.extend(
    f"L={size} mean_abs_M(T={temperature:.6f})={mean_abs_m:.8f}"
    for size, (temperature, mean_abs_m) in sorted(lowest.items())
)
report = "\n".join(lines) + "\n"
print(report, end="")
(root / "evidence" / "peaks.txt").write_text(report)
