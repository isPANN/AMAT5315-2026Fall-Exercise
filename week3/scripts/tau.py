"""Draw with: uv run --with matplotlib week3/scripts/tau.py"""

import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).parents[1]
output = subprocess.run(
    [sys.executable, root / "scripts" / "errors.py"],
    check=True,
    capture_output=True,
    text=True,
).stdout
series = defaultdict(list)
for row in csv.DictReader(output.splitlines(), delimiter="\t"):
    series[int(row["L"])].append((float(row["T"]), float(row["tau_int"])))

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, axis = plt.subplots(figsize=(7.4, 4.8), layout="constrained")
for size, values in sorted(series.items()):
    values.sort()
    axis.plot(
        [value[0] for value in values],
        [value[1] for value in values],
        marker="o",
        markersize=4,
        linewidth=1.8,
        label=f"L = {size}",
    )
axis.set(yscale="log", xlabel="Temperature T", ylabel=r"Integrated autocorrelation time $\tau_{\rm int}$")
axis.grid(color="#dedede", linewidth=0.7, which="both")
axis.legend(frameon=False)
fig.savefig(root / "evidence" / "tau.png", dpi=200)
