"""Draw with: uv run --with matplotlib week3/scripts/trace.py"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).parents[1]
temperatures = (2.3, 3.0)
traces = {temperature: [] for temperature in temperatures}
with (root / "artifacts" / "coarse-l64" / "series.jsonl").open() as rows:
    for line in rows:
        row = json.loads(line)
        if row["T"] in traces and row["sweep"] <= 2000:
            traces[row["T"]].append((row["sweep"], abs(row["M"])))

if any(len(trace) != 2000 for trace in traces.values()):
    raise RuntimeError("coarse L=64 ramp lacks 2000 recorded sweeps at T=2.3 or T=3.0")

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True, sharey=True, layout="constrained")
for axis, (temperature, trace), color in zip(axes, traces.items(), ("#b64b32", "#176ca4")):
    axis.plot([point[0] for point in trace], [point[1] for point in trace], color=color,
              linewidth=0.8)
    axis.set(ylabel=r"$|M|$", title=f"L = 64, T = {temperature:.1f}")
    axis.grid(color="#dedede", linewidth=0.7)
axes[-1].set_xlabel("Recorded sweep")
fig.savefig(root / "evidence" / "trace.png", dpi=200)
