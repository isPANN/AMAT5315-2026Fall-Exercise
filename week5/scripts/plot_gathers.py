"""Run with: uv run --no-project --with numpy --with matplotlib python week5/scripts/plot_gathers.py"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


week = Path(__file__).resolve().parents[1]
out = week / "artifacts/forward"
traces = np.load(out / "traces.npy")
experiment = json.loads((out / "run.json").read_text())["experiment"]
length_km = experiment["dx"] * experiment["length_unit_m"] / 1000
dt_s = experiment["dt"] * experiment["time_unit_s"]
receivers = np.asarray(experiment["receivers"])
receiver_x = receivers[:, 0] * length_km
times = np.arange(1, experiment["steps"] + 1) * dt_s
scale = np.max(np.abs(traces))

fig, axes = plt.subplots(1, traces.shape[0], figsize=(13, 5.5), sharey=True,
                         constrained_layout=True)
for shot, ax in enumerate(axes):
    image = ax.pcolormesh(receiver_x, times, traces[shot], shading="nearest",
                          cmap="RdBu_r", vmin=-scale, vmax=scale)
    shot_x = experiment["shots"][shot][0] * length_km
    ax.set(title=f"Shot {shot} · x = {shot_x:g} km",
           xlabel="Receiver position (km)")
    ax.set_ylim(times[-1] + dt_s / 2, 0)
axes[0].set_ylabel("Time (s)")
fig.colorbar(image, ax=axes, label="Pressure (reduced units)", shrink=0.85)
fig.savefig(out / "gathers.png", dpi=180)
plt.close(fig)

print(f"All traces L2 norm: {np.linalg.norm(traces):.12g}")
for shot, gather in enumerate(traces):
    sample, trace = np.unravel_index(np.abs(gather).argmax(), gather.shape)
    receiver = receivers[trace]
    x_km, z_km = receiver * length_km
    print(f"Shot {shot}: max |pressure| = {abs(gather[sample, trace]):.12g}; "
          f"trace index {trace}; receiver [{receiver[0]}, {receiver[1]}] "
          f"(x = {x_km:g} km, depth = {z_km:g} km); "
          f"sample index {sample}, time = {times[sample]:.3f} s")
