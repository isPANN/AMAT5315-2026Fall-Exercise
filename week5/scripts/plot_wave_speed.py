"""Run with: uv run --no-project --with numpy --with matplotlib python week5/scripts/plot_wave_speed.py"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


week = Path(__file__).resolve().parents[1]
out = week / "artifacts/forward"
run = json.loads((out / "run.json").read_text())
frames = np.load(out / "wavefield.npy")
experiment = run["experiment"]
recording = run["recording"]
cell_km = experiment["dx"] * experiment["length_unit_m"] / 1000
shot_x, shot_z = experiment["shots"][0]
k = np.arange(24)
x, z = 10 + k, 8 + k
distance = np.hypot(x - shot_x, z - shot_z) * cell_km

fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
peaks = []
for step in (108, 144):
    frame_index = recording["steps"].index(step)
    time_s = recording["times"][frame_index] * experiment["time_unit_s"]
    pressure = frames[frame_index, z, x]
    peak = int(np.argmax(pressure))
    if pressure[peak] <= 0:
        raise ValueError(f"step {step} has no positive sampled pressure")
    line, = ax.plot(distance, pressure, marker="o", markersize=3,
                    label=f"Step {step}, t = {time_s:.2f} s")
    ax.scatter(distance[peak], pressure[peak], marker="*", s=180,
               color=line.get_color(), edgecolor="black", zorder=3)
    peaks.append((step, time_s, peak, distance[peak], pressure[peak]))

interval = peaks[1][1] - peaks[0][1]
displacement = peaks[1][3] - peaks[0][3]
ax.axhline(0, color="0.5", linewidth=0.8)
ax.set(title="Pressure along the shot diagonal", xlabel="Distance from shot (km)",
       ylabel="Pressure (reduced units)")
ax.grid(alpha=0.25)
ax.legend()
ax.text(0.98, 0.04, f"Peak displacement: {displacement:.3f} km in {interval:.2f} s",
        transform=ax.transAxes, ha="right", va="bottom")
fig.savefig(out / "wave-speed.png", dpi=180)
plt.close(fig)

for step, _, peak, peak_distance, peak_pressure in peaks:
    print(f"Step {step}: peak k={peak}, distance={peak_distance:.9f} km, "
          f"pressure={peak_pressure:.9f}")
print(f"Time between frames: {interval:.6f} s")
print(f"Peak displacement: {displacement:.9f} km")
