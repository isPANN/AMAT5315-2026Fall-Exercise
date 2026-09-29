"""Run with: uv run --no-project --with numpy --with matplotlib python week5/scripts/plot_inputs.py"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


week = Path(__file__).resolve().parents[1]
experiment = json.loads((week / "inputs/reflector.json").read_text())
cell_km = experiment["dx"] * experiment["length_unit_m"] / 1000
speed_km_s = experiment["length_unit_m"] / (1000 * experiment["time_unit_s"])
extent = (-cell_km / 2, (experiment["nx"] - 0.5) * cell_km,
          (experiment["nz"] - 0.5) * cell_km, -cell_km / 2)
shots = np.asarray(experiment["shots"]) * cell_km
receivers = np.asarray(experiment["receivers"]) * cell_km

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
for ax, field, title, colorbar in (
    (axes[0], "background", "Background speed", "Speed (km/s)"),
    (axes[1], "perturbation", "Velocity perturbation", "Δ speed (km/s)"),
):
    values = np.asarray(experiment[field]) * speed_km_s
    image = ax.imshow(values, origin="upper", extent=extent, cmap="viridis",
                      vmin=0, vmax=values.max())
    ax.scatter(receivers[:, 0], receivers[:, 1], marker="v", s=35,
               facecolor="#5eead4", edgecolor="black", linewidth=0.6,
               label="Receivers", zorder=3)
    ax.scatter(shots[:, 0], shots[:, 1], marker="*", s=140,
               facecolor="#ef4444", edgecolor="white", linewidth=0.6,
               label="Shots", zorder=4)
    ax.set(title=title, xlabel="Horizontal position (km)", ylabel="Depth (km)")
    ax.set_aspect("equal")
    fig.colorbar(image, ax=ax, label=colorbar, shrink=0.82)

axes[0].legend(loc="lower right")
output = week / "artifacts/inputs.png"
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=180)
plt.close(fig)
