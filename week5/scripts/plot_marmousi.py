"""Run with: uv run --no-project --with numpy --with matplotlib python week5/scripts/plot_marmousi.py"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


week = Path(__file__).resolve().parents[1]
experiment = json.loads((week / "inputs/marmousi.json").read_text())
artifacts = week / "artifacts"
born = np.load(artifacts / "marmousi-born/born_data.npy")
image = np.load(artifacts / "marmousi-image/image.npy")
statistics = json.loads((artifacts / "marmousi-image/result.json").read_text())["statistics"]

cell_km = experiment["dx"] * experiment["length_unit_m"] / 1000
speed_km_s = experiment["length_unit_m"] / (1000 * experiment["time_unit_s"])
dt_s = experiment["dt"] * experiment["time_unit_s"]
perturbation = np.asarray(experiment["perturbation"]) * speed_km_s
shot_x = np.asarray(experiment["shots"])[:, 0] * cell_km
shot_indices = np.flatnonzero(np.isclose(shot_x, 10, rtol=0, atol=1e-12))
assert len(shot_indices) == 1, "Expected one shot at x = 10 km"
receiver_x = np.asarray(experiment["receivers"])[:, 0] * cell_km
times = np.arange(1, experiment["steps"] + 1) * dt_s
extent = (-cell_km / 2, (experiment["nx"] - 0.5) * cell_km,
          (experiment["nz"] - 0.5) * cell_km, -cell_km / 2)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), constrained_layout=True)
perturbation_scale = np.max(np.abs(perturbation))
perturbation_plot = axes[0].imshow(
    perturbation, origin="upper", extent=extent, aspect="auto", cmap="RdBu_r",
    vmin=-perturbation_scale, vmax=perturbation_scale)
axes[0].set(title="Velocity perturbation", xlabel="Horizontal position (km)",
            ylabel="Depth (km)")
fig.colorbar(perturbation_plot, ax=axes[0], label="Velocity perturbation (km/s)")

gather = born[int(shot_indices[0])]
gather_scale = np.max(np.abs(gather))
gather_plot = axes[1].pcolormesh(
    receiver_x, times, gather, shading="nearest", cmap="RdBu_r",
    vmin=-gather_scale, vmax=gather_scale)
axes[1].set(title="Born gather · shot at x = 10 km",
            xlabel="Receiver position (km)", ylabel="Time (s)")
axes[1].set_ylim(times[-1] + dt_s / 2, 0)
fig.colorbar(gather_plot, ax=axes[1], label="Pressure (reduced units)")

image_scale = np.max(np.abs(image))
image_plot = axes[2].imshow(
    image, origin="upper", extent=extent, aspect="auto", cmap="RdBu_r",
    vmin=-image_scale, vmax=image_scale)
axes[2].set(title="Raw image", xlabel="Horizontal position (km)",
            ylabel="Depth (km)")
fig.colorbar(image_plot, ax=axes[2], label="Raw RTM amplitude (arbitrary units)")

fig.savefig(artifacts / "marmousi.png", dpi=180)
plt.close(fig)

print(f"Image L2 norm: {np.linalg.norm(image):.12g}")
print(f"Peak saved states: {statistics['peak_saved_states']}")
print(f"Peak saved bytes: {statistics['peak_saved_bytes']}")
