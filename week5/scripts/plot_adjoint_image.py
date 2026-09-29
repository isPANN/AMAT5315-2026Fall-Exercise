"""Run with: uv run --no-project --with numpy --with matplotlib python week5/scripts/plot_adjoint_image.py"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


week = Path(__file__).resolve().parents[1]
experiment = json.loads((week / "inputs/reflector.json").read_text())
perturbation = np.asarray(experiment["perturbation"])
image = np.load(week / "artifacts/adjoint/image.npy")
assert perturbation.shape == image.shape

x0, x1 = 7, 33
z0, z1 = 10, 33
cell_km = experiment["dx"] * experiment["length_unit_m"] / 1000
speed_km_s = experiment["length_unit_m"] / (1000 * experiment["time_unit_s"])
depths = np.arange(z0, z1 + 1) * cell_km
true_row = z0 + np.linalg.norm(perturbation[z0:z1 + 1, x0:x1 + 1], axis=1).argmax()
true_depth = true_row * cell_km

perturbation_crop = perturbation[z0:z1 + 1, x0:x1 + 1] * speed_km_s
image_crop = image[z0:z1 + 1, x0:x1 + 1]
profile = np.linalg.norm(image_crop, axis=1)
peak = int(profile.argmax())
peak_depth = depths[peak]
extent = ((x0 - 0.5) * cell_km, (x1 + 0.5) * cell_km,
          (z1 + 0.5) * cell_km, (z0 - 0.5) * cell_km)

fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), constrained_layout=True)
for ax, field, title, cmap, low, high, label in (
    (axes[0], perturbation_crop, "Velocity perturbation", "viridis", 0,
     perturbation_crop.max(), "Velocity perturbation (km/s)"),
    (axes[1], image_crop, "Adjoint image", "RdBu_r", -np.abs(image_crop).max(),
     np.abs(image_crop).max(), "Raw RTM amplitude (arbitrary units)"),
):
    im = ax.imshow(field, origin="upper", extent=extent, cmap=cmap, vmin=low, vmax=high)
    ax.axhline(true_depth, color="white", linestyle="--", linewidth=1.2)
    ax.set(title=title, xlabel="Horizontal position (km)", ylabel="Depth (km)")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label=label, shrink=0.83)

ax = axes[2]
ax.plot(profile, depths, color="#1d4ed8", linewidth=2)
ax.axhline(true_depth, color="#b45309", linestyle="--", label="True reflector depth")
ax.scatter(profile[peak], peak_depth, marker="*", s=180, color="#dc2626",
           edgecolor="black", linewidth=0.5, zorder=3, label="Profile peak")
ax.set(title="Image depth profile", xlabel="Row L2 norm (arbitrary units)",
       ylabel="Depth (km)")
ax.legend(loc="lower right")
for ax in axes:
    ax.set_ylim(extent[2], extent[3])

output = week / "artifacts/adjoint/image.png"
fig.savefig(output, dpi=180)
plt.close(fig)

print(f"True reflector depth: {true_depth:.3f} km")
print(f"Image profile peak depth: {peak_depth:.3f} km")
print(f"Depth difference: {abs(true_depth - peak_depth):.3f} km")
