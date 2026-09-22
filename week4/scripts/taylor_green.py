# /// script
# dependencies = ["matplotlib", "numpy"]
# ///

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
artifact = ROOT / "artifacts" / "taylor-green"
frames = [json.loads(line) for line in (artifact / "fields.jsonl").read_text().splitlines()]
exact = json.loads((artifact / "exact-t1.json").read_text())
first, last = frames[0], frames[-1]
n = exact["n"]

computed_velocity = np.r_[last["u"], last["v"]]
exact_velocity = np.r_[exact["u"], exact["v"]]
relative_error = np.linalg.norm(computed_velocity - exact_velocity) / np.linalg.norm(
    exact_velocity
)
print(f"relative velocity error: {relative_error:.17e}")

assert first["t"] == 0 and last["t"] == 1
assert all(len(frame[name]) == n * n for frame in (first, last) for name in ("u", "v", "omega"))

coordinates = np.arange(n) * 2 * np.pi / n
arrow = slice(None, None, 4)
limit = max(np.max(np.abs(frame["omega"])) for frame in (first, last))
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=True)
images = []
for ax, frame in zip(axes, (first, last)):
    omega = np.asarray(frame["omega"]).reshape(n, n)
    u = np.asarray(frame["u"]).reshape(n, n)
    v = np.asarray(frame["v"]).reshape(n, n)
    images.append(
        ax.imshow(
            omega,
            origin="lower",
            extent=(0, 2 * np.pi, 0, 2 * np.pi),
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
    )
    ax.quiver(
        coordinates[arrow],
        coordinates[arrow],
        u[arrow, arrow],
        v[arrow, arrow],
        color="black",
        angles="xy",
        scale_units="xy",
        scale=4,
        width=0.003,
    )
    ax.set(
        xlabel=r"$x$",
        ylabel=r"$y$",
        title=rf"$t={frame['t']:.0f}$",
        xticks=[0, np.pi, 2 * np.pi],
        yticks=[0, np.pi, 2 * np.pi],
        xticklabels=["0", r"$\pi$", r"$2\pi$"],
        yticklabels=["0", r"$\pi$", r"$2\pi$"],
    )
fig.colorbar(images[-1], ax=axes, shrink=0.82, label=r"vorticity $\omega$")
fig.suptitle("Taylor–Green vortex with velocity field", fontsize=15)

output = ROOT / "evidence" / "taylor-green.png"
output.parent.mkdir(exist_ok=True)
fig.savefig(output, dpi=220)
print(output)
