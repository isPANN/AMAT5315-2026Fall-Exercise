# /// script
# dependencies = ["matplotlib", "numpy"]
# ///

"""Plot selected random-flow vorticity snapshots."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
run = json.loads((ROOT / "artifacts" / "random" / "run.json").read_text())
times = [0.0, 2.0, 5.0, 10.0]
frames = {}
with (ROOT / "artifacts" / "random" / "fields.jsonl").open() as lines:
    for line in lines:
        frame = json.loads(line)
        if frame["t"] in times:
            frames[frame["t"]] = np.asarray(frame["omega"]).reshape(run["n"], run["n"])
assert list(frames) == times

limit = max(np.abs(omega).max() for omega in frames.values())
fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4), constrained_layout=True)
for ax, (time, omega) in zip(axes, frames.items()):
    image = ax.imshow(
        omega,
        origin="lower",
        extent=(0, 2 * np.pi, 0, 2 * np.pi),
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    ax.set(
        title=rf"$t={time:g}$",
        xlabel=r"$x$",
        xticks=[0, np.pi, 2 * np.pi],
        xticklabels=["0", r"$\pi$", r"$2\pi$"],
        yticks=[0, np.pi, 2 * np.pi],
        yticklabels=["0", r"$\pi$", r"$2\pi$"],
    )
axes[0].set_ylabel(r"$y$")
for ax in axes[1:]:
    ax.tick_params(labelleft=False)
fig.colorbar(image, ax=axes, label=r"vorticity $\omega$", shrink=0.9)

output = ROOT / "evidence" / "random.png"
output.parent.mkdir(exist_ok=True)
fig.savefig(output, dpi=220)
print(output)
