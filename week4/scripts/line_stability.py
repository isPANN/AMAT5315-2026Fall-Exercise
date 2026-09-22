# /// script
# dependencies = ["matplotlib", "numpy"]
# ///

from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
raw = subprocess.run(
    ["cargo", "run", "--quiet", "--bin", "stability-data"],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()
width, height = map(int, raw[0].split()[:2])
x_min, x_max, y_min, y_max = map(float, raw[0].split()[2:])
growth = np.loadtxt(raw[1 : height + 1]).reshape(height, width)
trajectories = {}
cursor = height + 1
while cursor < len(raw):
    _, step, rows, points = raw[cursor].split()
    rows, points = int(rows), int(points)
    values = np.loadtxt(raw[cursor + 1 : cursor + rows + 1])
    trajectories[float(step)] = (values[:, 0], values[:, 1:].reshape(rows, points))
    cursor += rows + 1

x = np.linspace(x_min, x_max, width)
y = np.linspace(y_min, y_max, height)
z = x[np.newaxis, :] + 1j * y[:, np.newaxis]
stability = {
    "Forward Euler": 1 + z,
    "Explicit midpoint": 1 + z + z**2 / 2,
    "RK4": 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24,
}
assert np.allclose(growth, np.abs(stability["RK4"]), rtol=1e-12, atol=1e-12)
line_colours = ["#67e8f9", "#fbbf24", "#f472b6"]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(17, 6.8),
    constrained_layout=True,
    gridspec_kw={"width_ratios": [0.7, 1, 1]},
)
ax = axes[0]
image = ax.imshow(
    np.maximum(growth, 1e-3),
    origin="lower",
    extent=(x_min, x_max, y_min, y_max),
    cmap="RdBu_r",
    norm=LogNorm(vmin=1e-2, vmax=1e2),
    interpolation="bilinear",
    aspect="equal",
)
for values, colour in zip(stability.values(), line_colours):
    ax.contour(x, y, np.abs(values), levels=[1], colors=[colour], linewidths=2)

n, c, nu = 64, 1.0, 0.05
indices = np.arange(n)
wave_numbers = np.where(indices <= (n - 1) // 2, indices, indices - n).astype(float)
advective_wave_numbers = wave_numbers.copy()
advective_wave_numbers[n // 2] = 0.0
eigenvalues = -nu * wave_numbers**2 - 1j * c * advective_wave_numbers
step_styles = [(0.045, "#f8fafc", "o"), (0.056, "#a3e635", "^")]
for step, colour, marker in step_styles:
    modes = step * eigenvalues
    ax.scatter(
        modes.real,
        modes.imag,
        s=24,
        c=colour,
        marker=marker,
        edgecolors="#111827",
        linewidths=0.45,
        zorder=4,
    )

legend = [
    Line2D([0], [0], color=colour, lw=2, label=name)
    for name, colour in zip(stability, line_colours)
]
legend += [
    Line2D(
        [0],
        [0],
        marker=marker,
        linestyle="none",
        markerfacecolor=colour,
        markeredgecolor="#111827",
        markersize=7,
        label=f"Fourier modes, h = {step:.3f}",
    )
    for step, colour, marker in step_styles
]
ax.legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.94)
ax.axhline(0, color="white", alpha=0.28, linewidth=0.8)
ax.axvline(0, color="white", alpha=0.28, linewidth=0.8)
ax.set(
    xlabel=r"$\operatorname{Re}(z)$",
    ylabel=r"$\operatorname{Im}(z)$",
    title=r"(a) Stability in the $z$-plane",
)
colourbar = fig.colorbar(image, ax=ax, shrink=0.8, pad=0.03)
colourbar.set_label(r"Growth factor per step $|R_{\mathrm{RK4}}(z)|$")
colourbar.set_ticks([1e-2, 1e-1, 1, 10, 100])

space = np.arange(n) * 2 * np.pi / n
solution_images = []
for panel, (step, (times, states)) in enumerate(trajectories.items(), start=1):
    solution_images.append(
        axes[panel].pcolormesh(
            space,
            times,
            states,
            shading="auto",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
    )
    axes[panel].invert_yaxis()
    axes[panel].set(
        xlabel=r"$x$",
        ylabel=r"$t$",
        title=f"({chr(97 + panel)}) RK4, $\\Delta t={step:.3f}$",
        xticks=[0, np.pi, 2 * np.pi],
        xticklabels=["0", r"$\pi$", r"$2\pi$"],
    )
fig.colorbar(solution_images[-1], ax=axes[1:], shrink=0.8, pad=0.03, label=r"$u(x,t)$")
fig.suptitle(
    r"Fourier advection–diffusion: $n=64$, $c=1$, $\nu=0.05$",
    fontsize=15,
)

output = ROOT / "evidence" / "line-stability.png"
output.parent.mkdir(exist_ok=True)
fig.savefig(output, dpi=220)
print(output)
