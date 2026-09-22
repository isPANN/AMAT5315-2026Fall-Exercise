# /// script
# dependencies = ["matplotlib", "numpy"]
# ///

from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
raw = subprocess.run(
    ["cargo", "run", "--quiet", "--release", "--bin", "accuracy-data"],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()
n = int(raw[0])
runs = {raw[i]: np.fromstring(raw[i + 1], sep=" ") for i in range(1, 7, 2)}
convergence = {}
for i in range(7, len(raw), 2):
    _, name, step = raw[i].split()
    convergence.setdefault(name, []).append(
        (float(step), np.fromstring(raw[i + 1], sep=" "))
    )


def exact_solution(x, sigma, nu, time):
    variance = sigma**2 + 2 * nu * time
    images = np.arange(-3, 4)
    distances = x[..., np.newaxis] - np.pi / 2 - time + 2 * np.pi * images
    return sigma / np.sqrt(variance) * np.exp(-distances**2 / (2 * variance)).sum(axis=-1)


x = np.arange(n) * 2 * np.pi / n
exact = exact_solution(x, 0.25, 0.002, 2 * np.pi)
labels = {
    "rk4-fourier": r"RK4, Fourier, $\Delta t=0.02$",
    "rk4-centered": r"RK4, centred, $\Delta t=0.02$",
    "euler-fourier": r"Euler, Fourier, $\Delta t=0.005$",
}
styles = {
    "rk4-fourier": dict(color="#2563eb", marker="o", markersize=3, linewidth=1.2),
    "rk4-centered": dict(color="#f59e0b", linestyle="--", linewidth=1.8),
    "euler-fourier": dict(color="#16a34a", linestyle=":", linewidth=2),
}

for name, values in runs.items():
    print(f"{labels[name]}: max error = {np.max(np.abs(values - exact)):.8e}")

dense_x = np.linspace(0, 2 * np.pi, 1000)
fig, (ax, error_ax) = plt.subplots(1, 2, figsize=(15, 5.2), constrained_layout=True)
ax.plot(
    dense_x,
    exact_solution(dense_x, 0.25, 0.002, 2 * np.pi),
    color="black",
    linewidth=2.5,
    label="Exact",
)
for name, values in runs.items():
    ax.plot(np.r_[x, 2 * np.pi], np.r_[values, values[0]], label=labels[name], **styles[name])
ax.set(
    xlabel=r"$x$",
    ylabel=r"$u(x,2\pi)$",
    title=r"(a) One lap: $\sigma=0.25$, $\nu=0.002$",
    xlim=(0, 2 * np.pi),
    xticks=[0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
    xticklabels=["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
)
ax.grid(alpha=0.2)
ax.legend(frameon=False)

convergence_exact = exact_solution(x, 0.35, 0.05, 1.0)
convergence_styles = {
    "Euler": ("#16a34a", "o"),
    "Midpoint": ("#f59e0b", "s"),
    "RK4": ("#2563eb", "^"),
    "Equal-weight-RK4": ("#db2777", "D"),
}
for name, samples in convergence.items():
    steps = np.array([sample[0] for sample in samples])
    errors = np.array(
        [np.max(np.abs(sample[1] - convergence_exact)) for sample in samples]
    )
    slope = np.polyfit(np.log(steps), np.log(errors), 1)[0]
    colour, marker = convergence_styles[name]
    error_ax.loglog(
        steps,
        errors,
        color=colour,
        marker=marker,
        linewidth=1.7,
        label=f"{name.replace('-', ' ')} (slope {slope:.2f})",
    )
    print(f"{name.replace('-', ' ')}: fitted slope = {slope:.6f}")
error_ax.set(
    xlabel=r"$\Delta t$",
    ylabel=r"maximum error at $t=1$",
    title=r"(b) Time-step convergence: $\sigma=0.35$, $\nu=0.05$",
)
error_ax.grid(which="both", alpha=0.2)
error_ax.legend(frameon=False)

output = ROOT / "evidence" / "line-accuracy.png"
output.parent.mkdir(exist_ok=True)
fig.savefig(output, dpi=220)
print(output)
