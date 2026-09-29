"""Run with: uv run --no-project --with jax --with matplotlib python week5/scripts/compare_modes.py"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from derivatives import passes


r = jnp.linspace(0.95, 2.5, 601, dtype=jnp.float64)
h = 1e-6
_, tangents, adjoints = passes(r)
analytic = 4 * (-12 * r**-13 + 6 * r**-7)
energy = lambda x: 4 * (x**-12 - x**-6)
finite_difference = (energy(r + h) - energy(r - h)) / (2 * h)

methods = {
    "Forward mode": tangents["U"],
    "Reverse mode": adjoints["r"],
    "Centered difference": finite_difference,
}
errors = {name: jnp.abs(values - analytic) for name, values in methods.items()}

fig, (derivative_ax, error_ax) = plt.subplots(2, 1, figsize=(7, 7), sharex=True, layout="constrained")
derivative_ax.plot(r, analytic, color="black", label="Analytic")
for offset, (name, values) in enumerate(methods.items()):
    derivative_ax.plot(r, values, linestyle="none", marker=("o", "^", "x")[offset],
                       markevery=(offset * 20, 60), markersize=4, label=name)
    error_ax.semilogy(r, errors[name], label=name)

derivative_ax.set_ylabel("dU/dr (reduced units)")
error_ax.set_xlabel("Separation r (reduced units)")
error_ax.set_ylabel("Absolute error (reduced units)")
derivative_ax.legend()
error_ax.legend()
derivative_ax.grid(alpha=0.25)
error_ax.grid(alpha=0.25)

output = Path(__file__).resolve().parents[1] / "artifacts/ad/modes.png"
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=160)
plt.close(fig)

for name, error in errors.items():
    print(f"{name}: {float(jnp.max(error)):.12g}")
