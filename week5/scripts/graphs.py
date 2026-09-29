"""Run with: uv run --no-project --with jax --with matplotlib python week5/scripts/graphs.py"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from derivatives import primal


def draw(jaxpr, positions, title, output_label, path):
    xs, ys = zip(*positions.values())
    fig, ax = plt.subplots(figsize=(max(xs) + 2, max(ys) - min(ys) + 2))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, pad=15)

    producer = {jaxpr.jaxpr.invars[0]: "r"}
    for index, eqn in enumerate(jaxpr.jaxpr.eqns):
        for var in eqn.outvars:
            producer[var] = index

    def arrow(source, target):
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        ax.add_patch(FancyArrowPatch((x0 + 0.7, y0), (x1 - 0.7, y1),
                                     arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1.2, color="#64748b"))

    for index, eqn in enumerate(jaxpr.jaxpr.eqns):
        for var in eqn.invars:
            if not hasattr(var, "val"):
                arrow(producer[var], index)
    arrow(producer[jaxpr.jaxpr.outvars[0]], "output")

    ax.text(*positions["r"], "r = 1.3", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#dcfce7", edgecolor="#16a34a"))
    for index, eqn in enumerate(jaxpr.jaxpr.eqns):
        details = [f"y={eqn.params['y']}"] if "y" in eqn.params else []
        details += [str(var.val) for var in eqn.invars if hasattr(var, "val")]
        label = eqn.primitive.name + ("\n" + ", ".join(details) if details else "")
        ax.text(*positions[index], label, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#dbeafe", edgecolor="#2563eb"))
    ax.text(*positions["output"], output_label, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f3e8ff", edgecolor="#9333ea"))

    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


r = jnp.float64(1.3)
energy = lambda x: primal(x)[-1]
energy_graph = jax.make_jaxpr(energy)(r)
gradient_graph = jax.make_jaxpr(jax.grad(energy))(r)

energy_positions = {"r": (0, 0), 0: (2, 0), 1: (4, 1), 2: (6, 0), 3: (8, 0), "output": (10, 0)}
gradient_positions = {
    "r": (0, 0),
    0: (2, 1), 1: (2, -3), 2: (4, -3), 3: (4, 2), 4: (4, 0.5),
    5: (6, 2), 6: (8, 2), 7: (6, -1), 8: (8, -1.7), 9: (8, 0),
    10: (10, -1), 11: (12, -3), "output": (14, -3),
}

output_dir = Path(__file__).resolve().parents[1] / "artifacts/ad"
output_dir.mkdir(parents=True, exist_ok=True)
draw(energy_graph, energy_positions, "JAX graph for U(r)", "U", output_dir / "graph.png")
draw(gradient_graph, gradient_positions, "JAX graph for grad U(r)", "dU/dr", output_dir / "grad-graph.png")
