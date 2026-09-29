"""Run with: uv run --no-project --with jax python week5/scripts/graphs.py"""

import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp

from derivatives import primal


def draw(jaxpr, title, output_label, path):
    lines = [
        "digraph G {",
        f"graph [rankdir=LR, dpi=160, pad=0.4, nodesep=0.35, ranksep=0.7, labelloc=t, label={json.dumps(title)}, fontname=Helvetica, fontsize=18];",
        'node [shape=box, style="rounded,filled", fillcolor="#dbeafe", color="#2563eb", fontname=Helvetica, fontsize=11];',
        'edge [color="#64748b", arrowsize=0.7];',
        'input [label="r = 1.3", fillcolor="#dcfce7", color="#16a34a"];',
        f'output [label={json.dumps(output_label)}, fillcolor="#f3e8ff", color="#9333ea"];',
    ]
    producer = {jaxpr.jaxpr.invars[0]: "input"}
    for index, eqn in enumerate(jaxpr.jaxpr.eqns):
        details = [f"y={eqn.params['y']}"] if "y" in eqn.params else []
        details += [str(var.val) for var in eqn.invars if hasattr(var, "val")]
        label = eqn.primitive.name + ("\n" + ", ".join(details) if details else "")
        lines.append(f"op{index} [label={json.dumps(label)}];")
        for var in eqn.invars:
            if not hasattr(var, "val"):
                lines.append(f"{producer[var]} -> op{index};")
        for var in eqn.outvars:
            producer[var] = f"op{index}"
    lines += [f"{producer[jaxpr.jaxpr.outvars[0]]} -> output;", "}"]
    subprocess.run(["dot", "-Tpng", "-o", str(path)], input="\n".join(lines), text=True, check=True)


r = jnp.float64(1.3)
energy = lambda x: primal(x)[-1]
output_dir = Path(__file__).resolve().parents[1] / "artifacts/ad"
output_dir.mkdir(parents=True, exist_ok=True)
draw(jax.make_jaxpr(energy)(r), "JAX graph for U(r)", "U", output_dir / "graph.png")
draw(jax.make_jaxpr(jax.grad(energy))(r), "JAX graph for grad U(r)", "dU/dr", output_dir / "grad-graph.png")
