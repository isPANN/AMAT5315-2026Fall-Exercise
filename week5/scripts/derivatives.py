"""Run with: uv run --no-project --with jax python week5/scripts/derivatives.py"""

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def passes(r):
    r = jnp.asarray(r, dtype=jnp.float64)

    # Primal pass
    a = r**-6
    b = a**2
    c = b - a
    U = 4 * c

    # Forward pass, seeded with dr = 1
    dr = jnp.ones_like(r)
    da = (-6 * r**-7) * dr
    db = (2 * a) * da
    dc = db - da
    dU = 4 * dc

    # Reverse pass, seeded with U_bar = 1
    U_bar = jnp.ones_like(U)
    c_bar = 4 * U_bar
    b_bar = c_bar
    a_bar = -c_bar
    a_bar += (2 * a) * b_bar
    r_bar = (-6 * r**-7) * a_bar

    tangents = dict(zip(("r", "a", "b", "c", "U"), (dr, da, db, dc, dU)))
    adjoints = dict(zip(("r", "a", "b", "c", "U"), (r_bar, a_bar, b_bar, c_bar, U_bar)))
    return U, tangents, adjoints


if __name__ == "__main__":
    r = jnp.float64(1.3)
    U, tangents, adjoints = passes(r)
    result = {
        "r": float(r),
        "energy": float(U),
        "tangents": {name: float(value) for name, value in tangents.items()},
        "adjoints": {name: float(value) for name, value in adjoints.items()},
        "jax_grad": float(jax.grad(lambda x: 4 * (x**-12 - x**-6))(r)),
    }

    output = Path(__file__).resolve().parents[1] / "artifacts/ad/derivatives.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
