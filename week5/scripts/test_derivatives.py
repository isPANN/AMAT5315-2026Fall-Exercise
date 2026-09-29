"""Run with: uv run --no-project --with jax python week5/scripts/test_derivatives.py"""

import json
import math
import subprocess
import sys
from pathlib import Path


week = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, str(week / "scripts/derivatives.py")], check=True)
result = json.loads((week / "artifacts/ad/derivatives.json").read_text())

r = 1.3
a = r**-6
expected_energy = 4 * (r**-12 - a)
expected_grad = 4 * (-12 * r**-13 + 6 * r**-7)
expected_tangents = {
    "r": 1,
    "a": -6 * r**-7,
    "b": -12 * r**-13,
    "c": -12 * r**-13 + 6 * r**-7,
    "U": expected_grad,
}
expected_adjoints = {
    "r": expected_grad,
    "a": 8 * a - 4,
    "b": 4,
    "c": 4,
    "U": 1,
}

assert set(result) == {"r", "energy", "tangents", "adjoints", "jax_grad"}
for actual, expected in ((result["r"], r), (result["energy"], expected_energy), (result["jax_grad"], expected_grad)):
    assert math.isclose(actual, expected, rel_tol=1e-14, abs_tol=1e-14)
for name, expected in expected_tangents.items():
    assert math.isclose(result["tangents"][name], expected, rel_tol=1e-14, abs_tol=1e-14), name
for name, expected in expected_adjoints.items():
    assert math.isclose(result["adjoints"][name], expected, rel_tol=1e-14, abs_tol=1e-14), name
