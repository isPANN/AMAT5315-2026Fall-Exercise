"""Run with: uv run --no-project --with numpy python week5/scripts/compare_adjoint.py"""

import json
from pathlib import Path

import numpy as np


week = Path(__file__).resolve().parents[1]
born = np.load(week / "artifacts/born/born_data.npy")
image = np.load(week / "artifacts/adjoint/image.npy")
experiment = json.loads((week / "inputs/reflector.json").read_text())
perturbation = np.asarray(experiment["perturbation"], dtype=np.float64)
assert perturbation.shape == image.shape

left = float(np.sum(born * born))
right = float(np.sum(perturbation * image))
relative_difference = abs(left - right) / max(abs(left), abs(right))
print(f"left: {left:.17e}")
print(f"right: {right:.17e}")
print(f"relative difference: {relative_difference:.17e}")
