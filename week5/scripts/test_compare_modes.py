"""Run with: uv run --no-project --with jax --with matplotlib python week5/scripts/test_compare_modes.py"""

import subprocess
import sys
from pathlib import Path


week = Path(__file__).resolve().parents[1]
plot = week / "artifacts/ad/modes.png"
run = subprocess.run(
    [sys.executable, str(week / "scripts/compare_modes.py")],
    check=True,
    capture_output=True,
    text=True,
)
errors = dict(line.split(": ") for line in run.stdout.splitlines())
assert set(errors) == {"Forward mode", "Reverse mode", "Centered difference"}
assert float(errors["Forward mode"]) < 1e-13
assert float(errors["Reverse mode"]) < 1e-13
assert 1e-9 < float(errors["Centered difference"]) < 1e-7
assert plot.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
