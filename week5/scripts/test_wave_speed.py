"""Run with: uv run --no-project --with numpy --with matplotlib python week5/scripts/test_wave_speed.py"""

import math
import re
import subprocess
import sys
from pathlib import Path


week = Path(__file__).resolve().parents[1]
run = subprocess.run([sys.executable, str(week / "scripts/plot_wave_speed.py")],
                     check=True, capture_output=True, text=True)
lines = run.stdout.splitlines()
assert len(lines) == 4
for line, (step, index, distance, pressure) in zip(lines[:2], (
    (108, 6, 0.8485281374238571, 0.2852701),
    (144, 15, 2.1213203435596424, 0.2076008),
)):
    match = re.fullmatch(r"Step (\d+): peak k=(\d+), distance=([\d.]+) km, pressure=([\d.]+)", line)
    assert match is not None
    assert (int(match[1]), int(match[2])) == (step, index)
    assert math.isclose(float(match[3]), distance, rel_tol=1e-6)
    assert math.isclose(float(match[4]), pressure, rel_tol=1e-6)
assert math.isclose(float(lines[2].removeprefix("Time between frames: ").removesuffix(" s")), 0.72, rel_tol=1e-12)
assert math.isclose(float(lines[3].removeprefix("Peak displacement: ").removesuffix(" km")), 1.2727922061357853, rel_tol=1e-6)
png = week / "artifacts/forward/wave-speed.png"
assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and png.stat().st_size > 10_000
