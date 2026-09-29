"""Run with: uv run --no-project --with jax --with matplotlib python week5/scripts/test_graphs.py"""

import subprocess
import sys
from pathlib import Path

from PIL import Image


week = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, str(week / "scripts/graphs.py")], check=True)
for filename in ("graph.png", "grad-graph.png"):
    with Image.open(week / "artifacts/ad" / filename) as image:
        assert image.format == "PNG"
        assert image.size[0] >= 800 and image.size[1] >= 300
        assert min(low for low, _ in image.convert("RGB").getextrema()) < 200
