"""Time the NumPy, Rust debug, and Rust release default workloads three times."""
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

week = Path(__file__).resolve().parent
commands = {
    "NumPy week2-sim.py": [sys.executable, str(week / "week2-sim.py")],
    "Rust debug": [str(week / "md/target/debug/md"), "run", "--out", "artifacts"],
    "Rust release": [str(week / "md/target/release/md"), "run", "--out", "artifacts"],
}
print("| Program | Median (s) | Range: min–max (s) |", flush=True)
print("| --- | ---: | ---: |", flush=True)
for name, command in commands.items():
    elapsed = []
    for trial in range(3):
        with tempfile.TemporaryDirectory(prefix="md-timing-") as directory:
            start = time.perf_counter()
            subprocess.run(command, cwd=directory, check=True, capture_output=True)
            elapsed.append(time.perf_counter() - start)
    print(f"| {name} | {statistics.median(elapsed):.3f} | "
          f"{min(elapsed):.3f}–{max(elapsed):.3f} |", flush=True)
