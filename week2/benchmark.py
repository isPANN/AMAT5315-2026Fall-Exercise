"""Print the force-method benchmark table and redraw scaling.png (run from week2/)."""
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sizes = [100, 400, 1600]
results = {n: {key: [] for key in ("naive", "cells", "speedup")} for n in sizes}
with tempfile.TemporaryDirectory(prefix="md-benchmark-") as temp:
    for n in sizes:
        for trial in range(3):
            times = {}
            order = ("naive", "cells") if trial % 2 == 0 else ("cells", "naive")
            for force in order:
                start = time.perf_counter()
                subprocess.run(
                    ["md", "run", "--force", force, "--n", str(n),
                     "--steps", "500", "--eq-steps", "100", "--out",
                     str(Path(temp) / f"{n}-{trial}-{force}")],
                    check=True, capture_output=True,
                )
                times[force] = time.perf_counter() - start
                results[n][force].append(times[force])
            results[n]["speedup"].append(times["naive"] / times["cells"])

print("| N | naive (s) | cells (s) | speedup = naive / cells |")
print("| ---: | ---: | ---: | ---: |")
for n in sizes:
    entries = []
    for key, values in results[n].items():
        median, low, high = statistics.median(values), min(values), max(values)
        entries.append(f"{median:.3f}× ({low:.3f}–{high:.3f}×)" if key == "speedup"
                       else f"{median:.6f} ({low:.6f}–{high:.6f})")
    print(f"| {n} | " + " | ".join(entries) + " |")

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7.4, 4.8), layout="constrained")
for force, color, marker in [("naive", "#b64b32", "o"), ("cells", "#176ca4", "s")]:
    values = [[t / 600 for t in results[n][force]] for n in sizes]
    medians = [statistics.median(v) for v in values]
    errors = [[m - min(v) for m, v in zip(medians, values)],
              [max(v) - m for m, v in zip(medians, values)]]
    ax.errorbar(sizes, medians, yerr=errors, label=force, color=color,
                marker=marker, markersize=6, linewidth=2, capsize=4)
ax.set(xscale="log", yscale="log", xlabel="Number of atoms, N",
       ylabel="Seconds per integration step", title="MD force-method scaling")
ax.set_xticks(sizes, labels=[str(n) for n in sizes])
ax.minorticks_off()
ax.grid(which="major", color="#dedede", linewidth=0.7)
ax.legend(title="Force method", frameon=False, loc="upper left")
fig.get_layout_engine().set(rect=(0, 0.06, 1, 0.94))
fig.text(0.5, 0.02, "Wall time / 600 steps · median and min–max of 3 runs",
         ha="center", fontsize=9, color="#555555")
fig.savefig("scaling.png", dpi=200)
