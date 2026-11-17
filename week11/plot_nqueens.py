#!/usr/bin/env python3
"""Check saved N-queens runs and plot their counts and sample frequencies."""
from collections import Counter
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
known = {4: 2, 5: 10, 6: 4, 7: 40, 8: 92}
counts = []
for n, expected in known.items():
    run = json.loads((RUNS / f"count_nq{n}" / "run.json").read_text())
    count = int(run["summary"]["count"])
    if count != expected:
        raise ValueError(f"N={n}: contracted count {count} != {expected}")
    counts.append((n, count))
    print(f"N={n}: count {count} matches")

frequencies = None
for path in sorted(RUNS.glob("sample_nq*/run.json")):
    run = json.loads(path.read_text())
    n = int(run["parameters"]["--n"])
    samples = run["evidence"]["samples"]
    frequencies_for_run = Counter()
    for number, sample in enumerate(samples, start=1):
        board = sample["board"]
        if len(board) != n or any(len(row) != n or any(x not in (0, 1) for x in row) or sum(row) != 1 for row in board):
            raise ValueError(f"{path}: invalid rows in sample {number}")
        columns = tuple(row.index(1) for row in board)
        if len(set(columns)) != n or len({row - col for row, col in enumerate(columns)}) != n or len({row + col for row, col in enumerate(columns)}) != n:
            raise ValueError(f"{path}: attacking queens in sample {number}")
        frequencies_for_run[columns] += 1
    print(f"N={n}: {len(samples)} valid boards in {path.relative_to(ROOT)}")
    if path == RUNS / "sample_nq6" / "run.json":
        frequencies = frequencies_for_run

if frequencies is None:
    raise ValueError("missing runs/sample_nq6/run.json")
print("N=6 frequencies:", {tuple(c + 1 for c in board): count for board, count in sorted(frequencies.items())})

script = """set terminal pngcairo size 1800,800 enhanced font 'Arial,18'
set output 'artifacts/nqueens.png'
set multiplot layout 1,2 title 'N-queens counts and uniform samples' font ',24'
set grid ytics lc rgb '#d9dfe5'
set border 3
set tics out nomirror
$counts << EOD
"""
script += "".join(f"{n} {count}\n" for n, count in counts)
script += "EOD\n$freq << EOD\n"
for board, count in sorted(frequencies.items()):
    label = ''.join(str(col + 1) for col in board)
    script += f'"{label}" {count}\n'
script += f"""EOD
set title 'Exact solution counts'
set xlabel 'Board size N'
set ylabel 'Valid boards'
set xtics 1
set xrange [3.5:8.5]
set yrange [0:105]
set style fill solid 0.8 border -1
set boxwidth 0.6
set key off
plot $counts using 1:2 with boxes lc rgb '#367aa2', $counts using 1:2:2 with labels offset 0,0.8 font ',15'
set title 'N = 6: 200 sampled boards'
set xlabel 'Queen columns by row (1-indexed)'
set ylabel 'Sample frequency'
set yrange [0:65]
set xrange [*:*]
set xtics autofreq
set key top right opaque
plot $freq using 0:2:xticlabels(1) with boxes lc rgb '#367aa2' title 'Samples', $freq using 0:2:2 with labels offset 0,0.7 font ',15' notitle, {sum(frequencies.values()) / known[6]} with lines lw 2 dt 2 lc rgb '#d14b40' title 'Uniform expectation'
unset multiplot
"""
(ROOT / "artifacts").mkdir(exist_ok=True)
subprocess.run(["gnuplot"], input=script, text=True, cwd=ROOT, check=True)
