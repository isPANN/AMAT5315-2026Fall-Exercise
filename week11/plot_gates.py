#!/usr/bin/env python3
"""Regenerate the gadget figure using tnsample and gnuplot."""
import csv
import io
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
output = subprocess.run(
    ["cargo", "run", "--quiet", "--", "--mode", "gadgets", "--seed", "2026", "--out", "target/plot-run"],
    cwd=ROOT, check=True, capture_output=True, text=True,
).stdout
rows = list(csv.DictReader(io.StringIO(output), delimiter="\t"))
panels = [
    ("AND", "AND", "Bits (a,b,z)"),
    ("OR", "OR", "Bits (a,b,z)"),
    ("NOT", "NOT", "Bits (a,z)"),
    ("COPY", "COPY", "Bits (a,z)"),
    ("SET0", "SET0", "Bit (z)"),
    ("SET1", "SET1", "Bit (z)"),
    ("XOR", "XOR", "Bits (a,b,z); minimized over w"),
    ("circuit", "AND + NOT + SET1 circuit", "Bits (a,b,t,out)"),
]
plot = """set terminal pngcairo size 2000,1600 enhanced font 'Arial,17'
set output 'artifacts/gates.png'
set multiplot layout 4,2 rowsfirst title 'Gadget and circuit energies' font ',25'
set style fill solid 0.85 border -1
set boxwidth 0.72
set grid ytics lc rgb '#d9dfe5'
set border 3
set tics out nomirror
set ylabel 'Energy'
"""
for gate, title, xlabel in panels:
    plot += f"set title '{title}'\nset xlabel '{xlabel}'\n"
    plot += "set yrange [-5:0]\n" if gate == "XOR" else "set yrange [*:*]\n"
    if gate == "circuit":
        plot += "set xtics rotate by -45 font ',12'\n"
    plot += "$data << EOD\n"
    for row in rows:
        name, bits = row["metric"].split("/")
        if name == gate:
            plot += f'"{bits}" {row["value"]}\n'
    plot += "EOD\nplot $data using 0:2:xticlabels(1) with boxes lc rgb '#367aa2' notitle\n"
plot += "unset multiplot\n"
(ROOT / "artifacts").mkdir(exist_ok=True)
subprocess.run(["gnuplot"], input=plot, text=True, cwd=ROOT, check=True)
