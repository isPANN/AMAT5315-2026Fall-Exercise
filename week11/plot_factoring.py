#!/usr/bin/env python3
"""Plot saved factoring runs and extract a lowest-energy sample."""
from collections import Counter
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"


def main():
    plans = {order: [] for order in ("greedy", "treesa")}
    for k in range(2, 9):
        for order in plans:
            run = json.loads((RUNS / f"plan_k{k}_{order}" / "run.json").read_text())
            plans[order].append((k, run["summary"]["sc"]))
    run = json.loads((RUNS / "sample_f15" / "run.json").read_text())
    beta = float(run["parameters"]["--beta"])
    samples = run["evidence"]["samples"]
    lowest = min(samples, key=lambda s: s["energy"])
    config = ",".join(str((s + 1) // 2) for s in lowest["spins"])
    result = subprocess.run(
        ["pred", "extract", str(RUNS / "pred/factor_15.bundle.json"), "--config", config, "--json"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    print(f"Lowest sampled energy: {lowest['energy']}", flush=True)
    print(result.stdout)
    counts = Counter(sample["energy"] for sample in samples)
    summary = run["summary"]
    expected = len(samples) * math.exp(math.log(int(summary["g_0"])) - beta * summary["e_min"] - summary["log_z"])
    print(f"Expected ground-state samples: {expected}", flush=True)
    plot(plans, counts, summary["e_min"], expected, beta)


def plot(plans, counts, ground, expected, beta):
    script = """set terminal pngcairo size 1800,750 enhanced font 'Arial,18'
set output 'artifacts/factoring.png'
set multiplot layout 1,2 title 'Factoring via spin-glass contraction' font ',24'
set grid ytics lc rgb '#d9dfe5'
set border 3
set tics out nomirror
$greedy << EOD
"""
    script += "".join(f"{k} {sc}\n" for k, sc in plans["greedy"])
    script += "EOD\n$treesa << EOD\n"
    script += "".join(f"{k} {sc}\n" for k, sc in plans["treesa"])
    script += "EOD\n$hist << EOD\n"
    script += "".join(f"{energy} {count}\n" for energy, count in sorted(counts.items()))
    script += f"""EOD
set title 'Contraction planning'
set xlabel 'Factor bit length k (m = n = k)'
set ylabel 'Space complexity sc (log_2 entries)'
set xtics 1
set key top left
plot $greedy using 1:2 with linespoints lw 3 pt 7 ps 1.2 lc rgb '#367aa2' title 'greedy', $treesa using 1:2 with linespoints lw 3 pt 5 ps 1.2 lc rgb '#d28342' title 'treesa'
set title 'Factor 15: 2,000 samples; expected ground = {expected:.2f}' font ',16'
set xlabel 'Energy (beta = {beta:g})'
set ylabel 'Samples'
set xtics autofreq
set yrange [0:*]
set style fill solid 0.7
set boxwidth 0.8 absolute
set key top right opaque
set arrow 1 from first {ground}, graph 0 to first {ground}, first {expected} nohead lw 3 lc rgb '#d14b40'
plot $hist using 1:2 with boxes lc rgb '#367aa2' title 'Sampled energies', '+' using ({ground}):({expected}) with points pt 7 ps 1.7 lc rgb '#d14b40' title 'Expected ground states'
unset multiplot
"""
    (ROOT / "artifacts").mkdir(exist_ok=True)
    subprocess.run(["gnuplot"], input=script, text=True, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
