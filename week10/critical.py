import csv
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def specific_heat(k, g):
    return k[1:-1]**2 * (g[2:] - 2*g[1:-1] + g[:-2]) / (k[1] - k[0])**2


def parabola_peak(t, c):
    a, b, d = np.polyfit(t, c, 2)
    if a >= 0:
        raise ValueError('peak parabola is not concave')
    peak = -b / (2*a)
    if not min(t) <= peak <= max(t):
        raise ValueError('peak is outside the three fitted points')
    return peak, np.polyval([a, b, d], peak)


def run(l, chi, k, out):
    subprocess.run(['target/release/thermo', '--method', 'boundary', '--l', str(l), '--k', f'{k:.8f}', '--chi', str(chi), '--out', str(out)], check=True, stdout=subprocess.DEVNULL)
    return json.loads((out / 'evidence.json').read_text())


def table(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['l', 'chi', 'k', 'g', 'log_z', 'solver_seconds'])
        writer.writeheader()
        writer.writerows(rows)


def sweep(l, chi, ks, stage, temp):
    rows = []
    for k in ks:
        out = Path('runs/critical') / f'l{l}-chi{chi}-k{k:.5f}' if stage == 'refined' and l == 64 else temp
        d = run(l, chi, k, out)
        rows.append({key: d[key] for key in ['l', 'chi', 'k', 'g', 'log_z', 'solver_seconds']})
    table(Path('results') / f'critical-{stage}-l{l}-chi{chi}.csv', rows)
    return np.array([row['g'] for row in rows])


def main():
    subprocess.run(['cargo', 'build', '--release', '--locked', '--bin', 'thermo'], check=True)
    Path('results').mkdir(exist_ok=True)
    Path('artifacts').mkdir(exist_ok=True)
    coarse_k = np.arange(160, 201) * 0.0025
    peaks = {}
    fig, (curve, fit) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    with tempfile.TemporaryDirectory(dir='.') as temporary:
        for l in (16, 32, 64):
            coarse_g = sweep(l, 16, coarse_k, 'coarse', Path(temporary))
            coarse_c = specific_heat(coarse_k, coarse_g)
            index = np.argmax(coarse_c) + 1
            if index == 1 or index == len(coarse_k)-2:
                raise ValueError(f'L={l}: coarse peak is too close to sweep edge')
            center = coarse_k[index]
            refined_k = center + np.arange(-4, 5) * 0.00125
            curve.plot(1/coarse_k[1:-1], coarse_c, label=f'L={l}, χ=16')
            for chi in (16, 32):
                g = sweep(l, chi, refined_k, 'refined', Path(temporary))
                c = specific_heat(refined_k, g)
                i = np.argmax(c)
                if i == 0 or i == len(c)-1:
                    raise ValueError(f'L={l}, chi={chi}: refined peak not bracketed')
                t = 1 / refined_k[1:-1]
                peaks[l, chi] = parabola_peak(t[i-1:i+2], c[i-1:i+2])
                print(f'L={l} chi={chi}: T_peak={peaks[l,chi][0]:.9f}, c_peak={peaks[l,chi][1]:.9f}', flush=True)
                if chi == 32:
                    curve.scatter(t[i], c[i], s=20)
    sizes = np.array([16, 32, 64])
    temperatures = np.array([peaks[l, 32][0] for l in sizes])
    a, tc = np.polyfit(1/sizes, temperatures, 1)
    x = np.linspace(0, 1/16, 100)
    fit.plot(x, tc+a*x, label=f'Linear fit: $T_c$ = {tc:.6f}')
    fit.scatter(1/sizes, temperatures, label='χ=32 peaks', zorder=3)
    fit.scatter([0], [tc], marker='x', s=70, label='Extrapolated intercept')
    curve.set(xlabel='Temperature T', ylabel='Specific heat per spin $c_V$', title='Open-boundary Ising')
    fit.set(xlabel='1/L', ylabel='$T_{peak}$', title='Peak extrapolation')
    for ax in (curve, fit):
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig('artifacts/boundary-critical.png', dpi=180)
    summary = {'peaks': {f'l{l}-chi{chi}': {'temperature': float(t), 'specific_heat': float(c)} for (l, chi), (t, c) in peaks.items()}, 'chi32_fit': {'tc': float(tc), 'slope': float(a)}}
    Path('results/critical-peaks.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'chi=32 extrapolated T_c={tc:.9f}, slope={a:.9f}')


if __name__ == '__main__':
    main()
