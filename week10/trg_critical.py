import json
from concurrent.futures import ThreadPoolExecutor
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from critical import parabola_peak, specific_heat, table
from verify_critical import fit_peak


def sweep(l, chi, ks, stage, temp):
    def sample(k):
        out = Path('runs/trg-critical') / f'l{l}-chi{chi}-k{k:.5f}' if stage == 'refined' and l == 1024 else temp / f'k{k:.5f}'
        subprocess.run(['target/release/thermo', '--method', 'trg', '--l', str(l), '--k', f'{k:.8f}', '--chi', str(chi), '--out', str(out)], check=True, stdout=subprocess.DEVNULL)
        d = json.loads((out / 'evidence.json').read_text())
        return {key: d[key] for key in ['l', 'chi', 'k', 'g', 'log_z', 'solver_seconds']}
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(sample, ks))
    table(Path('results') / f'trg-critical-{stage}-l{l}-chi{chi}.csv', rows)
    return np.array([row['g'] for row in rows])


def main():
    subprocess.run(['cargo', 'build', '--release', '--locked', '--bin', 'thermo'], check=True)
    Path('results').mkdir(exist_ok=True)
    Path('artifacts').mkdir(exist_ok=True)
    coarse_k = np.arange(160, 201) * 0.0025
    sizes = (64, 256, 1024)
    fig, (curve, fit) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    with tempfile.TemporaryDirectory(dir='.') as temp:
        for l in sizes:
            g = sweep(l, 16, coarse_k, 'coarse', Path(temp))
            c = specific_heat(coarse_k, g)
            index = np.argmax(c) + 1
            if index == 1 or index == len(coarse_k)-2:
                raise ValueError(f'L={l}: coarse peak is too close to sweep edge')
            center = coarse_k[index]
            ks = center + np.arange(-4, 5) * 0.00125
            curve.plot(1/coarse_k[1:-1], c, label=f'L={l}, χ=16')
            print(f'L={l}: coarse peak K={center:.5f}', flush=True)
            for chi in (16, 32):
                sweep(l, chi, ks, 'refined', Path(temp))
                print(f'L={l} χ={chi}: refined sweep complete', flush=True)
    peaks = {}
    for chi, stride in ((16, 1), (32, 1), (32, 2)):
        name = f'chi{chi}-step{0.00125*stride:.5f}'
        peaks[name] = [fit_peak(Path(f'results/trg-critical-refined-l{l}-chi{chi}.csv'), stride) for l in sizes]
    exact_tc = 2/np.log(1+np.sqrt(2))
    fits = {}
    for name, values in peaks.items():
        slope, tc = np.polyfit(1/np.array(sizes), values, 1)
        fits[name] = {'tc': float(tc), 'slope': float(slope), 'relative_error': float(abs(tc-exact_tc)/exact_tc)}
        print(f'{name}: peaks {values}; Tc={tc:.9f}; relative error={abs(tc-exact_tc)/exact_tc:.3%}', flush=True)
    rank_diff = np.abs(np.array(peaks['chi16-step0.00125'])/peaks['chi32-step0.00125']-1)
    step_diff = np.abs(np.array(peaks['chi32-step0.00250'])/peaks['chi32-step0.00125']-1)
    print(f'relative peak differences: rank {rank_diff}, step {step_diff}', flush=True)
    slope = fits['chi32-step0.00125']['slope']
    tc = fits['chi32-step0.00125']['tc']
    x = np.linspace(0, 1/64, 100)
    fit.plot(x, tc+slope*x, label=f'Linear fit: $T_c$ = {tc:.6f}')
    fit.scatter(1/np.array(sizes), peaks['chi32-step0.00125'], label='χ=32 peaks', zorder=3)
    fit.scatter([0], [tc], marker='x', s=70, label='Extrapolated intercept')
    curve.set(xlabel='Temperature T', ylabel='Specific heat per spin $c_V$', title='Periodic Ising (TRG)')
    fit.set(xlabel='1/L', ylabel='$T_{peak}$', title='Peak extrapolation')
    for ax in (curve, fit):
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig('artifacts/trg-critical.png', dpi=180)
    Path('results/trg-critical-summary.json').write_text(json.dumps({'exact_tc': float(exact_tc), 'peaks': peaks, 'fits': fits, 'rank_relative_differences': rank_diff.tolist(), 'step_relative_differences': step_diff.tolist()}, indent=2)+'\n')


if __name__ == '__main__':
    main()
