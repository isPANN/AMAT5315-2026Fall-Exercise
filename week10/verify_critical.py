import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from critical import parabola_peak, run, specific_heat


def enumerate_ising(l, k):
    states = np.arange(1 << (l*l), dtype=np.uint64)
    spins = 2 * ((states[:, None] >> np.arange(l*l, dtype=np.uint64)) & 1).astype(np.int8) - 1
    spins = spins.reshape(-1, l, l)
    energy = -(spins[:, :, :-1] * spins[:, :, 1:]).sum(axis=(1, 2)) - (spins[:, :-1, :] * spins[:, 1:, :]).sum(axis=(1, 2))
    weights = np.exp(-k*energy)
    z = weights.sum()
    mean = np.dot(weights, energy) / z
    variance = np.dot(weights, energy**2) / z - mean**2
    return np.log(z)/(l*l), k*k*variance/(l*l)


def fit_peak(path, stride):
    with path.open() as f:
        rows = list(csv.DictReader(f))[::stride]
    k = np.array([float(row['k']) for row in rows])
    g = np.array([float(row['g']) for row in rows])
    c = specific_heat(k, g)
    i = np.argmax(c)
    if i == 0 or i == len(c)-1:
        raise ValueError(f'peak not bracketed in {path} with stride {stride}')
    return float(parabola_peak((1/k[1:-1])[i-1:i+2], c[i-1:i+2])[0])


def main():
    sizes = np.array([16, 32, 64])
    peaks = {}
    for chi, stride in ((16, 1), (32, 1), (32, 2)):
        name = f'chi{chi}-step{0.00125*stride:.5f}'
        peaks[name] = [fit_peak(Path(f'results/critical-refined-l{l}-chi{chi}.csv'), stride) for l in sizes]
    exact_tc = 2/np.log(1+np.sqrt(2))
    fits = {}
    for name, values in peaks.items():
        slope, tc = np.polyfit(1/sizes, values, 1)
        fits[name] = {'tc': float(tc), 'slope': float(slope), 'relative_error': float(abs(tc-exact_tc)/exact_tc)}
        print(f'{name}: peaks {values}; Tc={tc:.9f}; relative error={abs(tc-exact_tc)/exact_tc:.3%}')
    rank_diff = np.abs(np.array(peaks['chi16-step0.00125'])/peaks['chi32-step0.00125']-1)
    step_diff = np.abs(np.array(peaks['chi32-step0.00250'])/peaks['chi32-step0.00125']-1)
    print('relative peak differences, rank:', rank_diff, 'step:', step_diff)
    assert np.all(rank_diff < 0.01) and np.all(step_diff < 0.01)
    assert fits['chi32-step0.00125']['relative_error'] < 0.01 and fits['chi32-step0.00250']['relative_error'] < 0.01

    checks = {}
    with tempfile.TemporaryDirectory(dir='.') as tmp:
        out = Path(tmp)
        for k in (0.0, 0.44, 0.7):
            expected, cv = enumerate_ising(4, k)
            actual = run(4, 8, k, out)['g']
            error = abs(actual-expected)
            print(f'4x4 K={k}: g solver={actual:.12f}, enumeration={expected:.12f}, error={error:.3g}')
            assert error < 1e-8
            checks[str(k)] = {'g_solver': actual, 'g_enumerated': float(expected), 'absolute_error': float(error)}
        h = 0.0025
        g = [run(4, 8, k, out)['g'] for k in (0.44-h, 0.44, 0.44+h)]
        cv_fd = 0.44**2 * (g[2]-2*g[1]+g[0])/h**2
        cv_exact = enumerate_ising(4, 0.44)[1]
        print(f'4x4 K=0.44: cV finite difference={cv_fd:.12f}, variance={cv_exact:.12f}, error={abs(cv_fd-cv_exact):.3g}')
        assert abs(cv_fd-cv_exact) < 1e-3
        checks['cv'] = {'finite_difference': cv_fd, 'variance': float(cv_exact), 'absolute_error': float(abs(cv_fd-cv_exact))}
    Path('results/critical-verification.json').write_text(json.dumps({'exact_tc': float(exact_tc), 'peaks': peaks, 'fits': fits, 'rank_relative_differences': rank_diff.tolist(), 'step_relative_differences': step_diff.tolist(), 'enumeration': checks}, indent=2)+'\n')


if __name__ == '__main__':
    main()
