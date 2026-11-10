import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def enumerate_periodic(l, k):
    states = np.arange(1 << (l*l), dtype=np.uint64)
    spins = 2*((states[:, None] >> np.arange(l*l, dtype=np.uint64)) & 1).astype(np.int8)-1
    spins = spins.reshape(-1, l, l)
    energy = -(spins*np.roll(spins, -1, axis=1)).sum(axis=(1, 2)) - (spins*np.roll(spins, -1, axis=2)).sum(axis=(1, 2))
    weights = np.exp(-k*energy)
    z = weights.sum()
    mean = np.dot(weights, energy)/z
    variance = np.dot(weights, energy**2)/z - mean**2
    return float(np.log(z)/(l*l)), float(k*k*variance/(l*l))


def run(l, chi, k, out):
    subprocess.run(['target/release/thermo', '--method', 'trg', '--l', str(l), '--k', str(k), '--chi', str(chi), '--out', str(out)], check=True, stdout=subprocess.DEVNULL)
    return json.loads((out/'evidence.json').read_text())['g']


def main():
    summary = json.loads(Path('results/trg-critical-summary.json').read_text())
    sizes = np.array([64, 256, 1024])
    exact_tc = 2/np.log(1+np.sqrt(2))
    windows = {}
    for name, peaks in summary['peaks'].items():
        for start in (0, 1):
            slope, tc = np.polyfit(1/sizes[start:], np.array(peaks)[start:], 1)
            label = f'{name}-L{sizes[start]}-{sizes[-1]}'
            windows[label] = {'tc': float(tc), 'slope': float(slope), 'relative_error': float(abs(tc-exact_tc)/exact_tc)}
            print(f'{label}: Tc={tc:.9f}, relative error={abs(tc-exact_tc)/exact_tc:.4%}')
    checks = {}
    with tempfile.TemporaryDirectory(dir='.') as tmp:
        out = Path(tmp)
        for l, chi, tol in ((2,16,1e-8),(4,8,2e-3)):
            actual = run(l,chi,0.44,out)
            expected, _ = enumerate_periodic(l,0.44)
            error = abs(actual-expected)
            print(f'L={l} chi={chi}: g={actual:.12f}, enumeration={expected:.12f}, difference={error:.3g}')
            assert error < tol
            checks[f'l{l}-g'] = {'trg': actual, 'enumeration': expected, 'absolute_error': error}
        h=0.0025
        g=[run(2,16,k,out) for k in (0.44-h,0.44,0.44+h)]
        cv=0.44**2*(g[2]-2*g[1]+g[0])/h**2
        exact_cv=enumerate_periodic(2,0.44)[1]
        print(f'L=2: cV={cv:.12f}, variance={exact_cv:.12f}, difference={abs(cv-exact_cv):.3g}')
        assert abs(cv-exact_cv) < 1e-3
        checks['l2-cv'] = {'finite_difference': cv, 'variance': exact_cv, 'absolute_error': abs(cv-exact_cv)}
    assert max(summary['rank_relative_differences']) < 0.01
    assert max(summary['step_relative_differences']) < 0.01
    assert all(fit['relative_error'] < 0.01 for fit in windows.values())
    Path('results/trg-critical-verification.json').write_text(json.dumps({'exact_tc':float(exact_tc), 'fits':windows, 'enumeration':checks},indent=2)+'\n')


if __name__ == '__main__':
    main()
