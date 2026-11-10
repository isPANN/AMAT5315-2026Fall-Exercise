from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def relative_error(matrix, reconstruction):
    return np.linalg.norm(matrix - reconstruction) / np.linalg.norm(matrix)


def discarded_errors(singular_values):
    squared = singular_values**2
    return np.sqrt(np.r_[np.cumsum(squared[::-1])[::-1], 0] / squared.sum())


def main():
    results = Path('results')
    matrix = np.fromfile(results / 'matrix.bin', '<f8').reshape(512, 512)
    singular_values = np.fromfile(results / 'singular-values.bin', '<f8')
    ranks = np.array([16, 64, 128, 256, 512])
    errors = [relative_error(matrix, np.fromfile(results / f'rank-{rank}.bin', '<f8').reshape(512, 512)) for rank in ranks]
    prediction = discarded_errors(singular_values)
    for rank, error in zip(ranks, errors):
        print(f'rank {rank}: measured {error:.8f}, predicted {prediction[rank]:.8f}')

    fig, (spectrum, error_plot) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    spectrum.plot(np.arange(1, 513), singular_values)
    spectrum.set(xlabel='Singular-value index', ylabel='Singular value', title='Gaussian matrix spectrum')
    error_plot.plot(np.arange(513), prediction, label='Discarded-spectrum prediction')
    error_plot.scatter(ranks, errors, color='tab:orange', zorder=3, label='Saved reconstructions')
    error_plot.axvline(256, color='0.35', linestyle='--', label='Two-factor storage = original (rank 256)')
    error_plot.axhline(0.1, color='tab:red', linestyle=':', label='10% relative error')
    error_plot.set(xlabel='Rank', ylabel='Relative Frobenius error', title='Truncation error', xlim=(0, 512), ylim=(0, 1.03))
    error_plot.legend(fontsize=8)
    for ax in (spectrum, error_plot):
        ax.grid(alpha=0.25)
    Path('artifacts').mkdir(exist_ok=True)
    fig.savefig('artifacts/svd.png', dpi=180)


if __name__ == '__main__':
    main()
